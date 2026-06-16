"""Optimized inference engine for the hierarchical Vietnamese spell corrector.

Builds on correct.SpellCorrector and adds the speed-oriented packaging knobs:

  * device auto-detect (cuda > mps > cpu)
  * fp16 weights on cuda (~2x throughput, half the VRAM)
  * int8 dynamic quantization of the Linear layers on cpu
  * torch.inference_mode + thread tuning
  * optional torch.compile
  * a warmup pass so the first real request isn't slow

Usage as a library:
    from infer import FastSpellCorrector
    sc = FastSpellCorrector("spelling_corr/best_327000.pt")   # picks best device + opt
    sc(["Cơn bảo dag đổ bôj vào đất lền ."])

CLI / benchmark:
    python infer.py --checkpoint spelling_corr/best_327000.pt --text "Tôi đi hocj ."
    python infer.py --checkpoint spelling_corr/best_327000.pt --benchmark
"""

import argparse
import json
import time

import os

import torch
import torch.nn as nn

from correct import SpellCorrector

# Hierarchical checkpoint published on the HF Hub — auto-downloaded (and cached)
# when no local file is found, so a fresh clone runs with zero manual steps.
HF_CKPT_REPO = "ANZ-Innovation/spell_correction_v1"
HF_CKPT_FILE = "best_327000.pt"
DEFAULT_CKPT = os.path.join("spelling_corr", HF_CKPT_FILE)


def pick_device(prefer=None):
    if prefer and prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_checkpoint(ckpt=None):
    """Return a local path to the checkpoint.

    - existing file path        -> used as-is
    - missing '*.pt' path       -> downloaded from HF_CKPT_REPO
    - a 'owner/repo' id (no .pt) -> that repo is downloaded instead
    """
    ckpt = ckpt or DEFAULT_CKPT
    if os.path.exists(ckpt):
        return ckpt
    from huggingface_hub import hf_hub_download
    repo = HF_CKPT_REPO if ckpt.endswith(".pt") else ckpt
    print(f"[info] '{ckpt}' not found locally — downloading {HF_CKPT_FILE} "
          f"from {repo} (cached for next runs)…")
    return hf_hub_download(repo_id=repo, filename=HF_CKPT_FILE)


class FastSpellCorrector(SpellCorrector):
    """SpellCorrector with inference-time optimizations applied after load.

    precision:
      "auto"  -> bf16/fp16 on cuda, int8 dynamic quant on cpu, fp32 on mps
      "fp16"  -> half weights (cuda/mps)
      "bf16"  -> bfloat16 weights (cuda Ampere+; more numerically robust)
      "int8"  -> dynamic int8 quantization (cpu only)
      "fp32"  -> no change
    """

    def __init__(self, checkpoint=None, device="auto", precision="auto",
                 compile=False, warmup=True, num_threads=None, **kw):
        checkpoint = resolve_checkpoint(checkpoint)
        device = pick_device(device)
        if device == "cpu" and num_threads:
            torch.set_num_threads(num_threads)
        if device == "cuda":
            self._tune_cuda()
        super().__init__(checkpoint, device=device, **kw)

        self.precision = self._resolve_precision(precision, device)
        if self.precision == "fp16":
            self.model = self.model.half()
        elif self.precision == "bf16":
            self.model = self.model.bfloat16()
        elif self.precision == "int8":
            self.model = self._try_quantize(self.model, explicit=precision == "int8")

        if compile:
            # reduce-overhead enables CUDA graphs (kills per-call launch latency);
            # the encoder re-traces per new (B,W,C) shape, then runs graph-fast.
            mode = "reduce-overhead" if device == "cuda" else None
            self.model = torch.compile(self.model, mode=mode)

        if warmup:
            self._warmup()

    @staticmethod
    def _tune_cuda():
        """Standard NVIDIA inference switches (set before the model loads)."""
        torch.backends.cudnn.benchmark = True          # autotune conv/attn kernels
        torch.set_float32_matmul_precision("high")     # TF32 for leftover fp32 matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Prefer fused flash / memory-efficient attention for the encoders.
        for fn in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
            try:
                getattr(torch.backends.cuda, fn)(True)
            except Exception:
                pass

    def _try_quantize(self, model, explicit=False):
        """Dynamic int8 needs a CPU quant engine (fbgemm/qnnpack). Some torch
        builds ship without one — fall back to fp32 instead of crashing."""
        engines = [e for e in torch.backends.quantized.supported_engines
                   if e != "none"]
        if not engines:
            if explicit:
                print("[warn] no int8 quant engine available; using fp32")
            self.precision = "fp32"
            return model
        torch.backends.quantized.engine = engines[0]
        try:
            qmodel = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8)
            # Some builds quantize fine but break the transformer fast path at
            # forward time — verify with a dummy pass before committing to it.
            with torch.inference_mode():
                qmodel(torch.ones(1, 2, dtype=torch.long),
                       torch.ones(1, 2, 4, dtype=torch.long))
            return qmodel
        except Exception as e:
            msg = str(e).splitlines()[-1]
            print(f"[warn] int8 unsupported on this build ({msg}); using fp32")
            self.precision = "fp32"
            return model

    @staticmethod
    def _resolve_precision(precision, device):
        if precision != "auto":
            return precision
        if device == "cuda":
            # bf16 on Ampere+ (cc>=8) — same speed as fp16 with no overflow risk;
            # fp16 on older cards (Turing/Volta) which lack fast bf16.
            major = torch.cuda.get_device_capability()[0]
            return "bf16" if major >= 8 else "fp16"
        if device == "cpu":
            return "int8"
        return "fp32"  # mps: fp16 is flaky for masked attention, keep fp32

    def _warmup(self):
        # A few shapes so torch.compile traces the common (B,W,C) buckets and
        # cuDNN autotune settles before the first real request.
        try:
            for toks in ([["xin", "chào"]],
                         [["xin", "chào", "thế", "giới"]] * 8):
                self._run_batch(toks)
        except Exception:
            pass

    @torch.inference_mode()
    def _run_batch(self, batch_tokens):
        return super()._run_batch(batch_tokens)

    def info(self):
        n = sum(p.numel() for p in self.model.parameters())
        return {"device": self.device, "precision": self.precision,
                "params": n, "max_len": self.model.max_len,
                "word_vocab": len(self.word_vocab),
                "char_vocab": len(self.char_vocab)}


def benchmark(checkpoint=None, n=2000, batch_size=64, device="auto",
              precision="auto", compile=False):
    """Compare the fp32 baseline against the optimized engine on synthetic text.
    Use n >> batch_size (and several batches) so large-batch throughput is real
    and not dominated by one partial batch."""
    base_sent = "Tôi đi hocj ở truờng đai hocj môic ngàu vgrong tuần ."
    sents = [base_sent for _ in range(n)]

    def run(corr, label):
        corr(sents[: min(batch_size, n)])  # warm one full batch
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.time()
        corr(sents)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t
        print(f"{label:<22} {dt*1000:8.1f} ms  |  {n/dt:8.1f} sent/s  |  {dt/n*1000:6.3f} ms/sent")
        return dt

    dev = pick_device(device)
    print(f"device = {dev}, n = {n}, batch_size = {batch_size}, "
          f"batches = {-(-n // batch_size)}\n")
    checkpoint = resolve_checkpoint(checkpoint)
    base = SpellCorrector(checkpoint, device=dev, batch_size=batch_size)
    t_base = run(base, "baseline (fp32)")
    fast = FastSpellCorrector(checkpoint, device=dev, batch_size=batch_size,
                              precision=precision, compile=compile)
    print("optimized:", fast.info())
    t_fast = run(fast, f"optimized ({fast.precision}{'+compile' if compile else ''})")
    print(f"\nspeedup: {t_base / t_fast:.2f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="local .pt; nếu thiếu sẽ tự tải từ HF (ANZ-Innovation/spell_correction_v1)")
    p.add_argument("--text", action="append", default=None)
    p.add_argument("--file", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="auto",
                   choices=["auto", "fp16", "bf16", "int8", "fp32"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--n", type=int, default=2000, help="số câu cho benchmark")
    args = p.parse_args()

    if args.benchmark:
        benchmark(args.checkpoint, n=args.n, batch_size=args.batch_size,
                  device=args.device, precision=args.precision, compile=args.compile)
        return

    sc = FastSpellCorrector(args.checkpoint, device=args.device,
                            precision=args.precision, compile=args.compile,
                            batch_size=args.batch_size, threshold=args.threshold,
                            iterations=args.iterations)
    print("engine:", sc.info())

    if args.file:
        sents = [l.rstrip("\n") for l in open(args.file, encoding="utf-8")]
        for res in sc(sents):
            print(json.dumps(res, ensure_ascii=False))
        return

    for res in sc(args.text or ["Cơn bảo dag đổ bôj vào đất lền ."]):
        print(f"input : {res['input']}")
        for e in res["errors"]:
            print(f"  từ {e['word_index']}: {e['token']} -> {e['suggestion']}"
                  f" (p={e['confidence']}, vòng {e['iteration']})")
        print(f"output: {res['output']}\n")


if __name__ == "__main__":
    main()
