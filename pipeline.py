"""Two correction pipelines behind one interface.

  mode="best"   : best_327000.pt detects AND corrects (the hierarchical model
                  alone — fast, formatting-preserving, token-level edits).

  mode="hybrid" : best_327000.pt only *detects* which sentences contain errors;
                  the flagged ones are rewritten by the seq2seq corrector
                  protonx-models/protonx-legal-tc (T5). Clean sentences skip
                  the expensive generation step, which also stops the seq2seq
                  model from hallucinating edits into correct text.

Both expose __call__(sentences) -> list[{"input","output","errors","mode"}],
so serve.py / app.py treat them identically.
"""

import difflib

import torch

from infer import FastSpellCorrector, pick_device

SEQ2SEQ_MODEL = "protonx-models/protonx-legal-tc"


def _diff_errors(src, tgt):
    """Word-level diff src->tgt (case-insensitive) as error entries the UI can
    render. word_index points into src.split()."""
    a, b = src.split(), tgt.split()
    sm = difflib.SequenceMatcher(a=[w.lower() for w in a], b=[w.lower() for w in b])
    errs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):                       # aligned 1:1 swaps
                errs.append({"word_index": i1 + k, "token": a[i1 + k],
                             "suggestion": b[j1 + k]})
        else:                                             # merge/split/insert/delete
            errs.append({"word_index": i1,
                         "token": " ".join(a[i1:i2]),
                         "suggestion": " ".join(b[j1:j2])})
    return errs


class Seq2SeqCorrector:
    """protonx-models/protonx-legal-tc — T5 whole-sentence corrector."""

    def __init__(self, model_name=SEQ2SEQ_MODEL, device="auto", precision="auto",
                 num_beams=4, max_length=256, batch_size=32):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.device = pick_device(device)
        self.num_beams = num_beams
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
        self.precision = "fp16" if (precision in ("auto", "fp16") and self.device == "cuda") else "fp32"
        if self.precision == "fp16":
            model = model.half()
        self.model = model.to(self.device)

    @torch.inference_mode()
    def generate(self, sentences):
        """Raw seq2seq rewrite for every sentence (no detector gate)."""
        out = [None] * len(sentences)
        for k in range(0, len(sentences), self.batch_size):
            chunk = sentences[k:k + self.batch_size]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_length).to(self.device)
            gen = self.model.generate(**enc, max_length=self.max_length,
                                      num_beams=self.num_beams)
            for i, text in enumerate(self.tokenizer.batch_decode(gen, skip_special_tokens=True)):
                out[k + i] = text.strip()
        return out

    def __call__(self, sentences, **kw):
        results = []
        for src, tgt in zip(sentences, self.generate(sentences)):
            results.append({"input": src, "output": tgt,
                            "errors": _diff_errors(src, tgt), "mode": "seq2seq"})
        return results

    def info(self):
        n = sum(p.numel() for p in self.model.parameters())
        return {"model": SEQ2SEQ_MODEL, "device": self.device,
                "precision": self.precision, "params": n,
                "num_beams": self.num_beams}


class HybridCorrector:
    """Detect with best_327000.pt, correct flagged sentences with protonx-legal-tc."""

    def __init__(self, checkpoint=None, device="auto", precision="auto",
                 threshold=0.5, num_beams=4, seq2seq_device=None, **kw):
        self.detector = FastSpellCorrector(checkpoint, device=device,
                                           precision=precision, threshold=threshold, **kw)
        self.seq2seq = Seq2SeqCorrector(device=seq2seq_device or device,
                                        precision=precision, num_beams=num_beams)

    @property
    def threshold(self):
        return self.detector.threshold

    @threshold.setter
    def threshold(self, v):
        self.detector.threshold = v

    def _detect_flags(self, sentences):
        """Return per-sentence list of detected error positions (pure detection,
        before any correction)."""
        det = self.detector
        flags = []
        bs = det.batch_size
        for k in range(0, len(sentences), bs):
            chunk = sentences[k:k + bs]
            batch_tokens = [det._split_with_map(s)[0] for s in chunk]
            # blank sentences -> no tokens; _run_batch needs non-empty rows
            idx = [i for i, t in enumerate(batch_tokens) if t]
            per = det._run_batch([batch_tokens[i] for i in idx]) if idx else []
            res = [[] for _ in chunk]
            for i, errs in zip(idx, per):
                res[i] = errs
            flags.extend(res)
        return flags

    def __call__(self, sentences, **kw):
        flags = self._detect_flags(sentences)
        todo = [i for i, f in enumerate(flags) if f]               # detector gate
        fixed = self.seq2seq.generate([sentences[i] for i in todo]) if todo else []
        fixed_map = dict(zip(todo, fixed))

        results = []
        for i, src in enumerate(sentences):
            if i in fixed_map:
                tgt = fixed_map[i]
                results.append({"input": src, "output": tgt,
                                "errors": _diff_errors(src, tgt),
                                "detected": len(flags[i]), "mode": "hybrid"})
            else:
                results.append({"input": src, "output": src, "errors": [],
                                "detected": 0, "mode": "hybrid"})
        return results

    def info(self):
        return {"mode": "hybrid", "detector": self.detector.info(),
                "corrector": self.seq2seq.info()}


def build_corrector(mode, checkpoint=None, device="auto", precision="auto",
                    threshold=0.5, num_beams=4, **kw):
    """mode: 'best' -> hierarchical model only; 'hybrid' -> detect + protonx-legal-tc."""
    if mode == "best":
        return FastSpellCorrector(checkpoint, device=device, precision=precision,
                                  threshold=threshold, **kw)
    if mode == "hybrid":
        return HybridCorrector(checkpoint, device=device, precision=precision,
                               threshold=threshold, num_beams=num_beams, **kw)
    raise ValueError(f"unknown mode {mode!r} (use 'best' or 'hybrid')")
