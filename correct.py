"""Inference: detect spelling errors and suggest corrections.

As a library:
    from correct import SpellCorrector
    sc = SpellCorrector("checkpoints/best.pt")
    results = sc(["Cơn bảo dag đổ bôj vào đất lền .", "Câu này đúng chính tả ."])
    # -> list of dicts: {"input", "output", "errors": [...]}

As a CLI:
    python correct.py --checkpoint checkpoints/best.pt --text "Cơn bảo dag đổ bôj ."
    python correct.py --checkpoint checkpoints/best.pt --file sentences.txt > out.jsonl
"""

import argparse
import json
import sys

import torch

from data import tokenize
from model import HierarchicalSC


class SpellCorrector:
    def __init__(self, checkpoint, device=None, batch_size=64,
                 threshold=0.5, max_word_len=24, iterations=2):
        """iterations: maximum correction passes. After a pass applies fixes,
        the cleaner context can expose errors that were masked by surrounding
        noise, so the changed sentences are re-checked. Stops early as soon as
        a pass changes nothing (clean sentences only ever run once)."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.threshold = threshold
        self.max_word_len = max_word_len
        self.iterations = iterations
        model, self.word_vocab, self.char_vocab = HierarchicalSC.load(checkpoint, map_location=device)
        self.model = model.to(device).eval()

    @torch.no_grad()
    def __call__(self, sentences, iterations=None):
        """sentences: list[str]. Returns one dict per sentence:
        {"input": str, "output": str,
         "errors": [{"token_index", "word_index", "token", "suggestion",
                     "confidence", "iteration"}]}
        """
        n_iter = self.iterations if iterations is None else iterations
        results = [None] * len(sentences)
        todo = []
        for i, text in enumerate(sentences):
            tokens, word_of = self._split_with_map(text)
            if tokens:
                todo.append((i, {"orig": tokens, "cur": list(tokens),
                                 "word_of": word_of, "errors": {}}))
            else:
                results[i] = {"input": text, "output": text, "errors": []}

        active = todo
        for it in range(1, max(1, n_iter) + 1):
            if not active:
                break
            changed = []
            for k in range(0, len(active), self.batch_size):
                chunk = active[k:k + self.batch_size]
                for (i, s), errors in zip(chunk, self._run_batch([s["cur"] for _, s in chunk])):
                    dirty = False
                    for e in errors:
                        ti = e["token_index"]
                        if e["suggestion"] == s["cur"][ti]:
                            continue
                        s["cur"][ti] = e["suggestion"]
                        s["errors"][ti] = {"token_index": ti,
                                           "word_index": s["word_of"][ti],
                                           "token": s["orig"][ti],
                                           "suggestion": e["suggestion"],
                                           "confidence": e["confidence"],
                                           "iteration": it}
                        dirty = True
                    if dirty:
                        changed.append((i, s))
            active = changed  # re-check only sentences a pass actually edited

        for i, s in todo:
            # drop positions the model eventually reverted to the original
            errors = [e for ti, e in sorted(s["errors"].items())
                      if s["cur"][ti] != s["orig"][ti]]
            results[i] = {"input": sentences[i],
                          "output": self._reconstruct(sentences[i], s["orig"], s["word_of"], errors),
                          "errors": errors}
        return results

    def _split_with_map(self, text):
        """Tokenize while remembering which whitespace word each token came
        from, so we can rebuild the output preserving untouched words."""
        tokens, word_of = [], []
        for wi, raw in enumerate(text.split()):
            for tok in tokenize(raw):
                tokens.append(tok)
                word_of.append(wi)
        L = self.model.max_len
        return tokens[:L], word_of[:L]

    def _run_batch(self, batch_tokens):
        W = max(len(t) for t in batch_tokens)
        C = min(max(max(len(w) for w in t) for t in batch_tokens), self.max_word_len)
        word_ids = torch.zeros(len(batch_tokens), W, dtype=torch.long)
        char_ids = torch.zeros(len(batch_tokens), W, C, dtype=torch.long)
        for b, toks in enumerate(batch_tokens):
            for i, tok in enumerate(toks):
                word_ids[b, i] = self.word_vocab[tok]
                for j, ch in enumerate(tok[:C]):
                    char_ids[b, i, j] = self.char_vocab[ch]
        det_logits, corr_logits = self.model(word_ids.to(self.device), char_ids.to(self.device))
        err_prob = det_logits.softmax(-1)[..., 1].cpu()
        corr_pred = corr_logits.argmax(-1).cpu()
        out = []
        for b, toks in enumerate(batch_tokens):
            errors = []
            for i, tok in enumerate(toks):
                p = err_prob[b, i].item()
                if p >= self.threshold:
                    errors.append({"token_index": i, "token": tok,
                                   "suggestion": self.word_vocab.id2word[corr_pred[b, i].item()],
                                   "confidence": round(p, 4)})
            out.append(errors)
        return out

    @staticmethod
    def _reconstruct(text, tokens, word_of, errors):
        """Rebuild the sentence: untouched words keep their original form
        (case, attached punctuation); corrected words are rebuilt from their
        lowercase tokens with suggestions substituted."""
        fixed = {e["token_index"]: e["suggestion"] for e in errors}
        if not fixed:
            return text
        words = text.split()
        word_tokens = {}  # word_index -> [token or suggestion]
        touched = set()
        for ti, tok in enumerate(tokens):
            wi = word_of[ti]
            word_tokens.setdefault(wi, []).append(fixed.get(ti, tok))
            if ti in fixed:
                touched.add(wi)
        out = []
        for wi, raw in enumerate(words):
            if wi in touched:
                out.append(" ".join(word_tokens[wi]))
            else:
                out.append(raw)
        return " ".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", action="append", default=None,
                   help="sentence to check; repeat the flag for several")
    p.add_argument("--file", default=None,
                   help="file with one sentence per line; prints JSON lines to stdout")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--iterations", type=int, default=2,
                   help="max correction passes; re-checks a sentence after fixing it (1 = single pass)")
    args = p.parse_args()
    if not args.text and not args.file:
        p.error("provide --text or --file")

    sc = SpellCorrector(args.checkpoint, batch_size=args.batch_size,
                        threshold=args.threshold, iterations=args.iterations)

    if args.file:
        sentences = [line.rstrip("\n") for line in open(args.file, encoding="utf-8")]
        for res in sc(sentences):
            print(json.dumps(res, ensure_ascii=False))
        return

    for res in sc(args.text):
        print(f"input : {res['input']}")
        for e in res["errors"]:
            print(f"  từ {e['word_index']}: {e['token']} -> {e['suggestion']}"
                  f" (p={e['confidence']}, vòng {e['iteration']})")
        print(f"output: {res['output']}\n")


if __name__ == "__main__":
    main()
