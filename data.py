"""Inference-only text utilities: Vietnamese normalization, tokenization, and
the Vocab container. This is the slim subset of the training repo's data.py —
the synthetic error generation, Dataset, samplers and collate logic used only
for training have been removed. `model.py` and `correct.py` import `Vocab` and
`tokenize` from here.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Text normalization (same convention as the training repo)
# ---------------------------------------------------------------------------

_TONE_NORM = [("òa", "oà"), ("óa", "oá"), ("ỏa", "oả"), ("õa", "oã"), ("ọa", "oạ"),
              ("òe", "oè"), ("óe", "oé"), ("ỏe", "oẻ"), ("õe", "oẽ"), ("ọe", "oẹ"),
              ("ùy", "uỳ"), ("úy", "uý"), ("ủy", "uỷ"), ("ũy", "uỹ"), ("ụy", "uỵ")]


def norm_text(text):
    text = unicodedata.normalize("NFC", text)
    for a, b in _TONE_NORM:
        text = text.replace(a, b)
    return text


_PUNCT_RE = re.compile(r"([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~“”‘’…–])")


def tokenize(text):
    """Lowercase, NFC-normalize, and split off punctuation as its own tokens.
    Identical to the tokenizer the model was trained with — keep it in sync."""
    text = _PUNCT_RE.sub(r" \1 ", norm_text(text.strip().lower()))
    return text.split()


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocab:
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, word2id=None):
        self.word2id = word2id or {self.PAD: 0, self.UNK: 1}
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __len__(self):
        return len(self.word2id)

    def __contains__(self, w):
        return w in self.word2id

    def __getitem__(self, w):
        return self.word2id.get(w, 1)  # 1 = <unk>

    def to_dict(self):
        return dict(self.word2id)

    @classmethod
    def from_dict(cls, d):
        return cls(dict(d))
