"""Faithful implementation of "Hierarchical Transformer Encoders for
Vietnamese Spelling Correction" (arXiv:2105.13578).

Architecture (paper Sec. 4):
  - Character-level Transformer encoder: self-attention over the characters
    of each word (order-aware), pooled to one vector per word.
  - Word-level Transformer encoder: word embedding concatenated with the
    char-level vector, summed with learned positional embeddings.
  - Detection head: two FC layers -> 2-way softmax per token.
  - Correction head: two FC layers -> softmax over the word vocabulary,
    weight-tied to the word embedding table, trained only on error positions.
"""

import torch
import torch.nn as nn


class CharEncoder(nn.Module):
    """Encode each word's character sequence with self-attention, then
    mean-pool the character states into a single vector per word."""

    def __init__(self, char_vocab_size, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, max_word_len=24):
        super().__init__()
        self.d_model = d_model
        self.max_word_len = max_word_len
        self.char_embed = nn.Embedding(char_vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_word_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

    def forward(self, char_ids):
        # char_ids: (B, W, C) with 0 = padding
        B, W, C = char_ids.shape
        x = char_ids.view(B * W, C)
        pad_mask = x.eq(0)
        # Rows that are pure padding (padded word slots) would make attention
        # produce NaN; expose position 0 and zero the result afterwards.
        all_pad = pad_mask.all(dim=1)
        pad_mask = pad_mask & ~all_pad.unsqueeze(1)
        pos = torch.arange(C, device=x.device).unsqueeze(0)
        h = self.char_embed(x) + self.pos_embed(pos)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        keep = (~pad_mask).unsqueeze(-1).to(h.dtype)
        pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        pooled = pooled.masked_fill(all_pad.unsqueeze(-1), 0.0)
        return pooled.view(B, W, self.d_model)


class HierarchicalSC(nn.Module):
    def __init__(self,
                 word_vocab_size,
                 char_vocab_size,
                 d_model=768,
                 nhead=12,
                 num_layers=12,
                 dim_feedforward=3072,
                 dropout=0.1,
                 char_d_model=256,
                 char_nhead=8,
                 char_num_layers=4,
                 char_dim_feedforward=1024,
                 max_len=192,
                 max_word_len=24,
                 head_hidden=512):
        super().__init__()
        assert d_model > char_d_model, "word embedding dim = d_model - char_d_model must be positive"
        self.config = dict(word_vocab_size=word_vocab_size, char_vocab_size=char_vocab_size,
                           d_model=d_model, nhead=nhead, num_layers=num_layers,
                           dim_feedforward=dim_feedforward, dropout=dropout,
                           char_d_model=char_d_model, char_nhead=char_nhead,
                           char_num_layers=char_num_layers,
                           char_dim_feedforward=char_dim_feedforward,
                           max_len=max_len, max_word_len=max_word_len,
                           head_hidden=head_hidden)
        self.max_len = max_len
        word_emb_dim = d_model - char_d_model
        self.word_embed = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=0)
        self.char_encoder = CharEncoder(char_vocab_size, char_d_model, char_nhead,
                                        char_num_layers, char_dim_feedforward,
                                        dropout, max_word_len)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)

        # Detection: 2 FC layers -> 2 classes (paper Sec. 4.2)
        self.detect_head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 2),
        )
        # Correction: FC down to the word-embedding dim, logits tied to the
        # word embedding weights (paper: "the correction classifier shares the
        # same weights as the word embedding").
        self.correct_proj = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, word_emb_dim),
            nn.LayerNorm(word_emb_dim),
        )
        self.correct_bias = nn.Parameter(torch.zeros(word_vocab_size))

    def forward(self, word_ids, char_ids, corr_mask=None):
        """word_ids: (B, W) long, 0 = pad. char_ids: (B, W, C) long, 0 = pad.
        Without corr_mask returns det_logits (B, W, 2) and corr_logits
        (B, W, V) — the inference path.
        With corr_mask (bool, B x W) the vocab projection runs only on the
        selected positions and corr_logits has shape (n_selected, V). The full
        B x W x V tensor at vocab size 30k+ dominates training memory, so the
        training loop selects just the true-error positions."""
        B, W = word_ids.shape
        if W > self.max_len:
            raise ValueError(f"sequence length {W} exceeds max_len {self.max_len}")
        pad_mask = word_ids.eq(0)  # (B, W)
        pos = torch.arange(W, device=word_ids.device).unsqueeze(0)
        x = torch.cat([self.word_embed(word_ids), self.char_encoder(char_ids)], dim=-1)
        x = x + self.pos_embed(pos)
        x = self.emb_dropout(self.emb_norm(x))
        h = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, W, d_model)
        det_logits = self.detect_head(h)
        h = h if corr_mask is None else h[corr_mask]  # (n, d_model) when masked
        corr_logits = self.correct_proj(h) @ self.word_embed.weight.t() + self.correct_bias
        return det_logits, corr_logits

    @property
    def device(self):
        return self.word_embed.weight.device

    def save(self, path, word_vocab, char_vocab, train_state=None):
        """train_state (optional dict with optimizer/scheduler/scaler/step)
        makes the checkpoint exactly resumable; without it the file is
        model-only (smaller, enough for inference)."""
        params = {"config": self.config,
                  "state_dict": self.state_dict(),
                  "word_vocab": word_vocab.to_dict(),
                  "char_vocab": char_vocab.to_dict()}
        if train_state is not None:
            params["train_state"] = train_state
        torch.save(params, path)

    @staticmethod
    def from_checkpoint(ckpt):
        from data import Vocab
        model = HierarchicalSC(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        word_vocab = Vocab.from_dict(ckpt["word_vocab"])
        char_vocab = Vocab.from_dict(ckpt["char_vocab"])
        return model, word_vocab, char_vocab

    @staticmethod
    def load(path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        return HierarchicalSC.from_checkpoint(ckpt)
