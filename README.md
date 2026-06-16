# Vietnamese Spell Correction — Inference

Sửa lỗi chính tả tiếng Việt bằng mô hình **Hierarchical Transformer** (word‑level +
char‑level encoder, detection head + correction head). Gói này **chỉ chứa code
inference** (không có code train), kèm engine tối ưu tốc độ, REST API và UI web.

Hai chế độ chạy:

| Mode | Detect | Correct | Đặc điểm |
|------|--------|---------|----------|
| **1 · `best`** | `best_327000.pt` | `best_327000.pt` | Nhanh, sửa ở mức token, **giữ nguyên định dạng** (hoa/thường, dấu câu). |
| **2 · `hybrid`** | `best_327000.pt` (chỉ phát hiện) | `protonx-models/protonx-legal-tc` (T5 seq2seq) | Câu sạch được bỏ qua → chỉ câu có lỗi mới đưa sang T5 rewrite. Tránh model seq2seq "bịa" trên câu đúng. |

---

## Cài đặt

```bash
pip install -r requirements.txt
```

> `transformers / sentencepiece / huggingface_hub` chỉ cần cho **mode 2**. Mode 1
> chỉ cần `torch`.

### Checkpoint — không cần tải thủ công

Cả hai mô hình **tự tải từ HuggingFace Hub** ở lần chạy đầu rồi được cache lại,
nên một bản clone mới chạy được ngay không cần bước nào:

- Mô hình hierarchical: [`ANZ-Innovation/spell_correction_v1`](https://huggingface.co/ANZ-Innovation/spell_correction_v1) → `best_327000.pt` (~123MB)
- Mode 2 — seq2seq: `protonx-models/protonx-legal-tc` (T5, ~0.9GB)

Muốn dùng checkpoint local thì đặt file vào `spelling_corr/best_327000.pt` (hoặc
truyền `--checkpoint /đường/dẫn.pt`, hoặc env `SC_CHECKPOINT`) — nếu có, nó được
ưu tiên thay cho bản tải về.

---

## Sử dụng

### 1. CLI / thư viện

```bash
# sửa nhanh bằng best_327000 (mode 1) — checkpoint tự tải lần đầu
python infer.py --text "Cơn bảo dag đổ bôj vào đất lền ."

# nhiều câu từ file (mỗi dòng 1 câu) -> JSON lines
python infer.py --file sentences.txt
```

```python
from pipeline import build_corrector

# mode 1 — không cần truyền checkpoint, tự tải từ HF + auto chọn device
sc = build_corrector("best")
print(sc(["Tôi đi hocj ở truờng đai hocj ."])[0]["output"])
# -> "Tôi đi học ở trường đại học ."

# mode 2 (detect best_327000 -> sửa bằng protonx-legal-tc)
sc = build_corrector("hybrid")
print(sc(["Cơn bảo dag đổ bôj vào đất lền ."])[0]["output"])
# -> "Cơn bão đang đổ bộ vào đất liền ."
```

Mỗi câu trả về `{"input", "output", "errors":[{word_index, token, suggestion, ...}], "mode"}`.

### 2. REST API + UI web (FastAPI)

```bash
python serve.py                 # checkpoint tự tải; --device auto
# hoặc: uvicorn serve:app --host 0.0.0.0 --port 8000
```

- Mở **http://localhost:8000** để dùng UI (chọn mode, ngưỡng, số vòng lặp).
- Gọi API:

```bash
curl -s localhost:8000/correct -H 'content-type: application/json' \
  -d '{"mode":"hybrid","sentences":["Tôi đi hocj ."],"threshold":0.5}' | jq
```

Cấu hình qua env: `SC_CHECKPOINT`, `SC_DEVICE` (`auto|cuda|mps|cpu`), `SC_PRECISION`.

### 3. UI Streamlit

```bash
streamlit run app.py
```

---

## Tối ưu tốc độ

`FastSpellCorrector` (trong `infer.py`) tự áp dụng theo thiết bị:

- **CUDA**: `bf16` (Ampere+, vd A4000/A100) hoặc `fp16` (Turing/Volta); bật **TF32**,
  **cuDNN autotune**, **flash/mem‑efficient SDPA**; `--compile` dùng CUDA graphs.
- **CPU**: thử **int8 dynamic quantization** (tự fallback fp32 nếu build không hỗ trợ).
- **MPS** (Apple): fp32 + `inference_mode` + warmup.

Benchmark:

```bash
python infer.py --benchmark --device cuda --batch_size 256 --n 4000
python infer.py --benchmark --device cuda --batch_size 256 --n 4000 --compile
```
Output in A4000 (16GB)
device = cuda, n = 10000, batch_size = 2048, batches = 5
baseline (fp32)         17260.2 ms  |     579.4 sent/s  |   1.726 ms/sent
optimized: {'device': 'cuda', 'precision': 'bf16', 'params': 30651444, 'max_len': 192, 'word_vocab': 30002, 'char_vocab': 402}
optimized (bf16+compile)  13867.1 ms  |     721.1 sent/s  |   1.387 ms/sent
speedup: 1.24x

---

## Cấu trúc file

```
vi-spell-correct/
├── model.py        # kiến trúc HierarchicalSC (word + char encoder, 2 head)
├── data.py         # slim: chỉ Vocab + tokenize (đã bỏ toàn bộ code train)
├── correct.py      # SpellCorrector: batched + iterative inference
├── infer.py        # FastSpellCorrector: tối ưu tốc độ + benchmark + CLI
├── pipeline.py     # build_corrector("best"|"hybrid") + Seq2Seq/Hybrid corrector
├── serve.py        # FastAPI: API /correct, /health + UI web nhúng
├── app.py          # UI Streamlit
├── requirements.txt
└── spelling_corr/  # (tùy chọn) đặt best_327000.pt vào đây để dùng bản local thay vì tải HF
```

Phụ thuộc nội bộ: `serve.py`,`app.py` → `pipeline.py` → `infer.py` → `correct.py`
→ `model.py` + `data.py`.

---

## Mô hình

- **Hierarchical Transformer** (`best_327000.pt`): word encoder 6 lớp (d=512),
  char encoder 4 lớp (d=256), detection head + correction head (weight‑tied với
  word embedding). ~30.6M tham số. Vocab: 30k từ / 402 ký tự. `max_len=192`.
- **protonx-models/protonx-legal-tc**: T5 seq2seq text‑correction tiếng Việt
  (~226M tham số), input là text thô, không cần task prefix.
