"""Production packaging: a FastAPI service that loads the optimized engine once
and serves both a JSON API and a self-contained web UI.

Run:
    python serve.py            # checkpoint tự tải từ HF nếu chưa có local
    # or: uvicorn serve:app --host 0.0.0.0 --port 8000
    # checkpoint/device via env: SC_CHECKPOINT, SC_DEVICE, SC_PRECISION

Then open http://localhost:8000 for the UI, or POST to /correct:
    curl -s localhost:8000/correct -H 'content-type: application/json' \
         -d '{"sentences":["Tôi đi hocj ."]}' | jq
"""

import argparse
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from pipeline import build_corrector

app = FastAPI(title="Vietnamese Spell Correction")
_engines = {}  # mode -> corrector, lazily built


def get_engine(mode="best"):
    """mode 'best' = best_327000 detect+correct; 'hybrid' = detect + protonx-legal-tc.
    The hybrid engine downloads/loads the seq2seq model on first use."""
    if mode not in _engines:
        _engines[mode] = build_corrector(
            mode,
            os.environ.get("SC_CHECKPOINT") or None,  # None -> auto-download from HF
            device=os.environ.get("SC_DEVICE", "auto"),
            precision=os.environ.get("SC_PRECISION", "auto"))
    return _engines[mode]


class CorrectRequest(BaseModel):
    sentences: list[str]
    threshold: float = 0.5
    iterations: int = 2
    mode: str = "best"  # "best" | "hybrid"


@app.on_event("startup")
def _startup():
    get_engine("best")  # load + warm the fast model at boot; hybrid loads on demand


@app.get("/health")
def health():
    return {"status": "ok", "loaded": list(_engines), **get_engine("best").info()}


@app.post("/correct")
def correct(req: CorrectRequest):
    sc = get_engine(req.mode)
    sc.threshold = req.threshold
    kw = {"iterations": req.iterations} if req.mode == "best" else {}
    return {"results": sc(req.sentences, **kw)}


@app.get("/", response_class=HTMLResponse)
def index():
    return _HTML


_HTML = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sửa lỗi chính tả tiếng Việt</title>
<style>
  :root { --bg:#0f1220; --card:#1a1f35; --acc:#6ea8fe; --good:#3ddc97; --bad:#ff6b6b; --txt:#e8eaf6; --mut:#9aa3c7; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--txt); }
  .wrap { max-width:820px; margin:0 auto; padding:32px 20px 60px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--mut); font-size:13px; margin-bottom:20px; }
  textarea { width:100%; min-height:130px; background:var(--card); color:var(--txt);
    border:1px solid #2b3358; border-radius:12px; padding:14px; font-size:15px; resize:vertical; }
  .row { display:flex; gap:16px; align-items:center; margin:14px 0; flex-wrap:wrap; }
  label { font-size:13px; color:var(--mut); display:flex; gap:8px; align-items:center; }
  input[type=range] { accent-color:var(--acc); }
  button { background:var(--acc); color:#0a0e1c; border:0; border-radius:10px;
    padding:11px 22px; font-size:15px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.6; cursor:wait; }
  .card { background:var(--card); border:1px solid #2b3358; border-radius:12px;
    padding:16px; margin-top:14px; }
  .out { font-size:16px; line-height:1.7; }
  .fix { background:rgba(61,220,151,.18); color:var(--good); border-radius:5px;
    padding:1px 5px; font-weight:600; }
  .err { text-decoration:line-through; color:var(--bad); opacity:.75; margin-right:4px; }
  .meta { color:var(--mut); font-size:12px; margin-top:10px; }
  .pill { display:inline-block; background:#262d4f; border-radius:20px; padding:2px 10px;
    font-size:12px; color:var(--acc); margin-left:8px; }
</style></head>
<body><div class="wrap">
  <h1>Sửa lỗi chính tả tiếng Việt <span id="dev" class="pill"></span></h1>
  <div class="sub">Mô hình sửa lỗi chính tả tiếng Việt — gõ câu (có dấu), mỗi dòng một câu.</div>
  <textarea id="in" placeholder="Cơn bảo dang đổ bôj vào đất lền .">Cơn bảo dang đổ bôj vào đất lền .
Tôi đi hocj ở truờng đai hocj .</textarea>
  <div class="row">
    <label>Chế độ <select id="mode" style="background:var(--card);color:var(--txt);
      border:1px solid #2b3358;border-radius:8px;padding:5px">
      <option value="best">1 · best_327000 (detect + sửa)</option>
      <option value="hybrid">2 · detect best_327000 → sửa protonx-legal-tc</option>
    </select></label>
    <label>Ngưỡng phát hiện <input id="th" type="range" min="0.1" max="0.9" step="0.05" value="0.5">
      <span id="thv">0.50</span></label>
    <label>Số vòng lặp <input id="it" type="number" min="1" max="5" value="2" style="width:54px;
      background:var(--card);color:var(--txt);border:1px solid #2b3358;border-radius:8px;padding:4px"></label>
    <button id="go">Sửa lỗi</button>
  </div>
  <div class="sub" id="hint">Mode 2 tải model protonx-legal-tc ở lần chạy đầu (~0.9GB, có thể mất một lúc).</div>
  <div id="results"></div>
</div>
<script>
const $=s=>document.querySelector(s);
$("#th").oninput=()=>$("#thv").textContent=(+$("#th").value).toFixed(2);
fetch("/health").then(r=>r.json()).then(d=>$("#dev").textContent=d.device+" · "+d.precision);
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function render(res){
  const fixed={}; res.errors.forEach(e=>fixed[e.word_index]=e);
  const words=res.input.split(/\\s+/);
  const html=words.map((w,i)=>{
    if(fixed[i]!==undefined){const e=fixed[i];
      return `<span class="err">${esc(w)}</span><span class="fix" title="p=${e.confidence}, vòng ${e.iteration}">${esc(e.suggestion)}</span>`;}
    return esc(w);
  }).join(" ");
  const n=res.errors.length;
  return `<div class="card"><div class="out">${html}</div>
    <div class="meta">${n? n+" lỗi đã sửa":"Không phát hiện lỗi"} ·
    <code>output:</code> ${esc(res.output)}</div></div>`;
}
$("#go").onclick=async()=>{
  const btn=$("#go"); btn.disabled=true; btn.textContent="Đang xử lý…";
  const sentences=$("#in").value.split("\\n").map(s=>s.trim()).filter(Boolean);
  try{
    const r=await fetch("/correct",{method:"POST",headers:{"content-type":"application/json"},
      body:JSON.stringify({sentences,threshold:+$("#th").value,iterations:+$("#it").value,mode:$("#mode").value})});
    const d=await r.json();
    $("#results").innerHTML=d.results.map(render).join("");
  }catch(e){$("#results").innerHTML=`<div class="card">Lỗi: ${esc(String(e))}</div>`;}
  btn.disabled=false; btn.textContent="Sửa lỗi";
};
</script></body></html>"""


def main():
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="local .pt; bỏ trống để tự tải từ HF (ANZ-Innovation/spell_correction_v1)")
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="auto")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    if args.checkpoint:
        os.environ["SC_CHECKPOINT"] = args.checkpoint
    os.environ["SC_DEVICE"] = args.device
    os.environ["SC_PRECISION"] = args.precision
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
