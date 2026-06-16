"""Streamlit demo UI for the Vietnamese spell corrector.

Run:
    streamlit run app.py
    # checkpoint tự tải từ HF; ghi đè: SC_CHECKPOINT=/path/to.pt streamlit run app.py
"""

import os
import time

import streamlit as st

from pipeline import build_corrector

CKPT = os.environ.get("SC_CHECKPOINT") or None  # None -> auto-download from HF

st.set_page_config(page_title="Sửa lỗi chính tả tiếng Việt", page_icon="✍️", layout="centered")


@st.cache_resource(show_spinner="Đang nạp mô hình…")
def load_engine(mode, ckpt):
    return build_corrector(mode, ckpt, device="auto", precision="auto")


st.title("✍️ Sửa lỗi chính tả tiếng Việt")

with st.sidebar:
    st.header("Tùy chọn")
    mode = st.radio("Chế độ", ["best", "hybrid"],
                    format_func=lambda m: {"best": "1 · best_327000 (detect + sửa)",
                                           "hybrid": "2 · detect best_327000 → sửa protonx-legal-tc"}[m])
    if mode == "hybrid":
        st.caption("Lần đầu tải protonx-legal-tc (~0.9GB).")
    threshold = st.slider("Ngưỡng phát hiện lỗi", 0.1, 0.9, 0.5, 0.05)
    iterations = st.number_input("Số vòng lặp sửa (mode 1)", 1, 5, 2)

sc = load_engine(mode, CKPT)
info = sc.info()
st.caption(f"Chế độ **{mode}** · " + (f"{info.get('device','')} · {info.get('precision','')}"
           if mode == "best" else "best_327000 detect + protonx-legal-tc correct"))
with st.sidebar:
    st.markdown("---")
    st.json(info)

text = st.text_area("Nhập văn bản (mỗi dòng một câu)", height=160,
                    value="Cơn bảo dag đổ bôj vào đất lền .\nTôi đi hocj ở truờng đai hocj .")

if st.button("Sửa lỗi", type="primary"):
    sentences = [s.strip() for s in text.splitlines() if s.strip()]
    if sentences:
        sc.threshold = threshold
        t = time.time()
        results = sc(sentences, iterations=int(iterations))
        dt = time.time() - t
        st.success(f"Xong {len(sentences)} câu trong {dt*1000:.0f} ms "
                   f"({len(sentences)/dt:.0f} câu/giây)")

        for res in results:
            fixed = {e["word_index"]: e for e in res["errors"]}
            parts = []
            for i, w in enumerate(res["input"].split()):
                if i in fixed:
                    e = fixed[i]
                    parts.append(f"~~{w}~~ **:green[{e['suggestion']}]**")
                else:
                    parts.append(w)
            with st.container(border=True):
                st.markdown(" ".join(parts))
                if res["errors"]:
                    st.caption(f"{len(res['errors'])} lỗi đã sửa")
                    with st.expander("Chi tiết"):
                        st.table([{"từ": e["token"], "sửa thành": e["suggestion"],
                                   "độ tin cậy": e["confidence"], "vòng": e["iteration"]}
                                  for e in res["errors"]])
                else:
                    st.caption("Không phát hiện lỗi")
