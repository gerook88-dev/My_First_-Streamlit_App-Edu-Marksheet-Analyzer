"""
EduLens - Marksheet Analyzer for AKU-EB and STBB students
Run with:  streamlit run app.py
"""
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG + STYLE
# ----------------------------------------------------------------------------
st.set_page_config(page_title="EduLens | Marksheet Analyzer", page_icon="🎓", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem; max-width: 1250px;}
.hero {
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    padding: 2.2rem 2.4rem; border-radius: 22px; color: white; margin-bottom: 1.4rem;
    box-shadow: 0 12px 30px rgba(79,70,229,.35);
}
.hero h1 {margin: 0; font-size: 2.4rem; color: white;}
.hero p {margin: .4rem 0 0 0; font-size: 1.05rem; opacity: .92;}
.card {
    background: rgba(127,127,127,.08); border: 1px solid rgba(127,127,127,.18);
    border-radius: 16px; padding: 1rem 1.2rem; height: 100%;
}
.kpi {text-align:center; padding: 1rem; border-radius: 16px; color: white;}
.kpi .v {font-size: 2rem; font-weight: 800; line-height: 1.1;}
.kpi .l {font-size: .85rem; opacity: .9; letter-spacing: .04em; text-transform: uppercase;}
.k1 {background: linear-gradient(135deg,#4f46e5,#6366f1);}
.k2 {background: linear-gradient(135deg,#059669,#10b981);}
.k3 {background: linear-gradient(135deg,#d97706,#f59e0b);}
.k4 {background: linear-gradient(135deg,#dc2626,#ef4444);}
.pill {display:inline-block; padding:.15rem .7rem; border-radius:999px; font-size:.8rem;
       font-weight:600; margin-right:.3rem; color:white;}
.p-s {background:#10b981;} .p-o {background:#3b82f6;} .p-w {background:#f59e0b;} .p-c {background:#ef4444;}
.insight {border-left: 5px solid #7c3aed; padding: .7rem 1rem; margin: .5rem 0;
          background: rgba(124,58,237,.08); border-radius: 0 12px 12px 0;}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# BOARD CONFIG  (verify against the board's latest official notification)
# ----------------------------------------------------------------------------
BOARDS = {
    "AKU-EB": {
        "full": "Aga Khan University Examination Board",
        "pass": 40,
        "grades": [(90, "A1"), (80, "A"), (70, "B"), (60, "C"), (50, "D"), (40, "E"), (0, "U")],
        "emoji": "🏛️",
        "note": "Concept-based, application-style questions. Understanding beats rote memorisation.",
    },
    "STBB": {
        "full": "Sindh Textbook Board pattern",
        "pass": 33,
        "grades": [(80, "A-1"), (70, "A"), (60, "B"), (50, "C"), (40, "D"), (33, "E"), (0, "F")],
        "emoji": "📘",
        "note": "Textbook-centred papers. Exact textbook coverage and writing presentation matter most.",
    },
}

SUBJECT_TIPS = {
    "math": ["Solve 10 past-paper questions daily, timed.", "Keep a 'mistake notebook' of every error and redo it after 3 days.", "Memorise formulas using a one-page sheet you rewrite weekly."],
    "phys": ["Draw a diagram for every numerical.", "Practise unit conversions and write formulas before substituting.", "Link each concept to a real-life example (great for AKU-EB style questions)."],
    "chem": ["Master the periodic table trends and balancing equations first.", "Make reaction flow-charts for organic/inorganic chapters.", "Practise mole-concept numericals every other day."],
    "bio": ["Use labelled diagrams and active recall (cover and redraw).", "Make comparison tables (e.g. mitosis vs meiosis).", "Learn keywords - examiners mark on exact terminology."],
    "eng": ["Read one short article daily and note 5 new words.", "Practise one essay or letter per week with a strict structure.", "Revise grammar rules (tenses, voice, narration) with exercises."],
    "urdu": ["Practise writing with correct spelling and idioms.", "Prepare summaries and central ideas of every prose/poetry chapter.", "Write one letter/essay per week."],
    "islam": ["Memorise key ayaat/ahadith with translation.", "Prepare short notes for each chapter and revise weekly.", "Practise short-question answers in exactly 3-4 lines."],
    "pak": ["Build a timeline of events and a table of key personalities.", "Practise map work and short notes.", "Write structured answers: intro - points - conclusion."],
    "comp": ["Code or trace programs daily, even for 20 minutes.", "Make flowcharts for each algorithm.", "Solve past-paper programming questions by hand."],
    "sindhi": ["Practise reading aloud and writing daily.", "Prepare chapter-wise summaries.", "Revise grammar and composition formats."],
    "default": ["Solve at least 5 past papers under exam conditions.", "Teach the topic to a friend - if you can't, you haven't learned it yet.", "Use spaced revision: day 1, day 3, day 7, day 21."],
}

STATUS_META = [
    (80, "Strength", "p-s", "🟢"),
    (65, "Stable", "p-o", "🔵"),
    (50, "Needs Attention", "p-w", "🟠"),
    (0, "Critical", "p-c", "🔴"),
]


def grade_of(pct: float, board: str) -> str:
    for cutoff, g in BOARDS[board]["grades"]:
        if pct >= cutoff:
            return g
    return BOARDS[board]["grades"][-1][1]


def next_cutoff(pct: float, board: str):
    cuts = sorted({c for c, _ in BOARDS[board]["grades"] if c > 0})
    for c in cuts:
        if pct < c:
            return c
    return None


def status_of(pct: float):
    for cutoff, label, css, icon in STATUS_META:
        if pct >= cutoff:
            return label, css, icon
    return STATUS_META[-1][1:]


def tips_for(subject: str):
    s = subject.lower()
    for key, tips in SUBJECT_TIPS.items():
        if key != "default" and key in s:
            return tips
    return SUBJECT_TIPS["default"]


# ----------------------------------------------------------------------------
# PARSING
# ----------------------------------------------------------------------------
SKIP_WORDS = ("total", "grand", "percent", "grade", "roll", "seat", "result", "name", "board", "marks obtained", "aggregate")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Subject", "Obtained", "Total"]:
        if col not in df.columns:
            df[col] = np.nan
    if "Exam" not in df.columns:
        df["Exam"] = "Latest"
    df["Subject"] = df["Subject"].astype(str).str.strip()
    df["Obtained"] = pd.to_numeric(df["Obtained"], errors="coerce")
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
    df = df.dropna(subset=["Obtained", "Total"])
    df = df[(df["Total"] > 0) & (df["Obtained"] >= 0) & (df["Obtained"] <= df["Total"]) & (df["Subject"] != "") & (df["Subject"].str.lower() != "nan")]
    return df[["Exam", "Subject", "Obtained", "Total"]].reset_index(drop=True)


def parse_tabular(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    raw = pd.read_csv(uploaded) if name.endswith(".csv") else pd.read_excel(uploaded)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    subj = next((c for c in raw.columns if "subject" in c or "course" in c), raw.columns[0])
    obt = next((c for c in raw.columns if any(k in c for k in ("obtain", "score", "marks", "got"))), None)
    tot = next((c for c in raw.columns if any(k in c for k in ("total", "max", "out of", "full"))), None)
    out = pd.DataFrame({"Subject": raw[subj]})
    if obt is None:  # fall back to first numeric column
        nums = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
        obt = nums[0] if nums else None
    out["Obtained"] = raw[obt] if obt else np.nan
    out["Total"] = raw[tot] if tot else 100
    return out


def parse_text(text: str) -> pd.DataFrame:
    rows = []
    pat = re.compile(r"^\s*([A-Za-z][A-Za-z &\-\(\)/\.,']{2,}?)[\s:|\-]+(\d{1,3}(?:\.\d+)?)(?:\s*(?:/|out of|\|)?\s*)(\d{1,3}(?:\.\d+)?)\s*$", re.I)
    for line in text.splitlines():
        m = pat.match(line)
        if not m:
            continue
        name, a, b = m.group(1).strip(), float(m.group(2)), float(m.group(3))
        if any(w in name.lower() for w in SKIP_WORDS):
            continue
        obtained, total = min(a, b), max(a, b)
        if total in (0,):
            continue
        rows.append({"Subject": name, "Obtained": obtained, "Total": total})
    return pd.DataFrame(rows, columns=["Subject", "Obtained", "Total"])


def parse_pdf(uploaded) -> pd.DataFrame:
    text = ""
    data = uploaded.read()
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        try:
            from pypdf import PdfReader
            text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
        except Exception:
            text = ""
    return parse_text(text)


def parse_image(uploaded) -> pd.DataFrame:
    try:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(uploaded))
        return parse_text(text)
    except Exception:
        return pd.DataFrame(columns=["Subject", "Obtained", "Total"])


def parse_file(uploaded) -> pd.DataFrame:
    n = uploaded.name.lower()
    if n.endswith((".csv", ".xlsx", ".xls")):
        return parse_tabular(uploaded)
    if n.endswith(".pdf"):
        return parse_pdf(uploaded)
    return parse_image(uploaded)


def demo_data() -> pd.DataFrame:
    rows = []
    exam1 = {"English": 62, "Urdu": 68, "Mathematics": 54, "Physics": 58, "Chemistry": 66, "Biology": 74, "Islamiat": 82, "Pakistan Studies": 76}
    exam2 = {"English": 70, "Urdu": 72, "Mathematics": 61, "Physics": 66, "Chemistry": 71, "Biology": 81, "Islamiat": 86, "Pakistan Studies": 79}
    for exam, d in (("Class IX Annual", exam1), ("Class X Annual", exam2)):
        for s, m in d.items():
            rows.append({"Exam": exam, "Subject": s, "Obtained": m, "Total": 100})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# ANALYSIS
# ----------------------------------------------------------------------------
def analyze(df: pd.DataFrame, board: str) -> pd.DataFrame:
    d = df.copy()
    d["Pct"] = (d["Obtained"] / d["Total"] * 100).round(1)
    d["Lost"] = d["Total"] - d["Obtained"]
    d["Grade"] = d["Pct"].apply(lambda p: grade_of(p, board))
    d["Status"] = d["Pct"].apply(lambda p: status_of(p)[0])

    def to_next(row):
        nc = next_cutoff(row["Pct"], board)
        if nc is None:
            return 0.0
        return max(0.0, round(nc / 100 * row["Total"] - row["Obtained"], 1))

    d["MarksToNextGrade"] = d.apply(to_next, axis=1)
    d["NextGrade"] = d["Pct"].apply(lambda p: grade_of(next_cutoff(p, board), board) if next_cutoff(p, board) else "Top")
    return d


def overall_pct(d: pd.DataFrame) -> float:
    return d["Obtained"].sum() / d["Total"].sum() * 100


def build_report(student, board, klass, exam, d, overall) -> str:
    best = d.sort_values("Pct", ascending=False).iloc[0]
    worst = d.sort_values("Pct").iloc[0]
    lines = [
        f"# EduLens Performance Report - {student or 'Student'}",
        f"Board: {board} | Class: {klass} | Exam: {exam}",
        f"Overall: {overall:.1f}% | Grade: {grade_of(overall, board)}",
        "",
        "## Subject results",
    ]
    for _, r in d.sort_values("Pct", ascending=False).iterrows():
        lines.append(f"- {r['Subject']}: {r['Obtained']:.0f}/{r['Total']:.0f} ({r['Pct']}%) - Grade {r['Grade']} - {r['Status']}")
    lines += ["", f"Strongest subject: {best['Subject']} ({best['Pct']}%)", f"Weakest subject: {worst['Subject']} ({worst['Pct']}%)", "", "## Action plan"]
    for _, r in d.sort_values("Pct").head(3).iterrows():
        lines.append(f"### {r['Subject']}")
        for t in tips_for(r["Subject"]):
            lines.append(f"- {t}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------
defaults = {"step": 1, "board": None, "student": "", "klass": "Class X (Matric)", "raw_df": pd.DataFrame(columns=["Exam", "Subject", "Obtained", "Total"]), "final_df": None, "celebrated": False}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.markdown("## 🎓 EduLens")
    steps = ["1 · Choose board", "2 · Upload marksheet", "3 · Your insights"]
    for i, s in enumerate(steps, 1):
        st.markdown(("✅ " if st.session_state.step > i else "👉 " if st.session_state.step == i else "⚪ ") + s)
    st.progress((st.session_state.step - 1) / 2)
    st.divider()
    if st.button("🔄 Start over", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()
    st.caption("Your marks stay in this browser session only and are never stored.")

st.markdown(
    """<div class="hero"><h1>🎓 EduLens</h1>
<p>Upload your marksheet and get a personal report: strengths, weak spots, a smart improvement plan and a what-if simulator.</p></div>""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# STEP 1 - BOARD
# ----------------------------------------------------------------------------
if st.session_state.step == 1:
    st.subheader("Step 1 · Which board are you from?")
    c1, c2 = st.columns(2)
    for col, key in zip((c1, c2), BOARDS):
        with col:
            b = BOARDS[key]
            st.markdown(
                f"""<div class="card"><h2>{b['emoji']} {key}</h2><b>{b['full']}</b><br>
                <span style="opacity:.8">{b['note']}</span><br><br>
                Pass mark: <b>{b['pass']}%</b></div>""",
                unsafe_allow_html=True,
            )
            if st.button(f"Select {key}", key=f"sel_{key}", use_container_width=True, type="primary" if st.session_state.board == key else "secondary"):
                st.session_state.board = key
    if st.session_state.board:
        st.success(f"Board selected: **{st.session_state.board}**")
    c1, c2 = st.columns(2)
    st.session_state.student = c1.text_input("Your name (optional)", st.session_state.student)
    classes = ["Class IX", "Class X (Matric)", "Class XI", "Class XII (Inter)"]
    st.session_state.klass = c2.selectbox("Class", classes, index=classes.index(st.session_state.klass) if st.session_state.klass in classes else 1)
    if st.button("Continue ➜", type="primary", disabled=st.session_state.board is None):
        st.session_state.step = 2
        st.rerun()

# ----------------------------------------------------------------------------
# STEP 2 - UPLOAD
# ----------------------------------------------------------------------------
elif st.session_state.step == 2:
    board = st.session_state.board
    st.subheader(f"Step 2 · Add your {board} marksheet(s)")
    st.caption("Upload one file per exam (e.g. Class IX and Class X) to also see your progress over time.")
    method = st.radio("How would you like to add marks?", ["📤 Upload files", "✍️ Type manually", "🎯 Try demo data"], horizontal=True)

    if method.startswith("📤"):
        files = st.file_uploader("PDF, image, CSV or Excel", type=["pdf", "png", "jpg", "jpeg", "csv", "xlsx", "xls"], accept_multiple_files=True)
        with st.expander("Best results: CSV/Excel format"):
            st.code("Subject,Obtained,Total\nEnglish,72,100\nMathematics,65,100", language="csv")
            st.caption("PDFs and images are read automatically (text PDFs work best; images need Tesseract OCR installed). You can always correct values in the table below.")
        if files and st.button("🔍 Extract marks", type="primary"):
            frames = []
            for f in files:
                try:
                    part = parse_file(f)
                except Exception as e:
                    st.warning(f"Couldn't read {f.name}: {e}")
                    continue
                part["Exam"] = Path(f.name).stem
                frames.append(part)
            merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Exam", "Subject", "Obtained", "Total"])
            merged = clean_df(merged) if len(merged) else merged
            if merged.empty:
                st.error("No marks could be detected automatically - please type them in the table below.")
                merged = pd.DataFrame([{"Exam": "Latest", "Subject": "", "Obtained": 0.0, "Total": 100.0}])
            st.session_state.raw_df = merged
    elif method.startswith("✍️"):
        if st.button("Create blank table"):
            st.session_state.raw_df = pd.DataFrame(
                [{"Exam": "Latest", "Subject": s, "Obtained": 0.0, "Total": 100.0} for s in ["English", "Urdu", "Mathematics", "Physics", "Chemistry", "Biology"]]
            )
    else:
        if st.button("Load demo marks", type="primary"):
            st.session_state.raw_df = demo_data()

    if not st.session_state.raw_df.empty:
        st.markdown("#### ✏️ Check & edit your marks")
        edited = st.data_editor(
            st.session_state.raw_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Exam": st.column_config.TextColumn("Exam"),
                "Subject": st.column_config.TextColumn("Subject", required=True),
                "Obtained": st.column_config.NumberColumn("Obtained", min_value=0.0, format="%.1f"),
                "Total": st.column_config.NumberColumn("Total", min_value=1.0, format="%.1f"),
            },
            key="editor",
        )
        cleaned = clean_df(edited)
        st.caption(f"{len(cleaned)} valid subject rows detected.")
        c1, c2 = st.columns([1, 5])
        if c1.button("⬅ Back"):
            st.session_state.step = 1
            st.rerun()
        if c2.button("✨ Analyze my results", type="primary", disabled=len(cleaned) < 2):
            st.session_state.final_df = cleaned
            st.session_state.celebrated = False
            st.session_state.step = 3
            st.rerun()
    else:
        if st.button("⬅ Back"):
            st.session_state.step = 1
            st.rerun()

# ----------------------------------------------------------------------------
# STEP 3 - DASHBOARD
# ----------------------------------------------------------------------------
else:
    board = st.session_state.board
    df = st.session_state.final_df
    exams = list(dict.fromkeys(df["Exam"]))
    exam = st.selectbox("📅 Exam to analyze", exams, index=len(exams) - 1) if len(exams) > 1 else exams[0]
    d = analyze(df[df["Exam"] == exam], board)
    overall = overall_pct(d)
    g = grade_of(overall, board)
    best = d.sort_values("Pct", ascending=False).iloc[0]
    worst = d.sort_values("Pct").iloc[0]
    pass_mark = BOARDS[board]["pass"]
    name = st.session_state.student.strip() or "Champion"

    if not st.session_state.celebrated:
        if overall >= 75:
            st.balloons()
        st.session_state.celebrated = True

    st.markdown(f"## Hello, {name}! 👋 Here's your {board} report")
    k = st.columns(4)
    k[0].markdown(f'<div class="kpi k1"><div class="v">{overall:.1f}%</div><div class="l">Overall</div></div>', unsafe_allow_html=True)
    k[1].markdown(f'<div class="kpi k2"><div class="v">{g}</div><div class="l">Grade</div></div>', unsafe_allow_html=True)
    k[2].markdown(f'<div class="kpi k3"><div class="v">{best["Subject"][:14]}</div><div class="l">Top subject · {best["Pct"]}%</div></div>', unsafe_allow_html=True)
    k[3].markdown(f'<div class="kpi k4"><div class="v">{worst["Subject"][:14]}</div><div class="l">Needs love · {worst["Pct"]}%</div></div>', unsafe_allow_html=True)
    st.write("")

    t1, t2, t3, t4, t5 = st.tabs(["📊 Overview", "💪 Strengths & Gaps", "🚀 How to Improve", "🎮 What-If Simulator", "📄 Report"])

    # ---------------- OVERVIEW ----------------
    with t1:
        c1, c2 = st.columns([1, 1.6])
        with c1:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=overall,
                    number={"suffix": "%"},
                    delta={"reference": pass_mark, "suffix": "% above pass"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#7c3aed"},
                        "steps": [
                            {"range": [0, pass_mark], "color": "#fecaca"},
                            {"range": [pass_mark, 65], "color": "#fde68a"},
                            {"range": [65, 80], "color": "#bfdbfe"},
                            {"range": [80, 100], "color": "#bbf7d0"},
                        ],
                    },
                )
            )
            fig.update_layout(height=320, margin=dict(t=40, b=10, l=20, r=20), title="Overall performance")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            ds = d.sort_values("Pct")
            colors = ds["Pct"].apply(lambda p: {"Strength": "#10b981", "Stable": "#3b82f6", "Needs Attention": "#f59e0b", "Critical": "#ef4444"}[status_of(p)[0]])
            fig = go.Figure(go.Bar(x=ds["Pct"], y=ds["Subject"], orientation="h", marker_color=colors, text=[f"{p}% · {gr}" for p, gr in zip(ds["Pct"], ds["Grade"])], textposition="outside"))
            fig.add_vline(x=pass_mark, line_dash="dash", line_color="red", annotation_text="Pass")
            fig.add_vline(x=80, line_dash="dot", line_color="green", annotation_text="Strength")
            fig.update_layout(height=320, title="Subject-wise percentage", xaxis=dict(range=[0, 112]), margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        if len(exams) > 1:
            st.markdown("#### 📈 Your progress over time")
            allm = pd.concat([analyze(df[df["Exam"] == e], board).assign(Exam=e) for e in exams])
            c1, c2 = st.columns(2)
            trend = allm.groupby("Exam", sort=False).apply(lambda x: x["Obtained"].sum() / x["Total"].sum() * 100).reset_index(name="Overall %")
            f1 = px.line(trend, x="Exam", y="Overall %", markers=True, title="Overall % per exam")
            f1.update_traces(line_color="#7c3aed", line_width=4, marker_size=12)
            c1.plotly_chart(f1, use_container_width=True)
            piv = allm.pivot_table(index="Subject", columns="Exam", values="Pct", sort=False)
            f2 = px.imshow(piv[exams], text_auto=".0f", color_continuous_scale="RdYlGn", aspect="auto", title="Subject heatmap")
            c2.plotly_chart(f2, use_container_width=True)
            first, last = exams[0], exams[-1]
            if first != last:
                delta = (piv[last] - piv[first]).dropna().sort_values()
                if len(delta):
                    st.markdown(
                        f'<div class="insight">📈 Biggest jump: <b>{delta.index[-1]}</b> ({delta.iloc[-1]:+.1f} points). '
                        f'Biggest slip or slowest growth: <b>{delta.index[0]}</b> ({delta.iloc[0]:+.1f} points).</div>',
                        unsafe_allow_html=True,
                    )

    # ---------------- STRENGTHS & GAPS ----------------
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            if len(d) >= 3:
                r = go.Figure()
                r.add_trace(go.Scatterpolar(r=list(d["Pct"]) + [d["Pct"].iloc[0]], theta=list(d["Subject"]) + [d["Subject"].iloc[0]], fill="toself", name="You", line_color="#7c3aed"))
                r.add_trace(go.Scatterpolar(r=[80] * (len(d) + 1), theta=list(d["Subject"]) + [d["Subject"].iloc[0]], name="Strength line (80%)", line=dict(dash="dot", color="#10b981")))
                r.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=420, title="Skill radar", margin=dict(t=50))
                st.plotly_chart(r, use_container_width=True)
        with c2:
            lost = d[d["Lost"] > 0]
            tm = px.treemap(lost, path=["Subject"], values="Lost", color="Pct", color_continuous_scale="RdYlGn", title="Where your marks are leaking (bigger = more marks lost)")
            tm.update_layout(height=420, margin=dict(t=50))
            st.plotly_chart(tm, use_container_width=True)

        st.markdown("#### 🧭 Subject health check")
        for _, r in d.sort_values("Pct", ascending=False).iterrows():
            label, css, icon = status_of(r["Pct"])
            st.markdown(
                f'<span class="pill {css}">{icon} {label}</span> <b>{r["Subject"]}</b> — {r["Obtained"]:.0f}/{r["Total"]:.0f} ({r["Pct"]}%) · Grade <b>{r["Grade"]}</b>',
                unsafe_allow_html=True,
            )
            st.progress(min(r["Pct"] / 100, 1.0))

        std = d["Pct"].std() if len(d) > 1 else 0
        spread = d["Pct"].max() - d["Pct"].min()
        st.markdown("#### 🔎 Key insights")
        insights = [
            f"🏆 Your strongest area is <b>{best['Subject']}</b> at {best['Pct']}%. Use the study habits that work here in other subjects.",
            f"⚠️ <b>{worst['Subject']}</b> is your weakest at {worst['Pct']}%. It has the biggest potential for improvement.",
            f"⚖️ Your subjects differ by <b>{spread:.0f} points</b> ({'very balanced' if spread < 12 else 'fairly balanced' if spread < 25 else 'uneven - closing the gap will lift your overall result fast'}).",
        ]
        fails = d[d["Pct"] < pass_mark]
        if len(fails):
            insights.append(f"🚨 Below the {pass_mark}% pass mark in: <b>{', '.join(fails['Subject'])}</b>. Make these your first priority.")
        near = d[(d["MarksToNextGrade"] > 0) & (d["MarksToNextGrade"] <= 5)]
        if len(near):
            insights.append("🎯 Quick wins: just a few marks away from the next grade in <b>" + ", ".join(near["Subject"]) + "</b>.")
        for t in insights:
            st.markdown(f'<div class="insight">{t}</div>', unsafe_allow_html=True)

    # ---------------- IMPROVE ----------------
    with t3:
        st.markdown("#### 🎯 Quick wins: marks needed for the next grade")
        qw = d[d["MarksToNextGrade"] > 0].sort_values("MarksToNextGrade")[["Subject", "Pct", "Grade", "NextGrade", "MarksToNextGrade"]]
        qw.columns = ["Subject", "Current %", "Grade", "Next grade", "Extra marks needed"]
        st.dataframe(qw, use_container_width=True, hide_index=True)

        st.markdown("#### 🧪 Priority matrix")
        d_plot = d.copy()
        d_plot["Effort"] = d_plot["Lost"]
        pm = px.scatter(d_plot, x="Pct", y="Lost", size="Total", color="Status", text="Subject", title="Right-bottom = comfortable · Left-top = highest priority",
                        color_discrete_map={"Strength": "#10b981", "Stable": "#3b82f6", "Needs Attention": "#f59e0b", "Critical": "#ef4444"})
        pm.update_traces(textposition="top center")
        pm.update_layout(height=420, xaxis_title="Current %", yaxis_title="Marks lost")
        st.plotly_chart(pm, use_container_width=True)

        st.markdown("#### 🧠 Personalised action plan")
        for _, r in d.sort_values("Pct").head(4).iterrows():
            label, css, icon = status_of(r["Pct"])
            with st.expander(f"{icon} {r['Subject']} - {r['Pct']}% ({label})", expanded=(r["Subject"] == worst["Subject"])):
                for t in tips_for(r["Subject"]):
                    st.markdown(f"- {t}")
                st.markdown(f"**Goal:** reach **{min(100, r['Pct'] + 10):.0f}%** (+{r['Total'] * 0.10:.0f} marks) in your next exam.")
        st.info(f"{BOARDS[board]['emoji']} **{board} tip:** {BOARDS[board]['note']}")

        st.markdown("#### 🗓️ Smart weekly study planner")
        hours = st.slider("Study hours you can give per week", 5, 40, 15)
        w = (100 - d["Pct"]) + 10
        d["Hours"] = (w / w.sum() * hours).round(1)
        d["Sessions"] = (d["Hours"] * 60 / 45).round().astype(int)
        c1, c2 = st.columns([1, 1.2])
        pie = px.pie(d, names="Subject", values="Hours", hole=0.5, title="Time allocation")
        c1.plotly_chart(pie, use_container_width=True)
        c2.dataframe(d.sort_values("Hours", ascending=False)[["Subject", "Hours", "Sessions", "Status"]].rename(columns={"Hours": "Hours/week", "Sessions": "45-min sessions"}), use_container_width=True, hide_index=True)

        st.markdown("#### 🏁 Target calculator")
        target = st.slider("Your target overall %", float(round(overall)), 100.0, float(min(100, round(overall) + 5)), 1.0)
        need = target / 100 * d["Total"].sum() - d["Obtained"].sum()
        head = d["Lost"]
        wgt = head * (100 - d["Pct"] + 10)
        add = (need * wgt / wgt.sum()).clip(upper=head) if wgt.sum() > 0 else head * 0
        plan = pd.DataFrame({"Subject": d["Subject"], "Now": d["Obtained"], "Aim for": (d["Obtained"] + add).round(0), "Extra marks": add.round(0)})
        st.success(f"To reach **{target:.0f}%** you need about **{max(need, 0):.0f} extra marks** in total. Here's a smart split (more from weaker subjects):")
        st.dataframe(plan.sort_values("Extra marks", ascending=False), use_container_width=True, hide_index=True)

    # ---------------- SIMULATOR ----------------
    with t4:
        st.markdown("#### 🎮 What if I score more in…?")
        st.caption("Drag the sliders to see how your overall percentage and grade would change.")
        new = {}
        cols = st.columns(3)
        for i, (_, r) in enumerate(d.iterrows()):
            new[r["Subject"]] = cols[i % 3].slider(r["Subject"], 0, int(r["Total"]), int(r["Obtained"]), key=f"sim_{i}")
        proj = sum(new.values()) / d["Total"].sum() * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Projected overall", f"{proj:.1f}%", f"{proj - overall:+.1f}")
        c2.metric("Projected grade", grade_of(proj, board), f"was {g}")
        c3.metric("Total marks", f"{sum(new.values()):.0f}/{d['Total'].sum():.0f}", f"{sum(new.values()) - d['Obtained'].sum():+.0f}")
        cmp = pd.DataFrame({"Subject": d["Subject"], "Current": d["Pct"], "Projected": [new[s] / t * 100 for s, t in zip(d["Subject"], d["Total"])]})
        fig = go.Figure()
        fig.add_bar(x=cmp["Subject"], y=cmp["Current"], name="Current", marker_color="#94a3b8")
        fig.add_bar(x=cmp["Subject"], y=cmp["Projected"], name="Projected", marker_color="#7c3aed")
        fig.update_layout(barmode="group", height=380, yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- REPORT ----------------
    with t5:
        report = build_report(st.session_state.student, board, st.session_state.klass, exam, d, overall)
        st.markdown(report)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download report (.md)", report, file_name="edulens_report.md", use_container_width=True)
        c2.download_button("⬇️ Download data (.csv)", d.to_csv(index=False), file_name="edulens_analysis.csv", use_container_width=True)
        st.markdown("---")
        st.markdown("### ⭐ Enjoyed EduLens?")
        st.feedback("stars")
        st.caption("Made to help students study smarter, not harder. Wishing you every success! 🌟")
