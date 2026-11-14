from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_risk_model.joblib"
META_PATH = ROOT / "models" / "metadata.json"

st.set_page_config(page_title="Student Support Check", page_icon="📋", layout="wide")

# ---------- Look & feel: warm paper, charcoal ink, signal red / yellow / green ----------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');

:root{
  --paper:#F3EDE2; --card:#FBF8F2; --line:#DDD4C3; --ink:#2A2826; --soft:#7A7468;
  --red:#C8553D; --yellow:#E0A526; --green:#5E8C61;
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{display:none!important;}
header[data-testid="stHeader"]{background:transparent;}
.stApp{background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif;}
.block-container{padding-top:2.4rem; max-width:1100px;}

/* input panel */
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--card); border:1.5px solid var(--ink)!important; border-radius:18px; padding:.6rem .9rem .4rem;}
.stApp label p{font-family:'DM Sans',sans-serif; font-size:.88rem; font-weight:500; color:var(--ink);}
div[data-baseweb="slider"] div[role="slider"]{background:var(--ink)!important; border-color:var(--ink)!important; box-shadow:none!important;}
div[data-baseweb="slider"] > div > div > div{background:var(--ink)!important;}
div[data-testid="stNumberInput"] input{background:var(--card); border-radius:8px; color:var(--ink)!important; -webkit-text-fill-color:var(--ink)!important; font-weight:600;}
div[data-testid="stNumberInput"] button{color:var(--ink);}
div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"]{color:var(--soft);}

div.stButton > button[kind="primary"]{
  background:var(--ink); color:var(--paper); border:none; border-radius:999px;
  padding:.7rem 0; font-weight:700; letter-spacing:.02em; margin-top:.6rem;
}
div.stButton > button[kind="primary"]:hover{background:var(--red); color:#fff;}

/* type */
.eyebrow{font-family:'DM Mono',monospace; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:var(--soft);}
.title{font-family:'Fraunces',serif; font-weight:700; font-size:2.7rem; line-height:1.05; margin:.35rem 0 .5rem;}
.lede{color:var(--soft); font-size:1.02rem; max-width:560px; margin-bottom:1.8rem;}
.rule{border:none; border-top:1.5px solid var(--ink); margin:0 0 1.6rem;}

/* hero result */
.hero{display:grid; grid-template-columns:1.4fr 1fr; gap:0; background:var(--card);
  border:1.5px solid var(--ink); border-radius:18px; overflow:hidden; margin-bottom:1.2rem;}
.hero-l{padding:1.6rem 1.8rem; border-right:1.5px solid var(--ink);}
.hero-r{padding:1.6rem 1.8rem; display:flex; flex-direction:column; justify-content:center;}
.band{font-family:'Fraunces',serif; font-weight:700; font-size:3.4rem; line-height:1; margin:.4rem 0 .6rem;}
.verdict{display:inline-flex; align-items:center; gap:.5rem; font-weight:700; font-size:.95rem;}
.dot{width:.7rem; height:.7rem; border-radius:50%; display:inline-block;}
.pct{font-family:'Fraunces',serif; font-weight:700; font-size:3.4rem; line-height:1;}
.pct-cap{color:var(--soft); font-size:.9rem; margin-top:.3rem;}

/* meter */
.meter{margin:1.3rem 0 .2rem; position:relative;}
.track{display:flex; height:14px; border-radius:999px; overflow:hidden; border:1.5px solid var(--ink);}
.track span{display:block; height:100%;}
.marker{position:absolute; top:-6px; width:4px; height:26px; background:var(--ink); border-radius:2px; transform:translateX(-2px);}
.ticks{display:flex; justify-content:space-between; font-family:'DM Mono',monospace; font-size:.68rem; color:var(--soft); margin-top:.35rem;}

/* snapshot tiles */
.sec{font-family:'DM Mono',monospace; font-size:.74rem; letter-spacing:.14em; text-transform:uppercase; color:var(--soft); margin:1.6rem 0 .7rem;}
.grid{display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem;}
.tile{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:.9rem 1.05rem;}
.tile .k{font-size:.8rem; color:var(--soft);}
.tile .v{font-family:'Fraunces',serif; font-weight:700; font-size:1.6rem; margin:.1rem 0 .35rem;}
.tile .s{font-size:.78rem; font-weight:700; display:flex; align-items:center; gap:.4rem;}

/* signals */
.sig{display:flex; gap:.9rem; align-items:flex-start; padding:.85rem 0; border-bottom:1px dashed var(--line);}
.sig:last-child{border-bottom:none;}
.sig .n{font-family:'DM Mono',monospace; font-size:.78rem; background:var(--ink); color:var(--paper);
  border-radius:50%; width:1.6rem; height:1.6rem; display:flex; align-items:center; justify-content:center; flex:none;}
.sig p{margin:0; font-size:.97rem;}
.next{background:var(--ink); color:var(--paper); border-radius:14px; padding:1.1rem 1.4rem; margin-top:1.2rem;}
.next .eyebrow{color:#B9B2A4;}
.next p{margin:.3rem 0 0; font-size:1.02rem;}

/* empty state */
.steps{display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem; margin-top:.4rem;}
.step{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:1.1rem 1.2rem;}
.step b{font-family:'Fraunces',serif; font-size:2rem; display:block; margin-bottom:.2rem;}
.step span{color:var(--soft); font-size:.92rem;}

.foot{color:var(--soft); font-size:.8rem; margin-top:2.2rem; padding-top:.9rem; border-top:1px solid var(--line);}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def html(s: str):
    """Render HTML without markdown mistaking indented lines for code blocks."""
    st.markdown("\n".join(line.strip() for line in s.strip().splitlines()), unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


model, metadata = load_assets()
if model is None:
    st.error("Trained model not found. First run: python -m src.train")
    st.stop()

# ---------- Header ----------
html("""
<div class="eyebrow">Academic support · early signal</div>
<div class="title">Who might need<br>a helping hand?</div>
<div class="lede">Enter a student's recent learning details below. You'll get a plain-language read
on whether extra support may help, and what to talk about first.</div>
<hr class="rule">
""")

# ---------- Inputs ----------
html('<div class="sec" style="margin-top:0">Student details</div>')
with st.container(border=True):
    c1, c2, c3 = st.columns(3, gap="large")
    attendance = c1.slider("Attendance (%)", 0, 100, 75)
    completion = c2.slider("Assignments completed (%)", 0, 100, 70)
    previous = c3.slider("Previous average score", 0, 100, 60)
    c4, c5, c6 = st.columns(3, gap="large")
    trend = c4.slider("Score change (points)", -30, 30, 0, help="Negative means scores are going down.")
    missed = c5.number_input("Missed assignments", min_value=0, max_value=20, value=2)
    study = c6.slider("Study hours per week", 0, 40, 7)
    predict = st.button("Check student", type="primary", use_container_width=True)

# ---------- Empty state ----------
if not predict:
    pass
else:
    # ---------- Same model logic as before ----------
    sample = pd.DataFrame([{
        "attendance_rate": attendance,
        "assignment_completion": completion,
        "previous_avg_score": previous,
        "score_trend": trend,
        "missed_assignments": missed,
        "study_hours_per_week": study,
    }])
    probability = float(model.predict_proba(sample)[0][1])
    prediction = int(probability >= 0.5)
    risk_band = "High" if probability >= 0.60 else ("Moderate" if probability >= 0.30 else "Low")
    color = {"High": "var(--red)", "Moderate": "var(--yellow)", "Low": "var(--green)"}[risk_band]
    pos = max(1.5, min(98.5, probability * 100))

    html(f"""
    <div class="hero">
      <div class="hero-l">
        <div class="eyebrow">Risk level</div>
        <div class="band" style="color:{color}">{risk_band}</div>
        <div class="verdict"><span class="dot" style="background:{color}"></span>
          {"Needs support" if prediction else "Looks on track"}</div>
        <div class="meter">
          <div class="track">
            <span style="width:30%;background:var(--green)"></span>
            <span style="width:30%;background:var(--yellow)"></span>
            <span style="width:40%;background:var(--red)"></span>
          </div>
          <div class="marker" style="left:{pos}%"></div>
          <div class="ticks"><span>0%</span><span>30%</span><span>60%</span><span>100%</span></div>
        </div>
      </div>
      <div class="hero-r">
        <div class="eyebrow">Chance of being at risk</div>
        <div class="pct">{probability:.0%}</div>
        <div class="pct-cap">An estimate, not a certainty.</div>
      </div>
    </div>
    """)

    # ---------- Snapshot: each input checked against the same thresholds ----------
    checks = [
        ("Attendance", f"{attendance}%", attendance < 75,
         "Low. Check what's getting in the way.",
         "Attendance is low. Check if something is getting in the way."),
        ("Assignments done", f"{completion}%", completion < 70,
         "Low. Review workload or access.",
         "Few assignments are being completed. Review workload or access issues."),
        ("Previous average", f"{previous}", previous < 50,
         "Low. Find the hard topics.",
         "Past scores are low. Look at which topics are hard."),
        ("Score change", f"{trend:+d}", trend < -5,
         "Falling. Early check-in helps.",
         "Scores are going down. An early check-in could help."),
        ("Missed assignments", f"{missed}", missed >= 4,
         "Many. Ask the student why.",
         "Many assignments were missed. Talk with the student about why."),
        ("Study hours / week", f"{study}", study < 4,
         "Low. Plan study time together.",
         "Study time is low. Build a realistic study plan together."),
    ]

    tiles = ""
    for name, value, flag, short, _ in checks:
        c = "var(--red)" if flag else "var(--green)"
        status = short if flag else "Fine"
        tiles += (f'<div class="tile"><div class="k">{name}</div><div class="v">{value}</div>'
                  f'<div class="s" style="color:{c}"><span class="dot" style="background:{c}"></span>{status}</div></div>')
    html(f'<div class="sec">Snapshot</div><div class="grid">{tiles}</div>')

    # ---------- Talking points ----------
    signals = [long for _, _, flag, _, long in checks if flag]
    if not signals:
        signals = ["Nothing stands out. Review the student's full situation."]
    rows = "".join(f'<div class="sig"><div class="n">{i}</div><p>{s}</p></div>'
                   for i, s in enumerate(signals, 1))
    html(f'<div class="sec">Worth talking about</div>{rows}')

    if prediction:
        nxt = "A teacher should review this student's situation and consider offering support."
    else:
        nxt = "No action flagged. This doesn't guarantee future success, so keep an eye on progress."
    html(f'<div class="next"><div class="eyebrow">Suggested next step</div><p>{nxt}</p></div>')

html("""
<div class="foot">A guide only, built on sample data. A teacher makes the final call.
Never use it for punishment, grading or admissions.</div>
""")
