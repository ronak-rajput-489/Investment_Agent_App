import streamlit as st
from backend import app
from langchain_core.messages import HumanMessage
import base64
import time
from PIL import Image
import pandas as pd
import numpy as np
from datetime import date, timedelta
import base64
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="BFSI KPIs | AI Assistant", page_icon="🏦", layout="wide")

DARK_BLUE_1 = "#0B1F4B"
DARK_BLUE_2 = "#1E3A8A"
TEXT_DARK   = "#111827"

# ------------------ GLOBAL STYLES ------------------
st.markdown(f"""
<style>
  .block-container {{ padding-top: 1rem; }}
  #MainMenu, footer {{ visibility: hidden; }}
  /* Header: dark blue gradient + white text */
  .hdr {{
    background: linear-gradient(135deg, {DARK_BLUE_1} 0%, {DARK_BLUE_2} 100%);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 16px;
    box-shadow: 0 10px 28px rgba(0,0,0,.18);
    color: #ffffff !important;
  }}
  .hdr h1 {{
    margin: 0;
    font-size: 22px;
    letter-spacing: .2px;
    color: #ffffff !important;
    text-shadow: 0 1px 0 rgba(0,0,0,.15);
  }}
  .filter-label {{ font-weight: 800; color: #6b7280; margin-bottom: 6px }}
  .metric-card {{ border:1px solid #e6e8ef; border-radius:12px; padding:16px 18px; background:#fff }}
  .metric-title {{ font-size:12px; color:#6b7280; font-weight:800; text-transform:uppercase; letter-spacing:.04em }}
  .metric-value {{ font-size:28px; font-weight:900; color:{TEXT_DARK}; margin-top:4px }}
  .chat-wrap {{ border:1px solid #e6e8ef; border-radius:12px; background:#fff }}
  .chat-head {{ padding:10px 14px; border-bottom:1px solid #eef0f5; font-weight:900; color:#111827 }}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='hdr'><h1> BFSI Marketing Insight Assistant </h1></div>", unsafe_allow_html=True)

# ------------------ DATA ------------------
DATA_PATH = Path("bfsi_gold_sample_1000.csv")  # keep CSV next to app.py

@st.cache_data(show_spinner=False, ttl=600)
def load_data(p: Path) -> pd.DataFrame:
    if not p.exists():
        st.error(f"CSV not found: {p}. Place bfsi_gold_sample_1000.csv next to app.py.")
        st.stop()
    df = pd.read_csv(p)
    # normalize dtypes / names
    for c in ["RESPONSE_STATUS", "REGION", "PRODUCT", "CAMPAIGN_NAME", "CHANNEL"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in ["REVENUE", "SPEND"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

df = load_data(DATA_PATH)

# ------------------ FILTERS ------------------
f1, f2, f3 = st.columns([1.2, 1.6, 1.2])
#, vertical_alignment="center"
with f1:
    st.markdown("<div class='filter-label'>🧩 Product</div>", unsafe_allow_html=True)
    prod_opts = sorted(df["PRODUCT"].dropna().unique().tolist()) if "PRODUCT" in df.columns else []
    sel_products = st.multiselect("", prod_opts, default=prod_opts, key="f_prod")

with f2:
    st.markdown("<div class='filter-label'>🎯 Campaign</div>", unsafe_allow_html=True)
    if "CAMPAIGN_NAME" in df.columns:
        camp_opts = sorted(df["CAMPAIGN_NAME"].dropna().unique().tolist())
        sel_campaigns = st.multiselect("", camp_opts, default=camp_opts, key="f_camp")
    else:
        sel_campaigns = []
        st.selectbox("", ["No campaign field found"], index=0, disabled=True)

with f3:
    st.markdown("<div class='filter-label'>🌍 Region</div>", unsafe_allow_html=True)
    region_opts = sorted(df["REGION"].dropna().unique().tolist()) if "REGION" in df.columns else []
    sel_regions = st.multiselect("", region_opts, default=region_opts, key="f_region")

mask = pd.Series(True, index=df.index)
if sel_products:   mask &= df["PRODUCT"].isin(sel_products)
if sel_regions:    mask &= df["REGION"].isin(sel_regions)
if sel_campaigns:  mask &= df["CAMPAIGN_NAME"].isin(sel_campaigns)
dff = df.loc[mask].copy()

st.markdown("---")  # (no records chip by request)

# ------------------ HELPERS ------------------
def pct(n, d): return 0.0 if d == 0 else (n / d) * 100.0
def fmt_money_myr(v: float) -> str: return f"RM {v:,.0f}"

def format_roi_from_roas(roas: float) -> str:
    """Display ROI as +X.X% based on ROAS (revenue/spend)."""
    if roas is None or np.isnan(roas) or np.isinf(roas):
        return "—"
    roi = (roas - 1.0) * 100.0
    sign = "+" if roi >= 0 else ""
    return f"{sign}{roi:.1f}% "

# ------------------ KPIs ------------------

total = len(dff)
resp = dff["RESPONSE_STATUS"].str.lower() if "RESPONSE_STATUS" in dff.columns else pd.Series("", index=dff.index)
engaged   = int((resp == "engaged").sum())
converted = int((resp == "converted").sum())

leads_pct = pct(engaged, total)
conv_pct  = pct(converted, total)
revenue   = float(dff["REVENUE"].sum()) if "REVENUE" in dff.columns else 0.0
spend     = float(dff["SPEND"].sum()) if "SPEND" in dff.columns else 0.0
roas_val  = (revenue / spend) if spend > 0 else None  # for ROI formatter

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Leads %</div>"
        f"<div class='metric-value'>{leads_pct:.2f}%</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Conversion %</div>"
        f"<div class='metric-value'>{conv_pct:.2f}%</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Revenue (RM)</div>"
        f"<div class='metric-value'>{fmt_money_myr(revenue)}</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>ROI</div>"
        f"<div class='metric-value'>{format_roi_from_roas(roas_val)}</div></div>", unsafe_allow_html=True)

st.markdown("")

# --------------- Chat bot section -------------------- 

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if "message_history" not in st.session_state:
        st.session_state.message_history = [{"role":"assistant","content":"👋 Welcome! Ask about revenue, approvals, ROAS, channels or regions."}]

with st.expander("🤖 AI Financial Assistant", expanded=True):
    #st.markdown("<div class='section-title'>🤖 AI Financial Assistant</div>", unsafe_allow_html=True)
    
    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = []

    if 'Predefine_history' not in st.session_state:
        st.session_state['Predefine_history'] = []
        
    # --- Scrollable message box ---
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
        
    # loading the conversation history
    for message in st.session_state['message_history']:
        with st.chat_message(message['role'], avatar="🧑" if message['role']=="user" else "🤖"):
            st.markdown(message['content'])

    st.markdown("</div>", unsafe_allow_html=True)
    
user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user',avatar="🧑"):
        st.markdown(f"<div class='user-msg'>{user_input}</div>", unsafe_allow_html=True)
        # st.markdown(f"<div style='font-size:28px;'>{user_input}</div>", unsafe_allow_html=True)
        #st.text(user_input)

    response = app.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    ai_message = response['messages'][-1].content
    
    # Append AI message
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    
    with st.chat_message('assistant',avatar="🤖"):
        with st.spinner("Thinking..."):
            time.sleep(1.8)
            st.markdown(f"<div class='assistant-msg'>{ai_message}</div>", unsafe_allow_html=True)

st.markdown("""
        <style>
        .user-msg {
            background-color: #C8FACC;
            padding: 12px;
            border-radius: 15px;
            margin: 8px 0px;
            font-size: 20px;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)        
        