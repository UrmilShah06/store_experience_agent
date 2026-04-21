# app.py — Streamlit frontend for the store experience agent
# run: streamlit run app.py

import os
import json
from pathlib import Path
import streamlit as st

# openai key — load from .env

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("API key not found. Add OPENAI_API_KEY to your .env file.")
    st.stop()

st.set_page_config(
    page_title="Store Experience Agent",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

from agent_backend import build_workflow, get_initial_state, load_data, get_latest_week
from tools import get_store_layout, get_mcp_tool_registry, log_vm_action

# dark theme styles — went through a few colour iterations, settled on this blue-purple palette

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family:'Source Sans 3',sans-serif; background:#0a1520; color:#cde0ef; }
.main, .block-container { background:#0a1520; }
.agent-header {
    background:linear-gradient(135deg,#0d1b2a 0%,#1b2a3d 60%,#2d1b3d 100%);
    padding:1.8rem 2.2rem; border-radius:14px; margin-bottom:1.2rem; border-left:5px solid #9c27b0;
}
.agent-header h1 { font-family:'Playfair Display',serif; font-size:1.8rem; color:#fff; margin:0; }
.agent-header p  { color:#8ab4c9; font-size:0.82rem; margin:0.3rem 0 0 0; }
.agent-card { background:#0d1b2a; border:1px solid #1e3a52; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.8rem; }
.agent-card h4 { color:#9c27b0; margin:0 0 0.5rem 0; font-size:0.9rem; }
.agent-complete { border-left:3px solid #4caf7d; }
.agent-running  { border-left:3px solid #f5a623; }
.agent-waiting  { border-left:3px solid #5ba8d9; }
.action-card { background:#0f1f2d; border:1px solid #2d4a6a; border-radius:8px; padding:1rem 1.2rem; margin-bottom:0.7rem; }
section[data-testid="stSidebar"] { background:#080f18; border-right:1px solid #1e3a52; }
.stButton>button { background:#9c27b0 !important; color:#fff !important; font-weight:600 !important; border:none !important; border-radius:8px !important; }
hr { border-color:#1e3a52 !important; }
.streamlit-expanderHeader p { color: #ffffff !important; }
.stSelectbox label { color: #ffffff !important; }
.stTextInput label { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# initialise keys that don't exist yet — avoids KeyError on first load
for k, v in {
    "workflow": None, "stage": "setup",
    "agent_outputs": {}, "action_decisions": {},
    "final_report": "", "run_log": [],
    "store_filter": "all", "week_filter": "latest",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 12 stores — "all" option runs chain-level analysis, useful for area managers
STORES = ["all","Mumbai_Thane","Mumbai_Andheri","Delhi_VasantKunj","Delhi_Saket",
          "Bangalore_Koramangala","Bangalore_Whitefield","Pune_Kothrud","Pune_Wakad",
          "Chennai_AnnaNagar","Hyderabad_Banjara","Kolkata_SaltLake","Ahmedabad_SG"]

# sidebar config
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.session_state.store_filter = st.selectbox("Store", STORES, index=0)
    st.session_state.week_filter  = st.selectbox("Week", ['latest', 'W13_FY25', 'W12_FY25', 'W11_FY25', 'W10_FY25', 'W09_FY25', 'W08_FY25', 'W07_FY25', 'W06_FY25', 'W05_FY25', 'W04_FY25', 'W03_FY25', 'W02_FY25', 'W01_FY25'])

    st.markdown("---")
    if st.button("🚀 Run Agent Workflow"):
        st.session_state.update({
            "stage":"running", "agent_outputs":{},
            "action_decisions":{}, "run_log":[], "final_report":"", "workflow":None
        })
        st.rerun()

    # reset clears everything including the cached workflow
    if st.button("🔄 Reset"):
        st.session_state.update({
            "stage":"setup", "agent_outputs":{}, "action_decisions":{},
            "run_log":[], "final_report":"", "workflow":None
        })
        st.rerun()

    st.markdown("---")
    colors = {"setup":"#5ba8d9","running":"#f5a623","awaiting_approval":"#9c27b0","complete":"#4caf7d"}
    c = colors.get(st.session_state.stage, "#5ba8d9")
    st.markdown(f"<span style='color:{c}'>● {st.session_state.stage.replace('_',' ').title()}</span>", unsafe_allow_html=True)

# page header
st.markdown(f"""
<div class="agent-header">
    <h1>🏪 Store Experience Optimisation Agent</h1>
    <p>IoT Behavioural · Sales · VM Compliance · Planogram (Multimodal) | Store: <b>{st.session_state.store_filter}</b> | Week: <b>{st.session_state.week_filter}</b></p>
</div>""", unsafe_allow_html=True)

tab_workflow, tab_approval, tab_report, tab_mcp, tab_data, tab_about = st.tabs([
    "🤖 Workflow", "✅ Approval", "📋 Report", "🔧 MCP Tools", "📂 Data", "ℹ️ About"
])

# workflow tab — agent status cards + run logic
with tab_workflow:
    
    agent_defs = [
        ("1","Footfall Intelligence Analyst","IoT camera data · dwell time · basket conversion · demographics","footfall_analysis"),
        ("2","Sales Performance Analyst","Weekly sales · brand sell-through · VM opportunity identification","sales_analysis"),
        ("3","VM & Planogram Analyst (Multimodal)","Gondola compliance · planogram PDF via GPT-4o Vision · face count","vm_analysis"),
        ("4","Store Action Plan Writer","Synthesises all agents · promo calendar · prioritised action plan","action_plan"),
    ]

    for num, role, desc, key in agent_defs:
        out = st.session_state.agent_outputs.get(key, "")
        css = "agent-complete" if out else ("agent-running" if st.session_state.stage=="running" else "agent-waiting")
        badge = "✅ Complete" if out else ("⏳ Running..." if st.session_state.stage=="running" else "⏸️ Waiting")
        st.markdown(f"""<div class="agent-card {css}">
<h4>Agent {num} — {role} &nbsp;<small style='color:#5a80a0;font-weight:400;'>{badge}</small></h4>
<div style='color:#7a9bb5;font-size:0.78rem;margin-bottom:0.4rem;'>{desc}</div>
</div>""", unsafe_allow_html=True)
        if out:
            with st.expander(f"View Agent {num} Full Output"):
                import re as _re
                clean = _re.sub(r'\*\*(.*?)\*\*', lambda m: '<strong style="color:#9c27b0;">' + m.group(1) + '</strong>', out)
                clean = clean.replace(chr(10), '<br>')
                st.markdown(f"<div style='color:#cde0ef;font-size:0.83rem;line-height:1.7;background:#0d1b2a;padding:1rem;border-radius:8px;'>{clean}</div>", unsafe_allow_html=True)

    if st.session_state.stage == "running":
        with st.spinner("Running 4 agent nodes — approximately 2-3 minutes..."):
            try:
                wf = build_workflow()
                st.session_state.workflow = wf
                import time as _time
                cfg = {"configurable": {"thread_id": f"cs2_{st.session_state.store_filter}_{int(_time.time())}"}}
                init = get_initial_state(
                    store=st.session_state.store_filter,
                    week=st.session_state.week_filter,
                    api_key=OPENAI_API_KEY,
                )
                result = wf.invoke(init, config=cfg)
                st.session_state.agent_outputs = {
                    "footfall_analysis": result.get("footfall_analysis",""),
                    "sales_analysis":    result.get("sales_analysis",""),
                    "vm_analysis":       result.get("vm_analysis",""),
                    "action_plan":       result.get("action_plan",""),
                }
                st.session_state.run_log  = result.get("session_log", [])
                st.session_state.stage    = "awaiting_approval"
                st.rerun()
            except Exception as e:
                st.error(f"Workflow error: {e}")
                import traceback; st.code(traceback.format_exc())
                st.session_state.stage = "setup"

    if st.session_state.run_log:
        with st.expander("Run Log"):
            for entry in st.session_state.run_log:
                st.markdown(f"<div style='color:#5a80a0;font-size:0.75rem;'>✓ {entry}</div>", unsafe_allow_html=True)

# approval tab — human in the loop checkpoint
with tab_approval:
    if st.session_state.stage not in ["awaiting_approval","complete"]:
        st.info("Run the agent workflow first.")
    else:
        st.markdown("""<div style='background:#0d1b2a;border:1px solid #9c27b0;border-radius:10px;padding:1rem 1.5rem;margin-bottom:1rem;'>
<div style='color:#9c27b0;font-weight:600;margin-bottom:0.4rem;'>⏸️ Human-in-the-Loop Checkpoint</div>
<div style='color:#8ab4c9;font-size:0.85rem;'>Review the action plan. Approve, modify, or reject each action. Decisions logged via MCP tools.</div>
</div>""", unsafe_allow_html=True)

        with st.expander("📋 Agent Action Plan", expanded=True):
            plan = st.session_state.agent_outputs.get("action_plan","")
            st.markdown(f"<div style='color:#cde0ef;font-size:0.85rem;line-height:1.7;'>{plan.replace(chr(10),'<br>')}</div>", unsafe_allow_html=True)

        st.markdown("---")
        # these actions are hardcoded for the demo — in the next version they should be
        # extracted dynamically from the Agent 4 action plan output
        # logic: one small extraction LLM call after Agent 4 completes, prompt asks it to
        # return exactly 5 actions as a JSON array [{id, description}, ...]
        # then app reads from session_state["extracted_actions"] instead of this list
        # tried a regex approach to parse the action plan text directly but the LLM output
        # format wasn't consistent enough — sometimes bullets, sometimes numbered, sometimes both
        # the JSON extraction call is cleaner
        # fallback to this hardcoded list if the extraction call fails or returns malformed JSON
        actions = [
            ("VM_001","Reposition top-performing brand to eye-level gondola in Zone A"),
            ("VM_002","Increase face count for high sell-through SKUs in Ladies Western"),
            ("VM_003","Move underperforming brand from eye-level to mid-level fixture"),
            ("VM_004","Add brand signage and POS material for upcoming promotion"),
            ("VM_005","Cross-merchandise basket affinity brands on adjacent fixtures"),
        ]

        for aid, default_desc in actions:
            dec = st.session_state.action_decisions.get(aid, {})
            st.markdown("<div class='action-card'>", unsafe_allow_html=True)
            c1, c2 = st.columns([3,2])
            with c1:
                desc = st.text_input(f"Action {aid}:", value=dec.get("description", default_desc), key=f"d_{aid}")
            with c2:
                idx = ["Approve","Modify","Reject"].index(dec.get("choice","Approve"))
                choice = st.radio("Decision", ["Approve","Modify","Reject"], key=f"c_{aid}", horizontal=True, index=idx)
            notes = ""
            if choice == "Modify":
                notes = st.text_input("Modification:", key=f"n_{aid}", placeholder="e.g. Reduce to 2 facings")
            elif choice == "Reject":
                notes = st.text_input("Reason:", key=f"r_{aid}", placeholder="e.g. Space committed elsewhere")
            st.session_state.action_decisions[aid] = {"description":desc,"choice":choice,"status":choice.lower(),"notes":notes}
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        if st.session_state.stage == "awaiting_approval":
            if st.button("Submit Decisions & Generate Report", type="primary"):
                with st.spinner("Logging actions and generating report..."):
                    try:
                        approved_list, rejected_list = [], []
                        for aid, dec in st.session_state.action_decisions.items():
                            if dec["status"] in ["approve","modify"]:
                                log_vm_action(
                                    store=st.session_state.store_filter,
                                    action_id=aid,
                                    description=dec["description"],
                                    status=dec["status"],
                                    notes=dec.get("notes","")
                                )
                                approved_list.append(f"{aid}: {dec['description']} ({dec['status']})")
                            else:
                                rejected_list.append(f"{aid}: {dec.get('notes','Rejected')}")

                        from langchain_openai import ChatOpenAI
                        from langchain_core.messages import HumanMessage, SystemMessage
                        llm = ChatOpenAI(model="gpt-4o", temperature=0.1, openai_api_key=OPENAI_API_KEY)

                        resp = llm.invoke([
                            SystemMessage(content="You are a retail operations report writer."),
                            HumanMessage(content=f"""Generate a session completion report.

Store: {st.session_state.store_filter} | Week: {st.session_state.week_filter}

APPROVED ({len(approved_list)}):
{chr(10).join(approved_list) or 'None'}

REJECTED ({len(rejected_list)}):
{chr(10).join(rejected_list) or 'None'}

ACTION PLAN CONTEXT:
{st.session_state.agent_outputs.get('action_plan','')[:500]}

Structure:
**SESSION SUMMARY** - what was analysed, key IoT + sales + VM findings
**APPROVED ACTIONS AND EXPECTED IMPACT** - what changes and projected benefit per action
**REJECTED ACTIONS** - what was declined and why it matters for future planning
**SUCCESS METRICS FOR NEXT WEEK** - 3 specific KPIs to track
**NEXT SESSION FOCUS** - what to prioritise next run""")
                        ])
                        st.session_state.final_report = resp.content
                        st.session_state.stage = "complete"
                        st.session_state["approved_list_for_doc"] = [
                            {"action_id": aid, "description": dec["description"], "status": dec["status"]}
                            for aid, dec in st.session_state.action_decisions.items()
                            if dec["status"] in ["approve", "modify"]
                        ]
                        st.session_state["rejected_list_for_doc"] = [
                            {"action_id": aid, "reason": dec.get("notes", "Rejected")}
                            for aid, dec in st.session_state.action_decisions.items()
                            if dec["status"] == "reject"
                        ]
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback; st.code(traceback.format_exc())
        else:
            a = sum(1 for d in st.session_state.action_decisions.values() if d.get("status")=="approve")
            m = sum(1 for d in st.session_state.action_decisions.values() if d.get("status")=="modify")
            r = sum(1 for d in st.session_state.action_decisions.values() if d.get("status")=="reject")
            st.success(f"✅ Approved: {a} | ✏️ Modified: {m} | ❌ Rejected: {r}")

with tab_report:
    if not st.session_state.final_report:
        st.info("Complete the approval workflow to generate the final report.")
    else:
        st.markdown("<div style='background:#0d3320;border:1px solid #1a6640;border-radius:8px;padding:0.7rem 1rem;margin-bottom:1rem;'><span style='color:#4caf7d;font-weight:600;'>✅ Workflow Complete</span></div>", unsafe_allow_html=True)
        from agent_backend import generate_word_document
        docx_path = generate_word_document(
            store=st.session_state.store_filter,
            week=st.session_state.week_filter,
            agent_outputs=st.session_state.agent_outputs,
            approved_actions=st.session_state.get("approved_list_for_doc", []),
            rejected_actions=st.session_state.get("rejected_list_for_doc", []),
            final_report=st.session_state.final_report,
        )
        if Path(docx_path).exists():
            with open(docx_path, "rb") as f:
                st.download_button(
                    label="📄 Download Session Report (.docx)",
                    data=f,
                    file_name=f"Store_Experience_{st.session_state.store_filter}_{st.session_state.week_filter}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
        st.markdown(f"<div style='background:#0d1b2a;border:1px solid #1e3a52;border-radius:10px;padding:1.5rem;color:#cde0ef;font-size:0.88rem;line-height:1.7;'>{st.session_state.final_report.replace(chr(10),'<br>')}</div>", unsafe_allow_html=True)
        log_path = Path("./vm_action_log.csv")
        if log_path.exists():
            import pandas as pd
            st.markdown("### 📋 VM Action Log (MCP Tool Output)")
            st.dataframe(pd.read_csv(log_path), use_container_width=True)

# mcp tools tab
with tab_mcp:
    st.markdown("<div style='color:#cde0ef;font-size:0.88rem;margin-bottom:1rem;'><strong style='color:#9c27b0;'>MCP (Model Context Protocol)</strong> — Agent capabilities exposed as callable tools for external systems.</div>", unsafe_allow_html=True)

    registry = get_mcp_tool_registry()
    st.markdown("<h3 style='color:#ffffff;'>Available MCP Tools</h3>", unsafe_allow_html=True)
    for tool in registry["tools"]:
        with st.expander(f"Tool: {tool['name']}"):
            st.markdown(f"<span style='color:#ffffff;font-weight:600;'>Description:</span> <span style='color:#cde0ef;'>{tool['description']}</span>", unsafe_allow_html=True)
            st.markdown("<span style='color:#ffffff;font-weight:600;'>Parameters:</span>", unsafe_allow_html=True)
            for param, details in tool["parameters"].items():
                req = "required" if details.get("required") else "optional"
                desc_text = details.get("description", details.get("type",""))
                req_color = "#4caf7d" if req == "required" else "#f5a623"
                st.markdown(
                                f"<div style='margin:0.3rem 0;'>"
                                f"<code style='background:#1e3a52;color:#ffffff;padding:2px 6px;border-radius:4px;font-size:0.85rem;'>{param}</code>"
                                f" <span style='color:{req_color};font-size:0.8rem;font-weight:600;'>({req})</span>"
                                f" <span style='color:#cde0ef;font-size:0.83rem;'>{desc_text}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                    )
            st.markdown(f"<span style='color:#ffffff;font-weight:600;'>Returns:</span> <span style='color:#ffffff;'>{tool['returns']}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h4 style='color:#ffffff;'>Test MCP Tools Live</h4>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<span style='color:#ffffff;font-weight:600;'>get_store_layout</span>", unsafe_allow_html=True)
        ts = st.selectbox("Store", STORES[1:], key="mcp_store")
        tc = st.selectbox("Category (optional)", ["","Men Formal","Ladies Western","Sportswear","Kids"], key="mcp_cat")
        if st.button("Call get_store_layout"):
            st.code(get_store_layout(ts, tc if tc else None)[:1000], language="json")
    with c2:
        st.markdown("<span style='color:#ffffff;font-weight:600;'>log_vm_action</span>", unsafe_allow_html=True)
        ta = st.text_input("Action description", "Move AND brand to eye level Zone A")
        if st.button("Call log_vm_action"):
            st.code(log_vm_action(store=ts, action_id="VM_TEST_001", description=ta, status="approve"), language="json")

# data preview tab
with tab_data:
    import pandas as pd
    fmap = {
        "IoT Store Behaviour":"./data/05_IoT_Store_Behaviour.xlsx",
        "Gondola Allocation":"./data/06_Gondola_Allocation.xlsx",
        "Promo Calendar":"./data/07_Promo_Calendar.xlsx",
        "Weekly Sales (CS1)":"./data/01_Weekly_Sales_Report.xlsx",
        "SKU Performance (CS1)":"./data/04_SKU_Performance_Report.xlsx",
    }
    chosen = st.selectbox("Select file:", list(fmap.keys()))
    fp = fmap[chosen]
    if Path(fp).exists():
        df = pd.read_excel(fp)
        st.caption(f"{len(df):,} rows · {len(df.columns)} columns")
        st.dataframe(df.head(30), use_container_width=True, height=400)
    else:
        st.warning(f"File not found: {fp}")

# about tab
with tab_about:
    st.markdown("""<div style='color:#cde0ef;line-height:1.75;font-size:0.88rem;'>
<h3 style='font-family:Playfair Display,serif;color:#9c27b0;margin-top:0;'>Store Experience Optimisation Agent</h3>

<h4 style='color:#5ba8d9;'>The Business Problem</h4>
<p>Store data lives in silos — IoT behavioural patterns, weekly sales performance, gondola allocation, and the promotional calendar are never synthesised together for a wholistic view on customer behaviour and sales performance. A VM manager reads the planogram. A store manager reads the sales report. Nobody sees that customers are spending 4 minutes in a category but converting at only 18% — and that the top-converting brand is on the bottom shelf. This agent connects all four data sources automatically and produces a monthly VM action plan.</p>

<h4 style='color:#5ba8d9;'>Architecture</h4>
<p>Four LangGraph agent nodes run sequentially as a state machine: <b>Footfall Intelligence Analyst</b> (IoT behavioural data — dwell time, basket conversion, demographics by zone) → <b>Sales Performance Analyst</b> (weekly sales correlation, brand sell-through, VM opportunity identification) → <b>VM & Planogram Analyst</b> (gondola compliance + planogram PDF read via GPT-4o Vision) → <b>Store Action Plan Writer</b> (synthesis + promotional calendar alignment). A human-in-the-loop interrupt pauses execution at the approval checkpoint. Approved actions are logged via MCP tools.</p>

<h4 style='color:#5ba8d9;'>IoT Observational Data</h4>
<p>The behavioural data layer is modelled on real in-store IoT camera deployments used in retail analytics — capturing dwell time per zone, product journey count (items touched), label reads, fitting room visits, basket conversion rate, and competitor brand interaction. This granular observational data, combined with sales and VM compliance analysis, enables recommendations that traditional sales reporting cannot surface.</p>

<h4 style='color:#5ba8d9;'>MCP Integration</h4>
<p>Two tools exposed as MCP endpoints: <code>get_store_layout</code> returns current gondola state as JSON. <code>log_vm_action</code> writes approved VM changes to the implementation tracker. This follows the Model Context Protocol pattern for enterprise agent integration — the agent's capabilities are discoverable and callable by external systems.</p>

<h4 style='color:#5ba8d9;'>Tech Stack</h4>
<p>LangGraph · GPT-4o · GPT-4o Vision (multimodal planogram reading) · pandas · Streamlit · MCP · python-docx</p>
</div>""", unsafe_allow_html=True)
