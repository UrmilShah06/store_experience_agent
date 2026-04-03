# agent_backend.py
# LangGraph workflow with 4 agent nodes — footfall, sales, VM/planogram, action plan
# human-in-the-loop interrupt pauses before process_decisions

import os
import json
import base64
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional
import operator

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- state ---

class StoreExperienceState(TypedDict):
    # shared state passed between all nodes — the agent's working memory
    # Inputs
    store_filter: str           # specific store or "all"
    week_filter: str            # "latest" or specific week
    api_key: str

    # Agent outputs (accumulated as workflow progresses)
    footfall_analysis: str      # Agent 1 output
    sales_analysis: str         # Agent 2 output
    vm_analysis: str            # Agent 3 output (includes planogram)
    action_plan: str            # Agent 4 draft

    # Human review
    human_decisions: dict       # {action_id: "approve"/"modify"/"reject"}
    approved_actions: List[dict]
    rejected_actions: List[dict]

    # Final output
    final_report: str
    session_log: List[str]      # Audit trail

    # Flow control
    current_node: str
    error: Optional[str]


# --- data loading ---

DATA_DIR = "./data"
_cache = {}  # reloads if server restarts — fine for demo use

def load_data(source):
    if source not in _cache:
        files = {
            "iot": "05_IoT_Store_Behaviour.xlsx",
            "gondola": "06_Gondola_Allocation.xlsx",
            "promo": "07_Promo_Calendar.xlsx",
            "sales": "01_Weekly_Sales_Report.xlsx",
            "sku": "04_SKU_Performance_Report.xlsx",
        }
        path = Path(DATA_DIR) / files[source]
        df = pd.read_excel(path)
        df.columns = df.columns.str.strip()
        _cache[source] = df
    return _cache[source].copy()


def get_latest_week(df):
    return sorted(df['Week'].unique())[-1]


def encode_pdf_page(pdf_path):
    # convert planogram PDF first page to base64 PNG for GPT-4o vision
    try:
        from pdf2image import convert_from_path
        import io
        pages = convert_from_path(pdf_path, first_page=1, last_page=1)
        if pages:
            buf = io.BytesIO()
            pages[0].save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        pass
    return ""


# --- agent 1: footfall intelligence analyst ---

def footfall_analysis_node(state):
    # analyses IoT behavioural data — dwell time, conversion, demographics by zone
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1,
                     openai_api_key=state["api_key"])

    df = load_data("iot")
    store = state.get("store_filter", "all")
    week = state.get("week_filter", "latest")

    # Apply filters
    if week == "latest":
        target_week = get_latest_week(df)
    else:
        target_week = week

    mask = df['Week'] == target_week
    if store != "all":
        mask &= df['Store'].str.lower() == store.lower()

    filtered = df[mask].copy()

    if filtered.empty:
        state["footfall_analysis"] = f"No IoT data found for store={store}, week={target_week}"
        state["session_log"].append("Agent 1 (Footfall): No data found")
        return state

    # Aggregate by zone + category
    agg = filtered.groupby(['Zone', 'Category']).agg(
        Total_Visitors=('Visitor_Count', 'sum'),
        Avg_Dwell_Sec=('Avg_Dwell_Time_Sec', 'mean'),
        Avg_Journey=('Product_Journey_Count', 'mean'),
        Avg_Conversion=('Basket_Conversion_Rate_%', 'mean'),
        Avg_Competitor_Int=('Competitor_Interaction_%', 'mean'),
        Female_Pct=('Gender_Female_%', 'mean'),
        Male_Pct=('Gender_Male_%', 'mean'),
        Peak_Visits=('Peak_Hour_Flag', lambda x: (x=='Y').sum()),
    ).reset_index()

    # Format as structured text for LLM
    data_text = f"IoT BEHAVIOURAL DATA — {target_week}"
    if store != "all":
        data_text += f" | Store: {store}"
    data_text += f"\nTotal observation records: {len(filtered)}\n\n"

    for _, row in agg.iterrows():
        data_text += (
            f"Zone: {row['Zone']} | Category: {row['Category']}\n"
            f"  Visitors: {int(row['Total_Visitors'])} | "
            f"Avg Dwell: {int(row['Avg_Dwell_Sec'])}s | "
            f"Avg Product Journey: {row['Avg_Journey']:.1f} products touched\n"
            f"  Basket Conversion: {row['Avg_Conversion']:.1f}% | "
            f"Competitor Interaction: {row['Avg_Competitor_Int']:.1f}%\n"
            f"  Gender Split: F={row['Female_Pct']:.0f}% M={row['Male_Pct']:.0f}%\n\n"
        )

    # Agent prompt
    response = llm.invoke([
        SystemMessage(content="""You are a Retail Footfall Intelligence Analyst specialising in 
IoT camera data interpretation. You have expertise in customer journey analytics, 
behavioural patterns, and conversion optimisation in large format retail stores.
Your analysis should identify: high-opportunity zones (high footfall, low conversion),
friction points (high dwell, low purchase), demographic mismatches with product range,
and peak hour patterns that should influence VM decisions."""),
        HumanMessage(content=f"""Analyse this IoT behavioural data and identify key insights:

{data_text}

Provide:
1. TOP 3 OPPORTUNITY ZONES: High visitor count but low basket conversion (untapped potential)
2. FRICTION POINTS: High dwell time but low conversion (customer interested but not buying)
3. DEMOGRAPHIC INSIGHTS: Where gender/age mix suggests VM or range adjustments
4. PEAK HOUR PATTERNS: Which zones need extra VM attention during peak hours
5. COMPETITOR THREAT ZONES: Where competitor interaction is high (risk of losing customers)

Be specific with zone names, category names, and percentages from the data.""")
    ])

    state["footfall_analysis"] = response.content
    state["session_log"].append(f"Agent 1 (Footfall Analysis): Completed — {target_week}")
    state["current_node"] = "footfall_analysis"
    return state


# --- agent 2: sales performance analyst ---

def sales_analysis_node(state):
    # correlates sales with footfall patterns to find VM opportunities
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1,
                     openai_api_key=state["api_key"])

    sales_df = load_data("sales")
    sku_df = load_data("sku")
    store = state.get("store_filter", "all")
    week = state.get("week_filter", "latest")

    if week == "latest":
        target_week = get_latest_week(sales_df)
    else:
        target_week = week

    # Sales summary
    mask = sales_df['Week'] == target_week
    if store != "all":
        mask &= sales_df['Store'].str.lower() == store.lower()

    sales = sales_df[mask].copy()

    # Category performance
    cat_perf = []
    for _, row in sales.iterrows():
        try:
            actual = float(row['Actual_INR_L'])
            target = float(row['Target_INR_L'])
            ach = round(actual/target*100, 1) if target > 0 else 0
            lw = float(row['LW_INR_L'])
            wow = round((actual/lw-1)*100, 1) if lw > 0 else 0
            cat_perf.append(f"{row['Category']} @ {row['Store']}: "
                           f"INR {actual}L, Ach {ach}%, WoW {wow:+.1f}%")
        except Exception:
            pass

    # Brand sell-through from SKU data
    sku_mask = sku_df['Week'] == target_week
    if store != "all":
        sku_mask &= sku_df['Store'].str.lower() == store.lower()

    sku_filtered = sku_df[sku_mask].copy()
    brand_agg = sku_filtered.groupby(['Category', 'Brand']).agg(
        Units=('Units_Sold', 'sum'),
        Avg_ST=('Sell_Through_%', 'mean'),
        Low_Stock=('Days_Cover', lambda x: (pd.to_numeric(x, errors='coerce') < 14).sum())
    ).reset_index().sort_values('Avg_ST', ascending=False)

    brand_text = ""
    for _, row in brand_agg.head(20).iterrows():
        brand_text += (f"{row['Brand']} ({row['Category']}): "
                      f"{int(row['Units'])} units, ST {row['Avg_ST']:.1f}%, "
                      f"Low stock SKUs: {int(row['Low_Stock'])}\n")

    footfall_context = state.get("footfall_analysis", "")[:500]

    response = llm.invoke([
        SystemMessage(content="""You are a Senior Retail Sales Performance Analyst.
You specialise in correlating customer behaviour data with sales outcomes to identify
where Visual Merchandising changes would have the highest revenue impact.
Focus on: brands performing above expectations despite poor placement,
categories with strong footfall but weak sales (VM opportunity),
and brands with high sell-through that need more gondola space."""),
        HumanMessage(content=f"""Correlate sales performance with IoT footfall patterns.

SALES DATA ({target_week}):
{chr(10).join(cat_perf[:15])}

BRAND SELL-THROUGH:
{brand_text}

FOOTFALL CONTEXT (from Agent 1):
{footfall_context}

Identify:
1. VM OPPORTUNITY BRANDS: High sell-through but likely poor placement (needs more space/better position)
2. CATEGORY MISMATCHES: High footfall zone but underperforming category (wrong VM for the audience)
3. QUICK WIN GONDOLA CHANGES: Specific brand repositioning that would lift sales
4. STOCK ALERT BRANDS: High sell-through + low days cover (risk of losing sales due to stockout)

Use specific brand names, category names, and numbers from the data.""")
    ])

    state["sales_analysis"] = response.content
    state["session_log"].append(f"Agent 2 (Sales Analysis): Completed — {target_week}")
    state["current_node"] = "sales_analysis"
    return state


# --- agent node 3 — vm and planogram analyst (multimodal) ---

def vm_analysis_node(state):
    # reads gondola data + planogram PDF via GPT-4o vision, finds compliance gaps
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1,
                     openai_api_key=state["api_key"])

    gondola_df = load_data("gondola")
    store = state.get("store_filter", "all")

    # Filter gondola data
    if store != "all":
        gondola = gondola_df[gondola_df['Store'].str.lower() == store.lower()].copy()
    else:
        gondola = gondola_df.copy()

    # Compliance summary
    compliance_gaps = gondola[gondola['Compliance_Gap'] == 'YES']
    compliance_text = f"Gondola Allocation Analysis:\n"
    compliance_text += f"Total fixtures: {len(gondola)} | Compliance gaps: {len(compliance_gaps)}\n\n"

    for _, row in compliance_gaps.head(15).iterrows():
        compliance_text += (
            f"GAP: {row['Brand']} in {row['Zone']} ({row['Store']})\n"
            f"  Current: {row['Shelf_Level']} | Required: {row['Planogram_Specified_Level']}\n"
            f"  VM Score: {row['VM_Compliance_Score_%']}%\n\n"
        )

    # Try multimodal planogram reading
    planogram_insight = ""
    pdf_path = str(Path(DATA_DIR) / "08_Store_Planogram.pdf")
    img_b64 = encode_pdf_page(pdf_path)

    if img_b64:
        try:
            vision_response = llm.invoke([
                SystemMessage(content="You are a VM planogram expert. Analyse retail store floor plans."),
                HumanMessage(content=[
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"}},
                    {"type": "text",
                     "text": """Analyse this retail store planogram floor plan.
Extract:
1. Which categories are assigned to which zones (Zone A, B, C, D, E)
2. Which brands are specified for premium positions (eye level, front bays)
3. Any special fixtures mentioned (shop-in-shop, brand walls, seated display)
4. VM compliance rules stated in the guidelines table
Be specific about zone names and brand names visible in the document."""}
                ])
            ])
            planogram_insight = f"\nPLANOGRAM ANALYSIS (GPT-4o Vision):\n{vision_response.content}"
        except Exception as e:
            planogram_insight = f"\nPlanogram vision analysis unavailable: {e}"
    else:
        # Fallback: text-based planogram guidelines
        planogram_insight = """
PLANOGRAM GUIDELINES (Standard):
- Zone A (Entrance): Ladies Western front bay — AND/W brands eye level priority
- Zone B Left: Men Formal — Van Heusen/Arrow eye level mandatory
- Zone C Centre: Ladies Ethnic — Biba shop-in-shop, minimum 4 linear feet
- Zone D Right: Ladies/Men Footwear — Premium brands at zone entrance
- Zone E Back: Sportswear — Nike/Adidas separate brand walls"""

    footfall_context = state.get("footfall_analysis", "")[:400]
    sales_context = state.get("sales_analysis", "")[:400]

    response = llm.invoke([
        SystemMessage(content="""You are a Senior Visual Merchandising and Planogram Compliance Analyst.
You review gondola allocation data against planogram guidelines to produce
prioritised VM action recommendations. Focus on compliance gaps that are
costing the most revenue, and quick wins that require minimal disruption."""),
        HumanMessage(content=f"""Review gondola compliance and recommend VM actions.

{compliance_text}

{planogram_insight}

FOOTFALL CONTEXT: {footfall_context}
SALES CONTEXT: {sales_context}

Produce:
1. CRITICAL COMPLIANCE GAPS: Brands in wrong shelf position that are hurting conversion
2. QUICK WIN VM CHANGES: Gondola adjustments that can be done this week
3. FACE COUNT RECOMMENDATIONS: Brands that deserve more/less gondola space
4. ZONE OPTIMISATION: Any zone-level changes to better match customer flow
5. VM CALENDAR ALIGNMENT: VM actions to prepare for upcoming promotions

Prioritise by revenue impact. Be specific with store names, brand names, zone names.""")
    ])

    state["vm_analysis"] = response.content
    state["session_log"].append("Agent 3 (VM Analysis): Completed — Planogram reviewed")
    state["current_node"] = "vm_analysis"
    return state


# --- agent 4: store action plan writer ---

def action_plan_node(state):
    # synthesises all 3 agents + promo calendar into a ranked action plan
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1,
                     openai_api_key=state["api_key"])

    promo_df = load_data("promo")
    store = state.get("store_filter", "all")
    week = state.get("week_filter", "latest")

    if week == "latest":
        from pandas import to_numeric
        target_week = get_latest_week(promo_df)
    else:
        target_week = week

    # Get upcoming promos
    weeks_sorted = sorted(promo_df['Week'].unique())
    try:
        curr_idx = weeks_sorted.index(target_week)
        upcoming_weeks = weeks_sorted[curr_idx:curr_idx+4]
    except ValueError:
        upcoming_weeks = weeks_sorted[-4:]

    promo_mask = promo_df['Week'].isin(upcoming_weeks)
    if store != "all":
        promo_mask &= (promo_df['Store'].str.lower() == store.lower()) | \
                      (promo_df['Store_Scope'].isin(['Chain', 'Regional']))

    upcoming_promos = promo_df[promo_mask].copy()
    promo_text = "UPCOMING PROMOTIONS (next 4 weeks):\n"
    for _, row in upcoming_promos.head(10).iterrows():
        promo_text += (f"  {row['Week']}: {row['Promo_Name']} — {row['Category']} | "
                      f"Type: {row['Offer_Type']} | Priority: {row['Priority_Level']} | "
                      f"VM Support: {row['VM_Support_Required']}\n")

    response = llm.invoke([
        SystemMessage(content="""You are a Senior Retail Store Operations Consultant.
You synthesise IoT behavioural data, sales performance, VM compliance analysis,
and the promotional calendar into a concrete monthly store action plan.
Each action must have: priority, owner (Store Manager/Category Manager/VM Team),
effort (hours), expected impact, and specific details.
Actions must be executable — not generic advice."""),
        HumanMessage(content=f"""Synthesise all analyses into a monthly store action plan.

FOOTFALL ANALYSIS SUMMARY:
{state.get('footfall_analysis', '')[:600]}

SALES ANALYSIS SUMMARY:
{state.get('sales_analysis', '')[:600]}

VM ANALYSIS SUMMARY:
{state.get('vm_analysis', '')[:600]}

{promo_text}

Generate a MONTHLY STORE ACTION PLAN with exactly this structure:

**EXECUTIVE SUMMARY**
2-3 sentences on the biggest opportunity this month.

**PRIORITY ACTIONS** (numbered 1-7, sorted by revenue impact):
For each action:
- Action: [specific gondola/VM/promo action]
- Category/Brand: [specific names]
- Zone/Fixture: [specific location]
- Owner: [Store Manager / Category Manager / VM Team]
- Effort: [estimated hours]
- Expected Impact: [projected conversion uplift or revenue]
- Rationale: [1 line — which data signal drives this]

**PROMO READINESS**
VM actions needed before each upcoming promotion.

**METRICS TO TRACK**
3 KPIs to measure success of these actions next week.""")
    ])

    state["action_plan"] = response.content
    state["session_log"].append("Agent 4 (Action Plan): Draft completed")
    state["current_node"] = "action_plan"
    return state


# --- human approval ---

def human_approval_node(state):
    # interrupt point — execution pauses here for human review
    # This node just marks that we are at the approval stage
    # The actual interrupt is handled by LangGraph's interrupt mechanism
    # In Streamlit, we check current_node == "awaiting_approval"
    state["current_node"] = "awaiting_approval"
    state["session_log"].append("Human approval checkpoint reached")
    return state


def process_decisions_node(state):
    # process approvals and log via MCP tools
    from tools import log_vm_action, get_store_layout

    decisions = state.get("human_decisions", {})
    approved = []
    rejected = []

    for action_id, decision in decisions.items():
        if decision["status"] in ["approve", "modified"]:
            action_record = {
                "action_id": action_id,
                "description": decision.get("description", ""),
                "quantity": decision.get("quantity", ""),
                "status": decision["status"],
                "approved_by": "Store Manager",
            }
            # Call MCP tool
            result = log_vm_action(
                store=state.get("store_filter", "all"),
                action_id=action_id,
                description=decision.get("description", ""),
                status=decision["status"]
            )
            action_record["tool_result"] = result
            approved.append(action_record)
        else:
            rejected.append({
                "action_id": action_id,
                "reason": decision.get("reason", "Rejected by manager"),
            })

    state["approved_actions"] = approved
    state["rejected_actions"] = rejected
    state["session_log"].append(
        f"Decisions processed: {len(approved)} approved, {len(rejected)} rejected"
    )
    state["current_node"] = "decisions_processed"
    return state


def generate_report_node(state):
    # generate session summary report
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1,
                     openai_api_key=state["api_key"])

    approved = state.get("approved_actions", [])
    rejected = state.get("rejected_actions", [])

    approved_text = "\n".join([f"- {a['action_id']}: {a['description']} ({a['status']})"
                               for a in approved]) or "None"
    rejected_text = "\n".join([f"- {r['action_id']}: {r.get('reason','')}"
                               for r in rejected]) or "None"

    response = llm.invoke([
        SystemMessage(content="You are a retail operations report writer."),
        HumanMessage(content=f"""Generate a concise session completion report.

APPROVED ACTIONS ({len(approved)}):
{approved_text}

REJECTED ACTIONS ({len(rejected)}):
{rejected_text}

ORIGINAL ACTION PLAN:
{state.get('action_plan', '')[:400]}

Write a 1-page session summary:
- What was analysed
- Key findings from IoT, sales, and VM data
- Actions approved and their expected combined impact
- Actions rejected and why this matters
- What to measure next week to track progress""")
    ])

    state["final_report"] = response.content
    state["session_log"].append("Final report generated")
    state["current_node"] = "complete"
    return state


# --- workflow builder ---

def build_workflow():
    # sequential graph: footfall → sales → vm → action_plan → approval → decisions → report
    # interrupt_before=process_decisions pauses for human input
    workflow = StateGraph(StoreExperienceState)

    # Add nodes
    workflow.add_node("footfall_analysis", footfall_analysis_node)
    workflow.add_node("sales_analysis", sales_analysis_node)
    workflow.add_node("vm_analysis", vm_analysis_node)
    workflow.add_node("action_plan", action_plan_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("process_decisions", process_decisions_node)
    workflow.add_node("generate_report", generate_report_node)

    # Add edges (sequential flow)
    workflow.set_entry_point("footfall_analysis")
    workflow.add_edge("footfall_analysis", "sales_analysis")
    workflow.add_edge("sales_analysis", "vm_analysis")
    workflow.add_edge("vm_analysis", "action_plan")
    workflow.add_edge("action_plan", "human_approval")
    workflow.add_edge("human_approval", "process_decisions")
    workflow.add_edge("process_decisions", "generate_report")
    workflow.add_edge("generate_report", END)

    # Compile with memory checkpointer for human-in-the-loop
    memory = MemorySaver()
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["process_decisions"]  # Pause before processing decisions
    )


def get_initial_state(store, week, api_key):
    # fresh state for a new session
    return StoreExperienceState(
        store_filter=store,
        week_filter=week,
        api_key=api_key,
        footfall_analysis="",
        sales_analysis="",
        vm_analysis="",
        action_plan="",
        human_decisions={},
        approved_actions=[],
        rejected_actions=[],
        final_report="",
        session_log=[],
        current_node="start",
        error=None,
    )
