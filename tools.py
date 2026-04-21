# tools.py
# MCP tool definitions for the store experience agent
# get_store_layout reads gondola data, log_vm_action writes approvals to CSV
# in production these would be proper API endpoints — this is the demo version
#
# worth noting: MCP here is really just a structured JSON contract, not a live server
# the real MCP integration would have these registered with a tool registry
# and called over HTTP — keeping it simple for the portfolio demo

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd

DATA_DIR = "./data"
ACTION_LOG = "./vm_action_log.csv"  # appends on each run


def _normalise_store_name(name):
    # store names in the gondola file have inconsistent casing sometimes
    # this just lowercases for the mask — simple but saved a KeyError once
    # considered doing fuzzy match here but that felt like overkill for a demo
    return name.strip().lower()

# tried a fancier version that also stripped underscores and did partial matching
# e.g. "koramangala" would match "Bangalore_Koramangala"
# worked fine until "Chennai" matched both Chennai_AnnaNagar and a test row — scrapped it
# def _fuzzy_store_match(df, name):
#     name_clean = name.lower().replace('_', '')
#     return df['Store'].apply(lambda s: name_clean in s.lower().replace('_', ''))


def get_store_layout(store, category=None):
    # returns gondola allocation for a store as JSON
    # tried returning full chain data when store filter is missing but
    # the payload got too large for the LLM context — filtered is better
    try:
        path = Path(DATA_DIR) / "06_Gondola_Allocation.xlsx"
        df = pd.read_excel(path)
        df.columns = df.columns.str.strip()

        mask = df['Store'].str.lower() == _normalise_store_name(store)
        if category:
            mask &= df['Category'].str.lower() == category.lower()

        filtered = df[mask]

        if filtered.empty:
            return json.dumps({"status": "not_found",
                               "message": f"No layout data for {store}"})

        layout = []
        for _, row in filtered.iterrows():
            layout.append({
                "gondola_id": row['Gondola_ID'],
                "zone": row['Zone'],
                "category": row['Category'],
                "brand": row['Brand'],
                "fixture_type": row['Fixture_Type'],
                "shelf_level": row['Shelf_Level'],
                "face_count": int(row['Face_Count']),
                "linear_feet": float(row['Linear_Feet']),
                "vm_compliance_score": int(row['VM_Compliance_Score_%']),  # stored as 0-100 integer
                "compliance_gap": row['Compliance_Gap'],
                "planogram_specified_level": row['Planogram_Specified_Level'],
            })

        return json.dumps({
            "status": "success",
            "store": store,
            "category": category or "all",
            "total_fixtures": len(layout),
            "compliance_gaps": sum(1 for l in layout if l['compliance_gap'] == 'YES'),
            "layout": layout
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def log_vm_action(store, action_id, description, status, notes=""):
    # logs approved VM action to CSV
    # considered SQLite here but CSV is easier to inspect and the volume is low
    try:
        log_path = Path(ACTION_LOG)

        if not log_path.exists():
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "store", "action_id", "description",
                    "status", "notes", "logged_by"
                ])

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                store, action_id, description, status, notes,
                "Store_Manager"
            ])

        return json.dumps({
            "status": "logged",
            "action_id": action_id,
            "store": store,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": f"VM action {action_id} logged successfully. "
                       f"Store VM team will receive notification.",
            "next_step": "VM team to execute within 48 hours. "
                        "Compliance check scheduled for next week."
        })

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def get_mcp_tool_registry():
    # tool manifest so clients know what tools are available
    
    return {
        "mcp_version": "1.0",
        "server_name": "store_experience_agent",
        "tools": [
            {
                "name": "get_store_layout",
                "description": "Returns current gondola allocation and VM compliance status for a store",
                "parameters": {
                    "store": {"type": "string", "required": True,
                              "description": "Store name e.g. Bangalore_Koramangala"},
                    "category": {"type": "string", "required": False,
                                 "description": "Optional category filter"}
                },
                "returns": "JSON with gondola layout, compliance gaps, and fixture details"
            },
            {
                "name": "log_vm_action",
                "description": "Logs an approved VM gondola change to the implementation tracker",
                "parameters": {
                    "store": {"type": "string", "required": True},
                    "action_id": {"type": "string", "required": True,
                                  "description": "Unique action ID e.g. VM_001"},
                    "description": {"type": "string", "required": True,
                                    "description": "What VM change was approved"},
                    "status": {"type": "string", "required": True,
                               "enum": ["approve", "modified"]},
                    "notes": {"type": "string", "required": False}
                },
                "returns": "JSON confirmation with timestamp and next steps"
            }
        ]
    }


def invoke_mcp_tool(tool_name, parameters):
    # routes incoming tool calls — straightforward dispatch
    # in a real MCP server this would be handled by the framework routing layer
    if tool_name == "get_store_layout":
        return get_store_layout(
            store=parameters.get("store", ""),
            category=parameters.get("category")
        )
    elif tool_name == "log_vm_action":
        return log_vm_action(
            store=parameters.get("store", ""),
            action_id=parameters.get("action_id", ""),
            description=parameters.get("description", ""),
            status=parameters.get("status", "approve"),
            notes=parameters.get("notes", "")
        )
    else:
        return json.dumps({"status": "error",
                           "message": f"Unknown tool: {tool_name}"})
