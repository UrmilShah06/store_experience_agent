# tools.py
# MCP tool definitions — get_store_layout and log_vm_action
# in production these would be FastAPI endpoints; here they read from Excel and write to CSV

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd

DATA_DIR = "./data"
ACTION_LOG = "./vm_action_log.csv" # appends on each run, does not overwrite


# --- tool 1: get_store_layout ---

def get_store_layout(store, category=None):
    # returns gondola allocation for a store as JSON
    # production: would call store ops DB via API; here reads from Excel
    try:
        path = Path(DATA_DIR) / "06_Gondola_Allocation.xlsx"
        df = pd.read_excel(path)
        df.columns = df.columns.str.strip()

        mask = df['Store'].str.lower() == store.lower()
        if category:
            mask &= df['Category'].str.lower() == category.lower()

        filtered = df[mask]

        # tried returning a default empty layout here but not_found is cleaner
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
                "vm_compliance_score": int(row['VM_Compliance_Score_%']),
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


# --- tool 2: log_vm_action ---

def log_vm_action(store: str, action_id: str, description: str,
                  status: str, notes: str = "") -> str:
    # logs approved VM action to CSV tracker
    # production: would write to store ops system and notify VM team
    try:
        log_path = Path(ACTION_LOG)
        
        # Create file with headers if it does not exist
        if not log_path.exists():
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "store", "action_id", "description",
                    "status", "notes", "logged_by"
                ])

        # Append the new action
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


# --- mcp server simulation ---

def get_mcp_tool_registry() -> dict:
    # tool manifest — clients call this to discover available tools
    # pattern: /tools to discover, /invoke/{name} to call
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
    # route tool calls to the right function
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
