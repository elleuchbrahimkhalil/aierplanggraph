#!/usr/bin/env python
import subprocess
import sys

# Install dependencies
print("[*] Installing python deps (pandas, seaborn)...")
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "pandas",
        "openpyxl",
        "matplotlib",
        "seaborn",
    ]
)
print("[OK] deps installed")

# Now test the transform
print("\n[*] Testing transform...")
sys.path.insert(0, ".")
from ai_assistant.langgraph_skeleton import _apply_transform_plan

data = [
    {"articleCode": "A1", "quantite": 2},
    {"articleCode": "A1", "quantite": 3},
    {"articleCode": "A2", "quantite": 5},
]
plan = {
    "steps": [
        {
            "op": "aggregate",
            "groupby": ["articleCode"],
            "aggs": [{"field": "quantite", "agg": "sum", "as": "total_qty"}],
        },
        {"op": "limit", "value": 100},
    ]
}

try:
    res = _apply_transform_plan(data, plan)
    print("[OK] Transform result:")
    for row in res:
        print(f"  {row}")
    print(f"\n[OK] Total rows: {len(res)}")
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
