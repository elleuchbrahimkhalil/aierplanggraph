from langgraph_skeleton import _load_endpoints, _load_swagger_generated_endpoints, _load_endpoint_overrides
import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent

endpoints = _load_endpoints()

# Save unified
unified_path = ROOT_DIR / 'data' / 'all_endpoints.json'
unified_path.write_text(json.dumps({'endpoints': endpoints}, indent=2, ensure_ascii=False), encoding='utf-8')

print(f'Saved {len(endpoints)} endpoints to {unified_path}')

