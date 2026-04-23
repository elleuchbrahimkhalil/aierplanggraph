import json
import os

filepath = "ai_assistant/data/endpoints.get.json"
if not os.path.exists(filepath):
    print(f"Error: File {filepath} not found.")
else:
    try:
        # Essayer de lire avec utf-8-sig pour gérer le BOM
        with open(filepath, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        
        if "endpoints" not in data or not isinstance(data["endpoints"], list):
            print("Error: Root key 'endpoints' is missing or not a list.")
        else:
            endpoints = data["endpoints"]
            ids = [ep.get("id") for ep in endpoints if ep.get("id")]
            method_urls = [(ep.get("method"), ep.get("url")) for ep in endpoints if ep.get("method") and ep.get("url")]
            missing_fields = []
            target_found = False

            for i, ep in enumerate(endpoints):
                eid, method, url = ep.get("id"), ep.get("method"), ep.get("url")
                intent, keywords = ep.get("intent"), ep.get("keywords")
                
                if not all([eid, method, url, intent, keywords]):
                    missing_fields.append({"index": i, "id": eid, "missing": [f for f in ["id", "method", "url", "intent", "keywords"] if not ep.get(f)]})

                if url == "/api/Stats/VenteParArticle":
                    target_found = True

            duplicate_ids = [x for x in set(ids) if ids.count(x) > 1]
            duplicate_combos = [x for x in set(method_urls) if method_urls.count(x) > 1]

            print(f"--- Summary for {filepath} ---")
            print(f"Total endpoints: {len(endpoints)}")
            print(f"JSON Parse: OK")
            print(f"Root 'endpoints' list: OK")
            print(f"Duplicate IDs: {duplicate_ids if duplicate_ids else 'None'}")
            print(f"Duplicate (method, url): {duplicate_combos if duplicate_combos else 'None'}")
            print(f"Missing required fields: {len(missing_fields)} cases found")
            if missing_fields:
                for entry in missing_fields[:5]: print(f"  - Index {entry['index']} (ID: {entry['id']}): Missing {entry['missing']}")
                if len(missing_fields) > 5: print("  - ...")
            print(f"Presence of /api/Stats/VenteParArticle: {'Found' if target_found else 'Missing'}")
    except Exception as e:
        print(f"An error occurred: {e}")