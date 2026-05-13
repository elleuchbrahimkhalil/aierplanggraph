# ERP AI Assistant (LangGraph + Ollama + Column Router)

Ce dossier contient l'assistant **Python** orchestré par **LangGraph**.

L'objectif: prendre une question métier (client), choisir automatiquement les bonnes routes WebApi, charger les données, appliquer des opérations (filtre / agrégation) via DataFrame, puis renvoyer un JSON exploitable par le frontend (tableau + dashboard Chart.js).

## Flow (solution)

1) **Question client (React)**
- Le frontend appelle `POST /assistant/query` avec `{ "question": "..." }`.

2) **Extraction (Ollama)**
- Le nœud `extract_user_request` appelle Ollama et récupère un JSON:
  - `entity`
  - `requested_fields` (colonnes à afficher)
  - `extracted_params` (filtres + `group_by` + `aggregations` + identifiants)
  - `missing`, `confidence`

3) **Routage local (column_router.py)**
- Le nœud `route_endpoint` passe l'analyse au router local `column_router`.
- Le router compare les **paramètres + colonnes** des endpoints disponibles et sélectionne **un ou plusieurs** endpoints.

4) **Appel WebApi**
- Le nœud `call_webapi` exécute les endpoints sélectionnés (GET), gère un bearer token si configuré, puis persiste les résultats.
- Sortie cache: `ai_assistant/data/cache/last_api_result.json`.

5) **Transformation (DataFrame)**
- Le nœud `generate_transform_plan` construit un plan JSON minimal à partir de `extracted_params`.
- Le nœud `evidence_filter` charge le JSON, convertit en DataFrame (pandas si dispo) et applique:
  - `filter_rows`
  - `aggregate` (group_by + sum/avg/count/min/max)
  - `select`, `sort`, `limit`
- Sortie cache: `ai_assistant/data/cache/last_transform_plan.json`.

6) **Résultat + Dashboard**
- Le backend persiste `ai_assistant/data/cache/last_display_result.json`.
- Le frontend affiche un tableau + des graphiques via **Chart.js** (ex: bar/pie) à partir des lignes JSON.

## Run (assistant seul)

Depuis la racine du projet:

```powershell
python ai_assistant/langgraph_skeleton.py --serve --host 127.0.0.1 --port 8000
```

## Mode conversationnel (thread_id)

Le endpoint `POST /assistant/query` accepte un identifiant de session:

```json
{
  "question": "affiche les clients",
  "thread_id": "<uuid>"
}
```

Si tu réutilises le même `thread_id` sur plusieurs requêtes, le backend conserve `history` (mémoire en RAM) et l'utilise pour améliorer l'extraction/routage.

## One-command stack (backend)

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\ai_assistant\start_stack.ps1
```

Ce script:
- démarre Ollama (optionnel) + WebApi (si nécessaire)
- configure les variables d'environnement
- démarre le serveur HTTP de l'assistant

## One-command stack (full project)

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\start_all.ps1
```

## Variables d'environnement utiles

- `ERP_ENDPOINTS_JSON`: fichier endpoints (par défaut: `ai_assistant/data/endpoints.get.json`)
- `ERP_API_BASE_URL`: base URL WebApi (ex: `http://localhost:5006`)
- `ERP_API_BEARER_TOKEN`: token bearer (ou utiliser auth ci-dessous)
- `ERP_API_AUTH_URL`, `ERP_API_USERNAME`, `ERP_API_PASSWORD`, `ERP_API_SOCIETE`: récupération automatique du token

Mode “solution” (par défaut dans `start_stack.ps1`):
- `ERP_ROUTER_MODE=columns` (router local)
- `ERP_LOCAL_TRANSFORM=1` (plan local + DataFrame)

Optionnel:
- `ERP_GENERATE_TEXT_ANSWER=1` pour générer une réponse texte via Ollama (sinon l'UI se base surtout sur le tableau + charts)

## Comparaison Chart.js vs Seaborn

- Le backend expose `GET /assistant/seaborn.png` : rend un PNG Seaborn basé sur le dernier cache `ai_assistant/data/cache/last_display_result.json`.
- Dépendances Python nécessaires: `seaborn`, `matplotlib`, `pandas`.
