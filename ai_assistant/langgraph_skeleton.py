from __future__ import annotations

import argparse
import json
import math
import numbers
import os
import re
import base64
import io
from functools import lru_cache
import threading
from http import HTTPStatus
from urllib.parse import parse_qs
from urllib.parse import urlencode
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import requests

try:
    from . import column_router
except ImportError:
    import column_router
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver


ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DISPLAY_RESULT_PATH = CACHE_DIR / "last_display_result.json"
TRANSFORM_PLAN_PATH = CACHE_DIR / "last_transform_plan.json"
DATASET_MAPPING_PATH = ROOT_DIR / "data" / "dataset_mapping_db.json"
ROUTING_CATALOG_PATH = ROOT_DIR / "data" / "routing_catalog.txt"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip("\"\'")
    except OSError:
        return


_load_env_file(ROOT_DIR / ".env")
_load_env_file(ROOT_DIR / ".env.auth")


def _is_jwt(value: str) -> bool:
    return value.count(".") == 2 and len(value) > 20


def _decode_jwt_exp(token: str) -> Optional[int]:
    try:
        payload = token.split(".")[1]
        pad = "=" * ((4 - len(payload) % 4) % 4)
        data = base64.urlsafe_b64decode((payload + pad).encode("utf-8"))
        obj = json.loads(data.decode("utf-8"))
        exp = obj.get("exp")
        return int(exp) if exp is not None else None
    except (ValueError, IndexError, json.JSONDecodeError):
        return None


def _is_token_expired(token: str, skew_seconds: int = 60) -> bool:
    exp = _decode_jwt_exp(token)
    if exp is None:
        return False
    return exp <= int(datetime.now(UTC).timestamp()) + skew_seconds


def _find_token_in_payload(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for key in ["token", "access_token", "jwt", "bearer"]:
            value = payload.get(key)
            if isinstance(value, str) and _is_jwt(value):
                return value
        for value in payload.values():
            found = _find_token_in_payload(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_token_in_payload(item)
            if found:
                return found
    elif isinstance(payload, str) and _is_jwt(payload):
        return payload
    return None


def _request_bearer_token(base_urls: List[str], errors: List[str]) -> Optional[str]:
    auth_url = os.getenv("ERP_API_AUTH_URL", "").strip()
    if not auth_url and base_urls:
        auth_url = f"{base_urls[0].rstrip('/')}/api/WebUser"
    if not auth_url:
        errors.append("ERP_API_AUTH_URL is not set.")
        return None

    payload_json = os.getenv("ERP_API_AUTH_PAYLOAD_JSON", "").strip()
    if payload_json:
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            errors.append("ERP_API_AUTH_PAYLOAD_JSON is invalid JSON.")
            return None
    else:
        username = os.getenv("ERP_API_USERNAME", "").strip()
        password = os.getenv("ERP_API_PASSWORD", "").strip()
        societe = os.getenv("ERP_API_SOCIETE", "").strip()
        if not username or not password:
            errors.append("ERP_API_USERNAME/ERP_API_PASSWORD are not set.")
            return None
        payload = {"username": username, "password": password}
        if societe:
            payload["societe"] = societe

    try:
        response = requests.post(auth_url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json() if response.content else {}
        token = _find_token_in_payload(data)
        if not token:
            errors.append("Auth response did not include a token.")
            return None
        os.environ["ERP_API_BEARER_TOKEN"] = token
        return token
    except requests.RequestException as exc:
        status = None
        body_preview = None
        try:
            resp = getattr(exc, "response", None)
            if resp is not None:
                status = resp.status_code
                text = (resp.text or "").strip()
                if text:
                    body_preview = text[:800]
        except Exception:
            status = None
            body_preview = None

        if status is not None:
            if body_preview:
                if status >= 500 and "Unable to connect to any of the specified MySQL hosts" in body_preview:
                    errors.append(
                        f"Auth request failed because MySQL is unreachable (WebApi DB down). HTTP {status} on {auth_url}: {body_preview}"
                    )
                else:
                    errors.append(f"Auth request failed: HTTP {status} on {auth_url}: {body_preview}")
            else:
                errors.append(f"Auth request failed: HTTP {status} on {auth_url}: {exc}")
        else:
            errors.append(f"Auth request failed: {exc}")
        return None


def _ensure_bearer_token(base_urls: List[str], errors: List[str]) -> Optional[str]:
    token = os.getenv("ERP_API_BEARER_TOKEN", "").strip()
    if token and not _is_token_expired(token):
        return token
    return _request_bearer_token(base_urls, errors)


def _safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize to JSON, converting NaN/Infinity to null to avoid invalid JSON output."""
    def _clean(item: Any) -> Any:
        if isinstance(item, numbers.Real) and not isinstance(item, bool):
            value = float(item)
            if math.isnan(value) or math.isinf(value):
                return None
            return item
        if isinstance(item, str) and item in {"NaN", "Infinity", "-Infinity"}:
            return None
        if isinstance(item, dict):
            return {k: _clean(v) for k, v in item.items()}
        if isinstance(item, list):
            return [_clean(v) for v in item]
        if isinstance(item, tuple):
            return [_clean(v) for v in item]
        return item
    kwargs.setdefault("allow_nan", False)
    return json.dumps(_clean(obj), **kwargs)


def _is_model_only_mode() -> bool:
    """Disable heuristic fallbacks to evaluate model-only behavior."""
    # Par défaut activé pour ce mode 'clean'
    return os.getenv("ERP_MODEL_ONLY_MODE", "1") == "1"


class AssistantState(TypedDict, total=False):
    # Input
    question: str
    history: List[Dict[str, str]]  # Historique des échanges

    # Model request understanding
    request_analysis: Dict[str, Any]
    extracted_params: Dict[str, Any]

    # Endpoint routing
    endpoint_candidates: List[Dict[str, Any]]
    selected_endpoints: List[Dict[str, Any]]
    selected_endpoint: Optional[Dict[str, Any]]

    # WebApi execution
    api_result_path: str

    # Model/data transformation
    transform_plan: Dict[str, Any]
    filtered_result: Dict[str, Any]

    # Answer/display persistence
    display_result_path: str
    display_result: Dict[str, Any]
    answer: str

    # Runtime status
    confidence: float
    errors: List[str]


# ---------- Utilities ----------

def _load_endpoints() -> List[Dict[str, Any]]:
    """
    Charge UNIQUEMENT les endpoints depuis le fichier JSON configurÃ©.
    Suppression de la logique Swagger et de la fusion.
    """
    configured = os.getenv("ERP_ENDPOINTS_JSON", "").strip()
    if configured:
        path = Path(configured)
    else:
        path = ROOT_DIR / "data" / "endpoints.get.json"

    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload.get("endpoints", []) if isinstance(payload, dict) else []
    except (OSError, json.JSONDecodeError):
        return []


def _load_routing_catalog_text(candidates: List[Dict[str, Any]]) -> str:
    # Charger les instructions du fichier si disponible, mais NE PAS retourner immÃ©diatement
    compact_lines = []
    if os.getenv("ERP_INCLUDE_STATIC_ROUTING_CATALOG", "0") == "1" and ROUTING_CATALOG_PATH.exists():
        try:
            static_instr = ROUTING_CATALOG_PATH.read_text(encoding="utf-8")
            if static_instr.strip():
                compact_lines.append("STATIC ROUTING NOTES")
                compact_lines.append(static_instr.strip())
                compact_lines.append("") # Add a blank line for separation
        except OSError:
            pass

    for endpoint in candidates:
        if not isinstance(endpoint, dict):
            continue
        if str(endpoint.get("method", "GET")).upper() != "GET":
            continue
        column_limit = int(os.getenv("ERP_ROUTER_COLUMN_LIMIT", "24"))
        columns = _endpoint_columns(endpoint, column_limit)
        columns_only = os.getenv("ERP_ROUTER_COLUMNS_ONLY", "1") == "1"
        if columns_only:
            compact_lines.append(
                f"- {endpoint.get('id')} | columns=[{', '.join(columns) or '-'}]"
            )
        else:
            examples_str = "" # SupprimÃ© pour rÃ©duire le contexte et Ã©viter les timeouts Ollama
            compact_lines.append(
                f"- {endpoint.get('id')} | GET {endpoint.get('url', '')} | role={endpoint.get('role', 'general')} | route=[{', '.join(_unique_strings(endpoint.get('routeParameters', []))) or '-'}] | query=[{', '.join(_unique_strings(endpoint.get('queryParameters', []))) or '-'}] | columns=[{', '.join(columns) or '-'}] | desc={_truncate_text(endpoint.get('description', ''), 160)}{examples_str}"
            )
    return "\n".join(compact_lines).strip() + "\n"


@lru_cache(maxsize=1)
def _load_dataset_mapping_db() -> List[Dict[str, Any]]:
    """
    Charge le fichier dataset_mapping_db.json pour fournir un contexte sÃ©mantique
    au modÃ¨le, sans filtrer les endpoints (laissant le modÃ¨le dÃ©cider).
    """
    if not DATASET_MAPPING_PATH.exists():
        return []

    try:
        payload = json.loads(DATASET_MAPPING_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return []

    return payload if isinstance(payload, list) else []


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _score_mapping_table(table: Dict[str, Any], question_tokens: set[str]) -> int:
    """Score a mapping table by lexical overlap with the question.

    This is intentionally simple and deterministic: it helps avoid giving the LLM
    a huge mapping context (which reduces precision).
    """
    if not isinstance(table, dict) or not question_tokens:
        return 0

    parts: List[str] = [
        str(table.get("table", "")),
        str(table.get("businessDomain", "")),
        str(table.get("description", "")),
    ]

    for col in (table.get("columns", []) or []):
        if not isinstance(col, dict):
            continue
        parts.append(str(col.get("name", "")))
        parts.extend([str(s) for s in (col.get("synonyms", []) or [])])

    text = " ".join(parts).lower()
    score = 0
    for tok in question_tokens:
        if tok and tok in text:
            score += 1

    # Prefer direct column-name hits.
    column_names = {
        str(c.get("name", "")).lower()
        for c in (table.get("columns", []) or [])
        if isinstance(c, dict) and c.get("name")
    }
    score += 2 * len(column_names.intersection(question_tokens))
    return score


def _pick_relevant_mapping_tables(
    mapping_db: List[Dict[str, Any]],
    question: str,
    max_tables: int,
) -> List[Dict[str, Any]]:
    if not mapping_db:
        return []

    tokens = set(_tokenize(question))
    decorated: List[tuple[int, int, Dict[str, Any]]] = []
    for idx, table in enumerate(mapping_db):
        if not isinstance(table, dict):
            continue
        decorated.append((_score_mapping_table(table, tokens), idx, table))

    decorated.sort(key=lambda item: (-item[0], item[1]))
    picked = [item[2] for item in decorated[: max(0, max_tables)]]

    # If everything scores 0 (no overlap), fall back to the first N to keep behavior stable.
    if picked and all(_score_mapping_table(t, tokens) == 0 for t in picked):
        return [t for t in mapping_db[: max(0, max_tables)] if isinstance(t, dict)]
    return picked



def _unique_strings(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for value in values:
        token = str(value).strip()
        if token and token not in out:
            out.append(token)
    return out


def _endpoint_columns(endpoint: Dict[str, Any], limit: Optional[int] = None) -> List[str]:
    columns = _unique_strings(endpoint.get("columns", []))
    if limit is None:
        return columns
    return columns[: max(0, limit)]



def _build_mapping_context(
    question: str,
    max_tables: int = 4,
    max_columns: int = 6,
) -> List[Dict[str, Any]]:
    # Default to curated mapping DB; endpoint-derived mapping is noisy and often
    # decreases extraction precision when many endpoints share similar columns.
    use_endpoints = os.getenv("ERP_MAPPING_FROM_ENDPOINTS", "0") == "1"
    if use_endpoints:
        endpoints = _load_endpoints()
        endpoint_limit = int(os.getenv("ERP_MAPPING_ENDPOINT_LIMIT", str(max_tables)))
        column_limit = int(os.getenv("ERP_MAPPING_COLUMN_LIMIT", str(max_columns)))
        return column_router.build_mapping_context_from_endpoints(
            endpoints,
            max_endpoints=endpoint_limit,
            max_columns=column_limit,
        )

    mapping_db = _load_dataset_mapping_db()
    context: List[Dict[str, Any]] = []
    selected_tables = _pick_relevant_mapping_tables(mapping_db, question=question, max_tables=max_tables)
    for table in selected_tables:
        if not isinstance(table, dict):
            continue
        columns: List[Dict[str, Any]] = []
        for column in table.get("columns", [])[:max_columns]:
            if not isinstance(column, dict):
                continue
            columns.append(
                {
                    "name": column.get("name"),
                    "description": column.get("description"),
                    "synonyms": list(column.get("synonyms", []))[:6],
                }
            )
        context.append(
            {
                "table": table.get("table"),
                "description": table.get("description"),
                "columns": columns,
            }
        )
    return context


def _is_supported_business_endpoint(endpoint: Dict[str, Any]) -> bool:
    if str(endpoint.get("method", "GET")).upper() != "GET":
        return False

    url = str(endpoint.get("url", "")).lower()
    endpoint_id = str(endpoint.get("id", "")).lower()
    description = str(endpoint.get("description", "")).lower()
    tags = " ".join(str(tag) for tag in endpoint.get("tags", [])).lower()
    endpoint_text = " ".join([url, endpoint_id, description, tags])

    blocked_terms = [
        "swagger", "openapi", "health", "generate-test", "testendpoint", 
        "debug", "token", "login", "signin", "auth"
    ]
    if any(term in endpoint_text for term in blocked_terms):
        return False

    return True


def _compute_endpoint_column_coverage(endpoint: Dict[str, Any], requested_fields: List[str]) -> int:
    if not requested_fields:
        return 0

    endpoint_text = " ".join(
        [
            str(endpoint.get("id", "")),
            str(endpoint.get("url", "")),
            str(endpoint.get("description", "")),
            " ".join(str(k) for k in endpoint.get("keywords", [])),
            " ".join(str(tag) for tag in endpoint.get("tags", [])),
            " ".join(_endpoint_columns(endpoint)),
        ]
    ).lower()
    endpoint_tokens = set(_tokenize(endpoint_text))

    score = 0
    for field in requested_fields:
        if str(field).lower() in endpoint_text:
            score += 3
        elif set(_tokenize(str(field).lower())).issubset(endpoint_tokens):
            score += 2

    return score


def _rerank_endpoints_by_requested_fields(
    endpoints: List[Dict[str, Any]],
    requested_fields: List[str],
) -> List[Dict[str, Any]]:
    normalized_fields = _unique_strings(requested_fields)
    if not normalized_fields or len(endpoints) <= 1:
        return endpoints

    decorated: List[tuple[int, int, Dict[str, Any]]] = []
    for idx, endpoint in enumerate(endpoints):
        coverage = _compute_endpoint_column_coverage(endpoint, normalized_fields)
        decorated.append((coverage, idx, endpoint))

    decorated.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in decorated]


def _normalize_request_analysis(
    analysis: Dict[str, Any],
    question: str,
    previous_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Nettoie l'analyse du modÃ¨le.
    """
    normalized = analysis if isinstance(analysis, dict) else {}
    extracted_params = normalized.get("extracted_params", {})
    if not isinstance(extracted_params, dict):
        extracted_params = {}

    # On garde ce que le modÃ¨le a renvoyÃ©
    requested_fields = _unique_strings(
        normalized.get("requested_fields", extracted_params.get("requested_fields", []))
    )
    if requested_fields:
        extracted_params["requested_fields"] = requested_fields
    missing = _unique_strings(normalized.get("missing", []))

    entity = str(normalized.get("entity", "") or "").strip().lower()
    
    # Logique de fusion conversationnelle simple
    if previous_analysis and not entity:
        entity = previous_analysis.get("entity", "")
    if previous_analysis and entity == previous_analysis.get("entity"):
        # Fusionne les anciens paramètres avec les nouveaux pour permettre l'affinement (drill-down)
        merged_params = dict(previous_analysis.get("extracted_params", {}))
        merged_params.update(extracted_params)
        extracted_params = merged_params

    try: # Ensure confidence is a float
        confidence_value = float(normalized.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence_value = 0.0

    return {
        "question": question,
        "entity": entity,
        "requested_fields": requested_fields,
        "extracted_params": extracted_params,
        "missing": missing,
        "confidence": max(0.0, min(confidence_value, 1.0)),
    }


def _analysis_is_weak(analysis: Dict[str, Any]) -> bool:
    if not isinstance(analysis, dict):
        return True

    entity = str(analysis.get("entity", "") or "").strip().lower()
    requested_fields = _unique_strings(analysis.get("requested_fields", []))
    extracted_params = analysis.get("extracted_params", {})
    if not isinstance(extracted_params, dict):
        extracted_params = {}

    usable_params = {
        str(key): value
        for key, value in extracted_params.items()
        if key not in {"requested_fields", "aggregations", "group_by", "groupby", "metrics"}
        and value not in (None, "", [], {})
    }

    try:
        confidence = float(analysis.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    if entity and (requested_fields or usable_params):
        return False
    if entity and confidence >= 0.55:
        return False
    if requested_fields and confidence >= 0.55:
        return False
    return True


def _extract_request_with_llama(question: str, history: List[Dict[str, str]] = None) -> tuple[Dict[str, Any], Optional[str]]:
    """
    Extraction purement basÃ©e sur le modÃ¨le.
    """
    extractor_model = os.getenv("OLLAMA_MODEL_PARAM_EXTRACTOR", "deepseek-coder:6.7b")
    mapping_context = _build_mapping_context(question=question) # Use default reduced values
    
    history_str = ""
    if history:
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
    
    system_prompt = (
        "You are an ERP request understanding assistant. "
        "Your job is to identify the business entity and parameters. "
        "Return only JSON with keys: entity, requested_fields, extracted_params, missing, confidence. Do not include 'intent'."
    )
    user_prompt = (
        f"Conversation History:\n{history_str}\n\n"
        f"Question: {question}\n"
        f"Business mapping context: {json.dumps(mapping_context, ensure_ascii=False)}\n"
        "Rules:\n"
        "- entity must be a short business noun like clients, articles, factures, paiements.\n"
        "- requested_fields: List columns to display. If empty, infer standard fields (e.g. name, code).\n"
        "- Do not invent default columns such as code, raison, tel unless the user asked for them.\n"
        "- Distinguish display columns from filters: 'with email' or 'qui ont un email' usually means a filter on email presence, not automatically a request for telephone or code.\n"
        "- extracted_params must contain only useful filters, identifiers, dates, codes, group_by fields, or aggregations clearly implied by the question.\n"
        "- For aggregate questions, prefer extracted_params.group_by and extracted_params.aggregations.\n"
        "- Never mix entities: fournisseurs is not clients, articles is not factures.\n"
        "- missing must list required business details still absent from the question.\n"
        "- confidence must be a float between 0 and 1.\n"
        "- If the user did not request specific fields, return requested_fields as []."
    )

    try:
        llm_json = _call_ollama_json(extractor_model, system_prompt, user_prompt)
        if not isinstance(llm_json, dict):
            raise Exception("Request extraction returned invalid JSON object.")
        
        normalized = _normalize_request_analysis(
            analysis=llm_json,
            question=question,
        )
        return normalized, None
    except Exception as exc:
        raise Exception(f"Request extraction Ollama error: {exc}")


def _call_ollama_chat(model: str, system_prompt: str, user_prompt: str, timeout_seconds: Optional[int] = None) -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    if timeout_seconds is None:
        timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600")) # Augmenter le timeout par dÃ©faut
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
    top_p = float(os.getenv("OLLAMA_TOP_P", "0.1"))
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", os.getenv("OLLAMA_MAX_GEN_LEN", "512")))
    payload = {
        "model": model,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    response = requests.post(url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()

    if not response.text or not response.text.strip():
        raise ValueError(f"Ollama returned an empty HTTP response for model '{model}'.")

    try:
        decoded = response.json()
    except ValueError as exc:
        snippet = response.text.strip().replace("\n", " ")[:220]
        raise ValueError(
            f"Ollama returned non-JSON content for model '{model}': {snippet}"
        ) from exc

    content = decoded.get("message", {}).get("content", "")
    return content.strip()


def _extract_first_json_object(raw: str) -> Optional[str]:
    text = raw.strip()
    if not text:
        return None

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "\"":
                in_string = False
            continue

        if char == "\"":
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1].strip()

    return None


def _call_ollama_json(model: str, system_prompt: str, user_prompt: str, timeout_seconds: Optional[int] = None) -> Optional[Dict[str, Any]]:
    # Auto-detect timeout based on model role (extractor, router, transformer)
    if timeout_seconds is None:
        extractor_model = os.getenv("OLLAMA_MODEL_PARAM_EXTRACTOR", "llama3.2")
        router_model = os.getenv("OLLAMA_MODEL_ROUTER", "deepseek-coder:6.7b")
        transformer_model = os.getenv("OLLAMA_MODEL_TRANSFORM", "deepseek-coder:6.7b")
        
        if model == router_model:
            # Router uses dedicated router timeout
            timeout_seconds = int(os.getenv("OLLAMA_ROUTER_TIMEOUT_SECONDS", "600")) # Augmenter le timeout
        elif model == transformer_model:
            # Transformer uses transform timeout
            timeout_seconds = int(os.getenv("OLLAMA_TRANSFORM_TIMEOUT_SECONDS", "600")) # Augmenter le timeout
        elif model == extractor_model or "llama" in model.lower() or "gemma" in model.lower():
            # Extractor/answer models use standard timeout
            timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600")) # Augmenter le timeout
        else:
            # Default to standard timeout for unknown models
            timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600")) # Augmenter le timeout

    raw = _call_ollama_chat(model, system_prompt, user_prompt, timeout_seconds=timeout_seconds)
    cleaned = raw.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    extracted_json = _extract_first_json_object(cleaned)
    if extracted_json:
        cleaned = extracted_json
    if not cleaned:
        return None
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


def _truncate_text(value: Any, max_length: int = 240) -> str:
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def _build_answer_evidence(filtered: Dict[str, Any]) -> Dict[str, Any]:
    records = filtered.get("records", [])
    compact_records: List[Dict[str, Any]] = []
    for item in records[:8]:
        if isinstance(item, dict):
            record = item.get("record", item)
            endpoint = item.get("endpoint")
            if isinstance(record, dict):
                trimmed_record: Dict[str, Any] = {}
                for idx, (key, value) in enumerate(record.items()):
                    if idx >= 8:
                        break
                    if isinstance(value, (dict, list)):
                        trimmed_record[key] = _truncate_text(json.dumps(value, ensure_ascii=False), 120)
                    else:
                        trimmed_record[key] = _truncate_text(value, 120)
                compact_records.append({"endpoint": endpoint, "record": trimmed_record})
            else:
                compact_records.append({"endpoint": endpoint, "record": _truncate_text(record, 120)})
        else:
            compact_records.append({"record": _truncate_text(item, 120)})

    return {
        "count": filtered.get("count", 0),
        "by_endpoint": filtered.get("by_endpoint", {}),
        "records": compact_records,
    }


def _build_candidate_pool(
    endpoints: List[Dict[str, Any]],
    analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Construit le pool de candidats sans filtrage par domaine ou rÃ´le.
    Le modÃ¨le de routing fera le tri.
    """
    requested_fields = _unique_strings(analysis.get("requested_fields", []))
    try:
        routing_params = _unique_strings(column_router.collect_routing_params(analysis))
    except Exception:
        routing_params = []

    candidates: List[Dict[str, Any]] = []
    for endpoint in endpoints:
        if not _is_supported_business_endpoint(endpoint):
            continue
        if str(endpoint.get("method", "GET")).upper() != "GET":
            continue
        
        # Aucun filtrage par domaine ou rÃ´le ici
        candidates.append(endpoint)

    if not candidates:
        candidates = [ep for ep in endpoints if str(ep.get("method", "GET")).upper() == "GET"]

    rank_fields = routing_params or requested_fields
    return _rerank_endpoints_by_requested_fields(candidates, rank_fields)


def _route_request_with_deepseek_resilient(
    question: str,
    analysis: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    extracted_params = dict(analysis.get("extracted_params", {}))
    requested_fields = _unique_strings(analysis.get("requested_fields", []))
    analysis_is_weak = _analysis_is_weak(analysis)
    if requested_fields:
        extracted_params["requested_fields"] = requested_fields

    router_model = os.getenv(
        "OLLAMA_MODEL_ROUTER",
        os.getenv("OLLAMA_MODEL_PARAM_EXTRACTOR", "llama3.2"),
    )
    if not candidates:
        return [], extracted_params, "No endpoint candidate available for routing."

    llm_candidate_limit = min(len(candidates), int(os.getenv("ERP_ROUTER_CANDIDATE_LIMIT", "24")))
    routing_catalog_text = _load_routing_catalog_text(candidates[:llm_candidate_limit])

    system_prompt = (
        "You are an ERP API router specialized in endpoint discrimination. "
        "Choose the best endpoint ids from the provided catalog. "
        "Use the structured analysis only when it is coherent. "
        "If the analysis is weak or inconsistent, route directly from the user message and the catalog. "
        "Return valid JSON ONLY with keys: endpoint_ids and extracted_params."
    )
    user_prompt = (
        f"Question: {question}\n"
        f"Structured analysis: {json.dumps(analysis, ensure_ascii=False)}\n" # L'analyse n'a plus d'intent
        f"Analysis reliability: {'weak' if analysis_is_weak else 'usable'}\n"
        f"\nRouting catalog (ordered by relevance):\n{routing_catalog_text}\n"
        "Selection rules:\n"
        "1. Match the user message against endpoint columns and requested_fields.\n"
        "2. If analysis reliability is weak, ignore wrong entity or wrong requested_fields and route directly from the question.\n"
        "3. endpoint_ids MUST contain ONLY ids present in the catalog.\n"
        "4. Return a single endpoint unless multiple routes are truly required.\n"
        "5. Prefer endpoints listed earlier when the match quality is similar.\n"
        "6. extracted_params may repair bad extraction when the route is obvious from the question.\n"
        "7. Use endpoint columns to distinguish similar routes and to preserve requested display/grouping fields.\n"
        "8. Do not invent parameters unrelated to the selected endpoint.\n"
        "9. If the user asks for stats, totals, count, sum, avg, grouped, monthly, yearly, or by-client results, keep that aggregation intent inside extracted_params.\n"
        "10. In columns-only mode, do not use url, tags, or description.\n"
        "Return JSON: {\"endpoint_ids\": [\"id\"], \"extracted_params\": {...}}"
    )

    try:
        llm_choice = _call_ollama_json(router_model, system_prompt, user_prompt) or {}
    except Exception as exc:
        raise Exception(f"Router Ollama error: {exc}")

    llm_params = llm_choice.get("extracted_params", {})
    if isinstance(llm_params, dict):
        extracted_params.update(llm_params)
    if requested_fields and "requested_fields" not in extracted_params:
        extracted_params["requested_fields"] = requested_fields

    endpoint_ids = llm_choice.get("endpoint_ids", [])
    selected_ids = [str(endpoint_id) for endpoint_id in endpoint_ids if endpoint_id]
    selected_endpoints = [candidate for candidate in candidates if str(candidate.get("id")) in selected_ids]

    if not selected_endpoints:
        endpoint_id = llm_choice.get("endpoint_id")
        if endpoint_id:
            selected_endpoints = [candidate for candidate in candidates if str(candidate.get("id")) == str(endpoint_id)]

    if not selected_endpoints:
        raise Exception("Router returned no valid endpoint_ids.")

    limit = min(len(selected_endpoints), int(os.getenv("ERP_ROUTER_MAX_SELECTED", "3")))
    return selected_endpoints[:limit], extracted_params, None



def _persist_display_result(
    question: str,
    selected_endpoints: List[Dict[str, Any]],
    filtered_result: Dict[str, Any],
    answer: str,
    errors: List[str],
    extracted_params: Optional[Dict[str, Any]] = None,
    request_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "generatedAt": datetime.now(UTC).isoformat(),
        "question": question,
        "endpoints": [
            {
                "id": ep.get("id"),
                "url": ep.get("url"),
                "method": ep.get("method", "GET"),
            }
            for ep in selected_endpoints
        ],
        "requestAnalysis": request_analysis or {},
        "extractedParams": extracted_params or {},
        "display": {
            "count": filtered_result.get("count", 0),
            "by_endpoint": filtered_result.get("by_endpoint", {}),
            "records": filtered_result.get("records", []),
        },
        "answer": answer,
        "errors": errors,
    }
    DISPLAY_RESULT_PATH.write_text(_safe_json_dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _persist_transform_plan_result(
    question: str,
    plan: Dict[str, Any],
    request_analysis: Optional[Dict[str, Any]] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    selected_endpoints: Optional[List[Dict[str, Any]]] = None,
    errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    payload = {
        "generatedAt": datetime.now(UTC).isoformat(),
        "question": question,
        "requestAnalysis": request_analysis or {},
        "extractedParams": extracted_params or {},
        "endpoints": [
            {
                "id": ep.get("id"),
                "url": ep.get("url"),
                "method": ep.get("method", "GET"),
            }
            for ep in (selected_endpoints or [])
        ],
        "plan": plan,
        "errors": errors or [],
    }
    TRANSFORM_PLAN_PATH.write_text(_safe_json_dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _flatten_api_records(records: List[Any]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for item in records:
        if isinstance(item, dict) and isinstance(item.get("record"), dict):
            row = dict(item.get("record", {}))
            row["_endpoint"] = item.get("endpoint")
            row["_sourceUrl"] = item.get("sourceUrl")
            flattened.append(row)
        elif isinstance(item, dict):
            flattened.append(dict(item))
    return flattened


def _build_api_result_context(records: List[Dict[str, Any]], max_rows: int = 5, max_columns: int = 12) -> Dict[str, Any]:
    columns: List[str] = []
    for row in records[:max_rows]:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            if str(key).startswith("_"):
                continue
            if key not in columns:
                columns.append(str(key))
    preview_rows: List[Dict[str, Any]] = []
    for row in records[:max_rows]:
        if not isinstance(row, dict):
            continue
        preview: Dict[str, Any] = {}
        for key in columns[:max_columns]:
            if key in row:
                preview[key] = row[key]
        preview_rows.append(preview)
    return {
        "record_count": len(records),
        "available_columns": columns[:max_columns],
        "sample_rows": preview_rows,
    }


def _normalize_transform_condition(condition: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(condition, dict):
        return None

    field = str(condition.get("field") or condition.get("column") or "").strip()
    if not field:
        return None

    raw_operator = str(condition.get("operator", "equals")).strip().lower()
    operator_map = {
        "=": "equals", "==": "equals", "eq": "equals", "is": "equals",
        "!=": "not_equals", "<>": "not_equals", "ne": "not_equals",
        "contains": "contains", "like": "contains",
        "is_null": "is_null", "not_null": "not_null",
        ">": "gt", ">=": "gte", "<": "lt", "<=": "lte",
    }
    operator = operator_map.get(raw_operator, raw_operator)
    if operator not in {"equals", "not_equals", "contains", "is_null", "not_null", "gt", "gte", "lt", "lte"}:
        operator = "equals"

    value = condition.get("value")
    # Reject conditions without value for operators that require it (not is_null, not_null)
    if operator not in {"is_null", "not_null"} and value is None:
        return None

    return {
        "field": field,
        "operator": operator,
        "value": value,
    }


def _normalize_transform_plan_for_execution(plan: Dict[str, Any]) -> Dict[str, Any]:
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    normalized_steps: List[Dict[str, Any]] = []

    for step in steps:
        if not isinstance(step, dict):
            continue
        op = str(step.get("op", "")).strip().lower()
        # Support aggregate operations executed locally
        if op not in {"select", "rename", "filter_rows", "sort", "limit", "aggregate"}:
            continue

        if op == "select":
            raw_columns = step.get("columns", step.get("fields", []))
            columns = [str(column).strip() for column in raw_columns if str(column).strip()] if isinstance(raw_columns, list) else []
            if columns:
                normalized_steps.append({"op": "select", "columns": columns})
            continue

        if op == "rename":
            mapping = step.get("mapping", {})
            if isinstance(mapping, dict) and mapping:
                normalized_steps.append(
                    {"op": "rename", "mapping": {str(key): str(value) for key, value in mapping.items()}}
                )
            continue

        if op == "filter_rows":
            raw_conditions = step.get("conditions", [])
            conditions = []
            if isinstance(raw_conditions, list):
                for condition in raw_conditions:
                    normalized = _normalize_transform_condition(condition) if isinstance(condition, dict) else None
                    if normalized:
                        conditions.append(normalized)
            if conditions:
                normalized_steps.append({"op": "filter_rows", "conditions": conditions})
            continue

        if op == "sort":
            field = str(step.get("field") or step.get("column") or "").strip()
            if field:
                direction = str(step.get("direction", "asc")).strip().lower()
                normalized_steps.append(
                    {"op": "sort", "field": field, "direction": "desc" if direction == "desc" else "asc"}
                )
            continue

        if op == "limit":
            try:
                limit_value = int(step.get("value", 20))
            except (TypeError, ValueError):
                limit_value = 20
            normalized_steps.append({"op": "limit", "value": max(1, limit_value)})
            continue

        if op == "aggregate":
            raw_groupby = step.get("groupby", step.get("group_by", []))
            if isinstance(raw_groupby, list):
                groupby = [str(item).strip() for item in raw_groupby if str(item).strip()]
            elif raw_groupby:
                groupby = [str(raw_groupby).strip()]
            else:
                groupby = []

            raw_aggs = step.get("aggs", step.get("aggregations", []))
            normalized_aggs: List[Dict[str, Any]] = []
            if isinstance(raw_aggs, list):
                for agg in raw_aggs:
                    if isinstance(agg, dict):
                        field = str(agg.get("field", "")).strip()
                        func = str(agg.get("agg", agg.get("function", ""))).strip().lower()
                        as_name = str(agg.get("as", "")).strip() or None
                        if field and func in {"sum", "avg", "mean", "count", "min", "max"}:
                            normalized_aggs.append({"field": field, "agg": func, "as": as_name})
                        continue
                    if isinstance(agg, str):
                        match = re.match(
                            r"\s*(sum|avg|mean|count|min|max)\s*\(?\s*([a-zA-Z0-9_]+)\s*\)?\s*(?:as\s+([a-zA-Z0-9_]+))?\s*",
                            agg,
                            re.IGNORECASE,
                        )
                        if match:
                            normalized_aggs.append(
                                {
                                    "field": match.group(2),
                                    "agg": match.group(1).lower(),
                                    "as": match.group(3) or None,
                                }
                            )
            if normalized_aggs:
                normalized_steps.append({"op": "aggregate", "groupby": groupby, "aggs": normalized_aggs})
            continue

    if normalized_steps and not any(step.get("op") == "limit" for step in normalized_steps):
        normalized_steps.append({"op": "limit", "value": 20})

    return {"steps": normalized_steps}


def _build_transform_plan_with_llama(
    question: str,
    analysis: Dict[str, Any],
    extracted_params: Dict[str, Any],
    api_result_context: Dict[str, Any],
) -> tuple[Dict[str, Any], Optional[str]]:
    model = os.getenv(
        "OLLAMA_MODEL_TRANSFORM",
        os.getenv("OLLAMA_MODEL_PARAM_EXTRACTOR", os.getenv("OLLAMA_MODEL_ANSWER", "llama3.2")),
    )
    mapping_context = _build_mapping_context(question=question)
    requested_fields = _unique_strings(analysis.get("requested_fields", []))
    analysis_is_weak = _analysis_is_weak(analysis)

    system_prompt = (
        "You generate strict JSON transformation plans for ERP tabular results. "
        "Think like a pandas planner, but return JSON only. "
        "Allowed ops are: select, filter_rows, sort, limit, aggregate. "
        "Do not write Python code. Do not invent columns. "
        "Return only JSON with key: steps."
    )
    user_prompt = (
        f"Question: {question}\n"
        f"Structured analysis: {json.dumps(analysis, ensure_ascii=False)}\n"
        f"Analysis reliability: {'weak' if analysis_is_weak else 'usable'}\n"
        f"Requested fields: {json.dumps(requested_fields, ensure_ascii=False)}\n"
        f"Extracted params: {json.dumps(extracted_params, ensure_ascii=False)}\n"
        f"Business mapping context: {json.dumps(mapping_context, ensure_ascii=False)}\n"
        f"Retrieved API result context: {json.dumps(api_result_context, ensure_ascii=False)}\n"
        "Planning rules:\n"
        "1. Build the plan only from real available_columns.\n"
        "2. If the analysis is weak, infer the display/filter/aggregate intent directly from the original question and the real columns.\n"
        "3. requested_fields must contain only columns explicitly requested for display.\n"
        "4. Expressions like 'qui ont un email', 'avec email', 'non vide', 'renseigne' usually mean filter_rows with operator not_null.\n"
        "5. For totals, counts, sums, averages, grouped or monthly questions, use aggregate with groupby and aggs.\n"
        "6. Use select only for display columns, not for hidden filter columns.\n"
        "7. Use sort only when the question implies order.\n"
        "8. Add limit only when needed to keep the result readable.\n"
        "9. Return the minimal valid plan.\n"
        "Example aggregate step: {\"op\":\"aggregate\",\"groupby\":[\"client\"],\"aggs\":[{\"field\":\"montant\",\"agg\":\"sum\",\"as\":\"total_montant\"}]}\n"
        "Return JSON: {\"steps\": [...]}"
    )

    try:
        raw_plan = _call_ollama_json(model, system_prompt, user_prompt) or {}
        plan = _normalize_transform_plan_for_execution(raw_plan)
        if not plan.get("steps"):
            raise Exception("Transform plan Ollama returned no executable steps.")
        return plan, None
    except Exception as exc:
        raise Exception(f"Transform plan Ollama error: {exc}")


def _apply_select_rows(rows: List[Dict[str, Any]], columns: List[str]) -> List[Dict[str, Any]]:
    valid_columns = [column for column in columns if isinstance(column, str) and column]
    if not valid_columns:
        return rows
    selected_rows: List[Dict[str, Any]] = []
    for row in rows:
        selected = {column: row[column] for column in valid_columns if column in row}
        if "_endpoint" in row:
            selected["_endpoint"] = row["_endpoint"]
        selected_rows.append(selected)
    return selected_rows


def _apply_rename_rows(rows: List[Dict[str, Any]], mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(mapping, dict) or not mapping:
        return rows
    renamed_rows: List[Dict[str, Any]] = []
    for row in rows:
        renamed: Dict[str, Any] = {}
        for key, value in row.items():
            renamed[str(mapping.get(key, key))] = value
        renamed_rows.append(renamed)
    return renamed_rows


def _apply_filter_rows(rows: List[Dict[str, Any]], conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(conditions, list) or not conditions:
        return rows

    def matches(row: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        field = str(condition.get("field", "")).strip()
        operator = str(condition.get("operator", "equals")).strip().lower()
        value = condition.get("value")
        if not field or field not in row:
            return True
        current = row.get(field)
        if operator == "equals":
            return current == value
        if operator == "not_equals":
            return current != value
        if operator == "contains":
            return str(value).lower() in str(current).lower()
        if operator == "is_null":
            return current is None
        if operator == "not_null":
            return current is not None
        if operator in {"gt", "gte", "lt", "lte"}:
            try:
                current_num = float(current)
                value_num = float(value)
            except (TypeError, ValueError):
                return True
            if operator == "gt":
                return current_num > value_num
            if operator == "gte":
                return current_num >= value_num
            if operator == "lt":
                return current_num < value_num
            return current_num <= value_num
        return True

    return [row for row in rows if all(matches(row, condition) for condition in conditions if isinstance(condition, dict))]


def _apply_sort_rows(rows: List[Dict[str, Any]], field: str, direction: str) -> List[Dict[str, Any]]:
    if not field:
        return rows
    reverse = str(direction).lower() == "desc"
    return sorted(rows, key=lambda row: str(row.get(field, "")), reverse=reverse)


def _coerce_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        try:
            num = float(value)
            if math.isnan(num) or math.isinf(num):
                return None
            return num
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip().replace(" ", "")
        if not text:
            return None
        try:
            num = float(text)
            if math.isnan(num) or math.isinf(num):
                return None
            return num
        except ValueError:
            return None
    return None


def _apply_aggregate_rows(rows: List[Dict[str, Any]], groupby: List[str], aggs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not aggs:
        return rows

    normalized_groupby = [str(item).strip() for item in (groupby or []) if str(item).strip()]

    # group_key -> accumulator
    groups: Dict[tuple, Dict[str, Any]] = {}

    def key_for(row: Dict[str, Any]) -> tuple:
        if not normalized_groupby:
            return ("__all__",)
        return tuple(row.get(field) for field in normalized_groupby)

    # Initialize accumulators
    for row in rows:
        if not isinstance(row, dict):
            continue
        gk = key_for(row)
        if gk not in groups:
            groups[gk] = {
                "__count_rows__": 0,
                "__sum__": {},
                "__count__": {},
                "__min__": {},
                "__max__": {},
            }
        acc = groups[gk]
        acc["__count_rows__"] += 1

        for agg in aggs:
            if not isinstance(agg, dict):
                continue
            field = str(agg.get("field", "")).strip()
            func = str(agg.get("agg", "sum")).strip().lower()
            if not field:
                continue
            value = row.get(field)
            num = _coerce_number(value)

            if func in {"sum", "avg", "mean"}:
                if num is None:
                    continue
                acc["__sum__"][field] = float(acc["__sum__"].get(field, 0.0)) + num
                acc["__count__"][field] = int(acc["__count__"].get(field, 0)) + 1
                continue

            if func == "count":
                # count non-null values for that field
                if value is None:
                    continue
                acc["__count__"][field] = int(acc["__count__"].get(field, 0)) + 1
                continue

            if func in {"min", "max"}:
                if value is None:
                    continue
                container_key = "__min__" if func == "min" else "__max__"
                current = acc[container_key].get(field)
                # Prefer numeric comparison when possible
                if num is not None:
                    current_num = _coerce_number(current)
                    if current is None or current_num is None:
                        acc[container_key][field] = num
                    else:
                        if func == "min":
                            acc[container_key][field] = min(float(current_num), float(num))
                        else:
                            acc[container_key][field] = max(float(current_num), float(num))
                else:
                    # Fallback to string comparison
                    if current is None:
                        acc[container_key][field] = value
                    else:
                        if func == "min":
                            acc[container_key][field] = min(str(current), str(value))
                        else:
                            acc[container_key][field] = max(str(current), str(value))

    # Build output rows
    out: List[Dict[str, Any]] = []
    for gk, acc in groups.items():
        row: Dict[str, Any] = {}
        if normalized_groupby:
            for idx, field in enumerate(normalized_groupby):
                row[field] = gk[idx] if idx < len(gk) else None

        for agg in aggs:
            if not isinstance(agg, dict):
                continue
            field = str(agg.get("field", "")).strip()
            func = str(agg.get("agg", "sum")).strip().lower()
            if not field:
                continue
            as_name = str(agg.get("as") or "").strip() or f"{func}_{field}"

            if func == "sum":
                row[as_name] = acc["__sum__"].get(field)
            elif func in {"avg", "mean"}:
                total = acc["__sum__"].get(field)
                count = acc["__count__"].get(field)
                row[as_name] = (float(total) / float(count)) if (total is not None and count) else None
            elif func == "count":
                row[as_name] = acc["__count__"].get(field, 0)
            elif func == "min":
                row[as_name] = acc["__min__"].get(field)
            elif func == "max":
                row[as_name] = acc["__max__"].get(field)
            else:
                # Unsupported agg function in fallback: ignore
                pass

        out.append(row)

    return out


def _extract_filter_conditions_from_extracted_params(extracted_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(extracted_params, dict):
        return []

    conditions: List[Dict[str, Any]] = []

    # Preferred explicit shape
    for key in ("filters", "conditions"):
        raw = extracted_params.get(key)
        if isinstance(raw, list):
            for item in raw:
                normalized = _normalize_transform_condition(item) if isinstance(item, dict) else None
                if normalized:
                    conditions.append(normalized)

    # Back-compat: interpret extracted_params.{field} values
    for field, raw_value in extracted_params.items():
        if field in {"requested_fields", "aggregations", "group_by", "groupby", "metrics", "aggs", "tables", "filters", "conditions"}:
            continue
        if raw_value is None:
            continue

        if isinstance(raw_value, dict):
            # { operator: 'not_null', value: ... }
            if "operator" in raw_value or "op" in raw_value:
                candidate = {
                    "field": field,
                    "operator": raw_value.get("operator", raw_value.get("op", "equals")),
                    "value": raw_value.get("value"),
                }
                normalized = _normalize_transform_condition(candidate)
                if normalized:
                    conditions.append(normalized)
                continue

            # Short forms: {not_null:true}, {is_null:true}, {gt:10}, {contains:'abc'}
            if raw_value.get("not_null") is True:
                conditions.append({"field": field, "operator": "not_null", "value": "__ignored__"})
                continue
            if raw_value.get("is_null") is True:
                conditions.append({"field": field, "operator": "is_null", "value": "__ignored__"})
                continue
            for op_key, mapped in [("gt", "gt"), ("gte", "gte"), ("lt", "lt"), ("lte", "lte")]:
                if op_key in raw_value:
                    conditions.append({"field": field, "operator": mapped, "value": raw_value.get(op_key)})
                    break
            else:
                if "contains" in raw_value:
                    conditions.append({"field": field, "operator": "contains", "value": raw_value.get("contains")})
            continue

        if isinstance(raw_value, str) and raw_value.strip().lower() in {"not_null", "notnull"}:
            conditions.append({"field": field, "operator": "not_null", "value": "__ignored__"})
            continue
        if isinstance(raw_value, str) and raw_value.strip().lower() in {"is_null", "null", "isnull"}:
            conditions.append({"field": field, "operator": "is_null", "value": "__ignored__"})
            continue

        conditions.append({"field": field, "operator": "equals", "value": raw_value})

    # Normalize + drop placeholders
    normalized_conditions: List[Dict[str, Any]] = []
    for cond in conditions:
        normalized = _normalize_transform_condition(cond)
        if normalized:
            normalized_conditions.append(normalized)
    return normalized_conditions


def _apply_transform_plan_with_pandas(records: List[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    try:
        import pandas as pd
    except ImportError:
        return None

    df = pd.DataFrame(records)
    normalized_plan = _normalize_transform_plan_for_execution(plan)
    for step in normalized_plan.get("steps", []):
        if not isinstance(step, dict):
            continue
        op = step.get("op")
        if op == "aggregate":
            # Aggregate handled by pandas implementation
            groupby = step.get("groupby") or []
            aggs = step.get("aggs") or []
            if not groupby or not aggs:
                continue
            # build agg dict for pandas
            agg_dict = {}
            rename_map = {}
            for agg in aggs:
                field = agg.get("field")
                func = str(agg.get("agg", "sum")).lower()
                as_name = agg.get("as") or f"{func}_{field}"
                if field not in df.columns:
                    continue
                # pandas accepts mapping like {'field': 'sum'} or {'field': ['sum']}
                agg_dict.setdefault(field, []).append(func)
                rename_map[f"{field}"] = None
            if not agg_dict:
                continue
            grouped = df.groupby([g for g in groupby if g in df.columns]).agg(agg_dict)
            # flatten multiindex columns
            grouped.columns = ["_".join([col for col in col_tuple if col]) for col_tuple in grouped.columns.to_flat_index()]
            grouped = grouped.reset_index()
            # rename aggregated columns using provided 'as' names when possible
            for agg in aggs:
                field = agg.get("field")
                func = str(agg.get("agg", "sum")).lower()
                as_name = agg.get("as") or f"{func}_{field}"
                generated_col = f"{field}_{func}"
                if generated_col in grouped.columns:
                    grouped = grouped.rename(columns={generated_col: as_name})
            df = grouped
            continue
        if op == "select":
            columns = [column for column in step.get("columns", []) if column in df.columns]
            if columns:
                df = df[columns]
        elif op == "rename":
            mapping = {key: value for key, value in step.get("mapping", {}).items() if key in df.columns}
            if mapping:
                df = df.rename(columns=mapping)
        elif op == "filter_rows":
            for condition in step.get("conditions", []):
                if not isinstance(condition, dict):
                    continue
                field = condition.get("field")
                if field not in df.columns:
                    continue
                operator = str(condition.get("operator", "equals")).lower()
                value = condition.get("value")
                if operator == "equals":
                    df = df[df[field] == value]
                elif operator == "not_equals":
                    df = df[df[field] != value]
                elif operator == "contains":
                    df = df[df[field].astype(str).str.contains(str(value), case=False, na=False)]
                elif operator == "is_null":
                    df = df[df[field].isna()]
                elif operator == "not_null":
                    df = df[df[field].notna()]
                elif operator in {"gt", "gte", "lt", "lte"}:
                    num_series = pd.to_numeric(df[field], errors="coerce")
                    try:
                        value_num = float(value)
                    except (TypeError, ValueError):
                        continue
                    if operator == "gt":
                        df = df[num_series > value_num]
                    elif operator == "gte":
                        df = df[num_series >= value_num]
                    elif operator == "lt":
                        df = df[num_series < value_num]
                    else:
                        df = df[num_series <= value_num]
        elif op == "sort":
            field = step.get("field")
            if field in df.columns:
                df = df.sort_values(by=field, ascending=str(step.get("direction", "asc")).lower() != "desc")
        elif op == "limit":
            df = df.head(int(step.get("value", 20)))

    return df.where(df.notna(), None).to_dict(orient="records")


def _apply_transform_plan(records: List[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    pandas_result = _apply_transform_plan_with_pandas(records, plan)
    if pandas_result is not None:
        return pandas_result

    normalized_plan = _normalize_transform_plan_for_execution(plan)
    rows = [dict(record) for record in records if isinstance(record, dict)]
    for step in normalized_plan.get("steps", []):
        if not isinstance(step, dict):
            continue
        op = step.get("op")
        if op == "select":
            rows = _apply_select_rows(rows, step.get("columns", []))
        elif op == "rename":
            rows = _apply_rename_rows(rows, step.get("mapping", {}))
        elif op == "filter_rows":
            rows = _apply_filter_rows(rows, step.get("conditions", []))
        elif op == "aggregate":
            rows = _apply_aggregate_rows(
                rows,
                step.get("groupby") or step.get("group_by") or [],
                step.get("aggs") or step.get("aggregations") or [],
            )
        elif op == "sort":
            rows = _apply_sort_rows(rows, str(step.get("field", "")), str(step.get("direction", "asc")))
        elif op == "limit":
            rows = rows[: max(0, int(step.get("value", 20)))]
    return rows


def _strip_internal_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned_rows: List[Dict[str, Any]] = []
    for row in rows:
        cleaned_rows.append({key: value for key, value in row.items() if not str(key).startswith("_")})
    return cleaned_rows


def _get_erp_api_base_urls() -> List[str]:
    explicit = os.getenv("ERP_API_BASE_URL", "").strip().rstrip("/")
    if explicit:
        return [explicit]

    webapi_project_dir = os.getenv("ERP_WEBAPI_PROJECT_DIR", "").strip()
    if not webapi_project_dir:
        return []

    launch_settings = Path(webapi_project_dir) / "Properties" / "launchSettings.json"
    if not launch_settings.exists():
        return []

    try:
        payload = json.loads(launch_settings.read_text(encoding="utf-8-sig"))
        profiles = payload.get("profiles", {})
        candidate_profiles: List[Dict[str, Any]] = []
        if isinstance(profiles, dict):
            if isinstance(profiles.get("WebApi"), dict):
                candidate_profiles.append(profiles["WebApi"])
            candidate_profiles.extend(p for p in profiles.values() if isinstance(p, dict) and p not in candidate_profiles)

        urls: List[str] = []
        for profile in candidate_profiles:
            app_urls = profile.get("applicationUrl", "")
            if app_urls:
                for raw_url in str(app_urls).split(";"):
                    cleaned = raw_url.strip().rstrip("/")
                    if cleaned and cleaned not in urls:
                        urls.append(cleaned)
        return urls
    except (json.JSONDecodeError, OSError):
        return []
    return []

# _build_endpoint_url et _collect_request_parts sont utilisÃ©s
def _build_endpoint_url(url_template: str, params: Dict[str, Any]) -> str:
    route_params = re.findall(r"\{([^{}]+)\}", url_template)
    rendered = url_template
    for route_key in route_params:
        value = params.get(route_key)
        if value is None:
            if route_key.lower() == "code" and "id" in params:
                value = params.get("id")
            elif route_key.lower() == "id" and "code" in params:
                value = params.get("code")
        if value is not None:
            rendered = rendered.replace(f"{{{route_key}}}", str(value))
    return rendered


def _collect_request_parts(selected: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    required = selected.get("requiredParameters", [])
    query_keys = selected.get("queryParameters", [])

    missing_required = [key for key in required if key not in params]
    query_params = {k: params[k] for k in query_keys if k in params}

    if "pageNumber" in query_keys and "pageNumber" not in query_params:
        query_params["pageNumber"] = 1
    if "pageSize" in query_keys and "pageSize" not in query_params:
        query_params["pageSize"] = 50

    return {
        "missing_required": missing_required,
        "query_params": query_params,
    }

def _normalize_data_field(payload: Any) -> List[Any]:
    # Simplified data normalization
    if isinstance(payload, dict) and payload.get("data") is not None:
        return payload["data"] if isinstance(payload["data"], list) else [payload["data"]]
    return payload if isinstance(payload, list) else ([payload] if payload is not None else [])


def _resolve_endpoint_path_from_swagger(
    requested_path: str,
    method: str,
    selected: Dict[str, Any],
    swagger_paths: Dict[str, List[str]],
) -> str:
    # Swagger paths est toujours vide, donc on retourne directement le chemin demandÃ©
    # Supprime la logique de recherche Swagger inutile
    return requested_path


# ---------- Graph Nodes ----------
def extract_user_request(state: AssistantState) -> AssistantState:
    question = state.get("question", "")
    # On récupère l'historique existant ou on crée une nouvelle liste
    history = list(state.get("history", [])) 
    previous_analysis = state.get("request_analysis", None)
    errors = state.get("errors", []).copy()
    
    try:
        analysis, extraction_error = _extract_request_with_llama(question, history)
        if extraction_error:
            errors.append(extraction_error)
        
        # On passe previous_analysis pour permettre la fusion des paramètres
        analysis = _normalize_request_analysis(analysis, question, previous_analysis)
    except Exception as e:
        errors.append(str(e))
        # Retourne une analyse vide/minimale pour ne pas crasher le graphe
        analysis = {"intent": "GET", "entity": "", "extracted_params": {}, "confidence": 0.0}

    history.append({"role": "user", "content": question})

    return {
        "request_analysis": analysis,
        "extracted_params": analysis.get("extracted_params", {}),
        "history": history,
        "errors": errors,
    }


def retrieve_candidate_endpoints(state: AssistantState) -> AssistantState:
    analysis = state.get("request_analysis", {})
    endpoints = _load_endpoints()
    candidates = _build_candidate_pool(endpoints, analysis)

    if not candidates:
        candidates = [ep for ep in endpoints if str(ep.get("method", "GET")).upper() == "GET"]

    max_candidates = int(os.getenv("ERP_MAX_CANDIDATES", "12"))
    return {"endpoint_candidates": candidates[:max_candidates]}


def route_endpoint(state: AssistantState) -> AssistantState:
    candidates = state.get("endpoint_candidates", [])
    question = state.get("question", "")
    analysis = state.get("request_analysis", {})
    errors = state.get("errors", []).copy()
    router_mode = os.getenv("ERP_ROUTER_MODE", "columns").strip().lower()
    
    if not candidates:
        errors.append("No endpoint candidate matched the question.")
        return {"selected_endpoints": [], "extracted_params": {}, "errors": errors}

    try:
        if router_mode == "columns":
            params = column_router.collect_routing_params(analysis)
            selected_endpoints, _ = column_router.route_by_columns(candidates, params)
            routed_params = dict(analysis.get("extracted_params", {}))
        else:
            selected_endpoints, routed_params, router_error = _route_request_with_deepseek_resilient(
                question=question,
                analysis=analysis,
                candidates=candidates,
            )
            if router_error:
                errors.append(router_error)
    except Exception as e:
        errors.append(f"Routing failed: {str(e)}")
        # Ne pas forcer candidates[0] car cela retourne souvent GetAllArticles par erreur
        selected_endpoints = []
        routed_params = dict(analysis.get("extracted_params", {})) # Conserver les params extraits par Llama
    selected = selected_endpoints[0] if selected_endpoints else None

    return {
        "selected_endpoints": selected_endpoints,
        "selected_endpoint": selected,
        "extracted_params": routed_params,
        "errors": errors,
    }


def generate_transform_plan(state: AssistantState) -> AssistantState:
    """
    NÅ“ud dÃ©diÃ© pour DeepSeek Code.
    GÃ©nÃ¨re le plan d'action JSON APRÃˆS rÃ©cupÃ©ration rÃ©elle des donnÃ©es API.
    S'il timeout, retourne un plan par dÃ©faut sans crash le graphe.
    """
    question = state.get("question", "")
    analysis = state.get("request_analysis", {})
    extracted_params = state.get("extracted_params", {})
    requested_fields = _unique_strings(analysis.get("requested_fields", []))
    api_result_path = state.get("api_result_path", "")
    selected_endpoints = state.get("selected_endpoints", [])
    errors = state.get("errors", []).copy()

    empty_plan = {"steps": []}
    transform_error = None

    if not state.get("selected_endpoints"):
        _persist_transform_plan_result(
            question=question,
            plan=empty_plan,
            request_analysis=analysis,
            extracted_params=extracted_params,
            selected_endpoints=selected_endpoints,
            errors=errors,
        )
        return {"transform_plan": empty_plan}

    if not api_result_path:
        errors.append("Cannot build transform plan: missing api_result_path.")
        _persist_transform_plan_result(
            question=question,
            plan=empty_plan,
            request_analysis=analysis,
            extracted_params=extracted_params,
            selected_endpoints=selected_endpoints,
            errors=errors,
        )
        return {"transform_plan": empty_plan, "errors": errors}

    try:
        payload = json.loads(Path(api_result_path).read_text(encoding="utf-8"))
        flattened_rows = _flatten_api_records(payload.get("data", []))
        api_result_context = _build_api_result_context(flattened_rows)
        # Optionally build the transform plan locally (skip LLM) when configured
        use_local_transform = os.getenv("ERP_LOCAL_TRANSFORM", "1") == "1" # Par défaut: local (DataFrame)
        if use_local_transform:
            local_steps: List[Dict[str, Any]] = []

            # 1) Filters inferred from extracted_params
            filter_conditions = _extract_filter_conditions_from_extracted_params(extracted_params)
            if filter_conditions:
                local_steps.append({"op": "filter_rows", "conditions": filter_conditions})

            # 2) Aggregate if requested
            aggs = extracted_params.get("aggregations") or extracted_params.get("aggs") or extracted_params.get("metrics")
            group_by = extracted_params.get("group_by") or extracted_params.get("groupby") or []
            normalized_aggs: List[Dict[str, Any]] = []
            if aggs:
                if isinstance(aggs, dict):
                    for alias, expr in aggs.items():
                        m = re.match(r"\s*(sum|avg|mean|count|min|max)\s*\(?\s*([a-zA-Z0-9_]+)\s*\)?\s*", str(expr), re.I)
                        if m:
                            normalized_aggs.append({"field": m.group(2), "agg": m.group(1).lower(), "as": alias})
                elif isinstance(aggs, list):
                    for item in aggs:
                        if isinstance(item, dict) and item.get("field"):
                            normalized_aggs.append(
                                {
                                    "field": item.get("field"),
                                    "agg": str(item.get("agg", item.get("function", "sum"))).lower(),
                                    "as": item.get("as"),
                                }
                            )
                        elif isinstance(item, str):
                            m = re.match(r"\s*(sum|avg|mean|count|min|max)\s*\(?\s*([a-zA-Z0-9_]+)\s*\)?\s*(?:as\s+([a-zA-Z0-9_]+))?\s*", item, re.I)
                            if m:
                                normalized_aggs.append({"field": m.group(2), "agg": m.group(1).lower(), "as": m.group(3) or None})
            if normalized_aggs:
                agg_step: Dict[str, Any] = {
                    "op": "aggregate",
                    "groupby": group_by if isinstance(group_by, list) else ([group_by] if group_by else []),
                    "aggs": normalized_aggs,
                }
                local_steps.append(agg_step)

            # 3) Display columns
            if requested_fields:
                local_steps.append({"op": "select", "columns": requested_fields})

            # 4) Limit for readability
            local_steps.append({"op": "limit", "value": 100})

            transform_plan = {"steps": local_steps}
        else: # Sinon, on utilise le LLM pour gÃ©nÃ©rer le plan
            transform_plan, transform_error = _build_transform_plan_with_llama(
                question, analysis, extracted_params, api_result_context
            )
        if transform_error:
            errors.append(transform_error)
        # Optionally persist the transform plan to disk. Set ERP_PERSIST_TRANSFORM=0 to disable.
        if os.getenv("ERP_PERSIST_TRANSFORM", "1") != "0":
            _persist_transform_plan_result(
                question=question,
                plan=transform_plan,
                request_analysis=analysis,
                extracted_params=extracted_params,
                selected_endpoints=selected_endpoints,
                errors=errors,
            )
    except Exception as e:
        # Si transform timeout, on continue avec un plan simple (limit)
        # Le frontend affiche les donnÃ©es brutes + chart, pas de message d'erreur bloquant
        transform_plan = {"steps": [{"op": "limit", "value": 20}]}
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            # Log but don't block - user still gets data + chart
            print(f"[WARN] DeepSeek transform timeout (non-blocking): {error_msg}")
        else:
            errors.append(f"Transform generation issue: {e}")
        if os.getenv("ERP_PERSIST_TRANSFORM", "1") != "0":
            _persist_transform_plan_result(
                question=question,
                plan=transform_plan,
                request_analysis=analysis,
                extracted_params=extracted_params,
                selected_endpoints=selected_endpoints,
                errors=errors,
            )

    return {"transform_plan": transform_plan, "errors": errors}


def call_webapi(state: AssistantState) -> AssistantState:
    selected = state.get("selected_endpoint")
    selected_endpoints = state.get("selected_endpoints", [])
    params = state.get("extracted_params", {})
    analysis = state.get("request_analysis", {})
    errors = state.get("errors", []).copy()

    if not selected_endpoints and selected:
        selected_endpoints = [selected]

    if not selected_endpoints:
        errors.append("Cannot execute WebApi: no endpoint selected.")
        payload = {
            "generatedAt": datetime.now(UTC).isoformat(),
            "requestAnalysis": analysis,
            "endpoints": [],
            "params": params,
            "calls": [],
            "data": [],
        }
        path = CACHE_DIR / "last_api_result.json"
        path.write_text(_safe_json_dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"api_result_path": str(path), "errors": errors}

    base_urls = _get_erp_api_base_urls()
    if not base_urls:
        errors.append("Missing ERP_API_BASE_URL environment variable.")
        payload = {
            "generatedAt": datetime.now(UTC).isoformat(),
            "requestAnalysis": analysis,
            "endpoints": [ep.get("id") for ep in selected_endpoints],
            "params": params,
            "calls": [],
            "data": [],
        }
        path = CACHE_DIR / "last_api_result.json"
        path.write_text(_safe_json_dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"api_result_path": str(path), "errors": errors}

    # Swagger paths sera vide car on ne charge plus swagger
    swagger_paths: Dict[str, List[str]] = {} 
    headers: Dict[str, str] = {}
    bearer = _ensure_bearer_token(base_urls, errors)
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    else:
        errors.append("WARNING: ERP_API_BEARER_TOKEN is not set. API calls will likely fail with 401.")

    call_results: List[Dict[str, Any]] = []
    merged_data: List[Dict[str, Any]] = []

    for endpoint in selected_endpoints:
        endpoint_url = _build_endpoint_url(endpoint.get("url", ""), params)

        request_parts = _collect_request_parts(endpoint, params)
        missing_required = request_parts["missing_required"]
        query_params = request_parts["query_params"]

        if missing_required:
            errors.append(
                f"Missing required parameters for endpoint {endpoint.get('id')}: {', '.join(missing_required)}"
            )
            call_results.append(
                {
                    "endpoint": endpoint.get("id"),
                    "url": endpoint.get("url"),
                    "resolvedUrl": endpoint_url,
                    "fullUrl": None,
                    "query": query_params,
                    "statusCode": None,
                    "dataCount": 0,
                    "error": f"Missing required parameters: {', '.join(missing_required)}",
                    "raw": None,
                }
            )
            continue

        method = endpoint.get("method", "GET").upper()
        resolved_path = _resolve_endpoint_path_from_swagger(
            requested_path=endpoint_url,
            method=method,
            selected=endpoint,
            swagger_paths=swagger_paths,
        )

        candidate_urls: List[str] = []
        for base_url in base_urls:
            full_url = f"{base_url}{resolved_path}"
            if query_params:
                full_url = f"{full_url}?{urlencode(query_params, doseq=True)}"
            candidate_urls.append(full_url)

        last_exc: Optional[Exception] = None
        endpoint_payload: Dict[str, Any] = {
            "endpoint": endpoint.get("id"),
            "url": endpoint.get("url"),
            "resolvedUrl": resolved_path,
            "fullUrl": None,
            "query": query_params,
            "statusCode": None,
            "dataCount": 0,
            "error": None,
            "raw": None,
        }

        for full_url in candidate_urls:
            try:
                response = requests.get(full_url, headers=headers, timeout=60)
                if response.status_code == 401:
                    refreshed = _ensure_bearer_token(base_urls, errors)
                    if refreshed:
                        headers["Authorization"] = f"Bearer {refreshed}"
                        response = requests.get(full_url, headers=headers, timeout=60)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if "json" in content_type:
                    body = response.json()
                else:
                    raw_text = response.text.strip()
                    try:
                        body = response.json()
                    except ValueError:
                        body = {"text": raw_text} if raw_text else {"text": ""}
                data = _normalize_data_field(body)
                endpoint_payload = {
                    "endpoint": endpoint.get("id"),
                    "url": endpoint.get("url"),
                    "resolvedUrl": resolved_path,
                    "fullUrl": full_url,
                    "query": query_params,
                    "statusCode": response.status_code,
                    "dataCount": len(data),
                    "error": None,
                    "raw": body,
                }
                for record in data:
                    merged_data.append(
                        {
                            "endpoint": endpoint.get("id"),
                            "sourceUrl": full_url,
                            "record": record,
                        }
                    )
                last_exc = None
                break
            except requests.RequestException as exc:
                last_exc = exc

        if last_exc is not None:
            errors.append(f"WebApi call failed for {endpoint.get('id')}: {last_exc}")
            endpoint_payload["error"] = str(last_exc)

        call_results.append(endpoint_payload)

    payload: Dict[str, Any] = {
        "generatedAt": datetime.now(UTC).isoformat(),
        "requestAnalysis": analysis,
        "endpoints": [ep.get("id") for ep in selected_endpoints],
        "params": params,
        "calls": call_results,
        "data": merged_data,
    }

    path = CACHE_DIR / "last_api_result.json"
    path.write_text(_safe_json_dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"api_result_path": str(path), "errors": errors}


def evidence_filter(state: AssistantState) -> AssistantState:
    path_str = state.get("api_result_path", "")
    selected_endpoints = state.get("selected_endpoints", [])
    transform_plan = state.get("transform_plan", {"steps": []})
    errors = state.get("errors", []).copy()

    if not path_str:
        return {
            "filtered_result": {"records": [], "count": 0},
            "confidence": 0.0,
        }

    path = Path(path_str)
    if (not path.exists()) or (not path.is_file()):
        return {
            "filtered_result": {"records": [], "count": 0},
            "confidence": 0.0,
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("data", [])
    flattened_rows = _flatten_api_records(records)

    # Ici on applique les opÃ©rations Pandas via le plan gÃ©nÃ©rÃ© par DeepSeek

    transformed_rows = _apply_transform_plan(flattened_rows, transform_plan)
    filtered = transformed_rows

    by_endpoint: Dict[str, int] = {}
    for record in filtered:
        endpoint_id = str(record.get("_endpoint", "unknown"))
        by_endpoint[endpoint_id] = by_endpoint.get(endpoint_id, 0) + 1
    display_rows = _strip_internal_columns(filtered)

    return {
        "filtered_result": {
            "records": display_rows,
            "count": len(display_rows),
            "by_endpoint": by_endpoint,
        },
        "confidence": 0.8 if filtered else 0.2,
        "errors": errors,
    }


def answer_generation(state: AssistantState) -> AssistantState:
    question = state.get("question", "")
    selected_endpoints = state.get("selected_endpoints", [])
    filtered = state.get("filtered_result", {})
    records = filtered.get("records", [])

    def build_data_status_answer() -> str:
        if records:
            return f"{filtered.get('count', 0)} lignes chargees."
        if state.get("errors"):
            return "Aucune donnee chargee."
        return "Aucune donnee trouvee."

    answer = build_data_status_answer()

    if os.getenv("ERP_GENERATE_TEXT_ANSWER", "0") == "1":
        selected = state.get("selected_endpoint")
        compact_evidence = _build_answer_evidence(filtered)
        mapping_context = _build_mapping_context(question=question, max_tables=4, max_columns=6)
        model = os.getenv("OLLAMA_MODEL_ANSWER", "llama3.2")
        system_prompt = (
            "You are an ERP support assistant. "
            "Answer only from provided evidence. "
            "Keep the answer short and never describe unavailable data."
        )
        user_prompt = (
            f"Question: {question}\n"
            f"Primary endpoint: {selected}\n"
            f"Selected endpoints: {json.dumps(selected_endpoints, ensure_ascii=False)}\n"
            f"Business mapping context: {json.dumps(mapping_context, ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(compact_evidence, ensure_ascii=False)}"
        )
        try:
            answer = _call_ollama_chat(model, system_prompt, user_prompt)
        except Exception as exc:
            history = list(state.get("history", []))
            errors = state.get("errors", []).copy()
            errors.append(f"Ollama answer error: {exc}")
            
            history.append({"role": "assistant", "content": answer})
            
            display_payload = _persist_display_result(
                question=question,
                selected_endpoints=selected_endpoints,
                filtered_result=filtered,
                answer=answer,
                errors=errors,
                extracted_params=state.get("extracted_params", {}),
                request_analysis=state.get("request_analysis", {}),
            )
            return {
                "answer": answer,
                "errors": errors,
                "display_result_path": str(DISPLAY_RESULT_PATH),
                "display_result": display_payload,
                "history": history
            }

    history = list(state.get("history", []))
    history.append({"role": "assistant", "content": answer})

    display_payload = _persist_display_result(
        question=question,
        selected_endpoints=selected_endpoints,
        filtered_result=filtered,
        answer=answer,
        errors=state.get("errors", []),
        extracted_params=state.get("extracted_params", {}),
        request_analysis=state.get("request_analysis", {}),
    )
    return {
        "answer": answer,
        "history": history,
        "display_result_path": str(DISPLAY_RESULT_PATH),
        "display_result": display_payload,
    }

def answer_validation(state: AssistantState) -> AssistantState:
    answer = state.get("answer", "")
    confidence = state.get("confidence", 0.0)

    if "could not" in answer.lower():
        confidence = min(confidence, 0.3)
    elif state.get("filtered_result", {}).get("count", 0) > 0:
        confidence = max(confidence, 0.75)

    return {"confidence": confidence}


# ---------- Graph Factory ----------
def build_graph():
    graph = StateGraph(AssistantState)

    graph.add_node("extract_user_request", extract_user_request)
    graph.add_node("retrieve_candidate_endpoints", retrieve_candidate_endpoints)
    graph.add_node("route_endpoint", route_endpoint)
    graph.add_node("generate_transform_plan", generate_transform_plan)
    graph.add_node("call_webapi", call_webapi)
    graph.add_node("evidence_filter", evidence_filter)
    graph.add_node("answer_generation", answer_generation)
    graph.add_node("answer_validation", answer_validation)

    graph.add_edge(START, "extract_user_request")
    graph.add_edge("extract_user_request", "retrieve_candidate_endpoints")
    graph.add_edge("retrieve_candidate_endpoints", "route_endpoint")
    graph.add_edge("route_endpoint", "call_webapi")
    graph.add_edge("call_webapi", "generate_transform_plan")
    graph.add_edge("generate_transform_plan", "evidence_filter")
    graph.add_edge("evidence_filter", "answer_generation")
    graph.add_edge("answer_generation", "answer_validation")
    graph.add_edge("answer_validation", END)

    return graph.compile(checkpointer=MemorySaver())


@lru_cache(maxsize=1)
def _get_compiled_app():
    # One app instance per process so MemorySaver can retain state by thread_id.
    return build_graph()


_APP_LOCK = threading.Lock()


def run_session(question: str, thread_id: str = "default") -> AssistantState:
    app = _get_compiled_app()
    config = {"configurable": {"thread_id": thread_id}}

    inputs = {"question": question, "errors": []}
    # MemorySaver lives in-process; guard invoke to reduce race risks under ThreadingHTTPServer.
    with _APP_LOCK:
        result = app.invoke(inputs, config=config)
    return result

def run_once(question: str) -> AssistantState:
    return run_session(question)


class AssistantRequestHandler(BaseHTTPRequestHandler):
    server_version = "ERPAssistantHTTP/1.0"

    def _send_bytes(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _render_seaborn_png(self, records: List[Dict[str, Any]], options: Optional[Dict[str, Any]] = None) -> bytes:
        """Render a simple Seaborn chart from tabular records.

        Heuristics:
        - categorical + numeric -> barplot (sum)
        - numeric only -> histplot
        - categorical only -> countplot
        """
        if not isinstance(records, list) or not records:
            raise ValueError("No records to plot")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
        except Exception as exc:
            raise RuntimeError(
                "Seaborn dependencies not installed. Install seaborn/matplotlib/pandas."
            ) from exc

        options = options or {}
        df = pd.DataFrame([row for row in records if isinstance(row, dict)])
        if df.empty:
            raise ValueError("No tabular rows available")

        df = df.copy()
        df = df.replace({"": None})

        def is_numeric_series(series: "pd.Series") -> bool:
            if series.dtype.kind in "biufc":
                return True
            coerced = pd.to_numeric(series, errors="coerce")
            return coerced.notna().sum() >= max(1, int(len(series) * 0.6))

        numeric_cols: List[str] = []
        categorical_cols: List[str] = []

        for col in df.columns:
            if col.startswith("_"):
                continue
            s = df[col]
            if is_numeric_series(s):
                numeric_cols.append(col)
            else:
                # Keep short-ish strings / bool as categorical
                if s.dropna().astype(str).map(len).mean() if len(s.dropna()) else 0 < 60:
                    categorical_cols.append(col)

        def clean_option(value: Any) -> str:
            if isinstance(value, list):
                value = value[0] if value else ""
            return str(value or "").strip()

        requested_kind = clean_option(options.get("kind")).lower() or "auto"
        requested_x = clean_option(options.get("x"))
        requested_y = clean_option(options.get("y"))
        requested_agg = clean_option(options.get("agg")).lower() or "sum"
        try:
            limit = int(clean_option(options.get("limit")) or "10")
        except ValueError:
            limit = 10
        limit = max(3, min(50, limit))

        if requested_x and requested_x not in df.columns:
            requested_x = ""
        if requested_y and requested_y not in df.columns:
            requested_y = ""
        if requested_agg not in {"sum", "mean", "count"}:
            requested_agg = "sum"
        if requested_kind not in {"auto", "bar", "line", "hist", "count", "box"}:
            requested_kind = "auto"

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 4.8), dpi=140)

        title = "Seaborn"

        x_candidates = [requested_x] if requested_x else []
        x_candidates += [c for c in categorical_cols if c not in x_candidates]
        x_candidates += [c for c in df.columns if c not in x_candidates]
        y_candidates = [requested_y] if requested_y else []
        y_candidates += [c for c in numeric_cols if c not in y_candidates]

        def numeric_frame(col: str) -> "pd.DataFrame":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            return df[[col]].dropna()

        def aggregate_frame(x: str, y: str) -> "pd.DataFrame":
            work = df[[x, y]].copy()
            work[y] = pd.to_numeric(work[y], errors="coerce")
            work = work.dropna()
            if work.empty:
                return work
            if requested_agg == "mean":
                grouped = work.groupby(x, dropna=True, as_index=False)[y].mean()
                grouped = grouped.sort_values(y, ascending=False)
                return grouped.head(limit)
            if requested_agg == "count":
                grouped = work.groupby(x, dropna=True, as_index=False)[y].count()
                grouped = grouped.rename(columns={y: "count"})
                grouped = grouped.sort_values("count", ascending=False)
                return grouped.head(limit)
            grouped = work.groupby(x, dropna=True, as_index=False)[y].sum()
            grouped = grouped.sort_values(y, ascending=False)
            return grouped.head(limit)

        if requested_kind in {"hist", "box"} or (requested_kind == "auto" and numeric_cols and not categorical_cols):
            col = requested_y or (numeric_cols[0] if numeric_cols else "")
            if not col:
                raise ValueError("No numeric data for plotting")
            df2 = numeric_frame(col)
            if df2.empty:
                raise ValueError("No numeric data for plotting")
            if requested_kind == "box":
                sns.boxplot(data=df2, x=col, ax=ax)
                title = f"Boîte de {col}"
            else:
                sns.histplot(data=df2, x=col, kde=True, ax=ax)
                ax.set_ylabel("count")
                title = f"Distribution de {col}"
            ax.set_xlabel(col)

        elif requested_kind == "count" or (requested_kind == "auto" and categorical_cols and not numeric_cols):
            col = requested_x or (categorical_cols[0] if categorical_cols else "")
            if not col:
                raise ValueError("No categorical data for plotting")
            df2 = df[[col]].dropna()
            if df2.empty:
                raise ValueError("No categorical data for plotting")
            top = df2[col].astype(str).value_counts().head(limit).index.tolist()
            df2 = df2[df2[col].astype(str).isin(top)]
            sns.countplot(data=df2, x=col, ax=ax)
            ax.set_xlabel(col)
            ax.set_ylabel("count")
            ax.tick_params(axis="x", rotation=25)
            title = f"Répartition de {col}"

        elif categorical_cols and numeric_cols:
            # Pick the most usable columns (most non-null values)
            cat_ranked = sorted(x_candidates, key=lambda c: int(df[c].notna().sum()), reverse=True)
            num_ranked = sorted(
                y_candidates,
                key=lambda c: int(pd.to_numeric(df[c], errors="coerce").notna().sum()) if c in df.columns else 0,
                reverse=True,
            )

            plotted = False
            for x in cat_ranked:
                for y in num_ranked:
                    if x not in df.columns or y not in df.columns:
                        continue
                    df2 = aggregate_frame(x, y)
                    if df2.empty:
                        continue
                    value_col = "count" if requested_agg == "count" else y
                    if requested_kind == "line":
                        sns.lineplot(data=df2, x=x, y=value_col, marker="o", ax=ax)
                    else:
                        sns.barplot(data=df2, x=x, y=value_col, errorbar=None, ax=ax)
                    ax.set_xlabel(x)
                    ax.set_ylabel(value_col)
                    ax.tick_params(axis="x", rotation=25)
                    agg_label = {"sum": "Somme", "mean": "Moyenne", "count": "Nombre"}[requested_agg]
                    title = f"{agg_label} de {y} par {x}"
                    plotted = True
                    break
                if plotted:
                    break

            if not plotted:
                # Fallback to numeric-only plot
                col = num_ranked[0]
                df2 = numeric_frame(col)
                if df2.empty:
                    raise ValueError("No non-null data for plotting")
                sns.histplot(data=df2, x=col, kde=True, ax=ax)
                ax.set_xlabel(col)
                ax.set_ylabel("count")
                title = f"Distribution de {col}"

        elif numeric_cols:
            col = numeric_cols[0]
            df2 = numeric_frame(col)
            if df2.empty:
                raise ValueError("No numeric data for plotting")
            sns.histplot(data=df2, x=col, kde=True, ax=ax)
            ax.set_xlabel(col)
            ax.set_ylabel("count")
            title = f"Distribution de {col}"

        elif categorical_cols:
            col = categorical_cols[0]
            df2 = df[[col]].dropna()
            if df2.empty:
                raise ValueError("No categorical data for plotting")
            top = df2[col].astype(str).value_counts().head(limit).index.tolist()
            df2 = df2[df2[col].astype(str).isin(top)]
            sns.countplot(data=df2, x=col, ax=ax)
            ax.set_xlabel(col)
            ax.set_ylabel("count")
            ax.tick_params(axis="x", rotation=25)
            title = f"Répartition de {col}"
        else:
            raise ValueError("No suitable columns to plot")

        ax.set_title(title)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = _safe_json_dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path_only = parsed.path

        if self.path == "/health":
            self._send_json({"status": "ok"})
            return

        if path_only == "/assistant/last-result":
            if DISPLAY_RESULT_PATH.exists() and DISPLAY_RESULT_PATH.is_file():
                try:
                    payload = json.loads(DISPLAY_RESULT_PATH.read_text(encoding="utf-8"))
                    self._send_json(payload)
                    return
                except (json.JSONDecodeError, OSError):
                    self._send_json({"error": "Invalid last display result file"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                    return

            self._send_json({"error": "No display result available yet"}, status=HTTPStatus.NOT_FOUND)
            return

        if path_only == "/assistant/seaborn.png":
            if not DISPLAY_RESULT_PATH.exists() or not DISPLAY_RESULT_PATH.is_file():
                self._send_json({"error": "No display result available yet"}, status=HTTPStatus.NOT_FOUND)
                return

            try:
                payload = json.loads(DISPLAY_RESULT_PATH.read_text(encoding="utf-8"))
                records = payload.get("display", {}).get("records", [])
                if not isinstance(records, list):
                    raise ValueError("Invalid records")
                image = self._render_seaborn_png(records, parse_qs(parsed.query))
                self._send_bytes(image, "image/png", status=HTTPStatus.OK)
                return
            except Exception as exc:
                self._send_json(
                    {"error": "Seaborn render failed", "details": str(exc)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        # Support direct transform application: accepts raw data + extracted_params or plan
        if self.path == "/assistant/apply-transform":
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
                return

            data = payload.get("data", [])
            extracted_params = payload.get("extracted_params", {}) or {}
            requested_fields = payload.get("requested_fields", []) or []
            plan = payload.get("plan")

            # If no explicit plan, build a simple local plan from extracted_params/requested_fields
            if not plan:
                local_steps: List[Dict[str, Any]] = []
                filter_conditions: List[Dict[str, Any]] = []
                for k, v in extracted_params.items():
                    if k in {"requested_fields", "aggregations", "group_by", "groupby", "metrics"}:
                        continue
                    if v is None:
                        continue
                    filter_conditions.append({"field": str(k), "operator": "equals", "value": v})
                if filter_conditions:
                    local_steps.append({"op": "filter_rows", "conditions": filter_conditions})
                if requested_fields:
                    local_steps.append({"op": "select", "columns": requested_fields})
                aggs = extracted_params.get("aggregations") or extracted_params.get("metrics")
                group_by = extracted_params.get("group_by") or extracted_params.get("groupby")
                normalized_aggs: List[Dict[str, Any]] = []
                if aggs:
                    if isinstance(aggs, dict):
                        for alias, expr in aggs.items():
                            m = re.match(r"\s*(sum|avg|mean|count|min|max)\s*\(?\s*([a-zA-Z0-9_]+)\s*\)?\s*", str(expr), re.I)
                            if m:
                                normalized_aggs.append({"field": m.group(2), "agg": m.group(1).lower(), "as": alias})
                    elif isinstance(aggs, list):
                        for item in aggs:
                            if isinstance(item, dict) and item.get("field"):
                                normalized_aggs.append({"field": item.get("field"), "agg": item.get("agg", "sum"), "as": item.get("as")})
                            elif isinstance(item, str):
                                m = re.match(r"\s*(sum|avg|mean|count|min|max)\s*\(?\s*([a-zA-Z0-9_]+)\s*\)?\s*", item, re.I)
                                if m:
                                    normalized_aggs.append({"field": m.group(2), "agg": m.group(1).lower(), "as": None})
                if normalized_aggs:
                    agg_step: Dict[str, Any] = {"op": "aggregate", "groupby": group_by if isinstance(group_by, list) else ([group_by] if group_by else []), "aggs": normalized_aggs}
                    local_steps.append(agg_step)
                local_steps.append({"op": "limit", "value": int(payload.get("limit", 100))})
                plan = {"steps": local_steps}

            try:
                # detect pandas availability for debugging
                try:
                    import pandas as _pd
                    pandas_available = True
                except Exception:
                    pandas_available = False
                # apply transform (pandas preferred)
                transformed = _apply_transform_plan(data, plan)
                resp = {"records": transformed, "count": len(transformed), "plan": plan, "pandas_available": pandas_available}
                self._send_json(resp)
            except Exception as exc:
                self._send_json({"error": "Transform failed", "details": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if self.path != "/assistant/query":
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
            return

        question = str(payload.get("question", "")).strip()
        if not question:
            self._send_json({"error": "Question is required"}, status=HTTPStatus.BAD_REQUEST)
            return

        thread_id = str(payload.get("thread_id") or payload.get("thread") or payload.get("session_id") or "default").strip() or "default"

        try:
            result = run_session(question, thread_id=thread_id)
            # Echo thread_id for UI convenience
            if isinstance(result, dict):
                result = {**result, "thread_id": thread_id}
            self._send_json(result)
        except Exception as exc:
            self._send_json(
                {
                    "error": "Assistant execution failed",
                    "details": str(exc),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve_http(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), AssistantRequestHandler)
    print(f"Assistant HTTP server running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Run the assistant as an HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("ERP_ASSISTANT_PORT", "8000")))
    parser.add_argument("--question", default="Quels sont mes clients ?")
    args = parser.parse_args()

    if args.serve:
        serve_http(args.host, args.port)
    else:
        output = run_once(args.question)
        print(_safe_json_dumps(output, indent=2, ensure_ascii=False))
