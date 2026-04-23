from __future__ import annotations

import argparse
import json
import os
import re
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlencode
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import requests
from langgraph.graph import END, START, StateGraph


ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DISPLAY_RESULT_PATH = CACHE_DIR / "last_display_result.json"
DATASET_MAPPING_PATH = ROOT_DIR / "data" / "dataset_mapping_db.json"


class AssistantState(TypedDict, total=False):
    question: str
    intent: str
    domain: str
    endpoint_candidates: List[Dict[str, Any]]
    selected_endpoints: List[Dict[str, Any]]
    selected_endpoint: Optional[Dict[str, Any]]
    extracted_params: Dict[str, Any]
    api_result_path: str
    display_result_path: str
    display_result: Dict[str, Any]
    filtered_result: Dict[str, Any]
    schema_hint: Dict[str, Any]
    answer: str
    confidence: float
    errors: List[str]


# ---------- Utilities ----------
def _load_endpoints() -> List[Dict[str, Any]]:
    source_mode = os.getenv("ERP_ENDPOINT_SOURCE", "swagger").strip().lower()

    if source_mode == "swagger":
        return _load_swagger_generated_endpoints()

    configured = os.getenv("ERP_ENDPOINTS_JSON", "").strip()
    if configured:
        path = Path(configured)
    else:
        path = ROOT_DIR / "data" / "endpoints.sample.json"

    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    configured_endpoints = payload.get("endpoints", [])

    # Optionally enrich with live WebApi routes from Swagger.
    load_swagger = os.getenv("ERP_LOAD_SWAGGER_ENDPOINTS", "1") == "1"
    if not load_swagger:
        return configured_endpoints

    extra = _load_swagger_generated_endpoints()
    if not extra:
        return configured_endpoints

    existing_ids = {str(ep.get("id", "")) for ep in configured_endpoints}
    merged = configured_endpoints.copy()
    for ep in extra:
        if ep["id"] not in existing_ids:
            merged.append(ep)
    return merged


def _path_to_generated_id(path: str, method: str) -> str:
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", path.lower()) if t]
    return "webapi_" + method.lower() + "_" + "_".join(tokens)


def _path_to_keywords(path: str) -> List[str]:
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", path)
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", expanded.lower()) if t]
    stop = {"api", "odata", "get", "all", "by", "id", "v1", "swagger"}
    uniq: List[str] = []
    for t in tokens:
        if t in stop:
            continue
        if t not in uniq:
            uniq.append(t)
    return uniq[:8]


def _load_swagger_payload() -> Dict[str, Any]:
    base_urls = _get_erp_api_base_urls()
    for base_url in base_urls:
        try:
            response = requests.get(f"{base_url}/swagger/v1/swagger.json", timeout=20)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
        except requests.RequestException:
            continue

    local_swagger = os.getenv("ERP_SWAGGER_JSON", "").strip()
    if local_swagger:
        fallback_path = Path(local_swagger)
    else:
        fallback_path = ROOT_DIR / "swagger_live.json"

    if fallback_path.exists():
        payload = json.loads(fallback_path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            return payload

    return {}


def _load_swagger_generated_endpoints() -> List[Dict[str, Any]]:
    generated: List[Dict[str, Any]] = []
    swagger_payload = _load_swagger_payload()
    swagger_paths = swagger_payload.get("paths", {})
    if not isinstance(swagger_paths, dict) or not swagger_paths:
        return generated

    for path, methods in swagger_paths.items():
        if not isinstance(methods, dict):
            continue

        for method_name, operation in methods.items():
            method = str(method_name).upper()
            if method != "GET":
                continue

            operation = operation if isinstance(operation, dict) else {}
            tags = [str(tag) for tag in operation.get("tags", []) if tag]
            summary = str(operation.get("summary", ""))
            description = str(operation.get("description", ""))
            operation_id = str(operation.get("operationId", ""))
            parameters = operation.get("parameters", [])
            query_parameters = [
                str(param.get("name"))
                for param in parameters
                if isinstance(param, dict) and param.get("in") == "query" and param.get("name")
            ]
            route_parameters = [
                str(param.get("name"))
                for param in parameters
                if isinstance(param, dict) and param.get("in") == "path" and param.get("name")
            ]

            keywords = _path_to_keywords(
                " ".join(
                    [
                        path,
                        summary,
                        description,
                        operation_id,
                        " ".join(tags),
                        " ".join(query_parameters),
                    ]
                )
            )
            role = _infer_domain_from_path(" ".join([path, summary, description, operation_id, " ".join(tags)]))

            generated.append(
                {
                    "id": _path_to_generated_id(path, method),
                    "method": method,
                    "url": path,
                    "intent": "AGGREGATE" if _contains_any(path + " " + summary + " " + description, ["stat", "report", "vente", "chiffre", "dashboard"]) else "GET",
                    "keywords": keywords,
                    "routeParameters": route_parameters or re.findall(r"\{([^{}]+)\}", path),
                    "queryParameters": query_parameters,
                    "requiredParameters": route_parameters or re.findall(r"\{([^{}]+)\}", path),
                    "responseFormat": "Object",
                    "description": description or summary or f"Generated from Swagger: {path}",
                    "examples": [],
                    "role": role,
                    "tags": tags,
                }
            )

    return generated


def _load_endpoint_overrides() -> Dict[str, str]:
    configured = os.getenv("ERP_ENDPOINT_OVERRIDES_JSON", "").strip()
    if configured:
        path = Path(configured)
    else:
        path = ROOT_DIR / "data" / "endpoint_overrides.json"

    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(k): str(v) for k, v in payload.items()}
    return {}


@lru_cache(maxsize=1)
def _load_dataset_mapping_db() -> List[Dict[str, Any]]:
    if not DATASET_MAPPING_PATH.exists():
        return []

    try:
        payload = json.loads(DATASET_MAPPING_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return []

    return payload if isinstance(payload, list) else []


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _contains_any(text: str, terms: List[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _is_client_variable_identification_question(question: str) -> bool:
    q = question.lower()
    has_client_context = _contains_any(q, ["client", "clients"])
    has_variable_intent = _contains_any(
        q,
        [
            "variable",
            "variables",
            "champ",
            "champs",
            "colonne",
            "colonnes",
            "attribut",
            "attributs",
            "structure",
            "schema",
            "schéma",
            "identifier",
        ],
    )
    has_field_intent = _contains_any(
        q,
        [
            "nom",
            "email",
            "e-mail",
            "mail",
            "telephone",
            "téléphone",
            "tel",
            "adresse",
            "code",
            "raison",
            "solde",
            "ca",
            "chiffre",
            "dette",
        ],
    )
    return has_client_context and (has_variable_intent or has_field_intent)


def _infer_domain_from_question(question: str) -> str:
    q = question.lower()
    # Priority anchors for common ambiguous phrases.
    if any(w in q for w in ["paiement", "paiements", "depense", "dépense", "transfert", "solde", "créance", "creance"]):
        return "finance"
    if any(w in q for w in ["fournisseur", "fournisseurs", "achat"]):
        return "achat"
    if any(w in q for w in ["employe", "employés", "paie", "congé", "conge", "salaire"]):
        return "rh"
    if any(w in q for w in ["stock", "inventaire", "depot", "dépôt", "lot"]):
        return "stock"

    domain_keywords = {
        "commercial": ["client", "clients", "commande", "commandes", "vente", "facture", "bl"],
        "stock": ["stock", "inventaire", "depot", "dépôt", "article", "articles", "lot"],
        "finance": ["paiement", "paiements", "depense", "dépense", "transfert", "solde", "creance", "créance"],
        "rh": ["employe", "employés", "paie", "conge", "congé", "salaire"],
        "achat": ["fournisseur", "fournisseurs", "achat", "frs"],
    }
    best_domain = "general"
    best_score = 0
    for domain, words in domain_keywords.items():
        score = sum(1 for w in words if w in q)
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain


def _infer_domain_from_path(path: str) -> str:
    tokens = set(_split_path_tokens(path))
    if tokens.intersection({"client", "clients", "commande", "commandes", "blclient", "statsvente", "fact"}):
        return "commercial"
    if tokens.intersection({"stock", "depot", "lot", "article", "articles", "bonentree", "bontransfert"}):
        return "stock"
    if tokens.intersection({"paiement", "paiements", "depense", "depenses", "finance", "transfert"}):
        return "finance"
    if tokens.intersection({"demandeconge", "conge", "paie", "employe", "employes"}):
        return "rh"
    if tokens.intersection({"fournisseur", "fournisseurs", "blfrs", "frs"}):
        return "achat"
    return "general"


def _extract_simple_params(question: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    id_match = re.search(r"\b(?:id|client|commande|article|employe)\s*(\d+)\b", question, re.IGNORECASE)
    if id_match:
        out["id"] = int(id_match.group(1))

    date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", question)
    if date_match:
        out["date"] = date_match.group(1)

    return out


def _extract_requested_fields(question: str) -> List[str]:
    q = question.lower()
    mapping = {
        "email": ["email", "e-mail", "mail", "courriel"],
        "telephone": ["telephone", "téléphone", "tel", "mobile", "gsm"],
        "nom": ["nom", "name"],
        "code": ["code", "reference", "référence", "ref"],
        "adresse": ["adresse", "address"],
        "ville": ["ville", "city"],
    }

    requested: List[str] = []
    for canonical, aliases in mapping.items():
        if any(alias in q for alias in aliases):
            requested.append(canonical)

    return requested


def _normalize_text_tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    return set(_tokenize(str(value)))


def _infer_dataset_table(question: str, selected_endpoints: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    mapping_db = _load_dataset_mapping_db()
    if not mapping_db:
        return None

    q_tokens = _normalize_text_tokens(question)
    endpoint_text = ""
    if selected_endpoints:
        endpoint_text = " ".join(
            " ".join(
                [
                    str(ep.get("id", "")),
                    str(ep.get("url", "")),
                    str(ep.get("description", "")),
                    " ".join(str(tag) for tag in ep.get("tags", [])),
                ]
            )
            for ep in selected_endpoints
        ).lower()

    best_table: Optional[Dict[str, Any]] = None
    best_score = 0

    for table in mapping_db:
        if not isinstance(table, dict):
            continue

        table_tokens = _normalize_text_tokens(table.get("table"))
        table_tokens.update(_normalize_text_tokens(table.get("businessDomain")))
        table_tokens.update(_normalize_text_tokens(table.get("description")))

        column_tokens: set[str] = set()
        for column in table.get("columns", []):
            if not isinstance(column, dict):
                continue
            column_tokens.update(_normalize_text_tokens(column.get("name")))
            column_tokens.update(_normalize_text_tokens(column.get("description")))
            for synonym in column.get("synonyms", []):
                column_tokens.update(_normalize_text_tokens(synonym))

        score = len(q_tokens.intersection(table_tokens)) * 4
        score += len(q_tokens.intersection(column_tokens)) * 2

        if endpoint_text:
            score += len(table_tokens.intersection(set(_tokenize(endpoint_text))))

        if score > best_score:
            best_score = score
            best_table = table

    return best_table


def _build_schema_hint(table: Optional[Dict[str, Any]], max_columns: int = 8) -> Dict[str, Any]:
    if not table:
        return {}

    columns: List[Dict[str, Any]] = []
    for column in table.get("columns", [])[:max_columns]:
        if not isinstance(column, dict):
            continue
        columns.append(
            {
                "name": column.get("name"),
                "type": column.get("type"),
                "description": column.get("description"),
                "synonyms": list(column.get("synonyms", []))[:5],
            }
        )

    return {
        "table": table.get("table"),
        "businessDomain": table.get("businessDomain"),
        "description": table.get("description"),
        "columns": columns,
    }


def _infer_requested_fields(question: str, table: Optional[Dict[str, Any]] = None) -> List[str]:
    requested = _extract_requested_fields(question)
    if not table:
        return requested

    q = question.lower()
    for column in table.get("columns", []):
        if not isinstance(column, dict):
            continue

        column_name = str(column.get("name", "")).lower()
        synonyms = [str(s).lower() for s in column.get("synonyms", [])]
        description = str(column.get("description", "")).lower()

        if column_name and column_name not in requested:
            if column_name in q or any(s in q for s in synonyms) or any(word in description for word in ["mail", "nom", "téléphone", "telephone", "code"] if word in q):
                requested.append(column_name)

    return requested


def _project_record_fields(record: Any, requested_fields: List[str]) -> Any:
    if not requested_fields or not isinstance(record, dict):
        return record

    selected: Dict[str, Any] = {}
    aliases = {
        "email": ["email", "mail", "courriel"],
        "telephone": ["telephone", "tel", "mobile", "gsm", "phone"],
        "nom": ["nom", "name", "intitule", "libelle"],
        "code": ["code", "reference", "ref", "id"],
        "adresse": ["adresse", "address", "addr"],
        "ville": ["ville", "city"],
    }

    for key, value in record.items():
        lowered_key = str(key).lower()
        for requested in requested_fields:
            if any(alias in lowered_key for alias in aliases.get(requested, [requested])):
                selected[key] = value
                break

    return selected if selected else record


def _project_record_with_schema(record: Any, requested_fields: List[str], table: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(record, dict) or not table:
        return _project_record_fields(record, requested_fields)

    selected: Dict[str, Any] = {}
    normalized_requested = {str(field).lower() for field in requested_fields}

    for key, value in record.items():
        lowered_key = str(key).lower()
        if lowered_key in normalized_requested:
            selected[key] = value
            continue

        for column in table.get("columns", []):
            if not isinstance(column, dict):
                continue
            column_name = str(column.get("name", "")).lower()
            if lowered_key == column_name and (not normalized_requested or column_name in normalized_requested):
                selected[key] = value
                break
            if any(str(s).lower() == lowered_key or str(s).lower() in lowered_key for s in column.get("synonyms", [])):
                if not normalized_requested or column_name in normalized_requested or any(s in normalized_requested for s in [column_name]):
                    selected[key] = value
                    break

    if selected:
        return selected
    return _project_record_fields(record, requested_fields)


def _fallback_domain_endpoints(endpoints: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
    fallbacks: List[Dict[str, Any]] = []
    for endpoint in endpoints:
        if str(endpoint.get("method", "GET")).upper() != "GET":
            continue

        role = str(endpoint.get("role", "general"))
        if domain != "general" and role not in {domain, "general"}:
            continue

        text = " ".join(
            [
                str(endpoint.get("id", "")),
                str(endpoint.get("url", "")),
                str(endpoint.get("description", "")),
            ]
        ).lower()

        if _contains_any(text, ["getall", "list", "all", "/api/"]):
            fallbacks.append({"score": 1, **endpoint})

    return fallbacks


def _prioritize_endpoint_candidates(question: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    q = question.lower()
    variable_identification_context = _is_client_variable_identification_question(question)
    bl_context = _contains_any(q, [" bl ", "bon de livraison", "livraison", "livraisons"])
    client_context = _contains_any(q, ["client", "clients"])

    def priority_key(endpoint: Dict[str, Any]) -> tuple[int, int, str]:
        url = str(endpoint.get("url", "")).lower()
        endpoint_id = str(endpoint.get("id", "")).lower()

        if variable_identification_context:
            if "/api/client/getallclients" in url:
                return (0, 0, endpoint_id)
            if "/api/blclient/getallclients" in url:
                return (2, 0, endpoint_id)
            return (1, 0, endpoint_id)

        if client_context and not bl_context:
            if "/api/blclient/getallclients" in url:
                return (0, 0, endpoint_id)
            if "/api/client/getallclients" in url:
                return (3, 0, endpoint_id)
            return (1, 0, endpoint_id)

        if bl_context:
            if "/api/blclient/getallclients" in url:
                return (0, 0, endpoint_id)
            if "/api/client/getallclients" in url:
                return (3, 0, endpoint_id)
            return (1, 0, endpoint_id)

        if _contains_any(q, ["stock", "inventaire"]):
            if _contains_any(url, ["/api/bonentree", "/api/stockdepot", "/api/bontransfert"]):
                return (0, 0, endpoint_id)

        if _contains_any(q, ["paiement", "paiements", "reglement", "règlement"]):
            if _contains_any(url, ["/api/paiements", "/api/depenses"]):
                return (0, 0, endpoint_id)

        return (1, 0, endpoint_id)

    return sorted(candidates, key=priority_key)


def _is_supported_business_endpoint(endpoint: Dict[str, Any]) -> bool:
    if str(endpoint.get("method", "GET")).upper() != "GET":
        return False

    url = str(endpoint.get("url", "")).lower()
    endpoint_id = str(endpoint.get("id", "")).lower()
    description = str(endpoint.get("description", "")).lower()
    tags = " ".join(str(tag) for tag in endpoint.get("tags", [])).lower()
    endpoint_text = " ".join([url, endpoint_id, description, tags])

    blocked_terms = [
        "swagger",
        "openapi",
        "health",
        "generate-test",
        "generatetest",
        "/test",
        "testendpoint",
        "debug",
        "token",
        "login",
        "signin",
        "auth",
    ]
    if _contains_any(endpoint_text, blocked_terms):
        return False

    if "/api/reports/" in url and not _contains_any(endpoint_text, ["vente", "commande", "client", "fact", "report"]):
        return False

    business_allow_terms = [
        "getall",
        "list",
        "all",
        "odata",
        "client",
        "commande",
        "vente",
        "stock",
        "paiement",
        "fournisseur",
        "fact",
        "article",
        "report",
        "stats",
    ]
    return _contains_any(endpoint_text, business_allow_terms)


def _compute_endpoint_score(
    endpoint: Dict[str, Any],
    q_tokens: set[str],
    question: str,
    intent: str,
    domain: str,
) -> int:
    # Scoring is intentionally disabled for now.
    # DeepSeek Coder receives the raw candidate pool and selects the route directly.
    # Keep this function returning a neutral score so the candidate order stays stable.
    return 0

    # NOTE: Legacy heuristic scoring kept below for future reactivation.
    ep_keywords = [str(keyword) for keyword in endpoint.get("keywords", [])]
    ep_examples = [str(example) for example in endpoint.get("examples", [])]
    ep_text = " ".join(
        [
            str(endpoint.get("id", "")),
            str(endpoint.get("url", "")),
            str(endpoint.get("description", "")),
            " ".join(ep_keywords),
            " ".join(ep_examples),
            " ".join(str(tag) for tag in endpoint.get("tags", [])),
        ]
    ).lower()
    ep_tokens = set(_tokenize(ep_text))

    overlap = len(q_tokens.intersection(ep_tokens))
    role_match = endpoint.get("role") == domain
    intent_match = endpoint.get("intent") == intent

    score = overlap * 4
    url = str(endpoint.get("url", "")).lower()

    if overlap > 0 and intent_match:
        score += 2
    if overlap > 0 and role_match:
        score += 1

    if _contains_any(ep_text, ["getall", "list", "odata"]):
        score += 4
    if _contains_any(url, ["/api/reports/commande_client-report", "/api/statsvente/"]):
        score += 5
    if _contains_any(ep_text, ["generate-test", "generatetest", "/test", "debug"]):
        score -= 20

    if _contains_any(question, ["vente", "ventes", "statistique", "statistiques", "chiffre", "ca", "rapport"]):
        if _contains_any(ep_text, ["vente", "ventes", "statsvente", "report", "reports", "commande_client", "fact"]):
            score += 6
        if _contains_any(ep_text, ["client/getallclients", "get_clients", "blclient/getallclients"]):
            score -= 4

    if _contains_any(question, ["client", "clients"]) and _contains_any(ep_text, ["client", "clients"]):
        score += 3

    bl_context = _contains_any(question, [" bl ", "bon de livraison", "livraison", "livraisons"])
    variable_identification_context = _is_client_variable_identification_question(question)
    generic_client_context = _contains_any(question, ["client", "clients"]) and not bl_context and not variable_identification_context
    if generic_client_context:
        if "/api/blclient/getallclients" in url:
            score += 8
        if "/api/client/getallclients" in url:
            score -= 12
        if "/api/commandeclient/getallclients" in url:
            score -= 2

    if variable_identification_context:
        if "/api/client/getallclients" in url:
            score += 16
        if "/api/blclient/getallclients" in url:
            score -= 4

    if _contains_any(question, ["email", "e-mail", "mail", "courriel"]):
        if _contains_any(ep_text, ["email", "mail", "client"]):
            score += 8
        if "/api/client/getallclients" in url:
            score += 8
        if "/api/blclient/getallclients" in url:
            score -= 4

    if _contains_any(question, ["stock", "inventaire"]) and _contains_any(ep_text, ["stock", "depot", "article"]):
        score += 5

    if _contains_any(question, ["paiement", "paiements", "reglement", "règlement"]) and _contains_any(ep_text, ["paiement", "payments", "fact"]):
        score += 5

    return score


def _score_endpoints(
    endpoints: List[Dict[str, Any]],
    question: str,
    intent: str,
    domain: str,
    apply_business_filter: bool,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []

    for ep in endpoints:
        if apply_business_filter and not _is_supported_business_endpoint(ep):
            continue

        url = str(ep.get("url", "")).lower()
        if "/api/client/getallclients" in url and not _is_client_variable_identification_question(question):
            continue

        if str(ep.get("id", "")).startswith("webapi_get_"):
            role = ep.get("role", "general")
            if domain != "general" and role not in {domain, "general"}:
                continue

        scored.append({"score": 0, **ep})

    return scored


def _determine_endpoint_limit(question: str, intent: str) -> int:
    q = question.lower()
    multi_markers = [
        " et ",
        " avec ",
        " ainsi que ",
        " compare ",
        " comparaison ",
        " resume ",
        " résumé ",
        " tableau de bord ",
        " dashboard ",
        " situation ",
        " synthese ",
        " synthèse ",
    ]
    if intent == "AGGREGATE":
        return 3
    if any(marker in f" {q} " for marker in multi_markers):
        return 3
    return 1


def _normalize_requested_fields(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if item is None:
            continue
        token = str(item).strip().lower()
        if token and token not in out:
            out.append(token)
    return out


def _compute_endpoint_column_coverage(endpoint: Dict[str, Any], requested_fields: List[str]) -> int:
    if not requested_fields:
        return 0

    alias_map = {
        "email": ["email", "e-mail", "mail", "courriel"],
        "telephone": ["telephone", "téléphone", "tel", "mobile", "gsm", "phone"],
        "nom": ["nom", "name", "raison", "raison sociale", "intitule", "libelle"],
        "code": ["code", "reference", "référence", "ref", "id"],
        "adresse": ["adresse", "address", "addr"],
        "ville": ["ville", "city"],
    }

    endpoint_text = " ".join(
        [
            str(endpoint.get("id", "")),
            str(endpoint.get("url", "")),
            str(endpoint.get("description", "")),
            " ".join(str(k) for k in endpoint.get("keywords", [])),
            " ".join(str(ex) for ex in endpoint.get("examples", [])),
            " ".join(str(tag) for tag in endpoint.get("tags", [])),
            " ".join(str(q) for q in endpoint.get("queryParameters", [])),
            " ".join(str(rp) for rp in endpoint.get("requiredParameters", [])),
        ]
    ).lower()
    endpoint_tokens = set(_tokenize(endpoint_text))

    score = 0
    for field in requested_fields:
        aliases = alias_map.get(field, [field])
        hit = False
        for alias in aliases:
            alias_lower = str(alias).lower()
            alias_tokens = set(_tokenize(alias_lower))
            if alias_lower in endpoint_text or (alias_tokens and alias_tokens.issubset(endpoint_tokens)):
                hit = True
                break
        if hit:
            score += 3

    url = str(endpoint.get("url", "")).lower()
    if any(f in requested_fields for f in ["email", "telephone", "nom", "code"]):
        if "/api/blclient/getallclients" in url:
            score += 4
        elif "/api/client/getallclients" in url:
            score += 2

    return score


def _rerank_endpoints_by_requested_fields(
    endpoints: List[Dict[str, Any]],
    requested_fields: List[str],
) -> List[Dict[str, Any]]:
    normalized_fields = _normalize_requested_fields(requested_fields)
    if not normalized_fields or len(endpoints) <= 1:
        return endpoints

    decorated: List[tuple[int, int, Dict[str, Any]]] = []
    for idx, endpoint in enumerate(endpoints):
        coverage = _compute_endpoint_column_coverage(endpoint, normalized_fields)
        decorated.append((coverage, idx, endpoint))

    decorated.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in decorated]


def _extract_params_with_llama(
    question: str,
    intent: str,
    candidates: List[Dict[str, Any]],
    requested_fields: List[str],
) -> tuple[Dict[str, Any], Optional[str]]:
    use_ollama = os.getenv("USE_OLLAMA", "0") == "1"
    if not use_ollama:
        return {}, None

    extractor_model = os.getenv("OLLAMA_MODEL_PARAM_EXTRACTOR", os.getenv("OLLAMA_MODEL_ANSWER", "llama3.2:latest"))
    llm_candidate_limit = min(len(candidates), int(os.getenv("ERP_ROUTER_CANDIDATE_LIMIT", "12")))
    compact_candidates = _build_router_candidates_payload(candidates, llm_candidate_limit)

    table_hint = _infer_dataset_table(question, candidates)
    schema_hint = _build_schema_hint(table_hint)

    system_prompt = (
        "You are an ERP parameter extraction assistant. "
        "Extract useful filters and requested columns from the user message. "
        "Use business wording and schema hints when available. "
        "Return strict JSON only with: extracted_params."
    )
    user_prompt = (
        f"Question: {question}\n"
        f"Intent: {intent}\n"
        f"Requested fields guess: {json.dumps(requested_fields, ensure_ascii=False)}\n"
        f"Schema hint: {json.dumps(schema_hint, ensure_ascii=False)}\n"
        f"Candidate endpoints (for parameter names): {json.dumps(compact_candidates, ensure_ascii=False)}\n"
        "Rules:\n"
        "- Put route/query filter values in extracted_params (codeArticle, numBlClient, id, date, etc.).\n"
        "- Put user requested columns in extracted_params.requested_fields as an array.\n"
        "- If nothing found, return extracted_params as an empty object."
    )

    try:
        llm_json = _call_ollama_json(extractor_model, system_prompt, user_prompt)
        if not isinstance(llm_json, dict):
            return {}, None
        llm_params = llm_json.get("extracted_params", {})
        if not isinstance(llm_params, dict):
            return {}, None
        return llm_params, None
    except Exception as exc:
        return {}, f"Parameter extraction Ollama error: {exc}"


def _select_best_endpoints(
    candidates: List[Dict[str, Any]],
    question: str,
    intent: str,
    use_ollama: bool,
    router_model: str,
    extracted_params: Optional[Dict[str, Any]] = None,
    requested_fields: Optional[List[str]] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    if not candidates:
        return [], {}, None

    limit = min(_determine_endpoint_limit(question, intent), len(candidates))
    requested_fields = requested_fields or []
    ranked_candidates = _rerank_endpoints_by_requested_fields(candidates, requested_fields)
    selected = ranked_candidates[:limit]
    llm_candidate_limit = min(len(candidates), int(os.getenv("ERP_ROUTER_CANDIDATE_LIMIT", "12")))
    router_error: Optional[str] = None
    merged_extracted_params: Dict[str, Any] = dict(extracted_params or {})

    if use_ollama:
        system_prompt = (
            "You are the primary endpoint router for an ERP API. "
            "Choose the exact business GET endpoint or endpoints that best answer the user question. "
            "The user asked for specific columns, so prefer endpoints likely to expose these columns. "
            "Prefer business list/report/statistics endpoints over technical, test, auth, or utility endpoints. "
            "Prefer endpoints like GetAll, OData list, stats, reports, and métier routes that directly match the question. "
            "Return strict JSON only with fields: endpoint_ids, extracted_params. "
            "endpoint_ids must be an array using only candidate ids. "
            "Choose multiple endpoints only when truly necessary."
        )
        compact_candidates = _build_router_candidates_payload(ranked_candidates, llm_candidate_limit)
        router_params = merged_extracted_params
        user_prompt = (
            f"Question: {question}\n"
            f"Intent: {intent}\n"
            f"Extracted params from user message: {json.dumps(router_params, ensure_ascii=False)}\n"
            f"Requested columns: {json.dumps(requested_fields, ensure_ascii=False)}\n"
            f"Candidates: {json.dumps(compact_candidates, ensure_ascii=False)}\n"
            "Pick the best endpoint_ids from candidates and extract useful parameters from the question. "
            "Prefer endpoints whose query/required parameters can use the extracted params. "
            "Prefer endpoints that can return requested columns. "
            "Do not choose test or utility endpoints unless the question explicitly asks for them."
        )
        try:
            llm_choice = _call_ollama_json(router_model, system_prompt, user_prompt)
            if llm_choice:
                llm_params = llm_choice.get("extracted_params", {})
                if isinstance(llm_params, dict):
                    merged_extracted_params.update(llm_params)
                endpoint_ids = llm_choice.get("endpoint_ids")
                if isinstance(endpoint_ids, list):
                    chosen_ids = [str(endpoint_id) for endpoint_id in endpoint_ids if endpoint_id]
                    chosen = [c for c in ranked_candidates if str(c.get("id")) in chosen_ids]
                    if chosen:
                        reranked = _rerank_endpoints_by_requested_fields(chosen, requested_fields)
                        return reranked[:limit], merged_extracted_params, None
                endpoint_id = llm_choice.get("endpoint_id")
                if endpoint_id:
                    chosen = next((c for c in ranked_candidates if c.get("id") == endpoint_id), None)
                    if chosen:
                        return [chosen], merged_extracted_params, None
        except Exception as exc:
            router_error = f"Router Ollama error: {exc}"

    selected = _rerank_endpoints_by_requested_fields(selected, requested_fields)
    return selected, merged_extracted_params, router_error


def _call_ollama_chat(model: str, system_prompt: str, user_prompt: str, timeout_seconds: Optional[int] = None) -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    if timeout_seconds is None:
        timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    response = requests.post(url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    return content.strip()


def _call_ollama_json(model: str, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
    router_timeout_seconds = int(os.getenv("OLLAMA_ROUTER_TIMEOUT_SECONDS", os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")))
    raw = _call_ollama_chat(model, system_prompt, user_prompt, timeout_seconds=router_timeout_seconds)
    # Tolerate fenced or plain JSON model outputs.
    cleaned = raw.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
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


def _build_router_candidates_payload(candidates: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for candidate in candidates[:limit]:
        compact.append(
            {
                "id": candidate.get("id"),
                "url": candidate.get("url"),
                "intent": candidate.get("intent"),
                "requiredParameters": candidate.get("requiredParameters", []),
                "queryParameters": candidate.get("queryParameters", [])[:6],
                "role": candidate.get("role", "general"),
                "tags": candidate.get("tags", [])[:4],
                "description": _truncate_text(candidate.get("description", ""), 140),
                "score": candidate.get("score", 0),
            }
        )
    return compact


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


def _persist_display_result(
    question: str,
    selected_endpoints: List[Dict[str, Any]],
    filtered_result: Dict[str, Any],
    answer: str,
    errors: List[str],
    schema_hint: Optional[Dict[str, Any]] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
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
        "schemaHint": schema_hint or filtered_result.get("schema_hint", {}),
        "extractedParams": extracted_params or {},
        "display": {
            "count": filtered_result.get("count", 0),
            "by_endpoint": filtered_result.get("by_endpoint", {}),
            "records": filtered_result.get("records", []),
        },
        "answer": answer,
        "errors": errors,
    }
    DISPLAY_RESULT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _get_erp_api_base_urls() -> List[str]:
    explicit = os.getenv("ERP_API_BASE_URL", "").strip().rstrip("/")
    if explicit:
        return [explicit]

    # Optional auto-discovery from the local WebApi project path.
    webapi_project_dir = os.getenv("ERP_WEBAPI_PROJECT_DIR", "").strip()
    if not webapi_project_dir:
        return []

    launch_settings = Path(webapi_project_dir) / "Properties" / "launchSettings.json"
    if not launch_settings.exists():
        return []

    try:
        payload = json.loads(launch_settings.read_text(encoding="utf-8-sig"))
        profiles = payload.get("profiles", {})

        # Prefer explicit WebApi profile, then fallback to any profile containing applicationUrl.
        candidate_profiles: List[Dict[str, Any]] = []
        if isinstance(profiles, dict):
            if isinstance(profiles.get("WebApi"), dict):
                candidate_profiles.append(profiles["WebApi"])
            candidate_profiles.extend(
                p for p in profiles.values() if isinstance(p, dict) and p not in candidate_profiles
            )

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

    # Safe defaults for list endpoints with pagination support.
    if "pageNumber" in query_keys and "pageNumber" not in query_params:
        query_params["pageNumber"] = 1
    if "pageSize" in query_keys and "pageSize" not in query_params:
        query_params["pageSize"] = 50

    return {
        "missing_required": missing_required,
        "query_params": query_params,
    }


def _normalize_data_field(payload: Any) -> List[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("value"), list):
            return payload["value"]
        if isinstance(payload.get("items"), list):
            return payload["items"]
        return [payload]
    return []


def _split_path_tokens(text: str) -> List[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    raw = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return [t for t in raw if t]


def _singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _fetch_swagger_paths_with_methods(base_urls: List[str]) -> Dict[str, List[str]]:
    for base_url in base_urls:
        try:
            response = requests.get(f"{base_url}/swagger/v1/swagger.json", timeout=20)
            response.raise_for_status()
            payload = response.json()
            path_map: Dict[str, List[str]] = {}
            for path, methods in payload.get("paths", {}).items():
                if isinstance(methods, dict):
                    path_map[path] = [m.upper() for m in methods.keys()]
            if path_map:
                return path_map
        except requests.RequestException:
            continue
    return {}


def _resolve_endpoint_path_from_swagger(
    requested_path: str,
    method: str,
    selected: Dict[str, Any],
    swagger_paths: Dict[str, List[str]],
) -> str:
    if not swagger_paths:
        return requested_path

    method = method.upper()

    if requested_path in swagger_paths and method in swagger_paths[requested_path]:
        return requested_path

    for path, methods in swagger_paths.items():
        if path.lower() == requested_path.lower() and method in methods:
            return path

    req_tokens = set(_singularize(t) for t in _split_path_tokens(requested_path))
    req_tokens.update(_singularize(t) for t in _split_path_tokens(selected.get("id", "")))
    for kw in selected.get("keywords", []):
        req_tokens.update(_singularize(t) for t in _split_path_tokens(str(kw)))

    requires_identifier = bool(selected.get("requiredParameters") or selected.get("routeParameters"))
    core_tokens = {
        t for t in (_singularize(x) for x in _split_path_tokens(requested_path)) if t not in {"api", "odata"}
    }
    requested_tail = requested_path.strip("/").split("/")[-1]
    requested_tail_norm = re.sub(r"[^a-zA-Z0-9]+", "", requested_tail.lower())

    best_path = requested_path
    best_score = -1
    for path, methods in swagger_paths.items():
        if method not in methods:
            continue
        path_tokens = set(_singularize(t) for t in _split_path_tokens(path))
        overlap = len(req_tokens.intersection(path_tokens))
        bonus = 0
        if requested_path.lower().startswith("/api/") and path.lower().startswith("/api/"):
            bonus += 1
        if method == "GET" and any(tok in path_tokens for tok in ["get", "all", "list"]):
            bonus += 1
        if not requires_identifier and any(tok in path_tokens for tok in ["all", "list"]):
            bonus += 2
        if not requires_identifier and ("byid" in path.lower() or "{" in path):
            bonus -= 3
        if requires_identifier and ("byid" in path.lower() or "{" in path):
            bonus += 1

        core_overlap = len(core_tokens.intersection(path_tokens))
        if core_overlap == 0:
            bonus -= 3

        path_norm = re.sub(r"[^a-zA-Z0-9]+", "", path.lower())
        if requested_tail_norm and requested_tail_norm in path_norm:
            bonus += 3

        score = overlap + (2 * core_overlap) + bonus
        if score > best_score:
            best_score = score
            best_path = path

    return best_path


# ---------- Graph Nodes ----------
def classify_question(state: AssistantState) -> AssistantState:
    question = state.get("question", "")
    q = question.lower()

    if any(word in q for word in ["stat", "top", "chiffre", "total", "vente", "ventes", "rapport"]):
        intent = "AGGREGATE"
    elif any(word in q for word in ["filtre", "ou", "where", "condition"]):
        intent = "FILTER"
    else:
        intent = "GET"

    domain = _infer_domain_from_question(question)

    return {"intent": intent, "domain": domain}


def retrieve_candidate_endpoints(state: AssistantState) -> AssistantState:
    question = state.get("question", "")
    intent = state.get("intent", "GET")
    domain = state.get("domain", "general")
    use_ollama = os.getenv("USE_OLLAMA", "0") == "1"

    endpoints = _load_endpoints()
    scored = _score_endpoints(
        endpoints=endpoints,
        question=question,
        intent=intent,
        domain=domain,
        apply_business_filter=True,
    )

    if not scored:
        scored = _score_endpoints(
            endpoints=endpoints,
            question=question,
            intent=intent,
            domain=domain,
            apply_business_filter=False,
        )

    if not scored:
        scored = _fallback_domain_endpoints(endpoints, domain)

    # Keep a simple deterministic order for the non-LLM fallback path.
    scored = _prioritize_endpoint_candidates(question, scored)

    max_candidates = 12 if use_ollama else 5
    return {"endpoint_candidates": scored[:max_candidates]}


def select_endpoint_and_params(state: AssistantState) -> AssistantState:
    candidates = state.get("endpoint_candidates", [])
    selected = candidates[0] if candidates else None
    question = state.get("question", "")
    intent = state.get("intent", "GET")
    params = _extract_simple_params(question)
    table_hint = _infer_dataset_table(question, candidates)
    requested_fields = _infer_requested_fields(question, table_hint)
    if requested_fields:
        params["requested_fields"] = requested_fields

    errors = state.get("errors", []).copy()
    if selected is None:
        errors.append("No endpoint candidate matched the question.")

    use_ollama = os.getenv("USE_OLLAMA", "0") == "1"
    router_model = os.getenv("OLLAMA_MODEL_ROUTER", "deepseek-coder:6.7b")

    llm_extracted_params, param_extract_error = _extract_params_with_llama(
        question=question,
        intent=intent,
        candidates=candidates,
        requested_fields=requested_fields,
    )
    if param_extract_error:
        errors.append(param_extract_error)
    if isinstance(llm_extracted_params, dict):
        params.update(llm_extracted_params)

    resolved_requested_fields = params.get("requested_fields", requested_fields)
    if not isinstance(resolved_requested_fields, list):
        resolved_requested_fields = requested_fields

    selected_endpoints, llm_params, router_error = _select_best_endpoints(
        candidates=candidates,
        question=question,
        intent=intent,
        use_ollama=use_ollama,
        router_model=router_model,
        extracted_params=params,
        requested_fields=resolved_requested_fields,
    )
    if router_error:
        errors.append(router_error)
    if isinstance(llm_params, dict):
        params.update(llm_params)

    if selected_endpoints:
        selected = selected_endpoints[0]

    return {
        "selected_endpoints": selected_endpoints,
        "selected_endpoint": selected,
        "extracted_params": params,
        "errors": errors,
    }


def call_webapi(state: AssistantState) -> AssistantState:
    selected = state.get("selected_endpoint")
    selected_endpoints = state.get("selected_endpoints", [])
    params = state.get("extracted_params", {})
    errors = state.get("errors", []).copy()

    if not selected_endpoints and selected:
        selected_endpoints = [selected]

    if not selected_endpoints:
        errors.append("Cannot execute WebApi: no endpoint selected.")
        payload = {
            "generatedAt": datetime.now(UTC).isoformat(),
            "endpoints": [],
            "params": params,
            "calls": [],
            "data": [],
        }
        path = CACHE_DIR / "last_api_result.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"api_result_path": str(path), "errors": errors}

    base_urls = _get_erp_api_base_urls()
    if not base_urls:
        errors.append("Missing ERP_API_BASE_URL environment variable.")
        payload = {
            "generatedAt": datetime.now(UTC).isoformat(),
            "endpoints": [ep.get("id") for ep in selected_endpoints],
            "params": params,
            "calls": [],
            "data": [],
        }
        path = CACHE_DIR / "last_api_result.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"api_result_path": str(path), "errors": errors}

    overrides = _load_endpoint_overrides()
    swagger_paths = _fetch_swagger_paths_with_methods(base_urls)
    headers: Dict[str, str] = {}
    bearer = os.getenv("ERP_API_BEARER_TOKEN", "").strip()
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"

    call_results: List[Dict[str, Any]] = []
    merged_data: List[Dict[str, Any]] = []

    for endpoint in selected_endpoints:
        endpoint_url = _build_endpoint_url(endpoint.get("url", ""), params)
        override_path = overrides.get(str(endpoint.get("id", "")))
        if override_path:
            endpoint_url = _build_endpoint_url(override_path, params)

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
        "endpoints": [ep.get("id") for ep in selected_endpoints],
        "params": params,
        "calls": call_results,
        "data": merged_data,
    }

    path = CACHE_DIR / "last_api_result.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"api_result_path": str(path), "errors": errors}


def evidence_filter(state: AssistantState) -> AssistantState:
    path_str = state.get("api_result_path", "")
    selected_endpoints = state.get("selected_endpoints", [])
    table_hint = _infer_dataset_table(state.get("question", ""), selected_endpoints)
    schema_hint = _build_schema_hint(table_hint)

    if not path_str:
        return {
            "filtered_result": {"records": [], "count": 0},
            "schema_hint": schema_hint,
            "confidence": 0.0,
        }

    path = Path(path_str)
    if (not path.exists()) or (not path.is_file()):
        return {
            "filtered_result": {"records": [], "count": 0},
            "schema_hint": schema_hint,
            "confidence": 0.0,
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("data", [])
    requested_fields = _infer_requested_fields(state.get("question", ""), table_hint)
    filtered = records[:20]

    if requested_fields:
        projected: List[Dict[str, Any]] = []
        for item in filtered:
            if isinstance(item, dict) and "record" in item:
                projected_record = _project_record_with_schema(item.get("record"), requested_fields, table_hint)
                projected.append({**item, "record": projected_record})
            else:
                projected.append(item)
        filtered = projected

    by_endpoint: Dict[str, int] = {}
    for record in filtered:
        endpoint_id = str(record.get("endpoint", "unknown"))
        by_endpoint[endpoint_id] = by_endpoint.get(endpoint_id, 0) + 1

    return {
        "filtered_result": {
            "records": filtered,
            "count": len(filtered),
            "by_endpoint": by_endpoint,
        },
        "schema_hint": _build_schema_hint(table_hint),
        "confidence": 0.6 if filtered else 0.2,
    }


def answer_generation(state: AssistantState) -> AssistantState:
    question = state.get("question", "")
    selected = state.get("selected_endpoint")
    selected_endpoints = state.get("selected_endpoints", [])
    filtered = state.get("filtered_result", {})
    schema_hint = state.get("schema_hint", {})
    compact_evidence = _build_answer_evidence(filtered)

    use_ollama = os.getenv("USE_OLLAMA", "0") == "1"
    model = os.getenv("OLLAMA_MODEL_ANSWER", "llama3.2:latest")
    records = filtered.get("records", [])

    def build_fallback_answer() -> str:
        endpoint_ids = [str(ep.get("id")) for ep in selected_endpoints] if selected_endpoints else []
        if records:
            return (
                f"J'ai trouvé {filtered.get('count', 0)} enregistrements utiles "
                f"via {', '.join(endpoint_ids) if endpoint_ids else 'les endpoints sélectionnés'}. "
                "La réponse détaillée Ollama n'est pas disponible, mais les données filtrées sont affichées."
            )
        if state.get("errors"):
            return (
                "Je n'ai pas pu produire une réponse complète. "
                "Les erreurs d'appel API ou de génération sont affichées, et vous pouvez vérifier les endpoints sélectionnés."
            )
        return (
            "Je n'ai pas trouvé de données suffisantes pour répondre à cette question avec certitude."
        )

    if use_ollama:
        system_prompt = (
            "You are an ERP support assistant. "
            "Answer only from provided evidence. "
            "Do not expose excessive internal data. "
            "Summarize only the information useful for the client. "
            "If evidence is missing, say what is missing. "
            "Use the provided schema hint to understand likely columns and business meaning before answering."
        )
        user_prompt = (
            f"Question: {question}\n"
            f"Primary endpoint: {selected}\n"
            f"Selected endpoints: {json.dumps(selected_endpoints, ensure_ascii=False)}\n"
            f"Schema hint: {json.dumps(schema_hint, ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(compact_evidence, ensure_ascii=False)}"
        )
        try:
            answer = _call_ollama_chat(model, system_prompt, user_prompt)
        except Exception as exc:
            errors = state.get("errors", []).copy()
            errors.append(f"Ollama error: {exc}")
            answer = build_fallback_answer()
            display_payload = _persist_display_result(
                question=question,
                selected_endpoints=selected_endpoints,
                filtered_result=filtered,
                answer=answer,
                errors=errors,
                schema_hint=schema_hint,
                extracted_params=state.get("extracted_params", {}),
            )
            return {
                "answer": answer,
                "errors": errors,
                "display_result_path": str(DISPLAY_RESULT_PATH),
                "display_result": display_payload,
            }
    else:
        answer = build_fallback_answer()
    display_payload = _persist_display_result(
        question=question,
        selected_endpoints=selected_endpoints,
        filtered_result=filtered,
        answer=answer,
        errors=state.get("errors", []),
        schema_hint=schema_hint,
        extracted_params=state.get("extracted_params", {}),
    )
    return {
        "answer": answer,
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

    graph.add_node("classify_question", classify_question)
    graph.add_node("retrieve_candidate_endpoints", retrieve_candidate_endpoints)
    graph.add_node("select_endpoint_and_params", select_endpoint_and_params)
    graph.add_node("call_webapi", call_webapi)
    graph.add_node("evidence_filter", evidence_filter)
    graph.add_node("answer_generation", answer_generation)
    graph.add_node("answer_validation", answer_validation)

    graph.add_edge(START, "classify_question")
    graph.add_edge("classify_question", "retrieve_candidate_endpoints")
    graph.add_edge("retrieve_candidate_endpoints", "select_endpoint_and_params")
    graph.add_edge("select_endpoint_and_params", "call_webapi")
    graph.add_edge("call_webapi", "evidence_filter")
    graph.add_edge("evidence_filter", "answer_generation")
    graph.add_edge("answer_generation", "answer_validation")
    graph.add_edge("answer_validation", END)

    return graph.compile()


def run_once(question: str) -> AssistantState:
    app = build_graph()
    result = app.invoke({"question": question, "errors": []})
    return result


class AssistantRequestHandler(BaseHTTPRequestHandler):
    server_version = "ERPAssistantHTTP/1.0"

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return

        if self.path == "/assistant/last-result":
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

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
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

        try:
            result = run_once(question)
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
    parser.add_argument("--question", default="Quels sont mes clients de Paris ?")
    args = parser.parse_args()

    if args.serve:
        serve_http(args.host, args.port)
    else:
        output = run_once(args.question)
        print(json.dumps(output, indent=2, ensure_ascii=False))
