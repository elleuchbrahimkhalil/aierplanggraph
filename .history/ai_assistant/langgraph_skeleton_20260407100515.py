from __future__ import annotations

import json
import os
import re
from urllib.parse import urlencode
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import requests
from langgraph.graph import END, START, StateGraph


ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class AssistantState(TypedDict, total=False):
    question: str
    intent: str
    endpoint_candidates: List[Dict[str, Any]]
    selected_endpoint: Optional[Dict[str, Any]]
    extracted_params: Dict[str, Any]
    api_result_path: str
    filtered_result: Dict[str, Any]
    answer: str
    confidence: float
    errors: List[str]


# ---------- Utilities ----------
def _load_endpoints() -> List[Dict[str, Any]]:
    configured = os.getenv("ERP_ENDPOINTS_JSON", "").strip()
    if configured:
        path = Path(configured)
    else:
        path = ROOT_DIR / "data" / "endpoints.sample.json"

    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("endpoints", [])


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _extract_simple_params(question: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    id_match = re.search(r"\b(?:id|client|commande|article|employe)\s*(\d+)\b", question, re.IGNORECASE)
    if id_match:
        out["id"] = int(id_match.group(1))

    date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", question)
    if date_match:
        out["date"] = date_match.group(1)

    return out


def _call_ollama_chat(model: str, system_prompt: str, user_prompt: str) -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    return content.strip()


def _call_ollama_json(model: str, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
    raw = _call_ollama_chat(model, system_prompt, user_prompt)
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
        if route_key in params:
            rendered = rendered.replace(f"{{{route_key}}}", str(params[route_key]))
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
        score = overlap + bonus
        if score > best_score:
            best_score = score
            best_path = path

    return best_path


# ---------- Graph Nodes ----------
def classify_question(state: AssistantState) -> AssistantState:
    question = state.get("question", "")
    q = question.lower()

    if any(word in q for word in ["stat", "top", "chiffre", "total"]):
        intent = "AGGREGATE"
    elif any(word in q for word in ["filtre", "ou", "where", "condition"]):
        intent = "FILTER"
    else:
        intent = "GET"

    return {"intent": intent}


def retrieve_candidate_endpoints(state: AssistantState) -> AssistantState:
    question = state.get("question", "")
    q_tokens = set(_tokenize(question))
    intent = state.get("intent", "GET")

    endpoints = _load_endpoints()
    scored: List[Dict[str, Any]] = []

    for ep in endpoints:
        ep_tokens = set(_tokenize(" ".join(ep.get("keywords", []))))
        overlap = len(q_tokens.intersection(ep_tokens))
        intent_bonus = 2 if ep.get("intent") == intent else 0
        score = overlap + intent_bonus
        if score > 0:
            scored.append({"score": score, **ep})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"endpoint_candidates": scored[:5]}


def select_endpoint_and_params(state: AssistantState) -> AssistantState:
    candidates = state.get("endpoint_candidates", [])
    selected = candidates[0] if candidates else None
    params = _extract_simple_params(state.get("question", ""))

    errors = state.get("errors", []).copy()
    if selected is None:
        errors.append("No endpoint candidate matched the question.")

    use_ollama = os.getenv("USE_OLLAMA", "0") == "1"
    router_model = os.getenv("OLLAMA_MODEL_ROUTER", "deepseek-coder:6.7b")
    if use_ollama and candidates:
        system_prompt = (
            "You are an endpoint router for an ERP API. "
            "Return strict JSON only with fields: endpoint_id, extracted_params."
        )
        compact_candidates = [
            {
                "id": c.get("id"),
                "url": c.get("url"),
                "intent": c.get("intent"),
                "requiredParameters": c.get("requiredParameters", []),
                "queryParameters": c.get("queryParameters", []),
            }
            for c in candidates[:5]
        ]
        user_prompt = (
            f"Question: {state.get('question', '')}\n"
            f"Candidates: {json.dumps(compact_candidates, ensure_ascii=False)}\n"
            "Pick one endpoint_id from candidates and extract parameters from question."
        )
        llm_choice = _call_ollama_json(router_model, system_prompt, user_prompt)
        if llm_choice:
            endpoint_id = llm_choice.get("endpoint_id")
            llm_params = llm_choice.get("extracted_params", {})
            chosen = next((c for c in candidates if c.get("id") == endpoint_id), None)
            if chosen:
                selected = chosen
            if isinstance(llm_params, dict):
                params.update(llm_params)

    return {
        "selected_endpoint": selected,
        "extracted_params": params,
        "errors": errors,
    }


def call_webapi(state: AssistantState) -> AssistantState:
    selected = state.get("selected_endpoint")
    params = state.get("extracted_params", {})
    errors = state.get("errors", []).copy()

    if not selected:
        errors.append("Cannot execute WebApi: no endpoint selected.")
        payload = {
            "generatedAt": datetime.now(UTC).isoformat(),
            "endpoint": None,
            "url": None,
            "fullUrl": None,
            "params": params,
            "query": {},
            "statusCode": None,
            "data": [],
            "raw": None,
        }
        path = CACHE_DIR / "last_api_result.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"api_result_path": str(path), "errors": errors}

    base_urls = _get_erp_api_base_urls()
    if not base_urls:
        errors.append("Missing ERP_API_BASE_URL environment variable.")
        payload = {
            "generatedAt": datetime.now(UTC).isoformat(),
            "endpoint": selected.get("id"),
            "url": selected.get("url"),
            "fullUrl": None,
            "params": params,
            "query": {},
            "statusCode": None,
            "data": [],
            "raw": None,
        }
        path = CACHE_DIR / "last_api_result.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"api_result_path": str(path), "errors": errors}

    endpoint_url = _build_endpoint_url(selected.get("url", ""), params)
    request_parts = _collect_request_parts(selected, params)
    missing_required = request_parts["missing_required"]
    query_params = request_parts["query_params"]

    if missing_required:
        errors.append(
            f"Missing required parameters for endpoint {selected.get('id')}: {', '.join(missing_required)}"
        )
        payload = {
            "generatedAt": datetime.now(UTC).isoformat(),
            "endpoint": selected.get("id"),
            "url": selected.get("url"),
            "fullUrl": None,
            "params": params,
            "query": query_params,
            "statusCode": None,
            "data": [],
            "raw": None,
        }
        path = CACHE_DIR / "last_api_result.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"api_result_path": str(path), "errors": errors}

    method = selected.get("method", "GET").upper()
    swagger_paths = _fetch_swagger_paths_with_methods(base_urls)
    resolved_path = _resolve_endpoint_path_from_swagger(
        requested_path=endpoint_url,
        method=method,
        selected=selected,
        swagger_paths=swagger_paths,
    )

    candidate_urls: List[str] = []
    for base_url in base_urls:
        full_url = f"{base_url}{resolved_path}"
        if query_params:
            full_url = f"{full_url}?{urlencode(query_params, doseq=True)}"
        candidate_urls.append(full_url)

    headers: Dict[str, str] = {}
    bearer = os.getenv("ERP_API_BEARER_TOKEN", "").strip()
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"

    payload: Dict[str, Any] = {
        "generatedAt": datetime.now(UTC).isoformat(),
        "endpoint": selected.get("id"),
        "url": selected.get("url"),
        "resolvedUrl": resolved_path,
        "fullUrl": None,
        "params": params,
        "query": query_params,
        "statusCode": None,
        "data": [],
        "raw": None,
    }
    last_exc: Optional[Exception] = None
    for full_url in candidate_urls:
        try:
            response = requests.get(full_url, headers=headers, timeout=60)
            response.raise_for_status()
            body = response.json()
            data = _normalize_data_field(body)
            payload = {
                "generatedAt": datetime.now(UTC).isoformat(),
                "endpoint": selected.get("id"),
                "url": selected.get("url"),
                "resolvedUrl": resolved_path,
                "fullUrl": full_url,
                "params": params,
                "query": query_params,
                "statusCode": response.status_code,
                "data": data,
                "raw": body,
            }
            last_exc = None
            break
        except requests.RequestException as exc:
            last_exc = exc

    if last_exc is not None:
        errors.append(f"WebApi call failed for {selected.get('id')}: {last_exc}")

    path = CACHE_DIR / "last_api_result.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"api_result_path": str(path), "errors": errors}


def evidence_filter(state: AssistantState) -> AssistantState:
    path_str = state.get("api_result_path", "")
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
    filtered = records[:20]

    return {
        "filtered_result": {"records": filtered, "count": len(filtered)},
        "confidence": 0.6 if filtered else 0.2,
    }


def answer_generation(state: AssistantState) -> AssistantState:
    question = state.get("question", "")
    selected = state.get("selected_endpoint")
    filtered = state.get("filtered_result", {})

    use_ollama = os.getenv("USE_OLLAMA", "0") == "1"
    model = os.getenv("OLLAMA_MODEL_ANSWER", "llama3.1:8b")

    if use_ollama:
        system_prompt = (
            "You are an ERP support assistant. "
            "Answer only from provided evidence. "
            "If evidence is missing, say what is missing."
        )
        user_prompt = (
            f"Question: {question}\n"
            f"Endpoint selected: {selected}\n"
            f"Evidence: {json.dumps(filtered, ensure_ascii=False)}"
        )
        try:
            answer = _call_ollama_chat(model, system_prompt, user_prompt)
        except Exception as exc:
            errors = state.get("errors", []).copy()
            errors.append(f"Ollama error: {exc}")
            answer = "I could not generate the final answer with Ollama."
            return {"answer": answer, "errors": errors}
    else:
        endpoint_id = selected.get("id") if selected else "none"
        answer = (
            "Skeleton response. "
            f"Question received: '{question}'. "
            f"Selected endpoint: '{endpoint_id}'. "
            f"Records considered: {filtered.get('count', 0)}."
        )

    return {"answer": answer}


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


if __name__ == "__main__":
    sample_question = "Quels sont mes clients de Paris ?"
    output = run_once(sample_question)
    print(json.dumps(output, indent=2, ensure_ascii=False))
