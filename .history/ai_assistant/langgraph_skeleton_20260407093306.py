from __future__ import annotations

import json
import os
import re
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

    return {
        "selected_endpoint": selected,
        "extracted_params": params,
        "errors": errors,
    }


def call_webapi_stub(state: AssistantState) -> AssistantState:
    selected = state.get("selected_endpoint")
    params = state.get("extracted_params", {})

    # Point 1 only: this is a skeleton stub. Real HTTP call comes in point 3.
    simulated_payload = {
        "generatedAt": datetime.now(UTC).isoformat(),
        "endpoint": selected.get("id") if selected else None,
        "url": selected.get("url") if selected else None,
        "params": params,
        "data": [
            {
                "sample": True,
                "message": "Replace this with real WebApi response in next step.",
            }
        ],
    }

    path = CACHE_DIR / "last_api_result.json"
    path.write_text(json.dumps(simulated_payload, indent=2), encoding="utf-8")

    return {"api_result_path": str(path)}


def evidence_filter(state: AssistantState) -> AssistantState:
    path = Path(state.get("api_result_path", ""))
    if not path.exists():
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
    model = os.getenv("OLLAMA_MODEL_ANSWER", "qwen2.5:7b-instruct")

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
    graph.add_node("call_webapi_stub", call_webapi_stub)
    graph.add_node("evidence_filter", evidence_filter)
    graph.add_node("answer_generation", answer_generation)
    graph.add_node("answer_validation", answer_validation)

    graph.add_edge(START, "classify_question")
    graph.add_edge("classify_question", "retrieve_candidate_endpoints")
    graph.add_edge("retrieve_candidate_endpoints", "select_endpoint_and_params")
    graph.add_edge("select_endpoint_and_params", "call_webapi_stub")
    graph.add_edge("call_webapi_stub", "evidence_filter")
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
