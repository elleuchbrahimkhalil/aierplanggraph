from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import threading
import time
import uuid
from pathlib import Path
from typing import Any
import sys

import psutil
import requests
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def normalize_scalar(value: Any) -> str:
    text = str(value).strip().lower().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.strip("/")


def normalize_entity(value: Any) -> str:
    text = normalize_scalar(value)
    aliases = {
        "clients": "client",
        "fournisseurs": "fournisseur",
        "articles": "article",
        "employes": "employe",
        "employés": "employe",
        "conges": "conge",
        "congés": "conge",
        "lots_stock": "lot_stock",
        "lots": "lot",
        "bl_clients": "bl_client",
        "bl_fournisseurs": "bl_fournisseur",
        "stats_ventes": "stats_vente",
    }
    return aliases.get(text, text)


def normalize_label(label: Any) -> str:
    text = str(label).strip().lower()
    if text.startswith("entity:"):
        return f"entity:{normalize_entity(text.split(':', 1)[1])}"
    return text


def flatten_params(params: dict[str, Any], prefix: str = "param") -> set[str]:
    labels: set[str] = set()
    if not isinstance(params, dict):
        return labels

    for key, value in params.items():
        key_norm = normalize_scalar(key)
        if key_norm in {"filters", "conditions", "aggregations", "aggs", "metrics", "group_by", "groupby"}:
            continue
        if value is None:
            continue
        if isinstance(value, dict):
            labels.update(flatten_params(value, f"{prefix}:{key_norm}"))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    labels.update(flatten_params(item, f"{prefix}:{key_norm}:{index}"))
                else:
                    labels.add(f"{prefix}:{key_norm}:{normalize_scalar(item)}")
        else:
            labels.add(f"{prefix}:{key_norm}:{normalize_scalar(value)}")
    return labels


def labels_from_expected(expected: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    explicit_labels = expected.get("labels") or expected.get("expected_labels")
    if isinstance(explicit_labels, list):
        labels.update(normalize_label(label) for label in explicit_labels if str(label).strip())
    expected_params = expected.get("extracted_params", {})
    labels.update(flatten_params(expected_params))
    labels.update(structured_labels_from_params(expected_params))
    requested_fields = expected.get("requested_fields", [])
    if isinstance(requested_fields, list):
        labels.update(f"requested_field:{normalize_scalar(field)}" for field in requested_fields)
    return labels


def labels_from_prediction(payload: dict[str, Any]) -> set[str]:
    labels: set[str] = set()

    display_result = payload.get("display_result", {}) if isinstance(payload.get("display_result"), dict) else {}

    selected = payload.get("selected_endpoints", [])
    if not selected and isinstance(display_result.get("endpoints"), list):
        selected = display_result.get("endpoints", [])
    if isinstance(selected, list):
        for endpoint in selected:
            if isinstance(endpoint, dict):
                for value in (endpoint.get("url"), endpoint.get("id")):
                    if value:
                        labels.add(f"endpoint:{normalize_scalar(value)}")
            else:
                if endpoint:
                    labels.add(f"endpoint:{normalize_scalar(endpoint)}")

    extracted_params = payload.get("extracted_params", {})
    if not isinstance(extracted_params, dict) or not extracted_params:
        extracted_params = display_result.get("extractedParams", {})
    if isinstance(extracted_params, dict):
        labels.update(flatten_params(extracted_params))
        labels.update(structured_labels_from_params(extracted_params))

    analysis = (
        payload.get("analysis")
        or payload.get("request_analysis")
        or payload.get("requestAnalysis")
        or display_result.get("requestAnalysis")
        or {}
    )
    if isinstance(analysis, dict):
        labels.update(structured_labels_from_params(analysis.get("extracted_params", {})))
        entity = analysis.get("entity")
        if entity:
            labels.add(f"entity:{normalize_entity(entity)}")
        requested_fields = analysis.get("requested_fields", [])
        if isinstance(requested_fields, list):
            labels.update(f"requested_field:{normalize_scalar(field)}" for field in requested_fields)
    transform_plan = payload.get("transform_plan", {})
    if isinstance(transform_plan, dict):
        labels.update(labels_from_transform_plan(transform_plan))

    return labels


def structured_labels_from_params(params: Any) -> set[str]:
    labels: set[str] = set()
    if not isinstance(params, dict):
        return labels

    for key in ("filters", "conditions"):
        raw = params.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            field = item.get("field") or item.get("column")
            operator = item.get("operator") or item.get("op") or "equals"
            value = item.get("value")
            operator_norm = normalize_scalar(operator)
            if field and operator_norm in {"not_null", "is_null"}:
                labels.add(f"filter:{normalize_scalar(field)}:{operator_norm}:__ignored__")
            elif field and value is not None:
                labels.add(f"filter:{normalize_scalar(field)}:{operator_norm}:{normalize_scalar(value)}")

    ignored = {"requested_fields", "aggregations", "group_by", "groupby", "metrics", "aggs", "tables", "filters", "conditions"}
    for field, value in params.items():
        if field in ignored or value is None:
            continue
        if isinstance(value, dict):
            operator = value.get("operator") or value.get("op")
            if operator and "value" in value:
                labels.add(f"filter:{normalize_scalar(field)}:{normalize_scalar(operator)}:{normalize_scalar(value.get('value'))}")
            for short_op in ("gt", "gte", "lt", "lte", "contains"):
                if short_op in value:
                    labels.add(f"filter:{normalize_scalar(field)}:{short_op}:{normalize_scalar(value.get(short_op))}")
        elif not isinstance(value, (list, dict)):
            labels.add(f"filter:{normalize_scalar(field)}:equals:{normalize_scalar(value)}")

    group_by = params.get("group_by") or params.get("groupby")
    if isinstance(group_by, str):
        labels.add(f"group_by:{normalize_scalar(group_by)}")
    elif isinstance(group_by, list):
        for item in group_by:
            if isinstance(item, str):
                labels.add(f"group_by:{normalize_scalar(item)}")
            elif isinstance(item, dict):
                fields = item.get("fields", [])
                if isinstance(fields, list):
                    labels.update(f"group_by:{normalize_scalar(field)}" for field in fields)

    aggregations = params.get("aggregations") or params.get("aggs") or params.get("metrics")
    if isinstance(aggregations, list):
        for agg in aggregations:
            if isinstance(agg, dict) and agg.get("field") and agg.get("agg"):
                labels.add(f"agg:{normalize_scalar(agg.get('field'))}:{normalize_scalar(agg.get('agg'))}")
    elif isinstance(aggregations, dict):
        for field, agg in aggregations.items():
            if isinstance(agg, str):
                labels.add(f"agg:{normalize_scalar(field)}:{normalize_scalar(agg)}")
    return labels


def labels_from_transform_plan(plan: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        return labels

    for step in steps:
        if not isinstance(step, dict):
            continue
        op = normalize_scalar(step.get("op", ""))
        if not op:
            continue
        labels.add(f"operation:{op}")
        if op == "filter_rows":
            for condition in step.get("conditions", []):
                if not isinstance(condition, dict):
                    continue
                field = condition.get("field") or condition.get("column")
                operator = condition.get("operator") or condition.get("op") or "equals"
                value = condition.get("value")
                operator_norm = normalize_scalar(operator)
                if field and operator_norm in {"not_null", "is_null"}:
                    labels.add(f"filter:{normalize_scalar(field)}:{operator_norm}:__ignored__")
                elif field and value is not None:
                    labels.add(f"filter:{normalize_scalar(field)}:{operator_norm}:{normalize_scalar(value)}")
        elif op == "select":
            columns = step.get("columns", [])
            if isinstance(columns, list):
                labels.update(f"requested_field:{normalize_scalar(column)}" for column in columns)
        elif op == "aggregate":
            groupby = step.get("groupby", [])
            if isinstance(groupby, str):
                groupby = [groupby]
            if isinstance(groupby, list):
                for field in groupby:
                    labels.add(f"group_by:{normalize_scalar(field)}")
            aggs = step.get("aggs", [])
            if isinstance(aggs, list):
                for agg in aggs:
                    if not isinstance(agg, dict):
                        continue
                    field = agg.get("field")
                    func = agg.get("agg")
                    if field and func:
                        labels.add(f"agg:{normalize_scalar(field)}:{normalize_scalar(func)}")
    return labels


def set_metrics(expected: set[str], predicted: set[str]) -> dict[str, float]:
    if not expected and not predicted:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "exact_match": 1.0,
            "hallucination_rate": 0.0,
            "hallucination_count": 0.0,
        }

    true_positive = len(expected & predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    hallucinations = predicted - expected
    hallucination_rate = (len(hallucinations) / len(predicted)) if predicted else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "exact_match": 1.0 if expected == predicted else 0.0,
        "hallucination_rate": float(hallucination_rate),
        "hallucination_count": float(len(hallucinations)),
    }


def monitor_resources(stop_event: threading.Event, samples: list[dict[str, float]], interval: float, local_ram_gb: float) -> None:
    psutil.cpu_percent(interval=None)
    local_ram_mb = local_ram_gb * 1024
    while not stop_event.is_set():
        cpu = psutil.cpu_percent(interval=interval)
        memory = psutil.virtual_memory()
        ram_used_mb = float(memory.used / (1024 * 1024))
        samples.append(
            {
                "cpu_percent": float(cpu),
                "ram_percent": float(memory.percent),
                "ram_used_mb": ram_used_mb,
                "ram_percent_of_configured_local": (ram_used_mb / local_ram_mb * 100) if local_ram_mb else 0.0,
            }
        )


def summarize_samples(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {
            "cpu_avg": 0.0,
            "cpu_max": 0.0,
            "ram_avg_percent": 0.0,
            "ram_max_percent": 0.0,
            "ram_max_mb": 0.0,
            "ram_avg_percent_of_16gb": 0.0,
            "ram_max_percent_of_16gb": 0.0,
        }
    cpu_values = [sample["cpu_percent"] for sample in samples]
    ram_percent_values = [sample["ram_percent"] for sample in samples]
    ram_mb_values = [sample["ram_used_mb"] for sample in samples]
    ram_configured_values = [sample["ram_percent_of_configured_local"] for sample in samples]
    return {
        "cpu_avg": statistics.mean(cpu_values),
        "cpu_max": max(cpu_values),
        "ram_avg_percent": statistics.mean(ram_percent_values),
        "ram_max_percent": max(ram_percent_values),
        "ram_max_mb": max(ram_mb_values),
        "ram_avg_percent_of_16gb": statistics.mean(ram_configured_values),
        "ram_max_percent_of_16gb": max(ram_configured_values),
    }


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("evaluation_cases", payload if isinstance(payload, list) else [])
    if not isinstance(cases, list):
        raise ValueError("Dataset must be a list or contain an evaluation_cases list.")
    label_prefixes = payload.get("label_prefixes", []) if isinstance(payload, dict) else []
    if not isinstance(label_prefixes, list):
        label_prefixes = []
    return (
        [case for case in cases if isinstance(case, dict) and case.get("prompt")],
        [str(prefix) for prefix in label_prefixes],
    )


def filter_label_scope(labels: set[str], prefixes: list[str]) -> set[str]:
    if not prefixes:
        return labels
    return {label for label in labels if any(label.startswith(prefix) for prefix in prefixes)}


def evaluate_case(
    case: dict[str, Any],
    assistant_url: str,
    timeout: float,
    sample_interval: float,
    local_ram_gb: float,
    mode: str,
    label_prefixes: list[str],
) -> dict[str, Any]:
    prompt = str(case["prompt"])
    expected = case.get("expected", {}) if isinstance(case.get("expected"), dict) else {}
    expected_labels = filter_label_scope(labels_from_expected(expected), label_prefixes)

    samples: list[dict[str, float]] = []
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=monitor_resources,
        args=(stop_event, samples, sample_interval, local_ram_gb),
        daemon=True,
    )

    started = time.perf_counter()
    monitor.start()
    status_code = 0
    error = ""
    payload: dict[str, Any] = {}
    try:
        if mode == "langgraph-llm":
            payload = evaluate_langgraph_llm(prompt)
            status_code = 200
            graph_errors = payload.get("errors", [])
            error = " | ".join(str(item) for item in graph_errors) if graph_errors else ""
        else:
            request_timeout = None if timeout <= 0 else timeout
            response = requests.post(
                f"{assistant_url.rstrip('/')}/assistant/query",
                json={"question": prompt, "thread_id": f"eval-{uuid.uuid4()}"},
                timeout=request_timeout,
            )
            status_code = response.status_code
            payload = response.json() if response.content else {}
            if not response.ok:
                error = payload.get("details") or payload.get("error") or response.text[:300]
    except Exception as exc:
        error = str(exc)
    finally:
        elapsed = time.perf_counter() - started
        stop_event.set()
        monitor.join(timeout=sample_interval * 3)

    predicted_labels = labels_from_prediction(payload) if isinstance(payload, dict) else set()
    predicted_labels = filter_label_scope(predicted_labels, label_prefixes)
    metrics = set_metrics(expected_labels, predicted_labels)
    resources = summarize_samples(samples)

    return {
        "prompt": prompt,
        "status_code": status_code,
        "error": error,
        "response_time_sec": elapsed,
        **resources,
        **metrics,
        "expected_labels": sorted(expected_labels),
        "predicted_labels": sorted(predicted_labels),
    }


def evaluate_langgraph_llm(prompt: str) -> dict[str, Any]:
    """Evaluate the same LangGraph LLM extraction/routing nodes without WebApi calls.

    This avoids timeouts from /assistant/query when the backend API or final answer
    generation is slow, while still using the LLM prompts/functions from
    ai_assistant.langgraph_skeleton.
    """
    from ai_assistant import langgraph_skeleton as graph

    state: dict[str, Any] = {"question": prompt, "errors": [], "history": []}
    state.update(graph.extract_user_request(state))
    state.update(graph.retrieve_candidate_endpoints(state))
    state.update(graph.route_endpoint(state))
    transform_plan = build_local_plan_from_extraction(graph, state)
    return {
        "request_analysis": state.get("request_analysis", {}),
        "extracted_params": state.get("extracted_params", {}),
        "selected_endpoints": state.get("selected_endpoints", []),
        "transform_plan": transform_plan,
        "errors": state.get("errors", []),
    }


def build_local_plan_from_extraction(graph: Any, state: dict[str, Any]) -> dict[str, Any]:
    analysis = state.get("request_analysis", {})
    extracted_params = state.get("extracted_params", {})
    steps: list[dict[str, Any]] = []

    filter_conditions = graph._extract_filter_conditions_from_extracted_params(extracted_params)
    if filter_conditions:
        steps.append({"op": "filter_rows", "conditions": filter_conditions})

    aggs = extracted_params.get("aggregations") or extracted_params.get("aggs") or extracted_params.get("metrics")
    group_by = extracted_params.get("group_by") or extracted_params.get("groupby")
    if aggs:
        normalized_aggs = []
        if isinstance(aggs, list):
            for item in aggs:
                if isinstance(item, dict) and item.get("field"):
                    normalized_aggs.append(
                        {
                            "field": item.get("field"),
                            "agg": str(item.get("agg", item.get("function", "sum"))).lower(),
                            "as": item.get("as"),
                        }
                    )
        elif isinstance(aggs, dict):
            for field, agg in aggs.items():
                normalized_aggs.append({"field": field, "agg": str(agg).lower(), "as": None})
        if normalized_aggs:
            steps.append(
                {
                    "op": "aggregate",
                    "groupby": group_by if isinstance(group_by, list) else ([group_by] if group_by else []),
                    "aggs": normalized_aggs,
                }
            )

    requested_fields = analysis.get("requested_fields", [])
    if isinstance(requested_fields, list) and requested_fields:
        steps.append({"op": "select", "columns": requested_fields})

    return {"steps": steps}


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    times = [row["response_time_sec"] for row in rows]
    successful_rows = [row for row in rows if not row["error"]]
    successful_times = [row["response_time_sec"] for row in successful_rows]
    return {
        "count": len(rows),
        "successful_count": len(successful_rows),
        "failed_count": len(rows) - len(successful_rows),
        "precision": statistics.mean(row["precision"] for row in rows) if rows else 0.0,
        "recall": statistics.mean(row["recall"] for row in rows) if rows else 0.0,
        "f1": statistics.mean(row["f1"] for row in rows) if rows else 0.0,
        "avg_case_precision": statistics.mean(row["precision"] for row in rows) if rows else 0.0,
        "avg_case_recall": statistics.mean(row["recall"] for row in rows) if rows else 0.0,
        "avg_case_f1": statistics.mean(row["f1"] for row in rows) if rows else 0.0,
        "avg_successful_case_f1": statistics.mean(row["f1"] for row in successful_rows) if successful_rows else 0.0,
        "exact_match_rate": statistics.mean(row["exact_match"] for row in rows) if rows else 0.0,
        "avg_response_time_sec": statistics.mean(times) if times else 0.0,
        "avg_successful_response_time_sec": statistics.mean(successful_times) if successful_times else 0.0,
        "max_response_time_sec": max(times) if times else 0.0,
        "avg_hallucination_rate": statistics.mean(row["hallucination_rate"] for row in rows) if rows else 0.0,
        "avg_hallucination_count": statistics.mean(row["hallucination_count"] for row in rows) if rows else 0.0,
        "avg_cpu_percent": statistics.mean(row["cpu_avg"] for row in rows) if rows else 0.0,
        "max_cpu_percent": max((row["cpu_max"] for row in rows), default=0.0),
        "avg_ram_percent": statistics.mean(row["ram_avg_percent"] for row in rows) if rows else 0.0,
        "max_ram_percent": max((row["ram_max_percent"] for row in rows), default=0.0),
        "avg_ram_percent_of_16gb": statistics.mean(row["ram_avg_percent_of_16gb"] for row in rows) if rows else 0.0,
        "max_ram_percent_of_16gb": max((row["ram_max_percent_of_16gb"] for row in rows), default=0.0),
        "error_rate": statistics.mean(1.0 if row["error"] else 0.0 for row in rows) if rows else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt",
        "status_code",
        "error",
        "precision",
        "recall",
        "f1",
        "exact_match",
        "hallucination_rate",
        "hallucination_count",
        "response_time_sec",
        "cpu_avg",
        "cpu_max",
        "ram_avg_percent",
        "ram_max_percent",
        "ram_max_mb",
        "ram_avg_percent_of_16gb",
        "ram_max_percent_of_16gb",
        "expected_labels",
        "predicted_labels",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["expected_labels"] = json.dumps(output["expected_labels"], ensure_ascii=False)
            output["predicted_labels"] = json.dumps(output["predicted_labels"], ensure_ascii=False)
            writer.writerow({key: output.get(key, "") for key in fieldnames})


def write_csv_with_summary(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_csv(path, rows)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([])
        writer.writerow([
            "SUMMARY",
            "",
            "",
            summary.get("avg_case_precision", 0.0),
            summary.get("avg_case_recall", 0.0),
            summary.get("avg_case_f1", 0.0),
            summary.get("exact_match_rate", 0.0),
            summary.get("avg_hallucination_rate", 0.0),
            summary.get("avg_hallucination_count", 0.0),
            summary.get("avg_response_time_sec", 0.0),
            summary.get("avg_cpu_percent", 0.0),
            summary.get("max_cpu_percent", 0.0),
            summary.get("avg_ram_percent", 0.0),
            summary.get("max_ram_percent", 0.0),
            summary.get("max_ram_percent", 0.0),
            summary.get("avg_ram_percent_of_16gb", 0.0),
            summary.get("max_ram_percent_of_16gb", 0.0),
            "",
            "",
        ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ERP assistant with F1, latency, CPU and RAM metrics.")
    parser.add_argument("--dataset", required=True, help="Path to api_test_evaluation.json.")
    parser.add_argument("--assistant-url", default="http://127.0.0.1:8000", help="Assistant base URL.")
    parser.add_argument("--output-dir", default="ai_assistant/data/eval", help="Directory for CSV/JSON reports.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout per prompt in seconds. Use 0 to disable the HTTP read timeout in full mode.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional number of cases to evaluate.")
    parser.add_argument("--sample-interval", type=float, default=0.25, help="CPU/RAM sample interval in seconds.")
    parser.add_argument("--local-ram-gb", type=float, default=16.0, help="Configured local RAM size used for RAM percentage reporting.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Ollama temperature used during evaluation.")
    parser.add_argument("--top-p", type=float, default=0.1, help="Ollama top_p used during evaluation.")
    parser.add_argument("--max-gen-len", type=int, default=256, help="Ollama num_predict/max generation length used during evaluation.")
    parser.add_argument(
        "--mode",
        choices=["full", "langgraph-llm"],
        default="full",
        help="full calls /assistant/query. langgraph-llm evaluates extraction/routing nodes directly without WebApi calls.",
    )
    args = parser.parse_args()
    os.environ["OLLAMA_TEMPERATURE"] = str(args.temperature)
    os.environ["OLLAMA_TOP_P"] = str(args.top_p)
    os.environ["OLLAMA_NUM_PREDICT"] = str(args.max_gen_len)

    cases, label_prefixes = load_dataset(Path(args.dataset))
    if args.limit > 0:
        cases = cases[: args.limit]

    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['prompt']}")
        rows.append(
            evaluate_case(
                case,
                args.assistant_url,
                args.timeout,
                args.sample_interval,
                args.local_ram_gb,
                args.mode,
                label_prefixes,
            )
        )

    summary = aggregate_results(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_with_summary(output_dir / "llm_eval_results.csv", rows, summary)
    (output_dir / "llm_eval_summary.json").write_text(
        json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV: {output_dir / 'llm_eval_results.csv'}")
    print(f"JSON: {output_dir / 'llm_eval_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
