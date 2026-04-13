from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
SWAGGER_PATH = ROOT_DIR / "swagger_live.json"
OUTPUT_PATH = ROOT_DIR / "data" / "endpoints.get.json"


def split_tokens(text: str) -> list[str]:
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [token for token in re.split(r"[^a-zA-Z0-9]+", expanded.lower()) if token]


def infer_role(text: str) -> str:
    tokens = set(split_tokens(text))
    if tokens.intersection({"client", "clients", "commande", "commandes", "blclient", "statsvente", "fact", "vente", "report"}):
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


def infer_intent(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ["stat", "report", "vente", "chiffre", "dashboard", "top", "total"]):
        return "AGGREGATE"
    if any(term in lowered for term in ["filter", "filtre", "search", "find"]):
        return "FILTER"
    return "GET"


def build_keywords(text: str) -> list[str]:
    stop_words = {"api", "odata", "get", "all", "by", "id", "v1", "swagger"}
    keywords: list[str] = []
    for token in split_tokens(text):
        if token in stop_words or token in keywords:
            continue
        keywords.append(token)
    return keywords[:16]


def normalize_label(path: str, tags: list[str]) -> str:
    tokens = split_tokens(" ".join(tags) if tags else path)
    token_set = set(tokens)
    if {"stats", "vente"}.issubset(token_set) or "statsvente" in token_set:
        return "statistiques de ventes"
    if {"commande", "client", "report"}.issubset(token_set):
        return "rapports des commandes client"

    mapped = {
        "blclient": "bons de livraison client",
        "client": "clients",
        "clients": "clients",
        "commande": "commandes client",
        "commandeclient": "commandes client",
        "article": "articles",
        "articles": "articles",
        "stock": "stock",
        "paiement": "paiements",
        "paiements": "paiements",
        "fournisseur": "fournisseurs",
        "fournisseurs": "fournisseurs",
        "statsvente": "statistiques de ventes",
        "report": "rapports",
        "reports": "rapports",
        "agencebancaire": "agences bancaires",
        "agence": "agences",
        "employe": "employés",
        "employes": "employés",
        "conge": "congés",
        "historiques": "historiques",
        "translation": "traductions",
    }
    for token in tokens:
        if token in mapped:
            return mapped[token]
    return "données"


def build_description(path: str, intent: str, tags: list[str], query_parameters: list[str], route_parameters: list[str]) -> str:
    label = normalize_label(path, tags)
    if intent == "AGGREGATE":
        base = f"Retourne un rapport ou des statistiques sur {label}."
    elif route_parameters or query_parameters:
        base = f"Retourne des informations sur {label} avec des critères de recherche."
    else:
        base = f"Retourne la liste des {label}."

    details: list[str] = []
    if query_parameters:
        details.append(f"Paramètres de requête disponibles: {', '.join(query_parameters[:6])}.")
    if route_parameters:
        details.append(f"Paramètres d'URL requis: {', '.join(route_parameters[:6])}.")
    return " ".join([base, *details]).strip()


def build_examples(path: str, tags: list[str], query_parameters: list[str]) -> list[str]:
    text = " ".join([path, " ".join(tags)]).lower()
    if "getall" not in text:
        return []

    if "commande_client-report" in text or ("report" in text and "commande" in text and "client" in text):
        return [
            "afficher le rapport des commandes client",
            "afficher le rapport d'une commande client",
            "afficher le détail d'une commande client",
            "afficher le rapport des ventes client",
            "afficher le document d'une commande client",
        ]
    if "client" in text:
        return [
            "afficher les clients",
            "afficher le nom de client",
            "afficher l'email de client",
            "afficher le code de client",
            "afficher le téléphone de client",
        ]
    if "statsvente" in text or "vente" in text:
        return [
            "afficher les ventes",
            "afficher les statistiques de vente",
            "afficher le chiffre d'affaires",
            "afficher les ventes par article",
            "afficher les ventes entre deux dates",
        ]
    if "stock" in text:
        return [
            "afficher le stock",
            "afficher le stock par dépôt",
            "afficher le stock des articles",
            "afficher la quantité disponible",
            "afficher le stock actuel",
        ]
    if "paiement" in text:
        return [
            "afficher les paiements",
            "afficher les paiements client",
            "afficher les règlements",
            "afficher les factures payées",
            "afficher les paiements par client",
        ]
    if "fournisseur" in text or "frs" in text:
        return [
            "afficher les fournisseurs",
            "afficher le nom de fournisseur",
            "afficher les factures fournisseurs",
            "afficher les paiements fournisseurs",
            "afficher les fournisseurs actifs",
        ]
    if "commande" in text:
        return [
            "afficher les commandes client",
            "afficher les commandes",
            "afficher les commandes d'un client",
            "afficher le détail des commandes",
            "afficher les commandes en cours",
        ]
    if "article" in text:
        return [
            "afficher les articles",
            "afficher le code article",
            "afficher le prix des articles",
            "afficher les articles disponibles",
            "afficher les détails des articles",
        ]
    if "agence" in text:
        return [
            "afficher les agences",
            "afficher les agences bancaires",
            "afficher le code agence",
            "afficher les détails des agences",
            "afficher une agence par code",
        ]

    generic = [
        "afficher les données",
        "afficher la liste complète",
        "afficher les informations disponibles",
    ]
    if query_parameters:
        generic.append(f"afficher avec le filtre {query_parameters[0]}")
    return generic


def build_endpoint(path: str, operation: dict[str, Any]) -> dict[str, Any]:
    tags = [str(tag) for tag in operation.get("tags", []) if tag]
    summary = str(operation.get("summary", ""))
    description = str(operation.get("description", ""))
    operation_id = str(operation.get("operationId", ""))
    parameters = operation.get("parameters", [])

    route_parameters = [
        str(param.get("name"))
        for param in parameters
        if isinstance(param, dict) and param.get("in") == "path" and param.get("name")
    ]
    query_parameters = [
        str(param.get("name"))
        for param in parameters
        if isinstance(param, dict) and param.get("in") == "query" and param.get("name")
    ]

    source_text = " ".join([path, summary, description, operation_id, " ".join(tags), " ".join(query_parameters)])
    endpoint_id = "webapi_get_" + "_".join(split_tokens(path))
    role_text = " ".join([path, summary, description, operation_id, " ".join(tags)])
    intent = infer_intent(role_text)

    return {
        "id": endpoint_id,
        "method": "GET",
        "url": path,
        "intent": intent,
        "keywords": build_keywords(source_text),
        "routeParameters": route_parameters,
        "queryParameters": query_parameters,
        "requiredParameters": route_parameters,
        "responseFormat": "Object",
        "description": build_description(path, intent, tags, query_parameters, route_parameters),
        "examples": build_examples(path, tags, query_parameters),
        "role": infer_role(role_text),
        "tags": tags,
        "operationId": operation_id,
    }


def main() -> None:
    payload = json.loads(SWAGGER_PATH.read_text(encoding="utf-8-sig"))
    paths = payload.get("paths", {})

    endpoints: list[dict[str, Any]] = []
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        operation = methods.get("get")
        if not isinstance(operation, dict):
            continue
        endpoints.append(build_endpoint(path, operation))

    endpoints.sort(key=lambda item: item["url"])
    OUTPUT_PATH.write_text(json.dumps({"endpoints": endpoints}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(endpoints)} GET endpoints to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
