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
    if tokens.intersection({"stock", "depot", "lot", "article", "articles", "bonentree", "bontransfert", "ordrecoupe"}):
        return "stock"
    if tokens.intersection({"paiement", "paiements", "depense", "depenses", "finance", "transfertargent"}):
        return "finance"
    if tokens.intersection({"demandeconge", "conge", "paie", "employe", "employes", "pointage", "fiche"}):
        return "rh"
    if tokens.intersection({"fournisseur", "fournisseurs", "blfrs", "frs"}):
        return "achat"
    return "general"


def infer_intent(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ["stat", "report", "vente", "chiffre", "dashboard", "top", "total", "count"]):
        return "AGGREGATE"
    if any(term in lowered for term in ["filter", "filtre", "search", "find", "getby", "byid", "bycode", "/{", "getbyid"]):
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


# Comprehensive label mapping keyed by lowercase tag name (camel-case merged or split token).
_TAG_LABEL_MAP: dict[str, str] = {
    # Commercial
    "blclient": "bons de livraison client",
    "blclientodata": "bons de livraison client",
    "blclientfilterarticleodata": "articles des bons de livraison client",
    "client": "clients",
    "clients": "clients",
    "clientodata": "clients",
    "clientsodata": "clients",
    "commandeclient": "commandes client",
    "commandeclientodata": "commandes client",
    "commandeclientarticletodata": "articles des commandes client",
    "commandearticleodata": "articles des commandes",
    "statsvente": "statistiques de ventes",
    "report": "rapports",
    "reports": "rapports",
    # Stock / Articles
    "article": "articles",
    "articles": "articles",
    "articleodata": "articles",
    "detailsarticleodata": "détails des articles",
    "stock": "stocks",
    "stockodata": "stocks",
    "stockdepot": "stocks par dépôt",
    "stockdepotodata": "stocks par dépôt",
    "stocklotodata": "stocks par lot",
    "bontransfert": "bons de transfert",
    "bontransfertodata": "bons de transfert",
    "bontransfertdemande": "demandes de transfert",
    "bontransfertdemandeodata": "demandes de transfert",
    "bonentree": "bons d'entrée en stock",
    "lot": "lots",
    "lots": "lots",
    "ordrecoupe": "ordres de coupe",
    # Finance / Paiements
    "paiement": "paiements",
    "paiements": "paiements",
    "paiementclientodata": "paiements clients",
    "paiementfrsodata": "paiements fournisseurs",
    "depense": "dépenses",
    "depenseodata": "dépenses",
    "depenses": "dépenses",
    "transfertargent": "transferts d'argent",
    "transfertargentodata": "transferts d'argent",
    # Achats
    "fournisseur": "fournisseurs",
    "fournisseurs": "fournisseurs",
    "frsodata": "fournisseurs",
    "blfrs": "bons de livraison fournisseurs",
    "blfrsarticleodata": "articles des bons de livraison fournisseurs",
    "blfrsodata": "bons de livraison fournisseurs",
    "frs": "fournisseurs",
    # RH / Paie
    "employe": "employés",
    "employes": "employés",
    "conge": "congés",
    "demandeconge": "demandes de congé",
    "demandecongeodata": "demandes de congé",
    "paieemployesodata": "paies des employés",
    "paieavancesodata": "avances sur salaire",
    "paiefichesodata": "fiches de paie",
    "paiepointageodata": "pointages",
    "paiechantierodata": "paies de chantier",
    "paiechantier": "paies de chantier",
    # Bancaire / Référentiel
    "agencebancaire": "agences bancaires",
    "agence": "agences",
    "agencebancaireodata": "agences bancaires",
    "historiques": "historiques",
    "historiquesodata": "historiques",
    "translation": "traductions",
    "translationodata": "traductions",
    # B2B / Web
    "b2b_intermediate": "articles B2B",
    "b2bintermediate": "articles B2B",
    "b2b": "articles B2B",
    "webusers": "utilisateurs web",
    "webuser": "utilisateurs web",
}

# Path-specific label overrides (for endpoints whose URL alone gives better context
# than their generic tag).
_PATH_LABEL_OVERRIDES: dict[str, str] = {
    "/api/B2b_intermediate/AllWebUsers": "utilisateurs web B2B",
    "/api/B2b_intermediate/GetCategories": "catégories du catalogue B2B",
    "/api/B2b_intermediate/GetDiscountPromotion": "promotions et remises B2B",
    "/api/B2b_intermediate/GetFamillesParents": "familles d'articles B2B",
    "/api/B2b_intermediate/GetMarques": "marques du catalogue B2B",
    "/api/B2b_intermediate/GetSousFamilles": "sous-familles d'articles B2B",
    "/api/B2b_intermediate/GetArticlesVoitures": "articles pour voitures (B2B)",
    "/api/Client/GetFamillesClient": "familles de clients",
    "/api/Client/GetZonesClient": "zones de clients",
    "/api/Depenses/GetAffectations": "affectations de dépenses",
    "/api/Depenses/GetAgences": "agences (référentiel dépenses)",
    "/api/Depenses/GetAnneeList": "années disponibles (dépenses)",
    "/api/Depenses/GetCategories": "catégories de dépenses",
    "/api/Depenses/GetComptaLignes": "lignes comptables des dépenses",
    "/api/Depenses/GetCountDepense": "nombre de dépenses",
    "/api/Depenses/GetDepots": "dépôts (référentiel dépenses)",
    "/api/Depenses/GetEmployes": "employés (référentiel dépenses)",
    "/api/Depenses/GetFichesPaie": "fiches de paie liées aux dépenses",
    "/api/Depenses/GetModes": "modes de paiement des dépenses",
    "/api/Depenses/GetMoisPaie": "mois de paie (dépenses)",
    "/api/BlClient/GetAllBlClients": "bons de livraison client",
    "/api/BlClient/GetAllClients": "clients (référentiel BL)",
    "/api/BlClient/GetAllCommandes": "commandes client (référentiel BL)",
    "/api/BlClient/GetAllDepots": "dépôts (référentiel BL)",
    "/api/BlClient/GetAllLots": "lots (référentiel BL)",
    "/api/BonTransfert/GETAllDepo": "dépôts (référentiel transfert)",
    "/api/BonTransfert/GetAllArticles": "articles (référentiel transfert)",
    "/api/BonTransfert/GetAllLots": "lots (référentiel transfert)",
    "/api/BonTransfertDemande/GETAllDepo": "dépôts (référentiel demande de transfert)",
    "/api/BonTransfertDemande/GetAllArticles": "articles (référentiel demande de transfert)",
    "/api/Paiements/GetAllClients": "clients (référentiel paiements)",
    "/api/Paiements/GetAllEtats": "états des paiements",
    "/api/Paiements/GetAllFrs": "fournisseurs (référentiel paiements)",
    "/api/StockDepot/GetAllFamille": "familles d'articles (stock)",
    "/api/StockDepot/GetAllFamilleP": "familles parentes d'articles (stock)",
    "/api/StockDepot/GetAllLot": "lots en stock",
    "/api/CommandeClient/GetAllClients": "clients (référentiel commandes)",
    "/api/DemandeConge/GetAll": "demandes de congé",
    "/api/DemandeConge/GetAllEmployes": "employés (référentiel congés)",
    "/api/Client/GetAllClients": "clients",
    "/api/Translation/GetAllToFront/{language}": "traductions par langue",
    "/api/Translation/GetAllToFront/{language}/{item}": "traductions par langue et item",
    "/api/Lot/etats": "états des lots",
    "/api/BonTransfertDemande/non-valides": "demandes de transfert non validées",
    "/api/Paiements/GetParametresf": "paramètres financiers des paiements",
    "/api/CommandeClient/GetDevisClientList": "devis clients",
    "/api/TransfertArgent/GetCountTransfert": "nombre de transferts d'argent",
}


def normalize_label(path: str, tags: list[str]) -> str:
    # Special OData root paths
    if path == "/odata":
        return "ressources OData"
    if "/$metadata" in path:
        return "métadonnées OData"

    # For $count paths, derive label from the parent path
    if path.endswith("/$count"):
        return normalize_label(path[: -len("/$count")], tags)

    # Path-specific overrides take priority
    if path in _PATH_LABEL_OVERRIDES:
        return _PATH_LABEL_OVERRIDES[path]

    # Try direct (merged-lowercase) tag lookup first
    for tag in tags:
        key = tag.lower().replace("_", "").replace("-", "")
        if key in _TAG_LABEL_MAP:
            return _TAG_LABEL_MAP[key]

    # Try split-token tag lookup
    tag_tokens = split_tokens(" ".join(tags) if tags else "")
    token_set = set(tag_tokens)
    if {"stats", "vente"}.issubset(token_set) or "statsvente" in token_set:
        return "statistiques de ventes"
    if {"commande", "client", "report"}.issubset(token_set):
        return "rapports des commandes client"
    for token in tag_tokens:
        if token in _TAG_LABEL_MAP:
            return _TAG_LABEL_MAP[token]

    # Fallback: try the first meaningful path segment (e.g. /api/Depenses/GetX → "Depenses")
    parts = [p for p in path.split("/") if p and p.lower() not in ("api", "odata")]
    for part in parts:
        key = part.lower().replace("_", "").replace("-", "")
        if key in _TAG_LABEL_MAP:
            return _TAG_LABEL_MAP[key]

    return "données"


def build_description(path: str, intent: str, tags: list[str], query_parameters: list[str], route_parameters: list[str]) -> str:
    # OData root paths
    if path == "/odata":
        return "Liste les ressources OData disponibles."
    if path == "/odata/$metadata":
        return "Retourne le schéma des métadonnées OData décrivant les entités disponibles."

    # $count paths
    if path.endswith("/$count"):
        label = normalize_label(path, tags)
        return f"Retourne le nombre total de {label}."

    label = normalize_label(path, tags)

    # "Nombre de …" paths (e.g. GetCountDepense, GetCountTransfert)
    if intent == "AGGREGATE" and "nombre" in label:
        return f"Retourne le {label}."

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


def _examples_for_label(label: str, query_parameters: list[str]) -> list[str]:
    """Return domain-specific example queries for a given resource label."""
    if "bon" in label and "livraison" in label and "client" in label:
        return [
            "afficher les bons de livraison client",
            "afficher les BL d'un client",
            "afficher le détail d'un bon de livraison",
            "afficher les livraisons en cours",
            "afficher les bons de livraison par date",
        ]
    if "bon" in label and "livraison" in label and "fournisseur" in label:
        return [
            "afficher les bons de livraison fournisseurs",
            "afficher les BL fournisseur",
            "afficher le détail d'un BL fournisseur",
            "afficher les livraisons fournisseurs",
            "afficher les articles reçus des fournisseurs",
        ]
    if "commande" in label and "article" in label:
        return [
            "afficher les articles des commandes",
            "afficher le détail des articles commandés",
            "afficher la quantité commandée par article",
        ]
    if "commande" in label and "rapport" in label:
        return [
            "afficher le rapport des commandes client",
            "afficher le rapport d'une commande client",
            "afficher le détail d'une commande client",
            "afficher le rapport des ventes client",
            "afficher le document d'une commande client",
        ]
    if "commande" in label:
        return [
            "afficher les commandes client",
            "afficher les commandes",
            "afficher les commandes d'un client",
            "afficher le détail des commandes",
            "afficher les commandes en cours",
        ]
    if "statistique" in label or "vente" in label:
        return [
            "afficher les statistiques de vente",
            "afficher le chiffre d'affaires",
            "afficher les ventes par article",
            "afficher les ventes entre deux dates",
            "afficher les performances commerciales",
        ]
    if "client" in label:
        return [
            "afficher les clients",
            "afficher le nom du client",
            "afficher l'email du client",
            "afficher le code client",
            "afficher le téléphone du client",
        ]
    if "détail" in label and "article" in label:
        return [
            "afficher les détails des articles",
            "afficher la description d'un article",
            "afficher les caractéristiques d'un article",
        ]
    if "article" in label or "B2B" in label:
        return [
            "afficher les articles",
            "afficher le code article",
            "afficher le prix des articles",
            "afficher les articles disponibles",
            "afficher les détails des articles",
        ]
    if "stocks par dépôt" in label:
        return [
            "afficher le stock par dépôt",
            "afficher la quantité en stock dans un dépôt",
            "afficher le stock disponible dans chaque dépôt",
            "afficher le stock d'un article par dépôt",
        ]
    if "stocks par lot" in label:
        return [
            "afficher le stock par lot",
            "afficher les lots disponibles",
            "afficher la quantité par lot",
        ]
    if "stock" in label:
        return [
            "afficher le stock",
            "afficher le stock par dépôt",
            "afficher le stock des articles",
            "afficher la quantité disponible",
            "afficher le stock actuel",
        ]
    if "transfert" in label and "argent" in label:
        return [
            "afficher les transferts d'argent",
            "afficher le montant des transferts",
            "afficher les transferts entre agences",
            "afficher l'historique des transferts",
        ]
    if "transfert" in label:
        return [
            "afficher les bons de transfert",
            "afficher les transferts de stock",
            "afficher les transferts entre dépôts",
            "afficher le détail d'un bon de transfert",
        ]
    if "bon" in label and "entrée" in label:
        return [
            "afficher les bons d'entrée en stock",
            "afficher les réceptions de marchandises",
            "afficher le détail d'un bon d'entrée",
        ]
    if "lot" in label:
        return [
            "afficher les lots",
            "afficher les lots disponibles",
            "afficher l'état des lots",
        ]
    if "ordre" in label and "coupe" in label:
        return [
            "afficher les ordres de coupe",
            "afficher les articles à couper",
            "afficher le détail d'un ordre de coupe",
        ]
    if "paiement" in label and "client" in label:
        return [
            "afficher les paiements clients",
            "afficher les règlements clients",
            "afficher les factures payées par les clients",
            "afficher les paiements par client",
        ]
    if "paiement" in label and "fournisseur" in label:
        return [
            "afficher les paiements fournisseurs",
            "afficher les règlements fournisseurs",
            "afficher les factures payées aux fournisseurs",
        ]
    if "paiement" in label:
        return [
            "afficher les paiements",
            "afficher les paiements client",
            "afficher les règlements",
            "afficher les factures payées",
            "afficher les paiements par client",
        ]
    if "dépense" in label:
        return [
            "afficher les dépenses",
            "afficher les dépenses par catégorie",
            "afficher le montant des dépenses",
            "afficher les dépenses du mois",
            "afficher les dépenses par agence",
        ]
    if "fournisseur" in label:
        return [
            "afficher les fournisseurs",
            "afficher le nom du fournisseur",
            "afficher les factures fournisseurs",
            "afficher les paiements fournisseurs",
            "afficher les fournisseurs actifs",
        ]
    if "avance" in label:
        return [
            "afficher les avances sur salaire",
            "afficher les avances des employés",
            "afficher le montant des avances",
        ]
    if "fiche" in label and "paie" in label:
        return [
            "afficher les fiches de paie",
            "afficher la fiche de paie d'un employé",
            "afficher le salaire net",
        ]
    if "pointage" in label:
        return [
            "afficher les pointages",
            "afficher les heures travaillées",
            "afficher les pointages du mois",
        ]
    if "paie" in label and "chantier" in label:
        return [
            "afficher les paies de chantier",
            "afficher les salaires de chantier",
        ]
    if "paie" in label and "employé" in label:
        return [
            "afficher les paies des employés",
            "afficher le salaire des employés",
            "afficher les bulletins de paie",
        ]
    if "demande" in label and "congé" in label:
        return [
            "afficher les demandes de congé",
            "afficher les congés demandés",
            "afficher l'état des demandes de congé",
            "afficher les congés par employé",
        ]
    if "congé" in label:
        return [
            "afficher les congés",
            "afficher les congés des employés",
            "afficher les types de congé",
        ]
    if "employé" in label:
        return [
            "afficher les employés",
            "afficher la liste des employés",
            "afficher les informations d'un employé",
        ]
    if "agence" in label:
        return [
            "afficher les agences bancaires",
            "afficher le code agence",
            "afficher les détails des agences",
            "afficher une agence par code",
        ]
    if "historique" in label:
        return [
            "afficher les historiques",
            "afficher l'historique des opérations",
        ]
    if "traduction" in label:
        return [
            "afficher les traductions",
            "afficher les libellés traduits",
            "afficher les traductions par langue",
        ]
    if "rapport" in label:
        return [
            "afficher les rapports",
            "générer un rapport",
            "afficher le rapport de ventes",
        ]
    if "utilisateur" in label:
        return [
            "afficher les utilisateurs web",
            "afficher la liste des utilisateurs",
        ]
    generic = [
        "afficher les données",
        "afficher la liste complète",
        "afficher les informations disponibles",
    ]
    if query_parameters:
        generic.append(f"afficher avec le filtre {query_parameters[0]}")
    return generic


def build_examples(path: str, tags: list[str], query_parameters: list[str]) -> list[str]:
    # Skip $count and OData metadata endpoints – no natural-language examples needed
    if path.endswith("/$count") or "/$metadata" in path or path == "/odata":
        return []

    text = " ".join([path, " ".join(tags)]).lower()
    label = normalize_label(path, tags)

    # Commande-client reports (keep specific check before generic "report")
    if "commande_client-report" in text or ("report" in text and "commande" in text and "client" in text):
        return [
            "afficher le rapport des commandes client",
            "afficher le rapport d'une commande client",
            "afficher le détail d'une commande client",
            "afficher le rapport des ventes client",
            "afficher le document d'une commande client",
        ]

    return _examples_for_label(label, query_parameters)


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
