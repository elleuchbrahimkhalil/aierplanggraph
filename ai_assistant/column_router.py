from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


META_PARAM_KEYS = {"requested_fields", "aggregations", "group_by", "groupby", "metrics"}


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", value.lower()).strip()


def _normalize_list(values: List[str]) -> List[str]:
    normalized: List[str] = []
    for value in values:
        token = _normalize_token(str(value))
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def _extract_agg_field(agg: str) -> str:
    match = re.match(r"\s*(sum|avg|mean|count|min|max)\s*\(?\s*([a-zA-Z0-9_]+)\s*\)?", agg, re.IGNORECASE)
    return match.group(2) if match else ""


def collect_routing_params(analysis: Dict[str, Any]) -> List[str]:
    params: List[str] = []

    def add(value: Any) -> None:
        token = str(value).strip()
        if token and token not in params:
            params.append(token)

    entity = str(analysis.get("entity", "") or "").strip()
    if entity:
        add(entity)

    requested_fields = analysis.get("requested_fields", [])
    if isinstance(requested_fields, list):
        for field in requested_fields:
            add(field)

    extracted_params = analysis.get("extracted_params", {})
    if isinstance(extracted_params, dict):
        # Ajouter les clés des filtres (ex: ville, date, code) pour aider le router
        # à sélectionner un endpoint qui expose ces colonnes, même si ces champs ne
        # sont pas explicitement demandés dans requested_fields.
        for key in extracted_params.keys():
            if key in META_PARAM_KEYS:
                continue
            add(key)

        tables = extracted_params.get("tables")
        if isinstance(tables, list):
            for table in tables:
                add(table)

        group_by = extracted_params.get("group_by")
        if isinstance(group_by, list):
            for item in group_by:
                add(item)
        elif group_by:
            add(group_by)

        groupby = extracted_params.get("groupby")
        if isinstance(groupby, list):
            for item in groupby:
                add(item)
        elif groupby:
            add(groupby)

        aggregations = extracted_params.get("aggregations")
        if isinstance(aggregations, dict):
            # ex: { total: "sum(montant)" }
            for expr in aggregations.values():
                field = _extract_agg_field(str(expr))
                if field:
                    add(field)
        for agg in aggregations or []:
            if isinstance(agg, dict) and agg.get("field"):
                add(agg.get("field"))
            elif isinstance(agg, str):
                field = _extract_agg_field(agg)
                if field:
                    add(field)

    return params


def route_by_columns(
    candidates: List[Dict[str, Any]],
    params: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Algorithme de routage intelligent basé sur les colonnes.
    
    Étapes:
    1. Parcourt chaque endpoint et vérifie si les paramètres correspondent aux colonnes
    2. Stocke les correspondances dans un dictionnaire {endpoint_id: [params_trouvés]}
    3. Sélectionne l'endpoint avec le plus de correspondances
    4. Si des paramètres manquent, cherche d'autres endpoints pour les compléter
    5. Retourne la liste des endpoints sélectionnés et le dictionnaire des correspondances
    """
    if not candidates:
        return [], {}

    # Si pas de paramètres, retourner le premier endpoint par défaut
    if not params:
        return [candidates[0]], {}

    normalized_params = _normalize_list(params)
    
    # Dictionnaire pour mémoriser les résultats du parcours: {endpoint_id: [params_trouvés]}
    matched_by_endpoint: Dict[str, List[str]] = {}
    
    # Variable pour stocker l'ID du premier endpoint par défaut
    first_endpoint_id = str(candidates[0].get("id", ""))
    
    # Parcourir tous les endpoints et trouver les correspondances
    for endpoint in candidates:
        endpoint_id = str(endpoint.get("id", ""))
        columns = _normalize_list(endpoint.get("columns", []) or [])
        
        # Trouver les paramètres qui correspondent aux colonnes de cet endpoint
        matched = [param for param in normalized_params if param in columns]
        
        if matched:
            matched_by_endpoint[endpoint_id] = matched
    
    # Si aucune correspondance trouvée, retourner le premier endpoint par défaut
    if not matched_by_endpoint:
        return [candidates[0]], {}
    
    # Trouver l'endpoint avec le plus de correspondances (clé avec le plus de valeurs)
    best_endpoint_id = None
    best_match_count = -1
    
    for endpoint_id, matched_params in matched_by_endpoint.items():
        if len(matched_params) > best_match_count:
            best_match_count = len(matched_params)
            best_endpoint_id = endpoint_id
    
    # Si un endpoint contient TOUS les paramètres, le retourner directement
    if best_match_count == len(normalized_params):
        selected_endpoint = next(
            (ep for ep in candidates if str(ep.get("id", "")) == best_endpoint_id),
            candidates[0]
        )
        return [selected_endpoint], matched_by_endpoint
    
    # Liste des IDs d'endpoints sélectionnés (commence par le meilleur)
    selected_ids: List[str] = [best_endpoint_id] if best_endpoint_id else [first_endpoint_id]
    
    # Ensemble des paramètres déjà trouvés
    found_params: set = set(matched_by_endpoint.get(best_endpoint_id or "", []))
    
    # Chercher d'autres endpoints pour les paramètres restants
    remaining_params = [param for param in normalized_params if param not in found_params]
    
    # Trier les endpoints par nombre de correspondances (du plus grand au plus petit)
    sorted_endpoints = sorted(
        matched_by_endpoint.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    if remaining_params:
        for endpoint_id, matched_params in sorted_endpoints:
            # Ignorer les endpoints déjà sélectionnés
            if endpoint_id in selected_ids:
                continue
            
            # Vérifier si cet endpoint contient des paramètres restants
            new_matches = [param for param in remaining_params if param in matched_params]
            
            if new_matches:
                # Ajouter cet endpoint à la sélection
                selected_ids.append(endpoint_id)
                found_params.update(new_matches)
                
                # Mettre à jour les paramètres restants
                remaining_params = [param for param in normalized_params if param not in found_params]
                
                # Si tous les paramètres sont trouvés, arrêter la recherche
                if not remaining_params:
                    break
    
    # Construire la liste des endpoints sélectionnés
    selected_endpoints = [
        endpoint for endpoint in candidates 
        if str(endpoint.get("id", "")) in selected_ids
    ]
    
    return selected_endpoints, matched_by_endpoint


def build_mapping_context_from_endpoints(
    endpoints: List[Dict[str, Any]],
    max_endpoints: int = 24,
    max_columns: int = 12,
) -> List[Dict[str, Any]]:
    context: List[Dict[str, Any]] = []
    for endpoint in endpoints[:max_endpoints]:
        columns: List[Dict[str, Any]] = []
        for column in (endpoint.get("columns", []) or [])[:max_columns]:
            columns.append({"name": column, "description": None, "synonyms": []})
        context.append(
            {
                "table": endpoint.get("id"),
                "description": endpoint.get("description"),
                "columns": columns,
            }
        )
    return context


def write_mapping_db_from_endpoints(
    path: Path,
    endpoints: List[Dict[str, Any]],
    max_columns: int = 48,
) -> None:
    mapping: List[Dict[str, Any]] = []
    for endpoint in endpoints:
        columns: List[Dict[str, Any]] = []
        for column in (endpoint.get("columns", []) or [])[:max_columns]:
            columns.append(
                {
                    "name": column,
                    "type": "string",
                    "isPrimaryKey": False,
                    "isForeignKey": False,
                    "description": "Column from endpoint",
                    "synonyms": [],
                }
            )
        mapping.append(
            {
                "table": endpoint.get("id"),
                "businessDomain": endpoint.get("role"),
                "description": endpoint.get("description"),
                "columns": columns,
            }
        )

    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def explain_routing_result(
    selected_endpoints: List[Dict[str, Any]],
    matched_by_endpoint: Dict[str, List[str]],
    all_params: List[str],
) -> str:
    """
    Génère une explication lisible du résultat du routage.
    
    Args:
        selected_endpoints: Liste des endpoints sélectionnés
        matched_by_endpoint: Dictionnaire {endpoint_id: [params_trouvés]}
        all_params: Liste de tous les paramètres recherchés
    
    Returns:
        Une chaîne expliquant le résultat du routage
    """
    if not selected_endpoints:
        return "Aucun endpoint sélectionné."
    
    lines: List[str] = []
    lines.append("=== Résultat du routage ===")
    lines.append(f"Paramètres recherchés: {', '.join(all_params)}")
    lines.append(f"Nombre d'endpoints sélectionnés: {len(selected_endpoints)}")
    lines.append("")
    
    all_matched: List[str] = []
    for endpoint in selected_endpoints:
        endpoint_id = str(endpoint.get("id", ""))
        matched = matched_by_endpoint.get(endpoint_id, [])
        all_matched.extend(matched)
        
        lines.append(f"Endpoint: {endpoint_id}")
        lines.append(f"  - URL: {endpoint.get('url', 'N/A')}")
        lines.append(f"  - Paramètres trouvés ({len(matched)}): {', '.join(matched) if matched else 'Aucun'}")
        lines.append("")
    
    # Vérifier les paramètres manquants
    normalized_all = _normalize_list(all_params)
    normalized_matched = _normalize_list(all_matched)
    missing = [p for p in normalized_all if p not in normalized_matched]
    
    if missing:
        lines.append(f"⚠️ Paramètres non trouvés: {', '.join(missing)}")
    else:
        lines.append("✅ Tous les paramètres ont été trouvés.")
    
    return "\n".join(lines)
