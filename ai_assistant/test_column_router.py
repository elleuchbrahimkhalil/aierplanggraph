"""
Test du column_router avec la question: "affiche les ventes par article"
"""
from column_router import route_by_columns, collect_routing_params, explain_routing_result
import json

# Simuler des endpoints candidats avec leurs colonnes
MOCK_ENDPOINTS = [
    {
        "id": "api/articles",
        "url": "/api/articles",
        "description": "Liste des articles",
        "columns": ["cod_art", "desg_art", "pvht", "pvuttc", "pachatmoy", "qmin_art", "cod_fam", "est_vendable"]
    },
    {
        "id": "api/ventes",
        "url": "/api/ventes",
        "description": "Liste des ventes",
        "columns": ["num_vente", "cod_art", "qte_vendue", "prix_unitaire", "date_vente", "client", "total"]
    },
    {
        "id": "api/stats-ventes-article",
        "url": "/api/stats/ventes-article",
        "description": "Statistiques des ventes par article",
        "columns": ["cod_art", "desg_art", "quantite_vendue", "chiffre_affaire", "marge", "nombre_ventes"]
    },
    {
        "id": "api/factures",
        "url": "/api/factures",
        "description": "Liste des factures",
        "columns": ["num_fact", "client", "date_fact", "totalht", "net", "reste", "etat_fact"]
    },
    {
        "id": "api/clients",
        "url": "/api/clients",
        "description": "Liste des clients",
        "columns": ["cod_clt", "raison", "tel", "solde_reel", "soldemax", "chiffre_d_affaire"]
    }
]


def test_ventes_par_article():
    """Test: 'affiche les ventes par article'"""
    print("=" * 60)
    print("TEST: affiche les ventes par article")
    print("=" * 60)
    
    # Simuler l'analyse de Llama
    analysis = {
        "entity": "ventes",
        "requested_fields": ["article", "ventes", "quantite", "chiffre_affaire"],
        "extracted_params": {
            "group_by": ["article"],
            "aggregations": [
                {"field": "quantite_vendue"},
                {"field": "chiffre_affaire"}
            ]
        }
    }
    
    # Collecter les paramètres de routage
    params = collect_routing_params(analysis)
    print(f"\nParamètres collectés: {params}")
    
    # Exécuter le routage
    selected_endpoints, matched_by_endpoint = route_by_columns(MOCK_ENDPOINTS, params)
    
    # Afficher le résultat
    explanation = explain_routing_result(selected_endpoints, matched_by_endpoint, params)
    print("\n" + explanation)
    
    # Afficher les détails du dictionnaire des correspondances
    print("\n--- Détail des correspondances ---")
    for endpoint_id, matched_params in matched_by_endpoint.items():
        print(f"  {endpoint_id}: {matched_params}")
    
    return selected_endpoints, matched_by_endpoint


def test_clients_de_paris():
    """Test: 'quels sont mes clients de Paris'"""
    print("\n" + "=" * 60)
    print("TEST: quels sont mes clients de Paris")
    print("=" * 60)
    
    analysis = {
        "entity": "clients",
        "requested_fields": ["cod_clt", "raison", "ville"],
        "extracted_params": {
            "ville": "Paris"
        }
    }
    
    params = collect_routing_params(analysis)
    print(f"\nParamètres collectés: {params}")
    
    selected_endpoints, matched_by_endpoint = route_by_columns(MOCK_ENDPOINTS, params)
    
    explanation = explain_routing_result(selected_endpoints, matched_by_endpoint, params)
    print("\n" + explanation)
    
    return selected_endpoints, matched_by_endpoint


def test_factures_impayees():
    """Test: 'montre les factures impayées'"""
    print("\n" + "=" * 60)
    print("TEST: montre les factures impayées")
    print("=" * 60)
    
    analysis = {
        "entity": "factures",
        "requested_fields": ["num_fact", "client", "reste", "date_fact"],
        "extracted_params": {
            "etat": "impayee"
        }
    }
    
    params = collect_routing_params(analysis)
    print(f"\nParamètres collectés: {params}")
    
    selected_endpoints, matched_by_endpoint = route_by_columns(MOCK_ENDPOINTS, params)
    
    explanation = explain_routing_result(selected_endpoints, matched_by_endpoint, params)
    print("\n" + explanation)
    
    return selected_endpoints, matched_by_endpoint


def test_multi_endpoint():
    """Test: requête nécessitant plusieurs endpoints"""
    print("\n" + "=" * 60)
    print("TEST: affiche le stock et les ventes par article")
    print("=" * 60)
    
    analysis = {
        "entity": "articles",
        "requested_fields": ["cod_art", "desg_art", "qmin_art", "quantite_vendue", "chiffre_affaire"],
        "extracted_params": {}
    }
    
    params = collect_routing_params(analysis)
    print(f"\nParamètres collectés: {params}")
    
    selected_endpoints, matched_by_endpoint = route_by_columns(MOCK_ENDPOINTS, params)
    
    explanation = explain_routing_result(selected_endpoints, matched_by_endpoint, params)
    print("\n" + explanation)
    
    # Afficher combien d'endpoints ont été sélectionnés
    if len(selected_endpoints) > 1:
        print(f"\n✅ Multi-endpoint sélectionné: {len(selected_endpoints)} endpoints")
        for ep in selected_endpoints:
            print(f"   - {ep.get('id')}")
    
    return selected_endpoints, matched_by_endpoint


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TEST DU COLUMN ROUTER")
    print("=" * 60)
    
    # Exécuter tous les tests
    test_ventes_par_article()
    test_clients_de_paris()
    test_factures_impayees()
    test_multi_endpoint()
    
    print("\n" + "=" * 60)
    print("FIN DES TESTS")
    print("=" * 60)
