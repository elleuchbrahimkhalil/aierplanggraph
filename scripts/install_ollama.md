# Installer Ollama (guide rapide)

Ollama est optionnel mais recommandé si tu veux exécuter des modèles LLM localement sans API externe.

## Prérequis
- Système 64-bit
- Docker (facultatif mais utile)
- Espace disque et RAM suffisants selon le modèle (ex: 6-16GB+ RAM pour petits modèles, beaucoup plus pour gros modèles)

## Installation (Linux / macOS)
1. Installer Ollama (suivre la doc officielle) ; exemple pour macOS (Homebrew):

```bash
brew install ollama
```

Ou télécharger le binaire depuis : https://ollama.com/docs/install

2. Vérifier l'installation :

```bash
ollama version
```

3. Démarrer Ollama (service local) — sur la plupart des plateformes Ollama s'exécute comme service automatiquement après installation. Sinon :

```bash
ollama daemon start
```

## Installation (Windows)
- Télécharger l'installateur depuis https://ollama.com/docs/install ou utiliser le binaire fourni.
- Après installation, ouvrir PowerShell et vérifier:

```powershell
ollama version
ollama daemon start
```

## Télécharger et exécuter un modèle Llama/compatible
- Rechercher un modèle compatible (ex: `llama3`, `llama2`, ou un modèle tiers) sur le registre Ollama.
- Exemple d'installation d'un petit modèle (si disponible) :

```bash
ollama pull llama3
```

- Lancer le modèle en mode serveur (si nécessaire) :

```bash
ollama run llama3
```

ou utiliser l'API REST intégrée (par défaut `http://localhost:11434/api`)

## Variables d'environnement suggérées
- `OLLAMA_URL` : l'URL locale, typiquement `http://localhost:11434`
- `OLLAMA_MODEL_PARAM_EXTRACTOR` : nom du modèle pour extraction de paramètres
- `OLLAMA_MODEL_ROUTER` : nom du modèle pour routing
- `OLLAMA_MODEL_TRANSFORM` : nom du modèle pour transformation

## Remarques
- Les noms de modèle varient selon ton registre / versions disponibles. Vérifier `ollama list` et la doc officiell
- Si tu as des limites de RAM, privilégie des versions plus petites ou héberge le modèle sur un serveur dédié.
