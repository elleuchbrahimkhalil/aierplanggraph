# Installation et déploiement (guide rapide)

Ce fichier décrit les prérequis et les commandes pour installer et démarrer le projet "javaerp" sur une autre machine (Windows ou Linux).

## Résumé
- Backend assistant (Python) : `ai_assistant/langgraph_skeleton.py`
- Frontend (React + Vite) : `frontend/`
- WebApi Java (Spring Boot) : racine Maven (`pom.xml`) — build avec `mvnw`
- Optionnel : Airflow (docker-compose), Ollama local pour LLMs, MySQL pour la base.

## Prérequis logiciels
- Git
- JDK compatible (voir `pom.xml` propriété `<java.version>`). Java 17+ recommandé (le projet indique `25`, utiliser la JDK disponible et compatible avec Spring Boot utilisé).
- Maven wrapper (inclus: `mvnw`, `mvnw.cmd`)
- Node.js (>=18), `npm`
- Python 3.11+ (ou 3.10+), `pip`
- Virtualenv (optionnel mais recommandé)
- Docker & Docker Compose (si vous utilisez `docker-compose.airflow.yml` ou services conteneurisés)

## Variables d'environnement importantes
Configurer (exemples) — définir soit dans la session, soit dans un fichier `.env`/systemd/service :

- `ERP_API_BASE_URL` : base URL du WebApi (ex: `http://localhost:5006`).
- `ERP_API_BEARER_TOKEN` : token Bearer si l'API nécessite authentification automatique.
- Ollama (optionnel) :
  - `OLLAMA_URL` (ex: `http://localhost:11434`)
  - `OLLAMA_MODEL_PARAM_EXTRACTOR`, `OLLAMA_MODEL_ROUTER`, `OLLAMA_MODEL_TRANSFORM` (noms de modèles selon ton instance Ollama)

## Étapes d'installation (Linux / macOS)

1. Cloner le repo:

```bash
git clone <repo-url> javaerp
cd javaerp
```

2. Backend Python — créer un env et installer dépendances:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r ai_assistant/requirements.txt
```

3. (Optionnel) Configurer variables d'environnement, ex:

```bash
export ERP_API_BASE_URL="http://localhost:5006"
export OLLAMA_URL="http://localhost:11434"
# export ERP_API_BEARER_TOKEN="..."
```

4. Lancer le backend assistant (serve):

```bash
python ai_assistant/langgraph_skeleton.py --serve --host 0.0.0.0 --port 8000
```

5. WebApi Java — builder et démarrer (utilise le wrapper Maven inclus):

```bash
./mvnw -v
./mvnw package -DskipTests
java -jar target/*.jar
```

6. Frontend — installer et démarrer (dev) :

```bash
cd frontend
npm install
npm run dev  # ou `npm run build` pour build production
```

7. (Optionnel) Airflow / autres services :

```bash
docker compose -f docker-compose.airflow.yml up -d
```

## Étapes Windows (PowerShell)

1. Cloner et ouvrir PowerShell, puis :

```powershell
git clone <repo-url> javaerp
cd "javaerp"
```

2. Python venv (PowerShell) :

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r ai_assistant/requirements.txt
```

3. Définir variables d'environnement (PowerShell) :

```powershell
$Env:ERP_API_BASE_URL = 'http://localhost:5006'
$Env:OLLAMA_URL = 'http://localhost:11434'
# $Env:ERP_API_BEARER_TOKEN = '...'
```

4. Lancer le backend :

```powershell
python ai_assistant/langgraph_skeleton.py --serve --host 127.0.0.1 --port 8000
```

5. Build Java (Windows) :

```powershell
.\mvnw.cmd -v
.\mvnw.cmd package -DskipTests
java -jar target\*.jar
```

6. Frontend (Windows PowerShell):

```powershell
cd frontend
npm install
npm run dev
```

## Dépannage rapide
- Si `/assistant/seaborn.png` renvoie une erreur : vérifier que `pandas`, `matplotlib`, `seaborn` et `numpy` sont installés dans l'environnement Python.
- Si WebApi renvoie 500 avec message MySQL : vérifier que la base MySQL est accessible et que `application.properties` contient les bonnes connexions.
- Si `npm run dev` échoue : supprimer `node_modules` et `package-lock.json` puis `npm install`.

## Remarques et sécurité
- Ne commite jamais de secrets (tokens) dans le repo. Utilise des fichiers `.env` exclus via `.gitignore` ou variables d'environnement.
- Si tu utilises Ollama local, assure-toi d'avoir suffisamment de mémoire et les modèles requis.

## Fichiers ajoutés
- `ai_assistant/requirements.txt` : liste des dépendances Python nécessaires.

---
Si tu veux, je peux créer un script d'installation automatique (`install_all.sh` / `install_all.ps1`) pour automatiser ces étapes.
