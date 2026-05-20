# Installer et démarrer le frontend (React + Vite)

## Prérequis
- Node.js >= 18
- npm

## Installation
```bash
cd frontend
npm install
```

## Démarrer en mode développement
```bash
npm run dev
# ouvre normalement http://localhost:5173
```

## Build production
```bash
npm run build
# le dossier `dist/` sera créé
```

## Dépannage
- Si erreur de dépendances : supprimer `node_modules` et `package-lock.json`, puis `npm install`.
- Si le port 5173 est pris, exécuter `npm run dev -- --port 5174`.
