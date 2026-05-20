#!/usr/bin/env bash
# Check Ollama and start daemon (Unix)
set -e
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama not installed. See https://ollama.com/docs/install"
  exit 1
fi

echo "Ollama version: $(ollama version || true)"

echo "Attempting to start ollama daemon (may already be running)..."
if ! ollama daemon start; then
  echo "Daemon start failed or already running"
fi

echo "Done. Example: ollama pull llama3 && ollama run llama3"
