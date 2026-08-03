#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  set -- bash
fi

if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  echo "NVIDIA runtime detected. Running with GPU configuration."
  docker compose run --rm app "$@"
else
  echo "No NVIDIA runtime detected. Falling back to CPU override."
  docker compose -f docker-compose.cpu.yml run --rm app "$@"
fi
