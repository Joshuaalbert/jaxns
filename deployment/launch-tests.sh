#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
echo "Script dir $SCRIPT_DIR"

docker-compose -f "$SCRIPT_DIR"/docker-compose.yaml down

docker-compose -f "$SCRIPT_DIR"/docker-compose.yaml build

docker-compose -f "$SCRIPT_DIR"/docker-compose.yaml up tests
