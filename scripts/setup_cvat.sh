#!/usr/bin/env bash
# Installs and starts a self-hosted CVAT next to this project (needs Docker + git).
set -euo pipefail
VERSION="${1:-v2.25.0}"   # check https://github.com/cvat-ai/cvat/releases
DIR="$(dirname "$0")/../cvat"

[ -d "$DIR" ] || git clone --depth 1 --branch "$VERSION" https://github.com/cvat-ai/cvat.git "$DIR"
cd "$DIR"
CVAT_VERSION="$VERSION" docker compose up -d
echo "CVAT is starting at http://localhost:8080"
echo "Create the admin user: docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'"
