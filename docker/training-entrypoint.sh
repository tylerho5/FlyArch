#!/bin/bash
set -euo pipefail

# The mounted cache directory must exist before the container starts.
test -d /cache
mkdir -p -- "$HOME" "$XDG_CACHE_HOME"
/opt/conda/bin/python /opt/flyarch/record_training_environment.py "$@"
exec "$@"
