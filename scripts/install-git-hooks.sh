#!/bin/sh
# Point this clone at the repo-managed hooks (see .githooks/).
set -eu
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
echo "core.hooksPath=$(git config --get core.hooksPath)"
