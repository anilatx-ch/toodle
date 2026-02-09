#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ZIP="${1:-$ROOT_DIR/../full-stack-ai-solution.zip}"
if [[ "$OUTPUT_ZIP" != /* ]]; then
  OUTPUT_ZIP="$ROOT_DIR/$OUTPUT_ZIP"
fi
OUTPUT_ZIP="$(realpath -m "$OUTPUT_ZIP")"
STAGING_DIR="$(mktemp -d)"
PACKAGE_ROOT="$STAGING_DIR/full-stack-ai-solution"

cleanup() {
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

mkdir -p "$PACKAGE_ROOT"

INCLUDE_PATHS=(
  "0_OBJECTIVE.md"
  "0_schema.json"
  "README.md"
  "pyproject.toml"
  "poetry.lock"
  "Makefile"
  "Dockerfile"
  "docker-compose.yml"
  "run_evaluation.py"
  "schema.py"
  "src"
  "scripts"
  "tests"
  "dbt_project"
  "docs"
  "schemas"
  "exploration/README.md"
  "exploration/subcategory_independence"
)

for rel_path in "${INCLUDE_PATHS[@]}"; do
  src_path="$ROOT_DIR/$rel_path"
  if [[ -e "$src_path" ]]; then
    mkdir -p "$PACKAGE_ROOT/$(dirname "$rel_path")"
    cp -a "$src_path" "$PACKAGE_ROOT/$rel_path"
  fi
done

# Remove generated dbt artifacts from packaged source tree.
rm -rf "$PACKAGE_ROOT/dbt_project/target" "$PACKAGE_ROOT/dbt_project/logs"
rm -f "$PACKAGE_ROOT/dbt_project/.user.yml"

# Strip runtime cache artifacts if present.
find "$PACKAGE_ROOT" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$PACKAGE_ROOT" -type f \( -name "*.pyc" -o -name "*.pyo" -o -name ".DS_Store" \) -delete

rm -f "$OUTPUT_ZIP"
mkdir -p "$(dirname "$OUTPUT_ZIP")"
(
  cd "$STAGING_DIR"
  zip -rq "$OUTPUT_ZIP" "full-stack-ai-solution"
)

echo "Created package: $OUTPUT_ZIP"
