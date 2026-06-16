#!/usr/bin/env bash
set -euo pipefail

FOLDER_NAME="eric-matti-submission-code"
OUT="eric_matti_submission_code.zip"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

rm -f "$SCRIPT_DIR/$OUT"

cd "$PARENT_DIR"

zip -r "$SCRIPT_DIR/$OUT" "$FOLDER_NAME" \
    --include \
        "$FOLDER_NAME/*.py" \
        "$FOLDER_NAME/*.txt" \
        "$FOLDER_NAME/*.md" \
        "$FOLDER_NAME/*.sh" \
        "$FOLDER_NAME/logistic_regression/*.py" \
        "$FOLDER_NAME/random_forest/*.py" \
    --exclude \
        "*/__pycache__/*" \
        "*/.venv/*"

echo "Packaged -> $SCRIPT_DIR/$OUT"
zip -sf "$SCRIPT_DIR/$OUT"
