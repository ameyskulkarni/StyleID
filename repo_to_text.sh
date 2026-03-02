#!/bin/bash
# repo_to_text.sh - Concatenate repository into a single file for Claude

OUTPUT_FILE="repo_contents.txt"
REPO_DIR="${1:-.}"  # Default to current directory

# File extensions to include
EXTENSIONS="py|md|yaml|yml|json|txt|sh|toml|cfg"

# Directories/patterns to exclude
EXCLUDE_DIRS=".git|__pycache__|node_modules|.venv|venv|env|.eggs|*.egg-info|dist|build|.pytest_cache|.mypy_cache|wandb|outputs|checkpoints"

echo "# Repository Contents" > "$OUTPUT_FILE"
echo "# Generated on $(date)" >> "$OUTPUT_FILE"
echo "# Source: $REPO_DIR" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# First, add directory structure
echo "## Directory Structure" >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
find "$REPO_DIR" -type f \
    | grep -Ev "$EXCLUDE_DIRS" \
    | grep -E "\.($EXTENSIONS)$" \
    | sed 's|^\./||' \
    | sort \
    >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Then add file contents
echo "## File Contents" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

find "$REPO_DIR" -type f \
    | grep -Ev "$EXCLUDE_DIRS" \
    | grep -E "\.($EXTENSIONS)$" \
    | sort \
    | while read -r file; do
        echo "========================================" >> "$OUTPUT_FILE"
        echo "FILE: ${file#./}" >> "$OUTPUT_FILE"
        echo "========================================" >> "$OUTPUT_FILE"
        cat "$file" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
done

echo "Created $OUTPUT_FILE ($(wc -l < "$OUTPUT_FILE") lines, $(du -h "$OUTPUT_FILE" | cut -f1))"