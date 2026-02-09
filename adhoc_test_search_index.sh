#!/bin/bash
set -e

echo "=== Building search index with 1% sample (test) ==="
echo

START=$(date +%s)
make build-search-index ENV=test
END=$(date +%s)

DURATION=$((END - START))
echo
echo "=== Runtime Analysis ==="
echo "Test (1%) runtime: ${DURATION}s"

# Calculate entries in test corpus
ENTRIES=$(.venv/bin/python -c "import json; print(len(json.load(open('data/retrieval/corpus_test.json'))))")
echo "Test entries indexed: ${ENTRIES}"

# Estimate total entries from raw data
TOTAL=$(.venv/bin/python -c "import json; rows=json.load(open('support_tickets.json')); print(len([r for r in rows if isinstance(r,dict) and r.get('resolution')]))")
echo "Total eligible entries: ${TOTAL}"

# Estimate prod runtime (100% / 1% = 100x)
PROD_EST=$((DURATION * 100))
echo
echo "Estimated prod runtime (100%): ${PROD_EST}s (~$((PROD_EST/60)) minutes)"
