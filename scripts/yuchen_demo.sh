#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PROFILE=yuchen-test
HOST=https://dbc-9dcd6158-e299.cloud.databricks.com
ENDPOINT=databricks-glm-5-2

cd "$REPO_DIR"

usage() {
    printf '%s\n' \
        "Usage: ./scripts/yuchen_demo.sh precheck" \
        "       ./scripts/yuchen_demo.sh validate" \
        "       ./scripts/yuchen_demo.sh inspect-endpoint" \
        "       ./scripts/yuchen_demo.sh benchmark" \
        "       ./scripts/yuchen_demo.sh verify RUN_DIR"
}

need() {
    command -v "$1" >/dev/null 2>&1 || {
        printf 'Missing required command: %s\n' "$1" >&2
        exit 127
    }
}

case "${1:-}" in
    precheck)
        need databricks
        need jq
        databricks auth login --host "$HOST" --profile "$PROFILE"
        databricks current-user me --profile "$PROFILE" \
            | jq '{userName,active,displayName}'
        ;;
    validate)
        need python3
        python3 -m traffic_replay validate --port 0 --format json
        python3 -m traffic_replay adapters --format json
        ;;
    inspect-endpoint)
        need databricks
        need jq
        databricks serving-endpoints get "$ENDPOINT" \
            --profile "$PROFILE" --output json \
            | jq '{name,state,route_optimized,endpoint_type:(.endpoint_type//null),served_entities:[(.config.served_entities//[])[]|{name,entity_name,entity_version,workload_type,workload_size,scale_to_zero_enabled,foundation_model:(.foundation_model.name//null)}]}'
        ;;
    benchmark)
        need python3
        printf '%s\n' \
            'This sends paid traffic: 2 preflight + 1 calibration + 1 measured request.' \
            'Type RUN to continue:'
        IFS= read -r confirmation
        [ "$confirmation" = RUN ] || {
            printf '%s\n' 'Cancelled; no benchmark traffic sent.'
            exit 2
        }
        python3 -m traffic_replay benchmark \
            --host "$HOST" \
            --endpoint "$ENDPOINT" \
            --auth-profile "$PROFILE" \
            --profile configs/profile_glm52_canary_illustrative.json \
            --endpoint-adapter openai.chat_completions.sse/v1 \
            --temperature 0 \
            --extra-body '{"reasoning_effort":"none"}' \
            --fixed-rate 0.1 \
            --duration 12 \
            --calibrate-requests 1 \
            --max-concurrency 1 \
            --max-pending-requests 1 \
            --ttft-definition first_visible \
            --rate-limits configs/rate_limits_databricks_glm_5_2_enterprise_p2t_2026-08-07.json \
            --out-dir results/yuchen-glm52-instrument-canary \
            --label 'INSTRUMENT CONFORMANCE ONLY - NOT CUSTOMER DEMAND OR CAPACITY' \
            --format json
        ;;
    verify)
        need python3
        run_dir=${2:-}
        [ -n "$run_dir" ] || {
            printf '%s\n' 'verify requires the exact measured RUN_DIR.' >&2
            usage >&2
            exit 64
        }
        python3 -m traffic_replay verify-run "$run_dir" \
            --out "${run_dir}-verification" --format json
        printf 'Verification requested under: %s\n' \
            "${run_dir}-verification"
        ;;
    *)
        usage >&2
        exit 64
        ;;
esac
