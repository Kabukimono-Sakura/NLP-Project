#!/usr/bin/env bash
set -euo pipefail

# Download models needed by framework3.
#
# Default:
#   bash framework3/scripts/download_models.sh
#
# Optional environment variables:
#   HF_ENDPOINT=https://hf-mirror.com
#   HF_HOME=/path/to/hf_cache
#   TEACHER_REPO=Qwen/Qwen3-14B
#   TEACHER_DIR=models/Qwen3-14B
#   STUDENT_BASE_REPO=Qwen/Qwen2.5-7B-Instruct
#   STUDENT_BASE_DIR=models/Qwen2.5-7B-Instruct
#   DOWNLOAD_STUDENT_BASE=1

PYTHON_BIN="${PYTHON:-python3}"

TEACHER_REPO="${TEACHER_REPO:-Qwen/Qwen3-14B}"
TEACHER_DIR="${TEACHER_DIR:-models/Qwen3-14B}"

STUDENT_BASE_REPO="${STUDENT_BASE_REPO:-Qwen/Qwen2.5-7B-Instruct}"
STUDENT_BASE_DIR="${STUDENT_BASE_DIR:-models/Qwen2.5-7B-Instruct}"
DOWNLOAD_STUDENT_BASE="${DOWNLOAD_STUDENT_BASE:-0}"

mkdir -p models

echo "[INFO] Python: $($PYTHON_BIN --version)"
echo "[INFO] Installing/upgrading Hugging Face downloader dependencies..."
if ! "$PYTHON_BIN" -m pip install -U huggingface_hub certifi; then
  echo "[WARN] pip install from the default index failed."
  echo "       If this is an SSL or network issue, try one of:"
  echo "       1) export PIP_TRUSTED_HOST=pypi.org"
  echo "       2) export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"
  echo "       3) conda install -y certifi"
  echo ""
  echo "       Continuing anyway; if huggingface_hub is already installed, download may still work."
fi

CERT_FILE="$("$PYTHON_BIN" - <<'PY'
try:
    import certifi
    print(certifi.where())
except Exception:
    print("")
PY
)"
if [[ -n "$CERT_FILE" ]]; then
  export SSL_CERT_FILE="$CERT_FILE"
  export REQUESTS_CA_BUNDLE="$CERT_FILE"
  echo "[INFO] SSL cert bundle: $CERT_FILE"
fi

if "$PYTHON_BIN" - <<'PY'
try:
    import hf_transfer  # noqa: F401
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
then
export HF_HUB_ENABLE_HF_TRANSFER=1
  echo "[INFO] hf_transfer detected; fast transfer enabled."
else
  unset HF_HUB_ENABLE_HF_TRANSFER
  echo "[INFO] hf_transfer not installed; using standard Hugging Face download."
fi

# hf-xet can fail behind mirrors/proxies with "Local entry not found" even when
# the repository exists. Standard HTTP download is slower but more reliable.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-2}"
HF_DOWNLOAD_RETRIES="${HF_DOWNLOAD_RETRIES:-10}"
echo "[INFO] HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo "[INFO] max workers: $HF_MAX_WORKERS"
echo "[INFO] download retries: $HF_DOWNLOAD_RETRIES"

download_repo() {
  local repo="$1"
  local target_dir="$2"

  local attempt=1
  while true; do
    echo "[INFO] Download attempt $attempt/$HF_DOWNLOAD_RETRIES: $repo"
    if command -v hf >/dev/null 2>&1; then
      if hf download "$repo" \
        --local-dir "$target_dir" \
        --max-workers "$HF_MAX_WORKERS"; then
        return 0
      fi
    else
      if huggingface-cli download "$repo" \
        --local-dir "$target_dir" \
        --resume-download; then
        return 0
      fi
    fi

    if [[ "$attempt" -ge "$HF_DOWNLOAD_RETRIES" ]]; then
      echo "[ERROR] Download failed after $HF_DOWNLOAD_RETRIES attempts: $repo"
      return 1
    fi

    attempt=$((attempt + 1))
    echo "[WARN] Download interrupted. Retrying in 20 seconds..."
    sleep 20
  done
}

echo "[INFO] Downloading teacher model:"
echo "       repo: $TEACHER_REPO"
echo "       dir : $TEACHER_DIR"
download_repo "$TEACHER_REPO" "$TEACHER_DIR"

if [[ "$DOWNLOAD_STUDENT_BASE" = "1" ]]; then
  echo "[INFO] Downloading student base model:"
  echo "       repo: $STUDENT_BASE_REPO"
  echo "       dir : $STUDENT_BASE_DIR"
  download_repo "$STUDENT_BASE_REPO" "$STUDENT_BASE_DIR"
fi

echo "[DONE] Models downloaded."
echo "Teacher model path: $TEACHER_DIR"
if [[ "$DOWNLOAD_STUDENT_BASE" = "1" ]]; then
  echo "Student base path : $STUDENT_BASE_DIR"
fi
