#!/bin/zsh
# Deploy simulacra backend to UCL GPU machine (mallard-l)
# Usage: ./deploy_ucl.sh

REMOTE="ucl-gpu"
REMOTE_DIR="/cs/student/projects1/2022/osrivast/simulacra/backend"

echo "Syncing backend to ${REMOTE}:${REMOTE_DIR}..."

rsync -avz \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'runs/' \
  --exclude '.venv/' \
  backend/ "${REMOTE}:${REMOTE_DIR}/"

echo "Done. Now SSH in and start training:"
echo "  ssh ucl-gpu"
echo "  cd ${REMOTE_DIR}"
echo "  tmux new -s v4"
echo "  uv run python scripts/train_v4_highalt.py"
