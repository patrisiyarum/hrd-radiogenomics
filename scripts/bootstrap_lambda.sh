#!/usr/bin/env bash
#
# One-shot bootstrap for a fresh Lambda Labs A100 instance.
# Runs on the REMOTE machine after you SSH in, not on your laptop.
#
# Usage:
#   ssh ubuntu@<LAMBDA_IP>
#   curl -fsSL https://raw.githubusercontent.com/patrisiyarum/hrd-radiogenomics/main/scripts/bootstrap_lambda.sh | bash
#
# What it does:
#   1. Installs uv (fast Python package manager).
#   2. Clones the hrd-radiogenomics repo.
#   3. `uv sync --extra dev` — installs torch, MONAI, pydicom, nibabel, etc.
#   4. Sets up WandB logging if WANDB_API_KEY is present in env.
#   5. Prints the exact command to kick off the full training pipeline.
#
# Time budget: bootstrap itself takes ~5 minutes.

set -euo pipefail

echo "=== [1/4] installing uv ==="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for this shell
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
uv --version

echo ""
echo "=== [2/4] cloning hrd-radiogenomics ==="
if [ ! -d "$HOME/hrd-radiogenomics" ]; then
    git clone https://github.com/patrisiyarum/hrd-radiogenomics.git "$HOME/hrd-radiogenomics"
fi
cd "$HOME/hrd-radiogenomics"
git pull --rebase || true

echo ""
echo "=== [3/4] installing Python deps (uv sync --extra dev) ==="
uv sync --extra dev

echo ""
echo "=== [4/4] verifying CUDA + PyTorch see the GPU ==="
uv run python - <<'PY'
import torch
print(f"torch version:     {torch.__version__}")
print(f"CUDA available:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device 0:     {torch.cuda.get_device_name(0)}")
    print(f"CUDA compute cap:  {torch.cuda.get_device_capability(0)}")
PY

echo ""
echo "=== bootstrap complete ==="
echo ""
echo "Next step — kick off the full training pipeline:"
echo ""
echo "    cd ~/hrd-radiogenomics"
echo "    uv run snakemake --snakefile pipelines/Snakefile --cores 8"
echo ""
echo "Expect ~8-14 hours wall time total."
echo "Run inside tmux or screen so the pipeline survives SSH disconnects:"
echo ""
echo "    sudo apt-get install -y tmux  # if missing"
echo "    tmux new -s train             # start a session"
echo "    uv run snakemake --snakefile pipelines/Snakefile --cores 8"
echo "    # detach with Ctrl-B then D; reconnect later via: tmux attach -t train"
echo ""
echo "Cost check — run this periodically to see GPU utilisation:"
echo "    watch -n 5 nvidia-smi"
