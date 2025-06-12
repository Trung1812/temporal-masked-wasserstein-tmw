#!/bin/bash
# Kaggle environment setup script for TMW project
set -e

# Clone your repo (replace with your actual repo URL)
REPO_URL="https://github.com/YOUR_USERNAME/tmw_project.git"
PROJECT_ROOT="/kaggle/working/tmw_project"

if [ ! -d "$PROJECT_ROOT" ]; then
  git clone "$REPO_URL" "$PROJECT_ROOT"
fi
cd "$PROJECT_ROOT"

# Install Python dependencies
pip install --upgrade pip
pip install pyyaml
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

# Install additional useful packages for Kaggle
pip install optuna scikit-learn torch matplotlib pandas

# (Optional) Set up environment variables if needed
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Print project structure for verification
echo "Project structure:"
ls -R

echo "Setup complete. You are now in $PROJECT_ROOT."
