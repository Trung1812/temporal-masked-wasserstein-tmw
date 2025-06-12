#!/bin/bash
# Google Colab environment setup script for TMW project
set -e

# Clone your repo (replace with your actual repo URL)
REPO_URL="https://github.com/YOUR_USERNAME/tmw_project.git"
PROJECT_ROOT="/content/tmw_project"

if [ ! -d "$PROJECT_ROOT" ]; then
  git clone "$REPO_URL" "$PROJECT_ROOT"
fi
cd "$PROJECT_ROOT"

# Install conda if needed (Colab usually has pip, not conda)
# Install Python dependencies
if [ -f environment.yml ]; then
  pip install -q pyyaml
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
else
  pip install -q pyyaml
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
fi

# Install additional useful packages for Colab
pip install -q optuna scikit-learn torch matplotlib pandas

# (Optional) Set up environment variables if needed
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Print project structure for verification
echo "Project structure:"
ls -R

echo "Setup complete. You are now in $PROJECT_ROOT."
