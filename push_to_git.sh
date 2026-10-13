#!/usr/bin/env bash
# push_to_git.sh
# ----------------
# Initializes (if needed) a git repo in this folder, commits the project
# files, and pushes them to a remote GitHub repository.
#
# Usage:
#   1. Create an empty repository on GitHub first (do NOT initialize it
#      with a README/.gitignore on GitHub's side).
#   2. Edit REMOTE_URL below to point to that repo, e.g.:
#        REMOTE_URL="https://github.com/<your-username>/diffusion-six.git"
#   3. Run:  bash push_to_git.sh

set -e  # stop on first error

REMOTE_URL="https://github.com/xiaopan1204/DTD_PROGRAM_2026_lucia.git"
BRANCH="main"

cd "$(dirname "$0")"   # make sure we run from this project's folder

# 1) Initialize git repo if this folder isn't one already
if [ ! -d ".git" ]; then
    echo "Initializing new git repository..."
    git init
    git branch -M "$BRANCH"
fi

# 2) Add the remote if it isn't configured yet
if ! git remote get-url origin >/dev/null 2>&1; then
    echo "Adding remote origin -> $REMOTE_URL"
    git remote add origin "$REMOTE_URL"
else
    echo "Remote 'origin' already configured: $(git remote get-url origin)"
fi

# 3) Stage and commit all files
git add .
if git diff --cached --quiet; then
    echo "Nothing new to commit."
else
    git commit -m "Add MNIST denoising project (CNN + U-Net, train/app scripts)"
fi

# 4) Push to GitHub
echo "Pushing to $REMOTE_URL ($BRANCH)..."
git push -u origin "$BRANCH"

echo "Done. Your project is now on GitHub."
