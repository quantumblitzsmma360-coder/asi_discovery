#!/bin/bash
# Setup script to deploy reflective neuralese to GitHub

REPO_NAME="asi-reflective-neuralese"
GH_USER="thamizhamudhuj"

# Create repo
gh repo create $GH_USER/$REPO_NAME --public --description "Reflective Neuralese ASI - Autonomous Self-Modifying Neural Architecture"

# Push workflows
git init
git add deploy/github_actions/
git commit -m "Add reflective neuralese GitHub Actions workflow (100 parallel workers)"
git branch -M main
git remote add origin https://github.com/$GH_USER/$REPO_NAME.git
git push -u origin main

# Enable Actions
gh api repos/$GH_USER/$REPO_NAME/actions/permissions -X PUT -f enabled=true
gh api repos/$GH_USER/$REPO_NAME/actions/permissions/workflow -X PUT -f default_workflow_permissions=write

echo "GitHub Actions deployed: 100 parallel workers running every 6 hours"
