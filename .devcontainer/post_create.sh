#!/usr/bin/env bash

# Remove all editor settings for Python from the VS Code remote settings.
sed --in-place '/"\[python\]": {/,/}/d' /home/vscode/.vscode-server/data/Machine/settings.json

# Install additional packages.
sudo apt-get update && sudo apt-get install --yes shellcheck

# Install Python packages.
pip install --disable-pip-version-check --requirement .devcontainer/requirements_other.txt
pip install --disable-pip-version-check --requirement .devcontainer/requirements_torch.txt
