#!/usr/bin/env bash

git_root="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -n "${git_root}" ]; then
    echo "Git repository root found at: ${git_root}"
else
    echo "Error: Git repository root not found." >&2
    exit 1
fi

environment_file="${git_root}/environment.yml"
conda env export --no-builds | grep --invert-match "prefix" > "${environment_file}"
echo "Updated environment file: ${environment_file}"
