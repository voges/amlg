#!/usr/bin/env bash

conda env export --no-builds | grep --invert-match "prefix" > environment.yml
