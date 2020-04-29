#!/usr/bin/env bash

function echo_step {
  echo "Step: $1"
}

# Remove virtual environment
echo_step "Deleting virtual environment."
conda remove --name sirius_env --all

