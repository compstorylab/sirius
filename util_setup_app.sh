#!/usr/bin/env bash

function command_exists {
  if [ -x "$(command -v $1)" ]; then
    echo "Found $1 at $(command -v $1)"
  else
    echo "Error: Could not find $1" >&2
    exit 1
  fi
}

function echo_step {
  echo "Step: $1"
}

# Is there Python 3
command_exists "python3"

# Is there Conda?
command_exists "conda"

# Is there Git?
command_exists "git"

# Create the env using conda
echo_step "Create the conda environment"
conda create -y -n sirius_env python=3.7 pip

# Start it up
echo_step "Activate the conda environment"
source /anaconda3/etc/profile.d/conda.sh
conda activate sirius_env

# Add conda packages
echo_step "Add Pip to the conda environment."
conda install -y -n sirius_env pip
echo_step "Add Plotly to conda environment."
conda install -y -n sirius_env -c plotly plotly-orca

# Add Pip packages
echo_step "Install requirements using Pip."
pip install -r requirements.txt
echo_step "Install sirius as a package."
pip install -e .

# Create and/or append to the environment file
echo_step "Create the env file."
echo "SIRIUS_SETTINGS_SECRET_KEY=$(openssl rand -base64 66)" >> .env

# Setup database for Django
echo_step "Setup the database for Django."
python manage.py makemigrations
python manage.py migrate

echo_step "Start the conda environment"
echo "Run the following command: 'conda activate sirius_env'"