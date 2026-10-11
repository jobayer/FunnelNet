#!/bin/bash

ENV_NAME=$(grep "^name:" environment.yml | cut -d' ' -f2)

activate_env() {
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    if [[ $CONDA_DEFAULT_ENV != "$ENV_NAME" ]]; then
        echo "Failed to activate conda environment: $ENV_NAME"
        exit 1
    fi
}

if command -v conda &>/dev/null; then
    activate_env
    echo "Running train.py on \"$ENV_NAME\" environment"
    echo -e "Using Python version: $(python --version)\n"
    python train.py "$@"
    conda deactivate
else
    echo "Conda not found. Please install conda and create a conda environment using the environment.yml file using the following command:"
    echo "\"conda env create -f environment.yml\""
    exit 1
fi