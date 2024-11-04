#!/bin/bash

# Array of jaxns versions to be installed
#declare -a jaxns_versions=("2.4.6" "2.4.7" "2.4.8" "2.4.10" "2.4.11" "2.4.12" "2.4.13" "2.5.0" "2.5.1" "2.5.2" "2.5.3" "2.6.0" "2.6.1" "2.6.2")
declare -a jaxns_versions=("2.5.1" "2.5.2" "2.5.3" "2.6.0" "2.6.1" "2.6.2")

# Path to your benchmark script
benchmark_script="main.py"

# Name for the conda environment
conda_env_name="jaxns_benchmarks_env"

# Function to create Conda environment and install jaxns
create_and_activate_env() {
  local version=$1
  echo "Creating Conda environment for jaxns version $version with Python 3.11..."
  conda create --name $conda_env_name python=3.11 -y
  eval "$(conda shell.bash hook)"
  conda activate $conda_env_name
  pip install jaxns==$version 'numpy<2'
}

# Function to tear down Conda environment
tear_down_env() {
  echo "Tearing down Conda environment..."
  conda deactivate
  conda env remove --name $conda_env_name
}

# Main loop to install each version, run benchmark, and tear down env
for version in "${jaxns_versions[@]}"; do
  create_and_activate_env $version
  python $benchmark_script
  tear_down_env
done

echo "Benchmarking complete."
