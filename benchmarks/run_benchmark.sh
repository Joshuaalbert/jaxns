#!/bin/bash

# Array of jaxns versions to be installed
declare -a jaxns_versions=("2.3.0" "2.3.1" "2.3.2" "2.3.4" "2.4.0")

# Path to your benchmark script
benchmark_script="gh117.py"

# Function to create virtual environment and install jaxns
create_and_activate_env() {
  local version=$1
  echo "Creating virtual environment for jaxns version $version..."
  virtualenv temp_env
  source temp_env/bin/activate
  pip install jaxns==$version
}

# Function to tear down virtual environment
tear_down_env() {
  echo "Tearing down virtual environment..."
  deactivate
  rm -rf temp_env
}

# Main loop to install each version, run benchmark, and tear down env
for version in "${jaxns_versions[@]}"; do
  create_and_activate_env $version
  python $benchmark_script
  tear_down_env
done

echo "Benchmarking complete."
