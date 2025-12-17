#!/bin/bash

# switch to venv
source venv/bin/activate

# Find and run all Python scripts in the figures folder
echo "Running all Python scripts in figures folder..."

find figures -name "*.py" -type f | while read script; do
    echo "Running $script..."
    cd "$(dirname "$script")"
    python3 "$(basename "$script")"
    cd - > /dev/null
done

echo "All scripts completed!"
