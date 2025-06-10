#!/bin/bash

# Setup venv
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install current directory
pip install -e .
