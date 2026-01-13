#!/bin/bash
# Start Dashboard from project root
cd "$(dirname "$0")/.."
python -m dashboard.app
