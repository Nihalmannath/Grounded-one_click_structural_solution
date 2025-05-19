#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launch script for the integrated workflow
----------------------------------------
This script runs the integrated workflow from the src directory.
"""

import os
import sys

# Add the src directory to the Python path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

# Import and run the workflow
from integrated_workflow import run_gui

if __name__ == "__main__":
    run_gui()
