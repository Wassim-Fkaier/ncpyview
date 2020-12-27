#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:15:17 2020.
This module represente the main executer of the ncpyview program
"""
__author__ = "Wassim Fkaier"
__maintainer__ = "Wassim Fkaier"
__email__ = "Wassim Fkaier@outlook.com"
__status__ = "Dev"

import pkg_resources
import subprocess
from .readparam import parserarg

def main():
    """
    This is main function excecute the ncpyview program

    Returns
    -------
    None.

    """
    # Try to parse files paths
    ncfiles = parserarg.parsearg_func()["ncfiles"]
    # Get the path of the ncpyview_app.py script
    script_path = pkg_resources.resource_filename("ncpyview", "ncpyview_app.py")
    # if not parsed files paths
    if ncfiles is not None:
        subprocess.check_call(
            f"streamlit run {script_path} -- -f {','.join(ncfiles)}",
            shell=True,
            stderr=subprocess.STDOUT,
        )
    else:
        subprocess.check_call(
            f"streamlit run {script_path}",
            shell=True, stderr=subprocess.STDOUT
        )
