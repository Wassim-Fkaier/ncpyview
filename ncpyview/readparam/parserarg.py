#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 11:11:02 2020

This module parse the arguments of the ncpyview program
"""
__author__ = "Wassim Fkaier"
__maintainer__ = "Wassim Fkaier"
__email__ = "Wassim Fkaier@outlook.com"
__status__ = "Dev"
__package_name__ = "ncpyview"

from argparse import ArgumentParser, RawTextHelpFormatter


def parsearg_func():

    """Argparse object: define which options are available and return the user 's choices for options and arguments"""

    examples = """ncpyview \n ncpyview -f file_name

    Examples:

    IMPORTANT: Check the configuration file etc/config.json to set the setup parameters 
    """

    parser = ArgumentParser(description=examples, formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "-f" "--ncfiles",
        dest="ncfiles",
        action="store",
        default=None,
        help="The netcdf file path",
        type=str,
    )

    args = parser.parse_args()

    args_dict = {}
    args_dict["ncfiles"] = args.ncfiles
    if args.ncfiles is not None:
        args.ncfiles = args_dict["ncfiles"].replace(" ", "")
        args_dict["ncfiles"] = args_dict["ncfiles"].split(",")

    return args_dict


if __name__ == "__main__":
    print(parsearg_func())
