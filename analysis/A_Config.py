#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to configure GPM API settings.

@author: shadya
"""

import gpm # type: ignore

def main():
    """
    Set up GPM API configuration.
    """

    username = "shadya.gamal@gmail.com"  # likely your mail
    password = "shadya.gamal@gmail.com"  # likely your mail
    gpm_base_dir = "/home/gamal/data/GPM"  # path to the directory where to download the data

    gpm.define_configs(gpm_username=username, gpm_password=password, gpm_base_dir=gpm_base_dir)

    print("Configuration completed.")
    return None

if __name__ == "__main__":
    main()