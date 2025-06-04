#!/usr/bin/env python3
"""
Script to configure GPM API settings.

@author: shadya
"""

import gpm  # type: ignore


def main():
    """
    Set up GPM API configuration.
    """
    username = "shadya.gamal@gmail.com"  # likely your mail
    password = "shadya.gamal@gmail.com"  # likely your mail
    gpm_base_dir = "/ltenas2/data/GPM"  # path to the directory where to download the data

    gpm.define_configs(username_pps=username, password_pps=password, base_dir=gpm_base_dir)

    print("Configuration completed.")


if __name__ == "__main__":
    main()


