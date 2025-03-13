#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:12:37 2025

@author: gamal
"""

import gpm

username = "shadya.gamal@gmail.com"  # likely your mail
password = "shadya.gamal@gmail.com"  # likely your mail
gpm_base_dir = "/home/gamal/data/GPM"  # path to the directory where to download the data
gpm.define_configs(gpm_username=username, gpm_password=password, gpm_base_dir=gpm_base_dir)