#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import subprocess
import os
import sys

def test_identification():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    try:
        output = subprocess.check_output('./examples/identify_kuka_lwr4.sh', cwd=path, shell=True, env=os.environ.copy())
    except subprocess.CalledProcessError as e:
        print (e.output)
        sys.exit(e.returncode)

if __name__ == '__main__':
    test_identification()
