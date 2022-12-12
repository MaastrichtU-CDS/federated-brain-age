""" Util functions for the wrapper.
"""

import subprocess

def run_command(command, success_message, logger=None):
    """ Runs a bash command """
    process = subprocess.run(command, capture_output=True, check=False)
    if logger:
        logger(success_message) if process.returncode == 0 else logger(process.stderr.decode("utf-8"))
    return process.returncode
