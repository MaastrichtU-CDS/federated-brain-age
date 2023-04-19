""" Util functions for the wrapper.
"""

import subprocess
import time

def run_command(command, success_message, logger=None, exit_code=None, sleep=120):
    """ Runs a bash command """
    return_code = None
    while return_code is None or (exit_code is not None and return_code != exit_code):
        process = subprocess.run(command, capture_output=True, check=False)
        return_code = process.returncode
        if logger:
            logger(success_message) if return_code == 0 else logger(process.stderr.decode("utf-8"))
        if exit_code is not None and return_code != exit_code:
            time.sleep(sleep)
    return return_code
