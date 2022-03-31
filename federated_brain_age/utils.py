""" Utility functions.
"""
import os

from federated_brain_age.constants import *

def run_sql(db_client, sql_statement, parameters=None, fetch_all=False):
    """ Execute the sql query and retrieve the results
    """
    db_client.execute(sql_statement, parameters)
    if fetch_all:
        return db_client.fetchall()
    else:
        return db_client.fetchone()

def parse_error(error_message):
    """ Parse an error message.
    """
    return {
        ERROR: error_message 
    }

def folder_exists(path):
    """ Check if a folder exists.
    """
    return os.path.exists(path)

def get_parameter(parameter, parameters, default_parameters):
    """ Get parameter from the environmnet variables, otherwise
        use the default value provided.
    """
    return parameters[parameter] if parameter in parameters \
        else default_parameters[parameter]
