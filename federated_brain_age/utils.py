""" Utility functions.
"""
import os

import pandas as pd

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

def read_csv(path, header = 0, column_names = None, separator = ",", 
    columns = None, filterKey = None, filter = None):
    """ Read a csv file and return the pandas dataframe.
        Optionally, it's possible to filter the data by a column.
    """
    data = pd.read_csv(path, header=header, names=column_names, sep=separator, usecols=columns)
    if filter and filterKey:
        data = data.loc[~data[filterKey].isin(filter)]
    return data
