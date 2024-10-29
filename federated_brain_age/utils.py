""" Utility functions.
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from federated_brain_age.constants import *

def run_sql(db_client, sql_statement, parameters=None, fetch_one=False, fetch_all=False):
    """ Execute the sql query and retrieve the results
    """
    result = None
    db_client.execute(sql_statement, parameters)
    if fetch_all:
        result = db_client.fetchall()
    elif fetch_one:
        result = db_client.fetchone()
    db_client.execute("COMMIT")
    return result

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

def check_errors(results):
    """ Parse the errors obtained from the results.
    """
    output = None
    if any([ERROR in result for result in results]):
        output = [result[ERROR] for result in results if ERROR in result]
    return output

def validate_parameters(input, parameters):
    """ Identify the parameters missing.
    """
    missing_parameters = []
    for parameter in parameters.keys():
        if parameter not in input:
            missing_parameters.append(parameter)
        elif len(parameters[parameter].keys()) > 0:
            missing_parameters.extend(validate_parameters(input[parameter], parameters[parameter]))
    return missing_parameters

def np_array_to_list(array):
    """ Convert a numpy array to a list.
        Necessary to send the results back to the client.
    """
    parsed_list = []
    if type(array) is list:
        for element in array:
            if type(element) in [np.ndarray, list]:
                parsed_list.append(np_array_to_list(element))
            # tensorflow.python.framework.ops.EagerTensor
            elif tf.is_tensor(element):
                parsed_list.append(element.numpy().tolist())
            else:
                parsed_list.append(element)
    elif type(array) is np.ndarray:
        parsed_list = array.tolist()
    return parsed_list
