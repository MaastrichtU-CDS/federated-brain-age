"""
Docker Wrapper
This module contains the `docker_wrapper` function for providing vantage6
algorithms with uniform input and output handling.
"""

import os
import pickle
import time

import psycopg2

from vantage6.tools.dispatch_rpc import dispact_rpc
from vantage6.tools.util import info, warn
from vantage6.tools import deserialization, serialization
from vantage6.tools.data_format import DataFormat
from vantage6.tools.exceptions import DeserializationException
from typing import BinaryIO

from federated_brain_age.constants import ERROR

# from sshtunnel import SSHTunnelForwarder

_DATA_FORMAT_SEPARATOR = '.'
_MAX_FORMAT_STRING_LENGTH = 10

PGDATABASE = "PGDATABASE"
PGURI = "postgresql://"
DATABASE_URI = "DATABASE_URI"
DATABASE_PASSWORD = "DATABASE_PASSWORD"
DATABASE_PORT = "DATABASE_PORT"
DATABASE_USER = "DATABASE_USER"
DATABASE_NAME = "DATABASE_NAME"

def docker_wrapper(module: str):
    """
    Wrap an algorithm module to provide input and output handling for the
    vantage6 infrastructure.
    Data is received in the form of files, whose location should be specified
    in the following environment variables:
    - `INPUT_FILE`: input arguments for the algorithm
    - `OUTPUT_FILE`: location where the results of the algorithm should be
      stored
    - `TOKEN_FILE`: access token for the vantage6 server REST api
    - `DATABASE_URI`: either a database endpoint or path to a csv file.
    The wrapper is able to parse a number of input file formats. The available
    formats can be found in `vantage6.tools.data_format.DataFormat`. When the
    input is not pickle (legacy), the format should be specified in the first
    bytes of the input file, followed by a '.'.
    It is also possible to specify the desired output format. This is done by
    including the parameter 'output_format' in the input parameters. Again, the
    list of possible output formats can be found in
    `vantage6.tools.data_format.DataFormat`.
    It is still possible that output serialization will fail even if the
    specified format is listed in the DataFormat enum. Algorithms can in
    principle return any python object, but not every serialization format will
    support arbitrary python objects. When dealing with unsupported algorithm
    output, the user should use 'pickle' as output format, which is the
    default.
    The other serialization formats support the following algorithm output:
    - built-in primitives (int, float, str, etc.)
    - built-in collections (list, dict, tuple, etc.)
    :param module: module that contains the vantage6 algorithms
    :return:
    """
    info(f"wrapper for {module}")

    # read input from the mounted inputfile.
    input_file = os.environ["INPUT_FILE"]
    info(f"Reading input file {input_file}")

    input_data = load_input(input_file)
    output_file = os.environ["OUTPUT_FILE"]
    output_format = input_data.get('output_format', None)

    db_client = None
    token = None
    if input_data.get("master"):
        token_file = os.environ["TOKEN_FILE"]
        info(f"Reading token file '{token_file}'")
        with open(token_file) as fp:
            token = fp.read().strip()
    else:
        # Nodes running the algorithm
        # - Get the database client
        info(f"Connecting to {os.getenv(PGDATABASE)}")
        try:
            connection = psycopg2.connect(PGURI)
            db_client = connection.cursor()
            info("Successfully connected to the database")       
        except Exception as error:
            info("Database unavailable")
            info(str(error))
            # write_output(
            #     output_format,
            #     {
            #         ERROR: f"DB connection error: {str(error)}",
            #     },
            #     output_file
            # )
            # return None

    # make the actual call to the method/function
    info("Dispatching ...")
    output = dispact_rpc(db_client, input_data, module, token)

    # Disconnecting from the database
    if db_client:
        info("Disconnecting from the database")
        db_client.close()
        connection.close()

    # write output from the method to mounted output file. Which will be
    # transfered back to the server by the node-instance.
    info(f"Writing output to {output_file}")
    write_output(output_format, output, output_file)

def write_output(output_format, output, output_file):
    """
    Write output to output_file using the format from output_format.
    If output_format == None, write output as pickle without indicating format (legacy method)
    :param output_format:
    :param output:
    :param output_file:
    :return:
    """
    with open(output_file, 'wb') as fp:
        if output_format:
            # Indicate output format
            fp.write(output_format.encode() + b'.')

            # Write actual data
            output_format = DataFormat(output_format.lower())
            serialized = serialization.serialize(output, output_format)
            fp.write(serialized)
        else:
            # No output format specified, use legacy method
            fp.write(pickle.dumps(output))

def load_input(input_file):
    """
    Try to read the specified data format and deserialize the rest of the
    stream accordingly. If this fails, assume the data format is pickle.
    :param input_file:
    :return:
    """
    with open(input_file, "rb") as fp:
        try:
            input_data = _read_formatted(fp)
        except DeserializationException:
            info('No data format specified. '
                 'Assuming input data is pickle format')
            fp.seek(0)
            try:
                input_data = pickle.load(fp)
            except pickle.UnpicklingError:
                raise DeserializationException('Could not deserialize input')
    return input_data

def _read_formatted(file: BinaryIO):
    data_format = str.join('', list(_read_data_format(file)))
    data_format = DataFormat(data_format.lower())
    return deserialization.deserialize(file, data_format)

def _read_data_format(file: BinaryIO):
    """
    Try to read the prescribed data format. The data format should be specified
    as follows: DATA_FORMAT.ACTUAL_BYTES. This function will attempt to read
    the string before the period. It will fail if the file is not in the right
    format.
    :param file: Input file received from vantage infrastructure.
    :return:
    """
    success = False

    for i in range(_MAX_FORMAT_STRING_LENGTH):
        try:
            char = file.read(1).decode()
        except UnicodeDecodeError:
            # We aren't reading a unicode string
            raise DeserializationException('No data format specified')

        if char == _DATA_FORMAT_SEPARATOR:
            success = True
            break
        else:
            yield char

    if not success:
        # The file didn't have a format prepended
        raise DeserializationException('No data format specified')
    