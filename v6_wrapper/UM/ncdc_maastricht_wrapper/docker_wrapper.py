"""
Docker Wrapper
This module contains the `docker_wrapper` function for providing vantage6
algorithms with uniform input and output handling.
"""

import os
import pickle
import uuid
import re

import psycopg2

# from vantage6.tools.dispatch_rpc import dispact_rpc
from ncdc_maastricht_wrapper.dispatch_rpc import dispact_rpc
from vantage6.tools.util import info, warn
from vantage6.tools import deserialization, serialization
from vantage6.tools.data_format import DataFormat
from vantage6.tools.exceptions import DeserializationException
from typing import BinaryIO

from ncdc_maastricht_wrapper.open_shift_manager import create_tasks, login, run_task
# from ncdc_maastricht_wrapper.pod_manager import create_tasks, login, run_task
from federated_brain_age import master

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
    - pandas DataFrames
    :param module: module that contains the vantage6 algorithms
    :return:
    """
    info(f"wrapper for {module}")

    # Read the input from the mounted input file.
    input_file = os.environ["INPUT_FILE"]
    output_file = os.environ["OUTPUT_FILE"]
    info(f"Reading input file {input_file}")
    input_data = None
    try:
        input_data = load_input(input_file)
    except Exception as error:
        info("Error loading the input!")
        info(str(error))
        write_output(
            None,
            {
                "ERROR": f"Error loading the input: {str(error)}",
            },
            output_file
        )

    if input_data is not None:
        # Output configurations
        output_format = input_data.get('output_format', None)
        # Validate docker image
        info(os.getenv("ALLOWED_ALGORITHMS", ""))
        allowed_images = os.getenv("ALLOWED_ALGORITHMS", "").split(",")
        docker_image_name= input_data.get("algorithm_image")
        if not allowed_images or allowed_images == "":
            warn("All docker images are allowed on this Node!")
        else:
            info("Docker image validation")
            # check if it matches any of the regex cases
            allowed_image = False
            for regex_expr in allowed_images:
                expr = re.compile(regex_expr)
                if expr.match(docker_image_name):
                    allowed_image = True
            if not allowed_image:
                info("Docker image not allowed")
                write_output(
                    output_format,
                    {
                        "ERROR": f"Docker image {docker_image_name} is not allowed!",
                    },
                    output_file
                )
                return None
        info(f"Running docker image: {docker_image_name}")
        # Check if it's the master
        master = input_data.get("master")
        token = None
        db_client = None
        if master:
            token_file = os.environ["TOKEN_FILE"]
            info(f"Reading token file '{token_file}'")
            with open(token_file) as fp:
                token = fp.read().strip()

            info(f"Connecting to {os.getenv(PGDATABASE)}")
            try:
                # Directly connecting
                connection = psycopg2.connect(PGURI)
                db_client = connection.cursor()
                # Using the Manager class
                # db_client = PostgresManager(db_env_var=os.getenv(PGDATABASE))
                info("Successfully connected to the database")       
            except Exception as error:
                info("Database unavailable")
                info(str(error))
                write_output(
                    output_format,
                    {
                        "ERROR": f"DB connection error: {str(error)}",
                    },
                    output_file
                )
                return None

            info("Dispatching ...")
            input_data["kwargs"]["algorithm_image"] = input_data.get("algorithm_image")
            output = dispact_rpc(
                db_client,
                input_data,
                module,
                token
            )

            # Disconnecting from the database
            if db_client:
                info("Disconnecting from the database")
                db_client.close()
                connection.close()

            info(f"Writing output to {output_file}")
            output_format = input_data.get('output_format', None)
            write_output(output_format, output, output_file)
        else:
            info("Sending task to cluster")
            # Validate environment variables
            missing_env_var = [
                var for var in ["TASK_FOLDER", "INPUT_FILE", "OUTPUT_FILE", "OC_TOKEN", "OC_SERVER"] if \
                    var not in os.environ
            ]
            if len(missing_env_var) > 0:
                write_output(
                    output_format,
                    {
                        "ERROR": f"Missing environment variables: {', '.join(missing_env_var)}",
                    },
                    output_file
                )
            # Send the tasks to the gpu cluster.
            login(os.environ["OC_TOKEN"], os.environ["OC_SERVER"])
            # Create the task id
            task_id = str(uuid.uuid1())
            # Build the tasks
            tasks = create_tasks(
                os.environ["TASK_FOLDER"],
                os.environ["INPUT_FILE"],
                os.environ["OUTPUT_FILE"],
                os.getenv("VOLUME"),
                #task_id,
                "brain-age-algorithm",
                input_data.get("algorithm_image")
            )
            # Create the output file
            write_output(
                output_format, {"INFO": "placeholder"}, output_file
            )
            # Run each task
            for task in tasks:
                retry = 0
                while 0 <= retry < 3:
                    try:
                        #task_id if task["task"] != "run-algorithm-app" else "brain-age-algorithm",
                        run_task(
                            #task_id if task["task"] != "run-algorithm-app" else "brain-age-algorithm",
                            "brain-age-algorithm",
                            task,
                        )
                        warn(f"Task {task_id} completed")
                        retry = -1
                    except Exception as error:
                        warn(f"Error running task {task['task']}: {str(error)}")
                        retry += 1
            input_data["method"] = "node"
            dispact_rpc(None, input_data, module, token)

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
