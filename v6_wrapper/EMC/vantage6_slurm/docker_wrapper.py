# Modified from https://gitlab.com/UM-CDS/projects/memorabel

"""
Docker Wrapper
This module contains the `docker_wrapper` function for providing vantage6
algorithms with uniform input and output handling.
"""

import os
import pickle
import uuid
import subprocess
import json
from time import sleep

from vantage6.tools.dispatch_rpc import dispatch_rpc
from vantage6.tools.util import info
from vantage6.tools import deserialization, serialization
from vantage6.tools.data_format import DataFormat
from vantage6.tools.exceptions import DeserializationException
from typing import BinaryIO

try:
    import psycopg2
except ImportError:
    master_image = False
else:
    master_image = True


_DATA_FORMAT_SEPARATOR = '.'
_MAX_FORMAT_STRING_LENGTH = 10

PGDATABASE = "PGDATABASE"
PGURI = "postgresql://"
DATABASE_URI = "DATABASE_URI"
DATABASE_PASSWORD = "DATABASE_PASSWORD"
DATABASE_PORT = "DATABASE_PORT"
DATABASE_USER = "DATABASE_USER"
DATABASE_NAME = "DATABASE_NAME"


class ConnectionSettings:
    def __init__(self, settings_path: str):
        connection_settings = json.load(open(settings_path, "r"))
        self.user = connection_settings["slurm_user_name"]
        self.login_node = connection_settings["slurm_login_node"]
        self.workdir_prefix = connection_settings["slurm_workdir_prefix"]
        self.singularity_exe = connection_settings["slurm_singularity_exe"]
        self.partitions = connection_settings["slurm_partitions"]
        self.node = connection_settings.get("slurm_node", "")
        self.node_exclude = connection_settings.get("slurm_node_exclude", "")
        self.env_file = connection_settings["slurm_env_file"]
        self.allowed_algorithms = connection_settings["allowed_algorithms"]
        self.bind_data_folder = connection_settings["singularity_bind_data_folder"]
        self.bind_images_folder = connection_settings["singularity_bind_images_folder"]
        self.bind_model_folder = connection_settings["singularity_bind_model_folder"]

    def check_algorithm_whitelist(self, algorithm_image: str):
        print(algorithm_image)
        print(self.allowed_algorithms)
        if algorithm_image not in self.allowed_algorithms:
            return False
        return True


def check_status(job_id: str, user: str, login_node: str) -> bool:
    command = ["ssh", f"{user}@{login_node}", "sacct", "-j", str(job_id), "--format", "State"]
    status_check = subprocess.run(command, capture_output=True)
    print(status_check.returncode)
    print(status_check.stdout)
    print(status_check.stderr)
    info("Within check status")
    if status_check.returncode != 0:
        # retry in a minute
        sleep(60)
        status_check = subprocess.run(command, capture_output=True)
        print(status_check.returncode)
        print(status_check.stdout)
        print(status_check.stderr)
        if status_check.returncode != 0:
            raise RuntimeError("Unable to check status!")
    status = status_check.stdout.split()
    print(status)
    if len(status) < 3:
        raise RuntimeError("Remote job id seems to be incorrect!")
    if b"PENDING" in status[2:] or b"RUNNING" in status[2:]:
        return True
    if all([status_state == b"COMPLETED" for status_state in status[2:]]):
        return False
    raise RuntimeError(f"Job seems to have failed: {status}")


def setup_and_copy_data(conn: ConnectionSettings, task_id: str, input_file: str, token_file: str, database_uri: str):
    # create workdir on slurm
    workdir_creation = subprocess.run(["ssh", f"{conn.user}@{conn.login_node}",
                                       "mkdir", f"{conn.workdir_prefix}/{task_id}"])
    if workdir_creation.returncode != 0:
        raise RuntimeError("Workdir  creation failed!")

    # copy input file, token file and data or database_uri (to be implemented) to slurm
    input_transfer = subprocess.run(["scp", input_file,
                                     f"{conn.user}@{conn.login_node}:{conn.workdir_prefix}/{task_id}/INPUT"])
    token_transfer = subprocess.run(["scp", token_file,
                                     f"{conn.user}@{conn.login_node}:{conn.workdir_prefix}/{task_id}/TOKEN"])
    data_transfer = subprocess.run(["scp", database_uri,
                                    f"{conn.user}@{conn.login_node}:{conn.workdir_prefix}/{task_id}/DATA"])
    dummy_file_creation = subprocess.run(["ssh", f"{conn.user}@{conn.login_node}",
                                          "touch", f"{conn.workdir_prefix}/{task_id}/OUTPUT"])
    if (input_transfer.returncode != 0 or token_transfer.returncode != 0
            or data_transfer.returncode != 0 or dummy_file_creation.returncode != 0):
        raise RuntimeError("File transfer failed!")


def submit_job(conn: ConnectionSettings, task_id: str, algorithm_image: str):
    submission_command = ["ssh", f"{conn.user}@{conn.login_node}"]

    # Build Slurm command
    node_exclude_part = f"--exclude={conn.node_exclude}"
    node_part = f"--nodelist={conn.node}"
    if not conn.node and conn.node_exclude:
        node_part = ""
    elif conn.node and not conn.node_exclude:
        node_exclude_part = ""
    elif not conn.node and not conn.node_exclude:
        node_part = ""
        node_exclude_part = ""
    else:
        pass

    slurm_command = ["sbatch", f"--partition={conn.partitions}", node_part, node_exclude_part,
                     f"--job-name=pht_{task_id}",
                     f"--export-file={conn.env_file}",
                     f"--chdir={conn.workdir_prefix}/{task_id}", "--time=1440", "--cpus-per-task=1",
                     "--gres=gpu:1", "--mem=32G"]
    singularity_command = ["--wrap="
                           + f"'{conn.singularity_exe} run "
                           + "--containall "
                           + "--env INPUT_FILE=/INPUT,OUTPUT_FILE=/OUTPUT,TOKEN_FILE=/TOKEN,DATABASE_URI=/DATA "
                           + "--env LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu "
                           + "--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES "
                           + f"--env-file {conn.env_file} "
                           + "--nv "
                           + f"--bind INPUT:/INPUT,OUTPUT:/OUTPUT,DATA:/DATA,TOKEN:/TOKEN,"
                           + f"{conn.bind_data_folder}:/data_folder,{conn.bind_images_folder}:/images_folder,"
                           + f"{conn.bind_model_folder}:/model_folder "
                           + f"\"'\"docker://{algorithm_image}\"'\"'"
                           ]
    submission_command.extend(slurm_command)
    submission_command.extend(singularity_command)

    submission = subprocess.run(submission_command, capture_output=True)
    print(submission.stdout)
    print(submission.stderr)
    if submission.returncode != 0:
        raise RuntimeError("Job submission failed!")
    job_id = submission.stdout.split()[-1]
    job_id = job_id.decode('utf-8')

    return job_id


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
    # Load environment variables
    input_file = os.environ["INPUT_FILE"]
    token_file = os.environ["TOKEN_FILE"]
    output_file = os.environ["OUTPUT_FILE"]
    database_uri = os.environ["DATABASE_URI"]

    # Read the input from the mounted input file.
    info(f"Reading input file {input_file}")
    input_data = load_input(input_file)
    master = input_data.get("master")
    output_format = input_data.get('output_format', None)

    # Check if it's the master
    token = None
    if master and master_image:
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
        print(input_data.get("algorithm_image"))
        input_data["kwargs"]["algorithm_image"] = input_data.get("algorithm_image")
        output = dispatch_rpc(
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
        write_output(output_format, output, output_file)
    else:
        conn = ConnectionSettings("/root/connection_settings.json")

        # Check algorithm whitelist
        algorithm_image = input_data.get("algorithm_image")
        allowed_bool = conn.check_algorithm_whitelist(algorithm_image)
        if not allowed_bool:
            write_output(
                output_format,
                {
                    "ERROR": f"Algorithm container {algorithm_image} is not allowed on this node.",
                },
                output_file
            )
            return None

        task_id = str(uuid.uuid1())
        info(f"Task id: {task_id}")
        setup_and_copy_data(conn, task_id, input_file, token_file, database_uri)
        job_id = submit_job(conn, task_id, algorithm_image)
        info(f"Slurm job: {job_id}")
        # periodically check if still running
        sleep(30)
        status_bool = check_status(job_id, conn.user, conn.login_node)
        info("Before while loop")
        while status_bool:
            # wait 2 minutes before checking again
            info("Inside check status while loop")
            sleep(120)
            status_bool = check_status(job_id, conn.user, conn.login_node)
        # copy output back to node
        output_transfer = subprocess.run(["scp",
                                          f"{conn.user}@{conn.login_node}:{conn.workdir_prefix}/{task_id}/OUTPUT",
                                          output_file])
        print("Transfer successful")
        if output_transfer.returncode != 0:
            raise RuntimeError("Copying output failed!")


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
