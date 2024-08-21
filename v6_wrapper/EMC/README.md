NEVER PUSH THE BUILD IMAGE TO A REPOSITORY, IT CONTAINS A PRIVATE KEY!!!!

## setup
First, copy the ssh key into the id_rsa file and make sure the permissions are set to 600.
Copy the host fingerprint into the known_hosts file and fill in the connection_settings.json:

```json
{
    "slurm_login_node": "The ip adress to the slurm login node.",
    "slurm_user_name": "The username to use for logging into the slurm cluster.",
    "slurm_workdir_prefix": "The path to the directory (in the slurm cluster) inside of which the tasks will be run.",
    "slurm_singularity_exe": "The path to the singularity executable (on the slurm cluster).",
    "slurm_partitions": "The partitions used to submit the jobs to on the slurm cluster.",
    "slurm_env_file": "A file (on the slurm cluster) containing additional environment variables (eg. XNAT credentials) to be used inside the singularity instances.",
    "slurm_node": "Nodes to select with --nodelist",
    "slurm_node_exclude": "Nodes to exclude with --exclude",
    "singularity_bind_data_folder": "Folder where dataset.csv is found",
    "singularity_bind_images_folder": "Folder where the images are found",
    "singularity_bind_model_folder": "Folder the model can be found",
    "allowed_algorithms": ["pmateus/brain-age-gpu:1.0.88"]
}
```

Then build the docker image:
```bash 
chmod 600 id_rsa
docker build . --tag vantage6_slurm:0.1.0
```

Make the node configuration as usual then add the following section to the relevant environment:

```yml
...
environments:
    ...
    test:
        ...
        docker_images_placeholders:
            gpu_image: vantage6_slurm:0.1.0
```

In order for this to work, you will need a special version of vantage6-node: pmateus/vantage6-node-whitelisted:2.0.0 
The node must be started using the following command in order to use this image:

```bash
vnode start --name config_name --image pmateus/vantage6-node-whitelisted:2.0.0 --environment test
```

## running a task
Run a task like this:

```python
from vantage6.client import UserClient

client = UserClient("http://localhost", 1337, "/api") #set the proper url and port for the server
client.authenticate("root", "root") # set the proper username and password
client.setup_encryption(None)

task = client.task.create(
  1, # collab id
  [1], # org id
  "my_task", # task name
  "gpu_image",
  "blah", # description
  { # inputs
      "master": "true", 
      "method":"master", 
      ["harbor2.vantage6.ai/demo/average", "average_partial"], # actual algorithm image, the actual function to run
      "kwargs":  {"column_name": "age"}}
)
```



