# Vantage6 base image is no longer available
# FROM harbor.vantage6.ai/algorithms/algorithm-base
FROM pmateus/algorithm-base:1.0.0

ARG PKG_NAME="federated_brain_age"

# Required for the psycopg2 dependency
RUN apt-get update
RUN apt-get install -y apt-utils gcc libpq-dev

# install the federated algorithm
# requirements filed used for development
COPY ./requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY . /app
RUN pip install /app

ENV PKG_NAME=${PKG_NAME}

# Execute the docker wrapper when running the image
CMD python -c "from federated_brain_age.docker_wrapper_v6 import docker_wrapper; docker_wrapper('${PKG_NAME}')"
