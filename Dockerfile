# vantage6 base image
FROM harbor.vantage6.ai/algorithms/algorithm-base

ARG PKG_NAME="federated_brain_age"

# Required for the psycopg2 dependency
RUN apt-get update
RUN apt-get install -y apt-utils gcc libpq-dev

# install the federated algorithm
COPY . /app
RUN pip install /app

ENV PKG_NAME=${PKG_NAME}

# Execute the docker wrapper when running the image
CMD python -c "from federated_brain_age.docker_wrapper import docker_wrapper; docker_wrapper('${PKG_NAME}')"
