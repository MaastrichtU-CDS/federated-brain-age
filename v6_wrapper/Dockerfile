# This specifies our base image. This base image contains some commonly used
# dependancies and an install from all vantage6 packages. You can specify a
# different image here (e.g. python:3). In that case it is important that
# `vantage6-client` is a dependancy of you project as this contains the wrapper
# we are using in this example.
FROM harbor.vantage6.ai/algorithms/algorithm-base

# Change this to the package name of your project. This needs to be the same
# as what you specified for the name in the `setup.py`.
ARG PKG_NAME="ncdc_maastricht_wrapper"

RUN apt-get update
#RUN apt-get install -y apt-utils gcc libpq-dev wget iputils-ping tar
RUN apt-get install -y wget tar apt-utils gcc libpq-dev

# Install the oc client
RUN wget https://mirror.openshift.com/pub/openshift-v4/clients/oc/latest/linux/oc.tar.gz && tar xvf oc.tar.gz
RUN mv oc kubectl /usr/local/bin/

# This will install your algorithm into this image.
COPY ./requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY . /app
RUN pip install --no-cache-dir /app

# This will run your algorithm when the Docker container is started. The
# wrapper takes care of the IO handling (communication between node and
# algorithm). You dont need to change anything here.
ENV TEMPLATES_FOLDER_PATH="/app/ncdc_maastricht_wrapper/json_templates"
ENV PKG_NAME=${PKG_NAME}
CMD python -c "from ncdc_maastricht_wrapper.docker_wrapper import docker_wrapper; docker_wrapper('${PKG_NAME}')"
