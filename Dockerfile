FROM python:3.9 as base

ARG PACKAGE_NAME="llama-lang"
ARG LLAMA_ENVIRONMENT

# Install Ubuntu libraries
RUN apt-get -yq update

# Install python packages
WORKDIR /app/${PACKAGE_NAME}
COPY ./requirements.txt /app/${PACKAGE_NAME}/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
# Copy all files to the container
COPY ./scripts /app/${PACKAGE_NAME}/scripts
COPY ./test /app/${PACKAGE_NAME}/test

WORKDIR /app/${PACKAGE_NAME}

RUN chmod a+x /app/${PACKAGE_NAME}/scripts/start.sh

ENV PACKAGE_NAME=$PACKAGE_NAME
ENV LLAMA_ENVIRONMENT=$LLAMA_ENVIRONMENT

ENTRYPOINT /app/${PACKAGE_NAME}/scripts/start.sh -e ${LLAMA_ENVIRONMENT}
