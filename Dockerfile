FROM python:3.9 as base

ARG PACKAGE_NAME="lamini-datasets"

# Install Ubuntu libraries
RUN apt-get -yq update

# Install python packages
WORKDIR /app/${PACKAGE_NAME}
COPY ./requirements.txt /app/${PACKAGE_NAME}/requirements.txt
RUN pip install -r requirements.txt

# Copy all files to the container
COPY ./seed_tasks.jsonl /app/${PACKAGE_NAME}/seed_tasks.jsonl
COPY ./scripts /app/${PACKAGE_NAME}/scripts

COPY ./generate_data.py /app/${PACKAGE_NAME}/generate_data.py

WORKDIR /app/${PACKAGE_NAME}

RUN chmod a+x /app/${PACKAGE_NAME}/scripts/start.sh

ENV PACKAGE_NAME=$PACKAGE_NAME
ENTRYPOINT ["/app/lamini-datasets/scripts/start.sh"]


