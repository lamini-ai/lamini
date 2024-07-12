#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# x all executed commands are printed to the terminal
# u unset variables are errors
# a export all variables to the environment
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuoxa pipefail

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

LLAMA_ENVIRONMENT="LOCAL"

while getopts "e:" option; do
    case ${option} in
        e) LLAMA_ENVIRONMENT=$OPTARG;;
    esac
done

PYTHONPATH=$LOCAL_DIRECTORY/.. python3 -m unittest discover -v -s tests -p "*.py"
PYTHONPATH=$LOCAL_DIRECTORY/.. python3 -m unittest discover -v -s examples -p "*.py"
