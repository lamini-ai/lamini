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

function get_version {
    local version=$(cat ../pyproject.toml | grep version | sed 's/version[^"]*//g' | sed 's/"//g')
    echo $version
}

function get_previous_revision {
    local revision=$(curl "https://pypi.org/pypi/lamini/json" | python3 -c "import sys, json, re; project = json.load(sys.stdin); versions = list(sorted(project['releases'], key=lambda x : tuple(int(component) for component in x.split('.') if all([x.isnumeric() for x in component]) ))); latest_version = versions[-1]; latest_release = project['releases'][latest_version]; filenames = [re.findall('-\d+-|$', release['filename'])[0].strip('-') for release in latest_release] ; print(max([0] + [int(version) for version in filenames if len(version) > 0]))")
    echo $revision
}

function get_old_name {
    local version="$(get_version)"
    local old_name="lamini-$version-py3-none-any.whl"
    echo $old_name
}

function get_new_name {
    local previous_revision="$(get_previous_revision)"
    local next_revision=$((previous_revision + 1))
    local version="$(get_version)"
    local new_name="lamini-$version-$next_revision-py3-none-any.whl"
    echo $new_name
}

# build
cd $LOCAL_DIRECTORY/..
mkdir -p lamini
touch lamini/__init__.py
pip3 install wheel build
python3 -m build --wheel
cd $LOCAL_DIRECTORY

old_name="$(get_old_name)"
new_name="$(get_new_name)"

echo "old version $old_name"
echo "new version $new_name"

# mv the build to the new minor version
mv $LOCAL_DIRECTORY/../dist/$old_name $LOCAL_DIRECTORY/../dist/$new_name

# upload it

