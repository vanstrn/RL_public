#!/bin/bash
set -e
CMD_ARGS=("$@")

#Executing setup file in the PWD. (Need to mount files.)
LOCAL_SETUP=${PWD}/install/setup.bash
if [ -f "$LOCAL_SETUP" ]; then
    source $LOCAL_SETUP
fi

exec "${CMD_ARGS[@]}"
