#!/usr/bin/env bash
set -e

if ! command -v java >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y default-jre-headless
fi

python3 -m pip install -r /autograder/source/requirements.txt
