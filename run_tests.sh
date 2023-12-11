#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/mech khulnasoft/mech:latest python3 -m pytest aikit_mech_tests/

