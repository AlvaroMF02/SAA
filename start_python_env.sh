#!/usr/bin/env bash

VENV_DIR="/home/albaro/python_envs/venv1"
source $VENV_DIR/bin/activate
jupyter lab --notebook-dir=$(pwd)

