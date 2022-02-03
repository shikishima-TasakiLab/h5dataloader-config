#!/bin/bash
RUN_DIR=$(dirname $(readlink -f $0))

docker build \
    -t h5dataloader-config \
    ${RUN_DIR}
