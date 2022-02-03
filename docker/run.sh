#!/bin/bash
RUN_DIR=$(dirname $(readlink -f $0))
XSOCK="/tmp/.X11-unix"
XAUTH="/tmp/.docker.xauth"

DATASET_DIR=""
CONFIG_DIR=""

function usage_exit {
  cat <<_EOS_ 1>&2
  Usage: run.sh -d DATASET_DIR -c CONFIG_DIR
  OPTIONS:
    -h, --help          Show this help
    -d, --dataset-dir   Specify the directory where the datasets are stored
    -c, --config-dir    Specify the directory where config file will be saved
_EOS_
  exit 1
}

while (( $# > 0 )); do
  if [[ $1 == "-h" ]] || [[ $1 == "--help" ]]; then
    usage_exit
  elif [[ $1 == "-d" ]] || [[ $1 == "--dataset-dir" ]]; then
    if [[ $2 == -* ]]; then
      echo "Invalid parameter"
      usage_exit
    fi
    DATASET_DIR=$2
    shift 2
  elif [[ $1 == "-c" ]] || [[ $1 == "--config-dir" ]]; then
    if [[ $2 == -* ]]; then
      echo "Invalid parameter"
      usage_exit
    fi
    CONFIG_DIR=$2
    shift 2
  else
    echo "Invalid parameter: $1"
    usage_exit
  fi
done

if [[ -z ${DATASET_DIR} ]]; then
  usage_exit
elif [[ -z ${CONFIG_DIR} ]]; then
  usage_exit
fi

DOCKER_VOLUME="${DOCKER_VOLUME} -v ${XSOCK}:${XSOCK}:rw"
DOCKER_VOLUME="${DOCKER_VOLUME} -v ${XAUTH}:${XAUTH}:rw"
DOCKER_VOLUME="${DOCKER_VOLUME} -v ${DATASET_DIR}:/root/datasets:ro"
DOCKER_VOLUME="${DOCKER_VOLUME} -v ${CONFIG_DIR}:/root/config:rw"

DOCKER_ENV="${DOCKER_ENV} -e XAUTHORITY=${XAUTH}"
DOCKER_ENV="${DOCKER_ENV} -e DISPLAY=$DISPLAY"
DOCKER_ENV="${DOCKER_ENV} -e TERM=xterm-256color"
DOCKER_ENV="${DOCKER_ENV} -e QT_X11_NO_MITSHM=1"

docker run \
    -it \
    --rm \
    ${DOCKER_VOLUME} \
    ${DOCKER_ENV} \
    h5dataloader-config
