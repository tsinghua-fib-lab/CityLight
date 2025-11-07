#!/bin/bash

set -e

print_usage() {
  echo "Usage: $0 [-d] <data-dir>"
}

if [ $# -eq 1 ] && [ $1 != "-d" ]; then
  DIR=$1
  DAEMON=0
elif [ $# -eq 2 ] && [ $1 = "-d" ]; then
  DIR=$2
  DAEMON=1
else
  print_usage
  exit 1
fi

DATAPATH=${DIR}/etcd
LOGPATH=${DIR}/etcd.log

echo "$0: create ${DATAPATH}"
mkdir -p ${DATAPATH}
if [ ${DAEMON} -eq 0 ]; then
  # foreground
  echo "$0: run etcd data-dir=${DATAPATH}"
  etcd --data-dir ${DATAPATH} --log-level=debug
else
  # background
  echo "$0: run etcd data-dir=${DATAPATH} log=${LOGPATH}"
  etcd --data-dir ${DATAPATH} --log-level=debug >>${LOGPATH} 2>&1 &
fi
