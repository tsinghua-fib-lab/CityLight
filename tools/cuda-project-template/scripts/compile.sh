#!/bin/bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" &>/dev/null && pwd)"

cd ${PROJECT_DIR}
mkdir -p build
cd build/

# config
cmake ..

make -j
