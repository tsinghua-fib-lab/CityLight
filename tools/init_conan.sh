#!/bin/bash

## keep this script re-runnable
## just call this in compile script without any condition check
## TODO: should rename as conan_config.sh, init_xxx should only be run one time
set -e

CONAN_HOME=${CONAN_USER_HOME}

# check conan_user_home
if [[ -z "${CONAN_USER_HOME}" ]]; then
  echo "$0: CONAN_USER_HOME is not set, use default value ~"
  CONAN_HOME=${HOME}
else
  echo "$0: detect CONAN_USER_HOME=${CONAN_HOME}"
  echo "$0: create ${CONAN_HOME}"
  mkdir -p ${CONAN_HOME}
fi

# check if conan init
if [ -d "${CONAN_HOME}/.conan/" ]
then
  echo "conan has been inited."
else
  echo "init conan with new profile"
  conan profile new default --detect
  conan remote add gitlab https://git.tsingroc.com/api/v4/packages/conan
fi

# re-config conan settings
# new settings can be added here
conan profile update settings.compiler.libcxx=libstdc++11 default
conan profile update settings.compiler.cppstd=17 default

echo "if there is auth problem, run the following commands to login gitlab conan"
echo "    conan user -p <personal_token> <gitlab_username> -r gitlab"
