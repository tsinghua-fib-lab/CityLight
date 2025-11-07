#!/bin/bash

set -e

print_usage() {
  echo "Usage: $0 <language> <project-name>"
  echo "language: cpp|go|gin|python"
  echo "project-name: both the directory and the name of project (cpp only)"
}

if [ $# -eq 2 ] && [ $1 != "-h" ] ; then
    LANG=$1
    NAME=$2
else
    print_usage
    exit 1
fi

if [ ${LANG} != "cpp" ] && [ ${LANG} != "go" ] && [ ${LANG} != "gin" ] && [ ${LANG} != "python" ] ; then
    echo "Invalid language type ${LANG}"
    print_usage
    exit 1
fi

if [ -d ${NAME} ]; then
    echo "Directory ${NAME} exists, please remove it or use another project-name."
    exit 1
fi

# 创建临时文件夹
temp_dir=$(mktemp -d)
echo "create temp directory ${temp_dir}"
# git clone模板
pushd ${temp_dir}
git clone git@git.tsingroc.com:general/${LANG}-project-example.git
# 删除.git文件夹
echo "remove git directory"
rm -r ./${LANG}-project-example/.git/
if [ ${LANG} != "gin" ] ; then
    rmdir ./${LANG}-project-example/submodules/protos
fi
# 替换项目名称字符串
replace-string.py --dir ${LANG}-project-example --from-string ${LANG}-project-example --to-string ${NAME}
popd
# 复制到目标目录
mkdir -p ./${NAME}
cp -r ${temp_dir}/${LANG}-project-example/* ./${NAME}
rm -r ${temp_dir}
echo "remove temp directory ${temp_dir}"
# 初始化新项目文件夹
pushd ./${NAME}
git init
if [ ${LANG} != "gin" ] ; then
    git submodule add git@git.tsingroc.com:general/protos.git ./submodules/protos
    # 修改.gitmodules
    OLD_ROOT="git@git.tsingroc.com:"
    NEW_ROOT="..\/..\/"
    sed -i "s/${OLD_ROOT}/${NEW_ROOT}/g" .gitmodules
fi
popd

# 后续提示
echo "
"
echo "*************** SUCCEEDED ***************"
echo "PLEASE CHECK THE REMAINING ITEMS:"
if [ ${LANG} == "cpp" ] ; then
    echo "1. Modify protos/CMakeLists.txt to add needed protos"
    echo "2. Uncomment add_subdirectory(protos) in CMakeLists.txt if proto is needed"
elif [ ${LANG} == "go" ] ; then
    echo "1. Modify module name in go.mod"
    echo "2. Modify buf prefix in buf.gen.yaml"
    echo "3. Modify PROTO_PATH in scripts/init.sh"
    echo "4. Check release-binary in .gitlab-ci.yaml"
    echo "5. Check Dockerfile"
elif [ ${LANG} == "gin" ] ; then
    echo "1. Modify module name in go.mod and code"
    echo "2. Modify database default settings in common/"
elif [ ${LANG} == "python" ] ; then
    cd ./${NAME}
    mv ./${LANG}-project-example ./${NAME}
    echo "1. Modify proto related path in scripts/init.sh"
    echo "2. Check Dockerfile"
fi
echo "*************** SUCCEEDED ***************"
