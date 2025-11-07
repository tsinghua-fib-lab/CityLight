#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
import os
from pathlib import Path

TEMPLATE = Path(__file__).parent / "cuda-project-template"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=str,
                        help="the directory that will be generated with the starting cpp code")
    parser.add_argument("--name", required=True, type=str, help="project name")
    return parser.parse_args()


def replace_file(path, project_name):
    with open(path, "r") as fin:
        text = fin.read()
    text = text.replace("{{PROJECT_NAME}}", project_name)
    with open(path, "w") as fout:
        fout.write(text)


if __name__ == "__main__":
    args = parse_args()
    code_dir = Path(args.dir)
    project_name = args.name
    if code_dir.exists():
        print(
            f"Error: {code_dir} already exists, pleasing choose another directory")
        exit(1)
    shutil.copytree(TEMPLATE, code_dir)
    for path, _, file_list in os.walk(code_dir):
        for file_name in file_list:
            replace_file(Path(path)/file_name, project_name)
    print("create-cuda-project succeeded!")
