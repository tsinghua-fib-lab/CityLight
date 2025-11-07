#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
import os
from pathlib import Path

TEMPLATE = Path(__file__).parent / "cpp-project-template"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        required=True,
        type=str,
        help="the directory that will be generated with the starting cpp code",
    )
    parser.add_argument(
        "--from-string", required=True, type=str, help="original string"
    )
    parser.add_argument("--to-string", required=True, type=str, help="target string")
    return parser.parse_args()


def replace_file(path, from_string: str, to_string: str):
    with open(path, "r") as fin:
        text = fin.read()
    text = text.replace(from_string, to_string)
    with open(path, "w") as fout:
        fout.write(text)


if __name__ == "__main__":
    args = parse_args()
    for root, _, file_list in os.walk(args.dir):
        for file_name in file_list:
            path: Path = Path(root) / file_name
            if path.is_file():
                print(f"[replace-string] Go to {path}")
                replace_file(path, args.from_string, args.to_string)
            else:
                print(f"[replace-string] Skip {path}")
