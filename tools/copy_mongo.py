#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pymongo import MongoClient


def copy_mongo(
    src_uri: str, src_db: str, src_col: str, dst_uri: str, dst_db: str, dst_col: str
):
    print(f"copy {src_db}.{src_col} at {src_uri} to {dst_db}.{dst_col} at {dst_uri}")
    src_client = MongoClient(src_uri)
    data = src_client[src_db][src_col].find()
    if data is not None:
        dst_client = MongoClient(dst_uri)
        col = dst_client[dst_db][dst_col]
        col.drop()
        col.insert_many(data, ordered=False)


def parse_args():
    parser = argparse.ArgumentParser(description="copy mongodb collection")
    parser.add_argument("--src_uri", type=str, required=True, help="Source MongoDB URI")
    parser.add_argument(
        "--src_db", type=str, required=True, help="Source MongoDB database name"
    )
    parser.add_argument(
        "--src_col", type=str, required=True, help="Source MongoDB collection name"
    )
    parser.add_argument(
        "--dst_uri", type=str, required=True, help="Destination MongoDB URI"
    )
    parser.add_argument(
        "--dst_db",
        type=str,
        default="",
        help="Destination MongoDB database name (keep it empty to reuse src_db)",
    )
    parser.add_argument(
        "--dst_col",
        type=str,
        default="",
        help="Destination MongoDB collection name (keep it empty to reuse src_col)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    copy_mongo(
        args.src_uri,
        args.src_db,
        args.src_col,
        args.dst_uri,
        args.dst_db if args.dst_db != "" else args.src_db,
        args.dst_col if args.dst_col != "" else args.src_col,
    )
