import struct
from collections import deque

import psycopg
import pyproj
from tqdm import tqdm


class Avg:
    def __init__(self, n):
        self.q = deque([], maxlen=n)

    def add(self, x):
        self.q.append(x)

    def get(self):
        return sum(self.q) / len(self.q)


proj = pyproj.Proj("+proj=tmerc +lat_0=39.90611 +lon_0=116.3911")
min_lon = 116.45466845743775
min_lat = 39.90635904559525
max_lon = 116.4727503404667
max_lat = 39.92235997926467
MAP_DB = "srt.map_beijing5ring_20230601"

INPUT_FILE = "_/output.bin"
OUTPUT_NAME = "nosignal_cuda"
SQL_URL = "postgres://sim:tnBB0Yf4tm2fIrUi1KB2LqqZTqyrztSa@121.36.37.114/simulation"
BATCH_SIZE = 1000
MOVING_AVERAGE = 300
OUTPUT_INTERVAL = 300

output = []
with open(INPUT_FILE, "rb") as f:
    step_start, step_total, step_interval, n_road = struct.unpack("=IIfI", f.read(16))
    road_id = [i[0] for i in struct.iter_unpack('=i', f.read(n_road * 4))]
    road_status = [Avg(MOVING_AVERAGE) for _ in range(n_road)]
    for step in iter(lambda: f.read(4), b''):
        step = struct.unpack('=i', step)[0]
        print(f'Parse {step}', end='\r')
        for avg, status in zip(road_status, f.read(n_road)):
            assert 1 <= status <= 5
            avg.add(status)
        if (step - step_start) % OUTPUT_INTERVAL == 0:
            for _id, avg in zip(road_id, road_status):
                output.append([step, _id, round(avg.get())])

with psycopg.connect(SQL_URL) as conn:
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS meta_simple (name TEXT PRIMARY KEY NOT NULL, start INT NOT NULL, steps INT NOT NULL, time FLOAT NOT NULL, total_agents INT NOT NULL, map TEXT NOT NULL, min_lng FLOAT NOT NULL, min_lat FLOAT NOT NULL, max_lng FLOAT NOT NULL, max_lat FLOAT NOT NULL)"
        )
        cur.execute(f"DELETE FROM meta_simple WHERE name='{OUTPUT_NAME}'")
        cur.execute(
            f"INSERT INTO meta_simple VALUES ('{OUTPUT_NAME}', {step_start}, {step_total}, {step_interval}, {0}, '{MAP_DB}', {min_lon}, {min_lat}, {max_lon}, {max_lat}, 0, {OUTPUT_INTERVAL})"
        )
        cur.execute(f"DROP TABLE IF EXISTS {OUTPUT_NAME}_s_road_status")
        cur.execute(
            f"CREATE TABLE {OUTPUT_NAME}_s_road (STEP INT4 NOT NULL,ID INT4 NOT NULL,LEVEL INT4 NOT NULL)"
        )
        cur.execute(
            f"CREATE TABLE {OUTPUT_NAME}_s_people (STEP INT4 NOT NULL,ID INT4 NOT NULL,PARENT_ID INT4 NOT NULL,DIRECTION FLOAT8 NOT NULL,LNG FLOAT8 NOT NULL,LAT FLOAT8 NOT NULL)"
        )
        cur.execute(
            f"CREATE TABLE {OUTPUT_NAME}_s_cars   (STEP INT4 NOT NULL,ID INT4 NOT NULL,PARENT_ID INT4 NOT NULL,DIRECTION FLOAT8 NOT NULL,LNG FLOAT8 NOT NULL,LAT FLOAT8 NOT NULL)"
        )
        cur.execute(
            f"CREATE TABLE {OUTPUT_NAME}_s_traffic_light (STEP INT4 NOT NULL,ID INT4 NOT NULL,STATE INT4 NOT NULL,LNG FLOAT8 NOT NULL,LAT FLOAT8 NOT NULL)"
        )
        cur.execute(
            f"CREATE INDEX {OUTPUT_NAME}_s_people_step_lng_lat_idx ON {OUTPUT_NAME}_s_people (step,lng,lat)"
        )
        cur.execute(
            f"CREATE INDEX {OUTPUT_NAME}_s_cars_step_lng_lat_idx ON {OUTPUT_NAME}_s_cars (step,lng,lat)"
        )
        cur.execute(
            f"CREATE INDEX {OUTPUT_NAME}_s_traffic_light_step_lng_lat_idx ON {OUTPUT_NAME}_s_traffic_light (step,lng,lat)"
        )

        for i in tqdm(range(0, len(output), BATCH_SIZE)):
            cur.execute(
                f"INSERT INTO {OUTPUT_NAME}_s_road_status VALUES "
                + ",".join(output[i: i + BATCH_SIZE])
            )
