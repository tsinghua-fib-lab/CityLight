import struct

import psycopg
import pyproj
from tqdm import tqdm

proj = pyproj.Proj("+proj=tmerc +lat_0=39.90611 +lon_0=116.3911")
min_lon = 116.45466845743775
min_lat = 39.90635904559525
max_lon = 116.4727503404667
max_lat = 39.92235997926467
# print(proj(min_lon, min_lat), proj(max_lon, max_lat))
# exit()
MAP_DB = "srt.map_beijing_extend_20230625"

INPUT_FILE = "_/output.bin"
OUTPUT_NAME = "nosignal_cuda"
OUTPUT_ALL = True
SQL_URL = "postgres://sim:tnBB0Yf4tm2fIrUi1KB2LqqZTqyrztSa@121.36.37.114/simulation"
BATCH_SIZE = 1000

vehs = []
peds = []
tls = []
xys = []
f = open(INPUT_FILE, "rb").read()
step_start, step_total, step_interval, total_agents = struct.unpack("=IIfI", f[:16])
step = None
for _type, *data in tqdm(struct.iter_unpack("=iiiffff", f[16:]), total=(len(f) - 16) // 28):
    if _type == 0:
        step = data[0]
    elif _type <= 2:
        p_id, lane_id, s, x, y, _dir = data
        lon, lat = proj(x, y, True)
        xys.append([lon, lat])
        (vehs if _type == 1 else peds).append(f"({step}, {p_id}, {lane_id}, {_dir}, {lon}, {lat})")
    elif _type == 3:
        l_id, state, t, x, y, _ = data
        lon, lat = proj(x, y, True)
        tls.append(f"({step}, {l_id}, {state}, {lon}, {lat})")
    else:
        assert False
x, y = zip(*xys)
if OUTPUT_ALL:
    min_lon, max_lon, min_lat, max_lat = min(x), max(x), min(y), max(y)
print(min_lon, max_lon, min_lat, max_lat)


with psycopg.connect(SQL_URL) as conn:
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS meta_simple (name TEXT PRIMARY KEY NOT NULL, start INT NOT NULL, steps INT NOT NULL, time FLOAT NOT NULL, total_agents INT NOT NULL, map TEXT NOT NULL, min_lng FLOAT NOT NULL, min_lat FLOAT NOT NULL, max_lng FLOAT NOT NULL, max_lat FLOAT NOT NULL)"
        )
        cur.execute(f"DELETE FROM meta_simple WHERE name='{OUTPUT_NAME}'")
        cur.execute(
            f"INSERT INTO meta_simple VALUES ('{OUTPUT_NAME}', {step_start}, {step_total}, {step_interval}, {total_agents}, '{MAP_DB}', {min_lon}, {min_lat}, {max_lon}, {max_lat})"
        )
        cur.execute(f"DROP TABLE IF EXISTS {OUTPUT_NAME}_s_people")
        cur.execute(f"DROP TABLE IF EXISTS {OUTPUT_NAME}_s_cars")
        cur.execute(f"DROP TABLE IF EXISTS {OUTPUT_NAME}_s_traffic_light")
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
        for i in tqdm(range(0, len(vehs), BATCH_SIZE)):
            cur.execute(
                f"INSERT INTO {OUTPUT_NAME}_s_cars VALUES "
                + ",".join(vehs[i: i + BATCH_SIZE])
            )
        for i in tqdm(range(0, len(peds), BATCH_SIZE)):
            cur.execute(
                f"INSERT INTO {OUTPUT_NAME}_s_people VALUES "
                + ",".join(peds[i: i + BATCH_SIZE])
            )
        for i in tqdm(range(0, len(tls), BATCH_SIZE)):
            cur.execute(
                f"INSERT INTO {OUTPUT_NAME}_s_traffic_light VALUES "
                + ",".join(tls[i: i + BATCH_SIZE])
            )
