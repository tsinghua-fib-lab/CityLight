import argparse
import json
import os
import random
import struct
import subprocess
import threading
import time
import traceback
from collections import deque

import yaml
from flask import Flask, request

PATH = os.path.dirname(os.path.abspath(__file__))


class Server:
    def __init__(self):
        self.lock = threading.Lock()
        self.want_lock = False
        self.cuda_finished = [True]
        self.ret_start = {
            'ok': False,
            'start': -1,
            'steps': -1,
        }
        self.ret_road_id = {
            'ok': False,
            'id': []
        }
        self.cfg = json.loads(os.environ['ARGS'])
        self.n_road = -1
        self.n_steps = -1
        self.step = -1
        self.road_status = []
        if self.cfg['mock']:
            self.serve = self.serve_mock
            self.get_start = self.get_start_mock

    def serve(self):
        print(self.cfg)
        while True:
            print('\n=== Listening...')
            while self.cuda_finished[0]:
                time.sleep(0.1)
            try:
                with open(self.cfg['out'], 'rb') as f:
                    def read(n):
                        s = b''
                        while not self.cuda_finished[0] and len(s) < n:
                            a = f.read(n - len(s))
                            if len(a) == 0:
                                time.sleep(0.01)
                                continue
                            s += a
                        if self.cuda_finished[0] and len(s) < n:
                            s += f.read(n - len(s))
                        if len(s) < n:
                            raise EOFError
                        return s
                    read(12)
                    self.n_road = struct.unpack('=i', read(4))[0]
                    self.ret_road_id = {
                        'ok': True,
                        'id': [i[0] for i in struct.iter_unpack('=i', read(self.n_road * 4))]
                    }
                    self.road_status = [1] * self.n_road
                    for _ in range(self.n_steps):
                        while self.want_lock:
                            time.sleep(0.01)
                        with self.lock:
                            self.step = struct.unpack('=i', read(4))[0]
                            print(f'\n Parse {self.step}')
                            self.road_status = list(read(self.n_road))
                while not self.cuda_finished[0]:
                    time.sleep(0.1)
            except EOFError:
                print('Error: EOF')

    def serve_mock(self):
        print(self.cfg)
        while True:
            print('\n=== Listening...')
            while self.cuda_finished[0]:
                time.sleep(0.1)
            time.sleep(1)
            self.n_road = 54117
            self.ret_road_id = {
                'ok': True,
                'id': [2_0000_0000 + i for i in range(self.n_road)]
            }
            self.road_status = [1] * self.n_road
            for step in range(self.n_steps):
                time.sleep(0.01)
                with self.lock:
                    self.step = step + 100
                    print(f'Simulate: {self.step}')
                    for i in range(self.n_road):
                        if random.random() < 0.1:
                            self.road_status[i] = random.randint(1, 5)
            self.cuda_finished[0] = True

    def get_start(self):
        if self.cuda_finished[0]:
            cfg = yaml.safe_load(open(self.cfg['cfg'], 'r', encoding='utf8'))
            self.ret_start = {
                'ok': True,
                'start': cfg['start_step'],
                'steps': cfg['total_step']
            }
            self.n_steps = cfg['total_step']
            self.ret_road_id['ok'] = False
            self.step = -1
            open(self.cfg['out'], 'wb').close()
            self.cuda_finished[0] = False
            threading.Thread(target=run_cuda, args=(self.cfg['bin'], self.cfg['cfg'], self.cfg['cuda'], self.cuda_finished), daemon=True).start()
        else:
            self.ret_start['ok'] = False
        return self.ret_start

    def get_start_mock(self):
        if self.cuda_finished[0]:
            cfg = yaml.safe_load(open(self.cfg['cfg'], 'r', encoding='utf8'))
            self.ret_start = {
                'ok': True,
                'start': cfg['start_step'],
                'steps': cfg['total_step']
            }
            self.n_steps = cfg['total_step']
            self.ret_road_id['ok'] = False
            self.step = -1
            self.cuda_finished[0] = False
        else:
            self.ret_start['ok'] = False
        return self.ret_start

    def get_road_id(self):
        return self.ret_road_id

    def get_road_status(self):
        self.want_lock = True
        with self.lock:
            self.want_lock = False
            if self.step == -1:
                return {
                    'ok': False,
                    'step': -1,
                    'status': []
                }
            return {
                'ok': True,
                'step': self.step,
                'status': [round(i.get()) for i in self.road_status]
            }


app = Flask(__name__)
server = None


@app.route("/__ver__")
def ver():
    return 'v0.0.1'


@app.route("/start")
def get_start():
    return server.get_start()


@app.route("/road_id")
def get_road_id():
    return server.get_road_id()


@app.route("/road_status")
def get_road_status():
    return server.get_road_status()


def add_from_request(response, key, value):
    s = request.headers.get(key, "")
    response.headers[key] = ", ".join(
        set(f"{s},{value}".replace(" ", "").split(",")) - {""}
    )


@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    add_from_request(response, "Access-Control-Allow-Headers", "Content-Type")
    return response


def run_cuda(p_bin, p_cfg, n_cuda, finished):
    cmd = f'{p_bin} -c {p_cfg}'
    if n_cuda != 0:
        cmd = f'CUDA_VISIBLE_DEVICES={n_cuda} {cmd}'
    p = subprocess.Popen(cmd, cwd=os.path.dirname(p_bin), bufsize=1, encoding='utf8', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    last_r = False
    for line in p.stdout:
        # 对\r特殊处理
        if 'ETA:' in line:
            last_r = True
            print(line.strip(), end='           \r')
        else:
            if last_r:
                print()
            last_r = False
            print(line, end='')
    p.wait()
    finished[0] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bin', type=str, help='path to the simulet-cuda binary', default=os.path.abspath(PATH + '/../_/cuda'))
    parser.add_argument('-c', '--config', type=str, help='path to the config file', default=os.path.abspath(PATH + '/demo_config.yml'))
    parser.add_argument('--cuda', type=int, help='which gpu to use', default=0)
    parser.add_argument('-p', '--port', type=int, help='HTTP listening port', default=50004)
    parser.add_argument('--mock', action='store_true', help='mock for testing')
    args = parser.parse_args()

    p_bin = os.path.abspath(args.bin)
    p_cfg = os.path.abspath(args.config)

    # 模拟器输出文件路径
    cfg = yaml.safe_load(open(p_cfg, 'rb'))
    p_out = os.path.abspath(os.path.dirname(p_bin) + '/' + cfg['output_file'])
    if not p_out or not os.path.exists(os.path.dirname(p_out)):
        raise FileNotFoundError(f'Wrong output path: "{p_out}"')
    if cfg["output_type"] != 'lane':
        raise ValueError('Wrong output_type: ' + cfg["output_type"])
    open(p_out, 'w').close()

    # 启动HTTP服务
    env = os.environ
    env['ARGS'] = json.dumps({
        'bin': p_bin,
        'cfg': p_cfg,
        'out': p_out,
        'cuda': args.cuda,
        'mock': args.mock
    })
    subprocess.Popen(
        f'waitress-serve --host 127.0.0.1 --port {args.port} demo:app', cwd=PATH, shell=True, env=env,
    ).communicate()


if __name__ == '__main__':
    os.setpgrp()
    try:
        main()
    except:
        print(traceback.format_exc())
    finally:
        os.killpg(0, 9)
else:
    server = Server()
    threading.Thread(target=server.serve, daemon=True).start()
