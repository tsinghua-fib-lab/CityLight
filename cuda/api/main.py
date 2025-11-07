import argparse
import os
import subprocess
import threading
import time
import traceback

import yaml

PATH = os.path.dirname(os.path.abspath(__file__))


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


def run_http(p_out, port):
    env = dict(os.environ)
    env['OUTPUT'] = p_out
    p = subprocess.Popen(f'waitress-serve --host 127.0.0.1 --port {port} http_server:app', cwd=PATH, bufsize=1, encoding='utf8', shell=True, env=env)
    p.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bin', type=str, help='path to the simulet-cuda binary', default=os.path.abspath(PATH + '/../_/cuda'))
    parser.add_argument('-c', '--config', type=str, help='path to the config file', default=os.path.abspath(PATH + '/../_/config.yml'))
    parser.add_argument('--cuda', type=int, help='which gpu to use', default=0)
    parser.add_argument('--port-grpc', type=int, help='gRPC listening port', default=53005)
    parser.add_argument('--port-http', type=int, help='HTTP listening port', default=53006)
    args = parser.parse_args()

    p_bin = os.path.abspath(args.bin)
    p_cfg = os.path.abspath(args.config)

    # 模拟器输出文件路径
    p_out = os.path.abspath(os.path.dirname(p_bin) + '/' + yaml.safe_load(open(args.config, 'rb'))['output_file'])
    if not p_out or not os.path.exists(os.path.dirname(p_out)):
        raise FileNotFoundError(f'Wrong output path: "{p_out}"')
    open(p_out, 'w').close()

    # 启动模拟器
    finished = [False]
    # threading.Thread(target=run_cuda, args=(p_bin, p_cfg, args.cuda, finished), daemon=True).start()

    # 启动HTTP服务
    threading.Thread(target=run_http, args=(p_out, args.port_http), daemon=True).start()

    # 等待模拟完成
    while not finished[0]:
        time.sleep(1)


if __name__ == '__main__':
    os.setpgrp()
    try:
        main()
    except:
        print(traceback.format_exc())
    finally:
        os.killpg(0, 9)
