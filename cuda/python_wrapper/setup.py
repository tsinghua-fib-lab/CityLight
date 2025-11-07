import os
import shutil
import sys
from glob import glob

from setuptools import find_packages, setup

path = os.path.dirname(os.path.abspath(__file__))
ver = f'_simulet.cpython-{sys.version_info.major}{sys.version_info.minor}*.so'
if len(glob(f'{path}/../build/src/{ver}')) != 1:
    raise FileNotFoundError(f'Make sure you have built the project in {os.path.abspath(path + "/../build")}')
if not os.path.exists(path + '/../api/wolong'):
    raise FileNotFoundError('Make sure you you have the latest version of pip and install with `pip install ./python_wrapper --no-cache`')


try:
    for i in glob(path + '/src/simulet/_simulet*.so'):
        os.remove(i)
    so_file_in = glob(f'{path}/../build/src/{ver}')[0]
    so_file_out = path + '/src/simulet/' + os.path.split(so_file_in)[1]
    shutil.copy(so_file_in, so_file_out)
    shutil.copytree(path + '/../api/wolong', path + '/src/simulet/wolong')
    setup(
        name='simulet',
        version=open(path + '/src/simulet/__init__.py').read().strip().split()[-1].strip("'"),
        author='aowenxuan',
        author_email='aowx21@outlook.com',
        url='https://git.fiblab.net/sim/simulet-cuda',
        description='Simulet',
        zip_safe=False,
        python_requires='>=3.6',
        packages=find_packages(where='src'),
        package_dir={'': 'src'},
        package_data={'simulet': ['_simulet*.so', 'wolong/**/*']},
        install_requires=[
            'shapely>=2',
            'protobuf>=3.20,<4',
            'grpcio',
            'grpcio-tools',
            'numpy>=1.20',
            'igraph',
            'tqdm'
        ],
    )
finally:
    shutil.rmtree(path + '/src/simulet/wolong')
    os.remove(so_file_out)
