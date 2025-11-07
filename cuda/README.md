# Simulet-CUDA

模拟器的GPU版本

## cmake编译

1. 克隆本仓库到本地
2. `git submodules update --init`
3. 安装`gRPC`、`cudatoolkit>=11`(11.6.124)、`gcc>=9`(11.4.0)、`cmake>=3.18`
4. 设置CUDA环境变量并编译
```bash
# 下面的路径按实际情况修改
PATH=/xxx/cuda11.x/bin:$PATH
LD_LIBRARY_PATH=/xxx/cuda11.x/lib64:$LD_LIBRARY_PATH
# 编译
mkdir -p build
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## DEMO API

```bash
pip install -r requirements.txt
```

生成proto代码

```bash
zsh -c 'python -m grpc_tools.protoc -Iprotos --python_out=api --grpc_python_out=api --pyi_out=api protos/wolong/**/*.proto'
```

启动DEMO后端

```bash
python api/demo.py
```

## Python API

1. 完成上面的`CMAKE编译`

2. 生成proto文件

```bash
zsh -c 'python -m grpc_tools.protoc -Iprotos --python_out=api --grpc_python_out=api protos/wolong/**/*.proto'
```

3. 运行如下代码安装`simulet`包:

```bash
pip install ./python_wrapper --no-cache
```

### Debug

如果在使用python API时遇到如下问题
```log
ImportError: /.../libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /.../_simulet.cpython-...so)
```
需要在当前conda环境中运行
```bash
conda install -c conda-forge libstdcxx-ng=12
```

## 实现细节

### 输出格式

输出代码在`output.cu`中，根据`Option`不同输出不同的内容

`Option::AGENT`是输出 人+车+信号灯 信息
* 按经纬度过滤
* 格式为`AgentOutput`结构体
* 在一开始加上一些额外信息，见`output::Init`

`Option::LANE`是输出车道拥挤程度信息
* 为了GPU实现的简单，会直接输出当前时刻的拥挤程度，由外部程序实现滑动平均的计算
* 初始输出是车道id，之后按顺序输出每个车道的拥堵情况
