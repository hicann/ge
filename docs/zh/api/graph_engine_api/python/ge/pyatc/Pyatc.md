# Pyatc接口

## 产品支持情况

全量芯片支持。

## 功能说明

pyatc是ATC（Ascend Tensor Compiler）离线模型编译工具的Python封装接口，在功能上与ATC命令行工具完全等价，旨在为Python用户提供更灵活的集成体验。

pyatc与ATC命令行工具的区别是：

- ATC命令行工具：作为独立子进程启动，编译过程所使用的Python解释器由atc进程自行解析，可能与用户当前终端环境（如python3）不一致，容易引发依赖冲突或路径问题。
- pyatc：直接在当前主进程中执行编译逻辑，复用用户当前的Python解释器及环境变量，确保编译环境与运行环境完全一致，避免了进程间的环境隔离问题。

## 调用方式

先参见[环境准备](../../../../../user_guides/graph_dev/overview/environment_setup.md)完成相关环境变量的设置，然后根据实际使用场景选择以下任一入口：

- 使用命令行入口：

    ```bash
    pyatc [参数]
    ```

- 使用Python模块入口：

    ```bash
    python3 -m ge.pyatc [参数]
    ```

## 参数说明

pyatc的参数与ATC命令行工具完全一致，详细功能介绍请参见[《ATC离线模型编译工具》](../../../../../user_guides/atc_tools/README.md)。

## 返回值说明

调用结束后通过进程退出码（exit code）表示执行结果：

- 0：执行成功。
- 非 0：执行失败，错误信息输出至标准输出。

## 调用示例

```bash
pyatc --model=resnet50.onnx --framework=5 --soc_version=<soc_version> --output=resnet50
```

```bash
pyatc --model=resnet50.onnx --framework=5 --soc_version=MC62CM12A* --output=resnet50
```

```bash
pyatc --model=resnet50.onnx --framework=5 --soc_version=Ascend035 --output=resnet50
```

<soc\_version\>查询方法请参见[--soc_version](../../../../../user_guides/atc_tools/CLI_options/--soc_version.md)。
