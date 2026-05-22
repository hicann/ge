# MoveReluBeforeConcatPass Python 样例使用指导

本目录提供 `graph_base_pass/2_move_relu_before_concat_pass` 的 **纯 Python** 版本示例，逻辑与 C++ 版本一致：

- 扫描 `ConcatV2 -> Relu` 结构
- 构造 replacement 子图（将 Relu 前移到 Concat 的每个输入上）
- 使用 `SubgraphBoundary` + `SubgraphRewriter.replace()` 做子图替换

> **FusionBasePass与PatternFusionPass**
> 
> 本样例中的 pass 继承 FusionBasePass，与继承 PatternFusionPass 不同，这里通过重写 `run()` 函数实现 pass 逻辑。
> `run()` 函数入参包含 graph 对象的引用，本样例场景中 Concat 节点输入数量不固定，
> 难以用固定的 pattern 表示，而在 `run()` 函数里可以根据目标图中匹配到的 Concat 节点动态构造 Boundary，实现更高的灵活性。

## 目录结构

```
python/
├── README.md                     // Python 样例说明
├── CMakeLists.txt                // 生成 es_all Python ES API 的编译脚本
├── src
│   ├── python_move_relu_before_concat_pass.py // Python pass 实现文件
```

## 前置条件

- 已完成 CANN 环境变量设置，设置方式为 `source ${ASCEND_PATH}/set_env.sh`，更多指导请参考 [C++ 样例 README](../cpp/README.md) 的配置环境变量步骤
- **run 包编译使用的 Python 版本**与执行本样例的 Python 版本一致。原因是 Python pass 相关扩展模块等与编译时 Python ABI 相关，版本不一致时可能导入失败或加载异常
- Python 环境可导入 ES API（通常来自 run 包/ops 包）：
  - `ge.es.math.ConcatV2`
  - `ge.es.nn.Relu`（若环境中没有该符号，请确认 ES Python API 安装完整）
- 可导入 GE Python 包（含 `ge.graph`、`ge.passes` 及 pass 加载链路）

如果执行时提示缺少 `ConcatV2`、`Relu` 等 ES API，再按下文“ES API 缺失时处理（可选）”生成并加载 `es_all`。

## Conda 环境示例（Python 3.11）

如果本机没有现成的匹配环境，可以参考下面的方式创建：

```bash
conda create -n ge-pass-py311 python=3.11 -y
conda activate ge-pass-py311
python -m pip install --upgrade pip
python -m pip install attrs decorator sympy numpy psutil scipy
```

创建环境后，请确认：

- 该环境中的 Python 版本与 run 包编译时使用的 Python 版本一致
- 再按 [C++ 样例 README](../cpp/README.md) 的配置环境变量步骤，完成环境变量设置
- 最后按本文使用方式设置 `ASCEND_GE_PY_PASS_PATH` 并使用 `pyatc` 验证

## 使用方式

以下命令默认在 `2_move_relu_before_concat_pass` 目录执行。

1. 配置环境变量：

```bash
source ${ASCEND_PATH}/set_env.sh
export DUMP_GE_GRAPH=1
export ASCEND_GE_PY_PASS_PATH=$PWD/python/src/python_move_relu_before_concat_pass.py
```

2. 生成 AIR 模型：

```bash
cd cpp/data
python es_gen_air.py
```

3. 使用 `pyatc` 离线编译触发 Python pass，`soc_version` 请根据实际环境修改：

```bash
pyatc --model=./graph.air --framework=1 --soc_version=xxx --output=./model
cd ../..
```

4. 说明：

   - `pyatc` 命令行参数与 `atc` 一致，但会在当前 Python 解释器进程中运行
   - 如果执行时报缺少 `ConcatV2`、`Relu` 等 ES API，先按下文生成并加载 `es_all`，再重新运行

## ES API 缺失时处理（可选）

如果执行时报缺少 `ConcatV2`、`Relu` 等 ES API，可通过本 Python 目录的 `CMakeLists.txt` 生成 `es_all`：

```bash
cd python
cmake -S . -B build
cmake --build build --target build_es_all -j$(nproc)
```

安装生成的 Python 包，并让当前 Python 进程能找到包和对应的动态库：

```bash
pip install --force-reinstall --upgrade --target ./build/whl_package ./build/es_output/whl/es_all-1.0.0-py3-none-any.whl
export PYTHONPATH="$PWD/build/whl_package:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$PWD/build/es_output/lib64:${LD_LIBRARY_PATH:-}"
cd ..
```

## 预期现象

日志中会出现类似打印：

```text
PythonMoveReluBeforeConcatPass
Replacement of PythonMoveReluBeforeConcatPass succeeded
```
