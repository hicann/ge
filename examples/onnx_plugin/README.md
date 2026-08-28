# ONNX Python 插件图编译执行样例

本样例演示完整链路：PyTorch 自定义算子导出 ONNX，GE 通过 `onnx_plugin` 注册
Python 解析插件，将 `ThresholdedRelu` 分解为已有的 `Threshold` 和 `Mul` 算子，
再由 ATC 编译为 OM，最后使用 ACL Python 接口执行并校验结果。

## 目录结构

```text
onnx_plugin/
├── plugin/
│   └── thresholded_relu_plugin.py  # GE ONNX Python 插件
├── export_onnx.py                  # PyTorch 自定义算子 -> ONNX
├── run_model.py                    # ACL 加载并执行 OM
└── run.sh                          # 导出、编译、执行一键脚本
```

`plugin/` 与其他脚本分开是有意的。GE 会扫描 `ASCEND_CUSTOM_OPP_PATH` 指向目录
下的一层 Python 文件；导出器和执行器依赖 PyTorch、NumPy 或 ACL，不应作为插件
在 ATC 初始化时加载。

## 环境要求

- 已安装 CANN，并已执行对应版本的 `set_env.sh`；
- 可用的 `atc` 命令和 Ascend 设备；
- 与当前 CANN Python 环境兼容的 PyTorch、ONNX 和 NumPy。

## 运行样例

先加载 CANN 环境，再指定实际设备的 SoC 型号：

```bash
source /path/to/cann/set_env.sh
SOC_VERSION=Ascend910B1 ./run.sh
```

`SOC_VERSION` 默认值为 `Ascend910B1`，请按实际设备修改，例如：

```bash
SOC_VERSION=Ascend910B2 ./run.sh
SOC_VERSION=Ascend910_9362 ./run.sh
```

已在如下环境端到端验证（导出、ATC 编译、ACL 执行、结果比对全部通过）：

```text
SoC: Ascend910_9362 (Atlas A3)   SOC_VERSION=Ascend910_9362
```

脚本依次完成以下步骤：

1. `export_onnx.py` 生成 `output/thresholded_relu.onnx`；
2. 设置 `ASCEND_CUSTOM_OPP_PATH`，让 ATC 发现 `plugin/thresholded_relu_plugin.py`；
3. ATC 将 ONNX 模型编译为 `output/thresholded_relu.om`；
4. `run_model.py` 通过 ACL 加载并执行 OM，校验输出。

输入为：

```text
[[-1.0,  0.5, 1.5],
 [ 2.0, -2.0, 3.0]]
```

预期输出为 `x` 中大于 `alpha=1.0` 的元素，其余元素为 `0`：

```text
[[0.0, 0.0, 1.5],
 [2.0, 0.0, 3.0]]
```

## 分步运行

需要分别观察导出、编译或执行结果时，可以手动运行：

```bash
python3 export_onnx.py --output output/thresholded_relu.onnx
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/plugin:${ASCEND_CUSTOM_OPP_PATH:-}"
atc --model=output/thresholded_relu.onnx \
    --framework=5 \
    --output=output/thresholded_relu \
    --soc_version="${SOC_VERSION:-Ascend910B1}"
python3 run_model.py --model output/thresholded_relu.om
```

该样例使用 `decompose` 回调生成已有 GE/ES 算子图，不提供新的设备 kernel，
因此重点验证 ONNX Python 插件注册、解析、子图展开、图编译和图执行链路。

`parse_node` 与 `decompose` 两个回调在本样例中配合工作：

- `parse_node` 在 ONNX 节点转换为目标算子时执行，负责把 `alpha` 属性
  写入目标算子，并为动态 IO 的目标算子 `PartitionedCall` 注册端口
  （parser 连线需要端口名）；
- `decompose` 收到的 `source` 就是 `parse_node` 产出的算子，其中读取的
  `alpha` 即由 `parse_node` 中转，因此 `decompose` 依赖 `parse_node`
  的属性传递与端口注册。
