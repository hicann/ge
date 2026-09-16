# ONNX Python 插件图编译执行样例

## 1、功能描述

本样例演示完整链路：PyTorch自定义算子导出ONNX模型，GE通过`onnx_plugin`注册Python解析插件，
将ONNX模型中的自定义算子接入GE图，再由ATC编译为OM模型，最后使用ACL Python接口加载执行并校验结果。

样例包含两个插件，一次运行覆盖全部三种回调：

| 自定义算子 | 插件 | 使用的回调 | 做的事 |
| --------- | ---- | ---------- | ------ |
| ThresholdedRelu | `thresholded_relu_plugin.py` | `parse_node` + `decompose` | GE没有现成算子，分解为已有的Threshold和Mul（无需写kernel） |
| MyElu | `my_elu_plugin.py` | `parse_operator` | GE有现成的Elu算子，把属性转写过去即可 |

参数解析优先使用`parse_node`，`parse_operator`只在少数情况使用，见第4章。

## 2、目录结构

```text
onnx_plugin/
├── plugin/
│   ├── thresholded_relu_plugin.py  // 插件一：parse_node + decompose
│   └── my_elu_plugin.py            // 插件二：parse_operator
├── export_onnx.py                  // PyTorch自定义算子导出ONNX模型
├── run_model.py                    // ACL加载并执行OM模型
└── run.sh                          // 导出、编译、执行一键脚本
```

`plugin/`目录与其他脚本分开是有意的：GE会扫描`ASCEND_CUSTOM_OPP_PATH`指向目录下的一层
Python文件，而导出器和执行器依赖PyTorch、NumPy或ACL，不应作为插件在ATC初始化时加载。
注意：目录内**任一**Python文件加载失败（如语法错误）都会中断整个编译，定位具体文件
需开启debug日志（见3.6节）。

## 3、使用方法

### 3.1、准备cann包

- 请参考[环境准备](../../docs/zh/quick_install.md#1-环境准备)中“方式三：手动安装软件包 > 场景1：体验master版本能力或基于master版本进行开发”，正确安装`toolkit`和`ops`包。
- 设置环境变量（假设包安装在/usr/local/Ascend/）：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

### 3.2、准备Python依赖

| 依赖 | 用途 | 版本要求 |
| ---- | ---- | -------- |
| PyTorch | 导出ONNX模型（`torch.onnx.export`） | 2.7~2.8，两个版本均已实测 |
| onnx | `torch.onnx.export`导出过程内部使用 | 1.21.0已实测 |
| NumPy | 构造输入与结果比对 | 无特殊要求 |

`acl`Python接口由CANN toolkit自带。安装命令：

```bash
pip3 install torch numpy onnx
```

已在如下环境完成端到端验证（导出、ATC编译、ACL执行、结果比对全部通过）：

| 项目 | 版本 |
| ---- | ---- |
| SoC | Ascend910_9362（Atlas A3） |
| CANN | 9.2.0 |
| Python | 3.12 |
| PyTorch | 2.7.1与2.8.0（CPU版，均已实测） |
| onnx | 1.21.0 |
| NumPy | 1.26.4 |

### 3.3、一键运行

在`examples/onnx_plugin`目录下执行：

```bash
SOC_VERSION=Ascend910B1 bash run.sh
```

`SOC_VERSION`是实际设备的SoC型号，默认值为`Ascend910B1`，请按实际设备修改，例如：

```bash
SOC_VERSION=Ascend910B2 bash run.sh
SOC_VERSION=Ascend910_9362 bash run.sh
```

`run.sh`会自动创建`output/`目录，并依次完成以下步骤：

1. 执行`export_onnx.py`，生成含ThresholdedRelu、MyElu两个自定义算子的`output/thresholded_relu.onnx`；
2. 设置`ASCEND_CUSTOM_OPP_PATH`指向`plugin/`目录，使ATC能发现两个插件；
3. 执行ATC，将ONNX模型编译为`output/thresholded_relu.om`（两个插件分别处理两个自定义算子）；
4. 执行`run_model.py`，通过ACL加载并执行OM模型，校验两路输出。

预期输出见3.5节。

### 3.4、分步运行

需要分别观察导出、编译或执行结果时，可按以下步骤手动运行。
注意：`output/`目录不会自动创建（一键脚本`run.sh`会自动创建），分步运行时需先手动创建：

```bash
# 0. 创建输出目录（分步运行必须；run.sh会自动创建）
mkdir -p output
# 1. 导出ONNX模型
python3 export_onnx.py --output output/thresholded_relu.onnx
# 2. 设置插件路径，使ATC能发现plugin/下的Python插件
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/plugin:${ASCEND_CUSTOM_OPP_PATH:-}"
# 3. 将ONNX模型编译为OM模型
atc --model=output/thresholded_relu.onnx \
    --framework=5 \
    --output=output/thresholded_relu \
    --soc_version="${SOC_VERSION:-Ascend910B1}"
# 4. 加载并执行OM模型
python3 run_model.py --model output/thresholded_relu.om
```

### 3.5、结果校验

执行成功后会看到：

```text
Input:
[[-1.   0.5  1.5]
 [ 2.  -2.   3. ]]
Output (ThresholdedRelu, decomposed into Threshold + Mul):
[[-0.   0.  1.5]
 [ 2.  -0.   3. ]]
Output (MyElu, mapped to Elu by parse_operator):
[[-0.6323242  0.5        1.5      ]
 [ 2.        -0.8647461  3.       ]]
[Success] GE graph compiled and executed; output matches PyTorch reference.
```

### 3.6、验证算子结果

GE支持图dump，编译后图文件保存在`graph_dump/pid_*/`目录（目录名因运行环境而异），
其中`ge_onnx_*.pbtxt`为ONNX格式的图文件，打开即可查看各编译阶段的图结构：

```bash
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/plugin:${ASCEND_CUSTOM_OPP_PATH:-}"
mkdir -p graph_dump
DUMP_GE_GRAPH=3 DUMP_GRAPH_PATH="$(pwd)/graph_dump" atc \
    --model=output/thresholded_relu.onnx \
    --framework=5 \
    --output=output/thresholded_relu \
    --soc_version="${SOC_VERSION:-Ascend910B1}"
```

两个插件的接入结果可在pbtxt图文件中确认：

- ThresholdedRelu：分解后的图中为Threshold和Mul节点，Threshold节点携带`/ThresholdedRelu`
  原始类型标记，可追溯分解来源；
- MyElu：一对一映射，图中为Elu节点，`alpha`属性已转写。

如需查看编译过程日志，在上述atc命令中增加`--log=debug`参数，并在执行前设置打屏环境变量
`export ASCEND_SLOG_PRINT_TO_STDOUT=1`，详见[atc --log参数说明](../../docs/zh/user_guides/atc_tools/CLI_options/--log.md)。
插件回调抛出Python异常时，默认报错只有错误码（如E19999），Python侧的错误信息
（异常类型、语句、插件文件与行号）同样需要上述debug日志才能看到。

## 4、插件与回调写法

接入一个自定义算子，插件要做两类事：

1. 参数解析：把ONNX节点上的属性搬到GE算子上——用`parse_node`或`parse_operator`，选一个绑定；
2. 算子分解：GE没有现成算子时，把这个算子替换成已有算子拼出的子图——用`decompose`。

| 回调 | 做的事 | 用例 |
| ---- | ------ | ---- |
| `parse_node` | 按名字取属性（优先推荐） | `thresholded_relu_plugin.py` |
| `parse_operator` | 整体解析属性（少数情况，见4.2） | `my_elu_plugin.py` |
| `decompose` | 用已有算子拼子图替换原节点 | `thresholded_relu_plugin.py` |

### 4.1、parse_node：按名字取属性

优先推荐使用`parse_node`。回调收到`node`和`target`两个对象：

- `node`：ONNX节点。属性用`node.attrs["属性名"]`按名取，取到的直接是Python值。属性类型由
  torch导出时的后缀决定：

  | torch导出写法 | node.attrs取到 |
  | ------------- | -------------- |
  | `alpha_f=1.0` | `1.0`（float） |
  | `axis_i=1` | `1`（int） |
  | `name_s="x"` | `"x"`（str） |
  | `dims_i=[1,2]` | `[1,2]`（int列表） |

- `target`：目标算子。用`target.set_attr("属性名", 值)`把属性写进去。

支持的属性类型：int、float、string和同类列表。节点上带tensor、子图属性时`node.attrs`会
直接报错，只能改用`parse_operator`（见4.2）。

端口：目标算子输入输出个数固定时不用注册；个数不固定时必须注册，用`register_input`、
`register_output`、`register_optional_input`、`register_dynamic_input`、
`register_dynamic_output`。

用例：`plugin/thresholded_relu_plugin.py`——取`alpha`写入目标算子，并注册端口。

### 4.2、parse_operator：整体解析属性

以下情况使用`parse_operator`：

1. 节点带tensor、子图属性——`parse_node`的attrs读到会报错（见4.1），`parse_operator`
   拿到的JSON里全部都有；
2. 需要一次拿到节点上的全部属性做整体转写，不逐个预知属性名。

回调收到`source`和`target`两个算子：

- `source`：ONNX节点转成的算子（只读）。它身上只有一个名为`attribute`的属性，内容是节点
  全部属性的JSON串。各属性类型在JSON中的呈现如下（`name`为属性名，`type`为类型码：

  1=float、2=int、3=string、4=tensor、5=子图、6=float列表、7=int列表、8=string列表）：

```json
{
  "attribute": [
    { "f": "1.5",            "name": "alpha",  "type": 1 },
    { "i": 3,                "name": "level",  "type": 2 },
    { "s": "x",              "name": "label",  "type": 3 },
    { "floats": [1.5, 0.5],  "name": "coeffs", "type": 6 },
    { "ints": [2, 3],        "name": "dims",   "type": 7 },
    { "strings": ["a", "b"], "name": "tags",   "type": 8 }
  ]
}
```

  标量属性（`f`/`i`/`s`）的值是字符串形式，取用时转成数值；列表属性（`floats`/`ints`/
  `strings`）直接是JSON数组；tensor（`t`）与子图（`g`）属性则是完整的结构体字典；

- `target`：目标算子，同样用`set_attr`写入。

用例：`plugin/my_elu_plugin.py`——从JSON里取出`alpha`转给Elu。这个用例只有一个float属性，
用`parse_node`同样能实现，这里用它演示JSON的读法。

### 4.3、decompose：用已有算子拼子图替换原节点

GE没有现成算子时用`decompose`。回调收到`source`（参数解析产出的算子，读到的属性即
`parse_node`或`parse_operator`写入的值），返回一段用已有算子拼成的子图，GE会把原节点
替换成这段子图，因此不用写设备kernel。

拼子图用的是GE的ES构图API：`GraphBuilder`负责建图（创建输入、设置输出、构建），已有算子
（Threshold、Mul、SplitD等）是直接调用的函数。可用算子清单与参数说明见
[ES Python API文档](../../docs/zh/user_guides/es_graph/api/es_python.md)。

写`decompose`时注意：

- 子图的输出要和参数解析阶段注册的输出对应；
- 子图输入不用手动指定dtype和shape，GE会按原节点的输入自动对齐。

用例：`plugin/thresholded_relu_plugin.py`——`alpha`中转后拼出Threshold×Mul子图。
