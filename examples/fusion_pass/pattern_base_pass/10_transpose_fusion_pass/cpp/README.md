# transpose_fusion_pass - GraphNodeSettedFormatPass 样例

## 功能描述

本样例演示如何继承 `FusionBasePass` 实现 **基于配置文件的算子格式修改与冗余 Transpose 消除**。

通过 `GraphNodeSettedFormatPass`，用户可以在 `kAfterOriginGraphOptimize` 阶段，
根据配置文件批量修改图中指定算子的输入/输出 format 和 shape，并通过 `CheckOpSupported` 校验修改是否合法，
校验通过后自动检查并删除因 format 变更而变冗余的 Transpose 节点。

### 典型场景

在昇腾 AI 处理器上，某些算子（如 Conv2D）更倾向于使用 NHWC 格式。若模型中存在
`Data(NCHW) -> Transpose(NCHW->NHWC) -> Conv2D(NHWC)` 的结构，用户可通过配置文件将 Data 节点
的输出格式直接改为 NHWC，使 Transpose 两侧 format 一致变为冗余，pass 自动删除该 Transpose 并直连
Data 与下游算子，减少中间数据转换。

```
Before:                              After:

Data(NCHW)                           Data(NHWC)
    |                                    |
 Transpose                             Conv2D
 (perm=[0,2,3,1])                     (NHWC)
    |
 Conv2D(NHWC)
```

### 配置文件格式

配置文件通过环境变量 `ASCEND_CUSTOM_FORMATS_CFG` 指定路径，格式为简单文本：

```ini
# 以 [NodeName] 作为 section 分隔，NodeName 为图中节点的名称
[conv1]
input.0=FORMAT_NCHW
output.0=FORMAT_NHWC

[matmul_custom]
input.0=FORMAT_FRACTAL_NZ
input.1=FORMAT_FRACTAL_NZ
output.0=FORMAT_FRACTAL_NZ
```

- 以 `[NodeName]` 作为 section 分隔，`NodeName` 为图中节点的名称（node_name）
- `input.<index>=<format>` 或 `output.<index>=<format>` 配置指定端口的格式
- format 值参考 `ge::Format` 枚举（如 `FORMAT_ND`, `FORMAT_NCHW`, `FORMAT_NHWC` 等）
- 空行和 `#` 开头的注释行会被忽略

### Pass 执行流程

```
Run()
  │
  ├─ 1. ParseConfigFile()         从环境变量读取配置文件路径，解析 node_name → FormatConfig 映射
  ├─ 2. 备份原图                   Graph origin_graph = *graph（用于整图回滚）
  ├─ 3. 遍历图中所有节点:
  │      └─ ApplyFormatAndCheck()  对匹配配置的节点执行：
  │           ├─ Step1: 备份当前 input/output format + shape
  │           ├─ Step2: 修改 input/output format + shape（联动转换 shape 维度）
  │           ├─ Step3: 传播 output format 到直连 NetOutput 节点
  │           ├─ Step4: CheckOpSupported 校验算子是否支持（Data 节点跳过）
  │           │           └─ 失败 → 节点级回退（RollbackFormats + RollbackNetOutput）
  │           └─ Step5: 检查并删除前后变冗余的 Transpose 节点
  │                       └─ 失败 → return false（触发整图回滚）
  └─ 4. if 任一节点失败:
         └─ 整图回滚 *graph = origin_graph，返回 FAILED
```

### 回滚机制

| 层级 | 触发条件 | 机制 |
|------|---------|------|
| **节点级回退** | CheckOpSupported 失败 / format 修改失败 | `RollbackFormats` + `RollbackNetOutput` 恢复当前节点及关联 NetOutput 的 format 和 shape |
| **整图回滚** | 任一节点 `ApplyFormatAndCheck` 返回 false（含 Transpose 删除失败） | `*graph = origin_graph` 恢复全部修改 |

### 已知限制

- **控制边不处理**：删除冗余 Transpose 节点时，仅处理数据边（data edge）的移除和重连，不处理控制边（control edge）。若 Transpose 节点存在控制边输入或输出，直接删除节点会导致控制依赖关系丢失，可能影响图的执行语义。因此，当前实现仅适用于 Transpose 无控制边的场景。

## 目录结构

```
├── src
│   └── graph_node_setted_format_pass.cpp     // GraphNodeSettedFormatPass 实现文件
├── CMakeLists.txt                             // 编译脚本
├── data
│   ├── custom_formats.cfg                     // 配置文件示例
│   ├── gen_onnx.py                            // ONNX 模型导出脚本
│   ├── verify_format.sh                       // 格式验证脚本
│   ├── run_atc.sh                             // ATC 转换脚本（带 DUMP）
│   ├── format_analysis.md                     // 格式分析结论
│   └── format_analysis_en.md                  // 格式分析结论（英文）
└── gen_es_api
    └── CMakeLists.txt                          // ES API 生成脚本
```

## 环境要求

- 编译器：GCC >= 7.3.x
- 使用 python 及其依赖库版本：python>=3.9、onnx
- 已完成[环境准备](../../../../../docs/zh/build.md#1-环境准备)。

## 实现步骤

1. 定义类 `GraphNodeSettedFormatPass` 继承 `FusionBasePass`。
2. 重写 `Run` 方法，主要逻辑包括：
   - 从环境变量 `ASCEND_CUSTOM_FORMATS_CFG` 指定的配置文件中解析节点格式配置。
   - 备份原图，遍历图中节点，对匹配配置的节点修改 input/output format 和 shape。
   - 修改 format 时同步重排 shape 维度（如 NCHW→NHWC 时 `[N,C,H,W]`→`[N,H,W,C]`）。
   - 若输出直连 NetOutput 节点，同步修改 NetOutput 对应输入端口的 format 和 shape。
   - 调用 `GeUtils::CheckNodeSupportOnAicore` 校验算子是否支持修改后的格式组合（Data 节点跳过校验）。
   - 校验失败时回退当前节点及关联 NetOutput 的修改；Transpose 删除失败时触发整图回滚。
   - 校验通过后检查并删除前后因 format 变更而变冗余的 Transpose 节点。
3. 注册 `GraphNodeSettedFormatPass` 为自定义融合 pass，执行阶段为 `kAfterOriginGraphOptimize`。

## 程序编译

假设 CANN 软件包的安装目录为 INSTALL_PATH，例如 `/home/HwHiAiUser/Ascend/`。

1. 配置环境变量。

   ```bash
   source ${ASCEND_HOME_PATH}/set_env.sh
   ```

   `${ASCEND_HOME_PATH}` 为 CANN 软件包安装目录下的 cann 路径。请替换相关软件包的实际安装路径。

2. 根据实际情况修改 `CMakeLists.txt` 中的如下信息。
   - ASCEND_PATH：可通过 `set_env.sh` 设置 `$ASCEND_HOME_PATH` 自动指定。
   - PASS_SO_DIR：自定义融合 pass 动态库安装目录名，默认为 `pass_so_dir`。

3. 编译并安装：

   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc) data_transpose_fusion_pass
   make install
   ```

4. 若 build 目录被删除或 pass so 迁移，需将 es so 拷贝到安装目录：

   ```bash
   cp build/es_output/lib64/libes_all.so ${ASCEND_PATH}/opp/vendors/${PASS_SO_DIR}/custom_fusion_passes/
   ```

## 程序运行

### 方式一：格式验证（不安装 pass）

使用 `data/verify_format.sh` 脚本可以直接分析 Data 节点的格式：

```bash
cd data
bash verify_format.sh Ascend910_9362
```

脚本会自动：
1. 生成 ONNX 模型
2. 使用 ATC 编译并 dump 图
3. 分析 Data 节点格式

### 方式二：ATC 离线推理（安装 pass 后验证融合效果）

1. 设置环境变量：

   ```bash
   export DUMP_GE_GRAPH=2
   export ASCEND_CUSTOM_FORMATS_CFG=/path/to/custom_formats.cfg
   ```

2. 生成 ONNX 模型：

   ```bash
   cd data
   python3 gen_onnx.py
   ```

3. 执行 ATC 编译：

   ```bash
   atc --model=./model.onnx --framework=5 --soc_version=xxx --output=./model
   ```

   或直接使用脚本：

   ```bash
   bash run_atc.sh Ascend910_9362
   ```

4. 日志中出现如下打印即表示 pass 已生效：

   ```
   GraphNodeSettedFormatPass is starting
   [GraphNodeSettedFormatPass] Parsed 4 op config(s) from /path/to/custom_formats.cfg
   [GraphNodeSettedFormatPass] Processing node[conv1] (type=Conv2D)
   [GraphNodeSettedFormatPass] Node[conv1]: format applied and CheckOpSupported passed
   [GraphNodeSettedFormatPass] Removed redundant Transpose[transpose_0]
   GraphNodeSettedFormatPass completed successfully
   ```

5. 对比 dump 图即可验证 Transpose 已被删除，对应算子 format 已修改。

## 清理

测试完成后，执行清理：

```bash
make clean_custom_pass
```
