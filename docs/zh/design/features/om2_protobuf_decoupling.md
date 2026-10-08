# OM2 执行态 Protobuf 解耦需求与详细设计（评审草稿）

> 状态：范围与总体策略已确认；Dump 与普通 AICPU 第一阶段穿刺已形成代码，跨组件错误码、AICPU 字段表和联合 golden 尚待冻结。本文整合 GE、普通 AICPU、TF AICPU 与 Dump 的边界，不替代各消费方实现设计。

# 简介

## 目的

面向 GE、RT、AICPU、opbase 开发与测试人员，规定 OM2 静态图加载/执行侧与 Protobuf 解耦的范围、载荷接口、实现位置、兼容策略和验收方法。

## 范围

- 覆盖 ACL 离线加载和 Session 在线加载 OM2；覆盖 GE `om2_executor.so` 自身的 Protobuf 直接依赖，以及 GE 生成并传至执行后处理模块的 Protobuf 载荷。
- 覆盖 OM2 的普通 AICPU `NodeDef`、TF AICPU `KernelRunParam`/`NodeDef`/`FunctionDefLibrary` 生成侧改造，以及 GE OM2 普通、溢出、custom kernel Dump 的 `OpMappingInfo` 改造。消费方须具备新 TLV 与旧 OM Protobuf 双格式能力；TF AICPU 消费方的内部实现另立详细设计。
- 不修改旧 OM 文件；OM2 尚在首版本开发，不要求读取旧 OM2 文件。已有第三方/自定义任务生成器无需升级，仍可生成新 OM2。
- 不处理 `ascend_dump` 本身及其传递 Protobuf 依赖；不以本需求宣称 `om2_executor.so` 的所有传递依赖均不含 Protobuf。不处理非 GE 可达的 `aicpu_cust_schedule` custom Dump。
- 不处理 `visual.json`、动态图/动态 shape、RT2 动态执行、FFTS+、不透明第三方载荷的 Protobuf 解耦；这些点纳入依赖清单与后续专题。

# 总体概述

## 软件概述

### 项目介绍

OM2 目前支持静态图。GE 图编译仍使用 Protobuf 描述图和任务；本需求限制的是 GE OM2 执行态对 Protobuf 的直接依赖，以及新 OM2 中由 GE 生产、运行时交给后处理模块解释的数据格式。仅在生成过程中使用、没有进入 OM2 执行数据的 Protobuf 暂保留。

### 产品环境介绍

```text
ACL 离线 / Session 在线 ──> GE OM2 codegen ──> OM2 包 + 模型 SO
                                                   │
                                  GE OM2 Executor ─┼─> RT/AICPU Dump
                                                   ├─> opbase 普通 AICPU
                                                   └─> TF AICPU
旧 OM ──> 原有加载执行路径 ────────────────────────────> 各消费方 Protobuf 入口
```

GE OM2 codegen 在生成边界转换已识别载荷；模型 SO 和 Executor 只搬运/修补运行参数，消费方按格式分派。旧 OM 路径保持原 Protobuf 数据。

## 软件功能

1. 移除 GE OM2 Executor 自身的 Protobuf 生成类型、解析/编码及链接依赖。
2. 对 GE 可识别的 OM2 AICPU、TF AICPU、Dump 载荷生成统一框架的 TLV；保持语义与原有执行时机。
3. 消费方兼容旧 OM Protobuf 与新 OM2 TLV；未知第三方载荷原样透传。

## 设计约束

- 不改变模型 SO↔Executor 既有 C ABI、导出符号、参数顺序和公开结构体布局；不把协议细节放进 `om2_model_data.h`。
- 共用 TLV 框架、分配不同业务 tag。代码允许各仓各自维护一份，不要求文件同源；字节协议和 golden 必须一致。
- 不靠任务类型（如 `AI_CPU`）单独识别业务协议，须结合实际 kernel SO、入口、载荷布局。无法确认者不转码。
- 已识别载荷转换失败须明确报错，不得把半转换或损坏的载荷写入 OM2；消费端识别为 TLV 后解析失败，不得回退 Protobuf。

## 假设和依赖关系

- GE TLV 生产与 RT/AICPU 双格式消费必须按同一发布门禁交付。旧 OM 回归和新 OM2 联合 golden 是放量前置条件。
- Dump 的业务字段与通道以 `C:\om2_dump_protobuf_decoupling_design.md` 和 `C:\om2_dump_ge_rt_aicpu_interface_design.md` 为专项输入。其“组件不兼容旧 Protobuf 载荷”的文字须按本文统一修订为“旧 OM PB、新 OM2 TLV，消费端双格式”；**不**表示一个载荷内混合两种格式。
- TF AICPU 消费实现、跨组件错误码、TLV 最大长度和字段编号仍需联合确认；未冻结前不能开启 OM2 TLV 产物交付。

# 特性1需求分析&设计

## 整体介绍

本特性以“编译内允许 PB、执行数据不带 GE 已知 PB”为界，先从 GE 自身剥离运行时 PB，再在 OM2 codegen 末端转换需由运行时消费的字段。编译期 PB 清单在“兼容性检查”中逐项列出，不扩大为图编译整体去 PB。

## 功能需求

### GE OM2 Executor 自身去 Protobuf

1. 介绍：`om2_executor.so` 的 GE 自有对象文件和链接项不直接使用 Protobuf；保留 `ascend_dump` 依赖。
2. 输入：OM2 包、模型 SO、运行参数；ACL 和 Session 两入口执行相同语义。
3. 处理：拆出/替换 `runtime/om2` 内的 PB 类型与工具；Dump 代码用 TLV 编码；`om2_codegen_types.h` 对 `TaskDef` 仅前置声明，编译实现文件显式包含定义。清理 `runtime/om2/CMakeLists.txt` 中的 `graphengine_protos`、PB 宏和直接链接项；检查 `type_utils_inner.cc` 等 glob 纳入的源码。不改变 `om2_model_data.h` 或模型 SO 的 C ABI。
4. 输出：GE 自身编译和 ELF 直接依赖不含 Protobuf，OM2 加载/执行结果保持一致。对第三方库传递依赖单独报告，不作为本次验收失败条件。

### 普通 AICPU 载荷

1. 介绍：新 OM2 的已识别普通 AICPU 任务携带 TLV `NodeDef`；旧 OM 仍携带 Protobuf。
2. 输入：任务生成器给出的 `AicpuParamHead + IO 地址数组 + u32 NodeDef 长度 + NodeDef`，以及 kernel SO/入口识别信息；输入可来自旧版第三方生成器。
3. 处理：仅对确认使用 `libcpu_kernels.so`/`RunCpuKernel` 协议的任务解析并转码 `aicpuops::NodeDef`；保留参数前缀、IO 地址布局和扩展信息，替换 NodeDef 后缀，重算长度并同步 `head.length`、`KernelDef.args_size`。旧生成器不需改动；不可识别的第三方载荷保持原字节。`RunCpuKernelWithBlock` 消费侧共享解码，但当前未找到 GE 对应明确生产路径，不能凭名称扩大生成范围。
4. 输出：新 OM2 已识别任务的 NodeDef 不含 PB 编码；opbase 两入口用格式分派并提供原语义对象。所有长度、偏移均做溢出和边界校验。当前未支持的 GE 属性类型仍按旧逻辑保留同名空 `AttrValue`，不能悄悄丢键。

### TF AICPU 载荷

1. 介绍：新 OM2 `KERNEL_EX` 中已确认的 TF 载荷不再含 PB 消息；TF 设备消费实现另行设计。
2. 输入：`task_info` 中依次编码的 `KernelRunParam`、TF `NodeDef`、可选 `FunctionDefLibrary`，以及 `STR_FWK_OP_KERNEL` 长度和偏移元信息。
3. 处理：仅当 SO/入口/布局确认 TF 协议时，将两个必需消息和可选第三消息分别转换成完整 TLV frame，再串接；逐层转换 TF 消息字段及嵌套消息，业务字节串保持原样；更新各段长度、偏移和 `task_info_size`。无 FunctionDefLibrary 时只输出两个 frame，第三段长度为 0。现有 OM2 codegen 对 `KERNEL_EX` 有 TF 假设；发现非 TF 协议或不合法长度时失败，不静默转码。
4. 输出：模型 SO 继续传送原始字节，不解析 PB/TLV；TF AICPU 新消费入口读取 TLV，旧 OM 入口仍读 PB。TF 动态 shape MemCopy 辅助任务暂不转换。

### Dump 载荷

1. 介绍：覆盖 GE 可达的模型级普通/溢出 Dump 与 custom kernel 执行前后 Dump；不重做专项业务设计。
2. 输入：既有 Dump 元数据、配置和执行时机。
3. 处理：GE OM2 按 Dump 专项设计构造 `DumpWire v1`；模型级继续走 `rtDatadumpInfoLoad`，custom 继续走 `DumpDataInfo` 启动。RT/AICPU 按首部识别 TLV 或旧 OM PB。识别 TLV 后遇到非法 version/长度/字段，应报 TLV 错，不能以 PB 重试。
4. 输出：两通道的业务语义、最终 dump 文件格式和用户配置不变。`ascend_dump` 内部解耦不属于本文。

### 旧 OM 与第三方生成器兼容

1. 介绍：旧 OM PB 不改写；第三方任务生成器无需升级。
2. 输入：旧 OM PB 载荷、新 OM2 TLV 载荷、无法确认业务协议的第三方字节载荷。
3. 处理：消费端依 payload 格式选两套解析；新 OM2 codegen 只转码 GE 已知、消费契约明确的载荷。不透明 DVPP、HCCL、第三方 AICPU 等按原样传递，不对所有 `TaskDef` 做盲转码。
4. 输出：旧 OM 在新组件上保持原行为；新 OM2 已识别载荷不再是 PB；不保证旧组件执行新 OM2。

## 非功能需求

### 可维护性

TLV frame、整数读写、边界校验、未知 tag 跳过、嵌套遍历和错误分类共用；业务 schema 由各域管理。未来压缩应在公共 framing 之上扩展，不在每种载荷复制压缩逻辑。

### 可测试性

同一批跨仓 golden 必须覆盖字节完全一致、字段 presence、重复 tag、未知 tag、嵌套与畸形输入；保留原 PB 路径回归。所有转换函数可用纯内存输入输出单测，不依赖设备。

### 可移植性

多字节整数明确为小端，固定宽度类型编码；不通过 `memcpy` C++ 对象作为 wire。结构体 packing、宿主端序不影响结果。

### 可靠性

解析先校验长度与层次，再原子提交结果；缓存项持有解析对象所需 backing，不能引用已释放的任务字节。

### 平台化要求

GE 不按芯片型号分支；RT/AICPU 的两个 Dump 通道和普通/TF AICPU 入口需在目标平台逐一确认支持能力。

### 特性交叉分析

涉及旧 OM、ACL/Session、普通/溢出/custom Dump、模型 SO ABI、普通 AICPU cache、TF 函数库、第三方任务、profiling/异常 Dump。动态图、`visual.json`、FFTS+ 已标记后续专题，不以本次实现的静态图通过代表其通过。

## 性能

### 模型编译时长

新增转换在 OM2 codegen 阶段、按识别出的任务线性执行；须对 optimizeStage1、optimizeStage2、build、loadmodelonline 分段量测，重点比较 build 和在线加载。当前没有实测值，性能门限待基线确认。

### OM 大小和加载占用内存

记录 PB→TLV 前后模型大小、峰值内存。任务视图只复制需转换的 `TaskDef`；转换完成释放临时缓冲，不对整个图和所有任务深拷贝。若 TLV 显著增大须评审，不默认引入压缩。

### 执行性能

静态图执行热路径不重复解码；普通 AICPU cache 命中时验证格式和载荷身份后复用。分别量测首执行解析、重复执行、并发执行及 Dump 开/关；阈值待性能基线冻结。

## 接口设计

### 新增/修改接口描述

跨组件载荷协议为 `DumpWire v1` framing：16 B 首部 `magic:u32=0x544C5601`、`version:u16=1`、`header_size:u16=16`、`total_size:u32`、`record_count:u32`；每条 record 是 `tag:u16 + payload_len:u32 + payload`，均小端。每个普通 AICPU 或 TF 消息为一个顶层 record 的独立 frame；TF `task_info` 串接 2～3 个 frame。Dump 字段/tag 以专项接口文档为准。分配范围暂定 Dump `0x0000–0x0FFF`、普通 AICPU `0x1000–0x1FFF`、TF `0x2000–0x3FFF`；普通 AICPU root 暂定 `0x1000`，TF 三个 root 暂定 `0x2000/0x2001/0x2002`。**除已在 Dump 专项冻结的字段外，AICPU/TF 的 tag、presence、重复语义和限制值尚未冻结，不可据此编码上线。**

普通 AICPU 至少表达 op、属性 map、输入/输出 tensor 和动态 map；属性覆盖 array、bytes、int64、float、bool、type、shape、tensor、list-list-int，区分空属性值和空数组。TF 转换覆盖 `KernelRunParam`、`NodeDef` 和 `FunctionDefLibrary` 的递归消息。map 按键排序以保证 golden 稳定；repeated 保持原出现顺序。未知字段按合法长度跳过；已知字段层次或宽度不合法则失败。

#### 第一阶段 Dump + 普通 AICPU 穿刺协议快照

第一阶段 PR 同时包含 Dump 生产侧穿刺和普通 AICPU 生产侧穿刺。二者复制维护相同的 16 B frame/6 B record wire 规则，但业务 schema 独立：Dump 使用 `0x0000-0x0FFF`，普通 AICPU 使用 `0x1000-0x1FFF`。这次公共化的是字节规则、校验原则和测试方法，不把两类业务对象合并，也暂不引入跨仓公共头文件。

Dump schema 已由专项穿刺定义：顶层 `MODEL=0x0001`、`TASK=0x0002`，子记录覆盖 INPUT、OUTPUT、WORKSPACE、BUFFER、ATTR、CONTEXT 和 DIM_RANGE。完整字段编号以 `dump_transport_info.h` 及其 golden/畸形输入 UT 为准；GE 侧保持模型级 `rtDatadumpInfoLoad` 和 custom `DumpDataInfo` 两条既有通道、时序及设备内存生命周期。

普通 AICPU 只识别 `libcpu_kernels.so` / `RunCpuKernel`，其已确认的 `TaskDef.kernel.args` 布局为：

```text
AicpuParamHead
uint64_t io_addrs[head.ioAddrNum]
uint32_t node_def_len
uint8_t node_def[node_def_len]
```

GE 生产证据位于 `cpu_kernel_builder.cpp` 的 `BuildArgs`、`BuildMemCopyInfo` 和 `BuildAndLaunchKernel`；opbase 消费证据位于 `cpu_kernel_cache.cc` 的 `ParseIoAddr`、`GetCpuKernelContext` 和 `GetCpuKernelContextWithBlock`。转换只发生在 OM2 任务副本，保持 IO 地址前缀不变，更新 `node_def_len`、`AicpuParamHead.length` 与 `KernelDef.args_size`；其他 SO、入口或无法确认布局的第三方任务按原字节透传。

下表是普通 AICPU 穿刺使用的最小字段表。除公共 frame 首部外，所有 tag 和 `value_type` 都是评审暂定值，不表示接口已冻结：

| tag | 名称 | payload |
| --- | --- | --- |
| `0x1000` | NodeDef root | 嵌套 record |
| `0x1001` | op | UTF-8 bytes |
| `0x1002` | input tensor | 嵌套 tensor record，可重复 |
| `0x1003` | output tensor | 嵌套 tensor record，可重复 |
| `0x1004` | attr | 嵌套 attr record，可重复 |
| `0x1011` | tensor.shape | 嵌套 shape record |
| `0x1012` | tensor.tensor_type | `i32` |
| `0x1013` | tensor.name | UTF-8 bytes |
| `0x1014` | tensor.data_ptr | `u64` |
| `0x1015` | tensor.data_size | `u64` |
| `0x1021` | shape.dims | `u32 count + i64[count]` |
| `0x1022` | shape.unknown_rank | `u8` |
| `0x1023` | shape.data_format | `i32` |
| `0x1031` | attr.name | UTF-8 bytes |
| `0x1032` | attr.value | `u16 value_type + typed payload` |

`attr.value` 暂定 `value_type` 为：`0 empty`、`1 string/bytes`、`2 int64`、`3 float32`、`4 bool`、`5 data_type(i32)`、`6 shape`、`7 tensor`、`8 list_string`、`9 list_int64`、`10 list_float32`、`11 list_bool`、`12 list_data_type`、`13 list_shape`、`14 list_tensor`、`15 list_list_int64`。

`list_shape` 与 `list_list_int64` 不是纯格式别名。现有 GE builder 对 `VT_LIST_LIST_INT` 的历史实现写入 PB `AttrValue.array.shape`，因此穿刺保持为 `list_shape`；`list_list_int64` 只对应 PB `AttrValue.list_list_int` oneof。空 `AttrValue` 继续表达“不支持的 GE 属性类型仍保留同名空键”，不能省略该 attr。

普通 AICPU 最小 golden 为 `NodeDef { op: "Relu" }`：

```text
01 56 4C 54 01 00 10 00 20 00 00 00 01 00 00 00
00 10 0A 00 00 00 01 10 04 00 00 00 52 65 6C 75
```

其含义为：frame `total_size=32`、`record_count=1`；root tag 为 `0x1000`、payload 长度为 10；内部 op tag 为 `0x1001`、payload 为 `Relu`。GE 编码和 opbase 解码必须逐字节使用同一 golden。消费端一旦识别 magic 即进入 TLV 路径；version、总长、record、层次或固定宽度非法时直接报错，不得回退 PB。无 TLV magic 的旧 OM 继续走 PB 路径。

公开 ACL/Session API、模型 SO↔Executor C ABI、`GertModelTaskDesc` 和 `DumpDataInfo` 参数个数不变。GE↔RT/AICPU/opbase 的载荷格式变更属于组件间接口，必须联合评审并维护字段级接口文档。

### 接口检查项

| 检查项 | 结论 |
| --- | --- |
| 接口评审与资料 | 涉及 GE↔RT/AICPU/opbase，必须联合评审；Dump 专项文档与 AICPU/TF 字段表均须冻结。 |
| 原型、返回值、时序 | 原有入口原型不变；新增 TLV 编解码内部接口与错误映射待字段表一同冻结。 |
| 行为兼容 | 旧 OM PB 与新 OM2 TLV 双解析；不兼容的旧消费方不得接收新 OM2。 |
| 错误与测试 | 非法 magic/version、长度、层次、重复及边界必须确定错误码并有负例。 |

## 软件设计

### 关键数据结构

- `Om2TaskView`（暂名）：与原 `GeModel` 并存的 OM2-only 任务视图，引用未改任务、持有需转换任务副本；其生命周期覆盖所有模型构建步骤。它不是跨 SO ABI 结构。
- `TlvFrame`/reader/writer（暂名）：只处理首部、record、宽度、溢出、层次和错误，不知道具体业务字段；各业务 mapper 负责语义与 presence。
- opbase 普通 AICPU 对外 PImpl 语义保持，新增 TLV native backing；PB 路径继续用原对象，TLV 路径不得通过 PB `ParseFromString` 或构造 PB 消息。

### 关键技术/算法

转换入口置于 `Om2Codegen::Om2CodegenAndCompile` 中创建 task builder、计算 args 布局之前。`om2_codegen_model_builder.cc` 的 task 读取点及 `om2_model_args_manager.cc` 的读取点必须统一引用同一任务视图；`BuildKernelRegistry`、`GenerateArgsData`、`BuildTaskSemantics`、`BuildConstInputs` 不得分读原/新任务。`ModelAdapter` 仍可从原 `GeModel` 读图与内存信息，不回写原任务，从而不影响旧 OM。

普通 AICPU 的 cache 不能只以 `kernel_id` 判定复用：格式、载荷长度和身份摘要/等价判定均须参与，先验中才允许绕过解析；`sessFlag` 与既有清理语义保持。`ParseIoAddr` 读 NodeDef 长度后必须再次检验剩余字节，防止越界。TF 各段偏移、长度以原始 `task_info` 上限检验，更新后再检验总长和字段宽度。

### 流程设计

```text
既有生成器 TaskDef ──> 识别业务协议 ─┬─ 已知：PB 语义解析→TLV 编码→OM2-only 任务副本
                                  └─ 未知：原字节引用/透传
                  ──> 统一任务视图 ──> 模型构建/参数布局/模型 SO
                  ──> GE Executor 搬运 ──> 消费方格式分派 ──> 原业务动作
旧 OM PB ──────────────────────────────────────> 消费方 PB 解析
```

### 对子模块的修改

GE OM2 codegen 增加已知任务载荷 mapper 与统一任务视图；OM2 runtime/CMake 清理自有 PB 依赖；Dump 按专项设计；opbase 增加普通 AICPU TLV native 解码及 cache 判等；TF AICPU、RT Dump 增加双格式入口。涉及主流程的模块级设计须同步更新。

### 错误处理

#### 系统错误

分配失败、长度加法溢出、无法容纳 `u32` 长度、任务视图构建失败均停止 OM2 生成或本次加载，释放临时对象；不输出部分有效的 OM2。设备内存生命周期沿用既有接口，Dump 载荷与长度对象的释放时点须 RT/AICPU 确认。

#### 接口错误

区分非法格式、不支持版本、长度越界、不支持的已识别业务协议、缺少必需字段和资源不足；消费方映射到既有 GE/RT/AICPU 错误码。具体数值待跨组件冻结，不新增未经评审的外部错误码。

## 安全检查

### 编码军规

所有不可信载荷先检查 frame 和 record 总长、嵌套深度、计数与整数溢出；不按未验证长度分配内存；不将 wire 指针跨缓存生命周期保存。模糊测试覆盖两个解码入口。

### 编码检查项

| 检查项 | 结论 |
| --- | --- |
| 资源生命周期 | 涉及任务视图、TLV 缓冲和 AICPU cache；随编译/模型/Session 既有生命周期释放。 |
| 新线程 | 不创建新线程。 |

## 兼容性检查

旧 OM 文件及其 PB 字节不修改，新消费方须继续解析。新 OM2 只生成 TLV 的已识别业务载荷，不要求旧 OM2 兼容。第三方不透明字节保持原样；如果其自身为 PB，不能因此宣称“OM2 全部字节零 PB”，验收按 GE 已知协议清单逐项判定。`ascend_dump` 的 PB 传递依赖另列残留。模型 SO↔Executor C ABI 不变，按 `inc/framework/README.md` 的 ABI 规则校验符号、结构体和行为。

编译期 Protobuf 依赖清单与处置：图/算子/属性的 `GraphDef`、`OpDef`、`TensorDef` 等内部表示和序列化仅参与图编译，暂不解耦；`TaskDef`/`KernelDef` 作为任务生成中间物仍可为 PB，但其中进入 OM2 运行时且由 GE 已知消费方解析的普通 AICPU 与 TF AICPU 子载荷必须转码；Dump `OpMappingInfo` 生成及运行侧按专项解耦；`visual.json` 单独确认；动态 shape/RT2 中依赖 PB 的执行数据单独记录并延期；FFTS+ 任务虽存在编译构造，但当前 OM2 不支持 `MODEL_TASK_FFTS_PLUS`，暂不纳入。此清单为当前静态图代码路径结论，后续 OM2 支持动态图时须重新逐项评估。

## DT设计

### 测试边界

GE 转换入口到 OM2 产物字节、模型 SO/Executor 搬运、RT/opbase/TF 消费与用户可见结果，分别设 UT、ST、跨仓 golden。构建依赖使用真实链接产物核对，不只用源码文本搜索。

### 测试设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
| --- | --- | --- | --- |
| 功能 | ACL/Session 静态图、普通 AICPU/TF/Dump、空/嵌套属性、可选函数库 | 编译新 OM2 并比较原业务结果和载荷字节 | UT/ST |
| 错误 | frame、段长、offset、嵌套、未知/重复 tag、非 TF `KERNEL_EX` | 畸形输入与 fuzz；确认不 PB fallback | UT |
| 兼容性 | 旧 OM PB、新 OM2 TLV、旧生成器、不透明第三方 | 双格式消费回归和任务原字节比较 | ST/BBIT |
| 性能 | 编译各阶段、首执行/缓存命中、Dump 开关、内存/OM 大小 | 同机同模型对照并记录基线 | ST |
| 特性交叉 | custom Dump、溢出 Dump、异常/profiling、模型 SO ABI | 联合 golden 与端到端回归 | ST/BBIT |

现有 TF faker 的 `task_info` 为任意字符串且长度信息为零，不满足严格转换输入；应改为合法最小 PB fixture，不得放松解析器。普通 AICPU faker 使用未确认 SO/入口，应保留作不透明透传用例，并补充真实已识别任务用例。

### 测试框架设计

GE 使用纯内存 codec/golden 测试与 OM2 codegen ST；RT/opbase/TF 消费仓读取相同 golden。现有在线 OM2 ST 使用 fake SO，需另补真实载荷端到端案例。构建时检查 `om2_executor.so` 直接 `NEEDED`、链接命令和符号引用；对 `ascend_dump` 传递依赖单独报告。

## 验收标准

1. ACL 离线、Session 在线生成和加载新 OM2 成功；普通/TF AICPU 与 GE 可达 Dump 已知载荷均为冻结版 TLV，业务行为与原路径等价。
2. GE OM2 Executor 自身不包含 PB 生成类型/调用及直接链接；保留的 `ascend_dump` 传递依赖有清晰清单。
3. 新消费方能分别解析旧 OM PB、新 OM2 TLV；损坏的 TLV 不回退 PB；第三方旧任务生成器不用升级。
4. 模型 SO↔Executor C ABI 和旧 OM 数据格式不变；跨仓 golden、负例、性能与旧 OM 回归通过。

## 待冻结事项

1. 普通 AICPU、TF AICPU 字段级 tag、presence、重复字段、最大长度/深度、错误码映射及 golden；其中 TF 消费实现由专项详细设计补齐。
2. `DumpWire` 专项文档中的“组件不兼容旧载荷”需修订为本文的旧 OM PB / 新 OM2 TLV 双解析边界；Dump 设备内存读取与释放时序由 RT/AICPU 复核。
3. GE、RT、AICPU、opbase 的联调版本与灰度门禁；性能与大小的量化基线。以上未完成前本文不能视为可直接实施的冻结版详细设计。
