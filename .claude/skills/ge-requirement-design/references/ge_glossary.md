# GE 术语映射表

供 `analysis.md` / `competitor.md` / `implementation.md` 共用，确保领域语言一致。**所有子文档在描述 GE 内部模块时必须使用本表的左列术语作为主语**，外部对比时再映射到右列。

## 数据模型

| GE 术语 | 通用图编译器对应 | 说明 |
|---------|------------------|------|
| `ComputeGraph` | High-level IR (Graph) | 图容器，包含 node list、input/output、subgraph |
| `SubGraph` | Subgraph / Region | 引擎分区后或控制流嵌套的子图 |
| `Node` | Operator Node | 算子节点，持有 OpDesc + Anchor |
| `OpDesc` | Operator metadata | 算子描述，包含输入/输出 TensorDesc、属性、推导函数 |
| `GeTensorDesc` | Tensor metadata | 张量元信息（shape、dtype、format） |
| `DataAnchor` (InData/OutData) | Data edge endpoint | 数据依赖端点 |
| `ControlAnchor` (InCtrl/OutCtrl) | Control edge endpoint | 控制依赖端点 |
| `AscendIR` | High-level IR Spec | GE 图表示规范，无独立 Edge 对象，连接由 Anchor 双向引用维护 |

## 编译管线

| GE 阶段 | 通用对应 | GE 模块路径 |
|---------|----------|-------------|
| Parser | Frontend Capture | `parser/` |
| Preprocess (GraphPrepare / InferShape / InferFormat) | Frontend Normalization | `compiler/graph/preprocess/` |
| PreRunOptimizeOriginalGraph | Pre-partition optimization | `compiler/graph/optimize/`、`compiler/graph/passes/` |
| EnginePartition | Backend partition | `compiler/graph/partition/` |
| PreRunOptimizeSubGraph | Per-subgraph optimization | `compiler/graph/optimize/` |
| PreRunAfterOptimizeSubGraph | Post-partition optimization | `compiler/graph/optimize/` |
| Build (StreamAllocator / MemoryPlanner / TaskBuilder) | Code generation + scheduling | `compiler/graph/build/` |
| Runtime V1 (静态) | AOT / Static runtime | `runtime/v1/` |
| Runtime V2 (RT2.0 动态) | Dynamic / Hybrid runtime | `runtime/v2/` |

## 优化与变换

| GE 术语 | 通用对应 | 说明 |
|---------|----------|------|
| `Pass` | Pass | 图变换单元 |
| `GraphPass` | Module-level pass | 作用于整图 |
| `NodePass` | Function-level pass | 作用于节点（含立即/延迟重遍历） |
| `Pattern Matcher` | Pattern rewriter | 模式匹配引擎，用于融合 Pass |
| `autofuse` | Auto fusion | 自动融合特性 |
| `Cluster` | Partition unit | EnginePartition 的最小单元 |
| `HasSecondPath` | Cycle check | 引擎分区时防止破坏结构的合法性检查 |
| `SymbolToAnchors` | Liveness equivalence class | 内存规划中共享地址的锚点等价类 |
| `TransData` | Layout conversion op | 格式转换算子，由 InferFormat 决定插入 |
| `Tiling` | Kernel parameter tuning | 算子下沉参数计算 |
| `Online Tiling` / `Dynamic Tiling` | Runtime kernel tuning | 运行时计算 tiling 参数 |

## 引擎与执行

| GE 术语 | 通用对应 | 说明 |
|---------|----------|------|
| `AICore Engine` | NPU tensor core | 主算力引擎 |
| `AICPU Engine` | NPU CPU engine | AICPU 算子引擎 |
| `VectorEngine` | Vector unit | 向量算力引擎 |
| `HCCL Engine` | Collective communication | 集合通信引擎 |
| `host_cpu_engine` | Host CPU | 主机 CPU 引擎 |
| `GE Local Engine` | Virtual/no-op engine | GE 内置虚拟引擎（NoOp、PhonyConcat 等） |
| `Stream` | Execution stream | 执行流，支持多流并行 |
| `Task` | Execution task | 最细粒度调度单元 |
| `EnginePlacer` | Backend selector | 引擎选择器 |
| `DnnEngine` / `OpsKernelInfoStore` | Backend plugin interface | 引擎插件接口 |

## 产物与对外接口

| GE 术语 | 通用对应 | 说明 |
|---------|----------|------|
| `om Model` | Compiled artifact | 离线编译产物（Offline Model） |
| `ATC` | AOT compiler tool | 离线编译工具 |
| `ACL` | C++ runtime API | 对外执行接口 |
| `Session` | Compilation/exec session | 编译/执行会话 |
| `Model Cache` / `JIT 缓存` | Compilation cache | 编译产物缓存 |
| `guard miss fallback` | Recompilation / fallback | guard 失配时回退原始 om 的兜底策略 |
| `external weight` / `FileConstant` | External weights | 权重外置，权重与图分离 |

## 硬件型号（规范名称）

> 必须使用规范名称，参考 `api-doc-generator` SKILL.md 规则 #2。多个型号按 950 → A3 → A2 顺序填写。

- `Ascend 950PR/Ascend 950DT`（需要时用 `cann-filter` 标签包裹）
- `Atlas A3 训练系列产品` / `Atlas A3 推理系列产品`
- `Atlas A2 训练系列产品` / `Atlas A2 推理系列产品`
- `Ascend 310` / `Ascend Lite`（边缘场景）

## 运行时版本

| 版本 | 适用场景 | 关键约束文档 |
|------|----------|--------------|
| RT1.0 (Known Shape Executor) | 静态 shape、AOT 编译、模型下沉 | `docs/en/design/constraints/known_shape_runtime.md` |
| RT2.0 (Unknown Shape Executor / Hybrid) | 动态 shape、Lowering、ExecuteGraph、ModelV2Executor | `docs/en/design/constraints/rt2_runtime.md` |

## 约束类文档索引（设计时必参考）

| 约束文档 | 触发场景 |
|----------|----------|
| `docs/en/design/constraints/memory-constraints.md` | 显存、内存复用、block_mem、零拷贝、内存排布冲突 |
| `docs/en/design/constraints/graph_split.md` | 图拆分、cluster、动态图拆分、执行器选择 |
| `docs/en/design/constraints/stream_allocator.md` | 流分配、多流、流复用、event 同步 |
| `docs/en/design/constraints/rt2_runtime.md` | RT2、动态 shape、rt2 executor、hybrid 执行 |
| `docs/en/design/constraints/known_shape_runtime.md` | 静态 shape、davinci model、sink 模式、地址刷新 |
| `docs/en/design/constraints/graph_metadef.md` | 图基础结构（`graph_metadef/` 禁止任意修改） |
