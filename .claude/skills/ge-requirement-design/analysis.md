# 子文档：GE 需求分析

## 描述

深度解析 GE (Graph Engine) 特性需求，提取编译目标、AscendIR 层级影响、引擎分区与 Pass 体系改动、内存与 om 约束、运行时版本选型。可独立调用，也可被 `SKILL.md` 主入口编排。

standalone: true

## 上下文加载（强制）

调用前必须先读取：
- `AGENTS.md` 的"架构文档加载"和"关键特性设计原则和软件约束"两张表
- `references/ge_glossary.md`

并根据 `requirement_text` 命中的触发词，按需加载 AGENTS.md 表中对应的特性/约束文档。

## 参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `requirement_text` | 是 | 原始需求描述 |
| `existing_analysis` | 否 | 已有分析的部分结果（用于补充或修正） |
| `depth` | 否 | `light` / `standard` / `deep`，默认 `standard` |
| `constraints` | 否 | 显式约束（参见 SKILL.md 中的 constraints 表） |

`depth` 含义：
- `light`：仅提取核心编译目标和 P0 需求
- `standard`：完整覆盖 AscendIR 影响、Pass、引擎分区、内存、兼容性
- `deep`：增加 InferShape/Format 推导细节、Tiling 策略、跨进程通信影响、性能建模

## 分层原则（重要）

本文档区分两层：

1. **需求层（`requirement_analysis`）**：业务视角的黑盒描述。读者是产品经理、架构评审人、用户支持。**禁止**出现 GE 内部模块/术语（Pass、IR、SoBin、stream、engine、dispatcher、ModelV2Executor 等）。
2. **设计衍生层（`technical_decomposition`）**：技术视角的白盒分解。这里的每一条都必须能反向追溯到需求层 `business_scope.in_scope` 的某一项；不能反向追溯的条目视为"设计自我膨胀"，要在 implementation 阶段被质疑。

**输出顺序**：`requirement_analysis` 必须在所有技术字段之前，且自身不依赖任何技术字段。

## 输出格式（标准 JSON）

```json
{
  "version": "1.0",
  "domain": "GE / Graph Engine on Ascend",
  "confidence": "high | medium | low",

  "requirement_analysis": {
    "background": "业务背景，1-3 段。说明用户为什么要这个能力、当前痛点。禁用内部模块/术语",
    "business_goal": "业务目标，用户视角能用上什么能力。一句话能讲清",
    "business_scope": {
      "in_scope": ["用户能感知/使用的能力 1（黑盒）", "用户能感知/使用的能力 2", "..."],
      "out_of_scope": ["明确不做的能力 1（与 in_scope 同语言层级）", "..."],
      "scope_rationale": "为什么这样划分边界（业务约束、依赖、阶段性、风险）"
    },
    "feature_type": "new | extension",
    "extends_what": "若 feature_type == extension：说明扩展的是哪个已有能力（一句话业务描述，**不要**引用代码路径）",
    "stakeholders": ["调用方/受影响方 1", "..."],
    "scope_validation_note": "校验：business_scope 内不得出现 Pass / IR / SoBin / stream / engine / dispatcher / so / ModelV2Executor 等内部术语。若出现，必须改写为业务语言"
  },

  "technical_decomposition": {
    "_note": "以下字段是设计衍生品，不是需求。每一条都必须能反向追溯到 requirement_analysis.business_scope.in_scope 的某一项。",

  "reusable_components": {
    "_note": "仅 extension 场景必填。识别可复用的现有组件，为 implementation 阶段提供输入。",
    "components": [
      {
        "component": "组件名",
        "location": "模块路径",
        "reuse_way": "如何复用（直接调用 / 扩展 / 适配）"
      }
    ]
  },

  "workload_estimation": {
    "_note": "可选。帮助用户评估需求规模。",
    "total_kloc": "总代码行数估算（千行）",
    "breakdown": [
      {"module": "模块名", "type": "新增/修改", "lines": "~N", "note": "简要说明"}
    ]
  },

  "compilation_target": {
    "source_framework": "ONNX | Caffe | TensorFlow PB | MindSpore | 直接构造 AscendIR",
    "target_hardware": ["Ascend 950PR/Ascend 950DT", "Atlas A3 训练系列产品", "Atlas A3 推理系列产品", "Atlas A2 训练系列产品", "Atlas A2 推理系列产品", "Ascend 310", "Ascend Lite"],
    "execution_mode": "training | inference | both",
    "deployment_mode": "online (ATC + Runtime) | offline (om) | cloud-edge mixed",
    "runtime_version": "RT1.0 (Known Shape) | RT2.0 (Unknown Shape / Hybrid) | both"
  },

  "ir_requirements": {
    "ascend_ir_scope": ["ComputeGraph", "SubGraph", "Node", "OpDesc", "GeTensorDesc"],
    "anchor_usage": ["DataAnchor (InData/OutData)", "ControlAnchor (InCtrl/OutCtrl)"],
    "metadef_impact": "本特性是否新增 OpDesc / TensorDesc / 属性定义；若需修改 graph_metadef/ 必须显式声明并提供原因",
    "shape_inference": "static | symbolic (符号化推导) | dynamic with RT2.0",
    "format_inference": "OriginFormat / StorageFormat 处理；NCHW / NHWC / FRACTAL_NZ / FRACTAL_Z / NC1HWC0；TransData 插入与消除策略"
  },

  "engine_partition_requirements": {
    "engines_involved": ["AICore", "AICPU", "VectorEngine", "HCCL", "host_cpu_engine", "GE Local Engine"],
    "partition_strategy": "Cluster-based (EnginePartitioner) | DynamicShapePartition | 自定义",
    "engine_placer_impact": "是否影响 EnginePlacer 决策逻辑",
    "tiling_strategy": "static tiling | dynamic tiling | online tiling | tiling sink",
    "stream_impact": "是否影响 StreamAllocator（多流、流复用、event 同步）"
  },

  "pass_requirements": {
    "new_passes": [
      {
        "name": "PassName",
        "type": "GraphPass | NodePass",
        "stage": "PreRunOptimizeOriginalGraph | PreRunOptimizeSubGraph | PreRunAfterOptimizeSubGraph",
        "purpose": "做什么变换",
        "module_path": "compiler/graph/passes/<sub-category>/"
      }
    ],
    "affected_existing_passes": ["compiler/graph/passes/shape_optimize/", "compiler/graph/passes/format_optimize/", "compiler/graph/passes/fusion/", "compiler/graph/passes/memory_optimize/", "compiler/graph/passes/memory_conflict/", "compiler/graph/passes/multi_batch/", "compiler/graph/passes/variable_optimize/", "compiler/graph/passes/symbolic/", "compiler/graph/passes/control_flow_and_stream/", "compiler/graph/passes/standard_optimize/", "compiler/graph/passes/feature/"],
    "pattern_matcher_usage": "是否新增融合 Pattern；列出 pattern 形态、benefit 排序、与现有融合 Pass 的执行顺序"
  },

  "memory_requirements": {
    "symbol_to_anchors_impact": false,
    "block_mem_impact": "是否影响 block_mem 内存分配",
    "zero_copy_impact": "是否影响零拷贝路径（用户输入/输出地址）",
    "memory_conflict_impact": "是否产生新的内存读写冲突场景（Inplace、连续内存、子图地址隔离）",
    "peak_memory_target": "如有：< X GB",
    "om_size_constraint": "如有，给出 om 大小上限和超限策略"
  },

  "compat_requirements": {
    "om_backward_compat": "是否需要兼容旧 om",
    "guard_miss_fallback": "guard 失配时是否需要 fallback 策略（如有，描述目标）",
    "cross_compilation": "是否需要交叉编译能力",
    "ascend_ir_compat": "是否对外暴露 AscendIR 结构变更（影响下游 parser/）",
    "acl_atc_api_change": "是否新增/修改 ACL / ATC 对外接口（需 api-doc-generator 串接）"
  },

  "runtime_requirements": {
    "v1_impact": "是否影响 Known Shape Executor / DavinciModel / 模型下沉 / 地址刷新",
    "v2_impact": "是否影响 Unknown Shape Executor / Lowering / ExecuteGraph / ModelV2Executor",
    "hccl_integration": "是否涉及集合通信",
    "acl_integration": "是否新增/修改 ACL 接口"
  },

  "performance_requirements": {
    "metrics": [
      {"name": "compilation_time", "target": "< 5min", "scenario": "cold compile / warm compile", "priority": "P1"},
      {"name": "throughput", "target": "+X% vs baseline", "baseline": "<指定基线>", "priority": "P0"},
      {"name": "latency", "target": "< X ms", "scenario": "<指定场景>", "priority": "P0"},
      {"name": "om_size", "target": "< X MB", "priority": "P1"},
      {"name": "peak_memory", "target": "< X GB", "priority": "P1"}
    ]
  },

  "feature_documentation_required": {
    "new_feature_doc": "是否需要在 docs/en/design/features/ 下新增特性文档（如 autofuse.md）",
    "new_constraint_doc": "是否需要在 docs/en/design/constraints/ 下新增约束文档",
    "module_doc_update": "需要更新的 docs/en/design/modules/ 文档"
  },

  "test_strategy_hints": {
    "ut_modules": ["对应 tests/ut/ 子目录"],
    "st_modules": ["对应 tests/st/ 子目录"],
    "cross_feature_regression": "需要回归的其他特性 ST（参考 cross_feature_check.md）"
  },

  "assumptions": ["设计假设"],
  "open_questions": ["待澄清问题（confidence < high 时必填）"],
  "risk_indicators": ["高风险点"]
  },

  "blockers": [
    "前置阻塞项：必须在详细设计前回答的问题（例如：扩展特性依赖的现有机制是否能跑通；某个对外接口的语义未定）。与 open_questions 的区别是：blockers 不解决则后续设计无法收敛"
  ]
}
```

## 执行逻辑

1. **先写需求层（`requirement_analysis`）**：
   - 从 `requirement_text` 提取 background / business_goal，**强制**用业务语言表达
   - 列 `business_scope.in_scope` 与 `out_of_scope`，每条都用"用户能/不能做什么"的句式
   - 判定 `feature_type`：如果需求是"在 X 已有能力之上加 Y / 把 X 扩展到新场景"，置为 `extension`
   - 自检：scope 内是否混入内部术语？混入则改写
2. **再做设计衍生层（`technical_decomposition`）**：
   - 解析 `requirement_text`，识别显性和隐性技术约束
   - 按 GE 4 阶段编译管线（Parse → Preprocess → Optimize/Partition → Build → Runtime）逐层评估影响
   - 根据下方关键词映射表自动推断技术字段
   - 加载 AGENTS.md 中命中的特性/约束文档，将其设计原则映射为 `assumptions` 或 `risk_indicators`
   - 自检：每条技术字段能否反向追溯到 `business_scope.in_scope` 的某一项？追溯不到则视为衍生膨胀，移到 `assumptions` 或删除
3. **识别前置阻塞项（`blockers`）**：
   - 扩展特性的"现有机制依赖能否成立"必须放 blockers 而非 open_questions
   - 涉及外部仓/外部团队对齐才能确认的能力前提
4. 如 `confidence < high`，必须输出 `open_questions` 请求补充
5. 按 `depth` 控制输出粒度

## 关键词 → GE 字段映射

| 关键词 | 推断字段 |
|--------|----------|
| `om` / `离线` / `ATC` / `pb` 加载 | `compilation_target.deployment_mode = offline` |
| `RT2.0` / `动态 shape` / `dynamic` / `hybrid` | `runtime_version = RT2.0`，加载 `rt2_runtime.md` |
| `静态 shape` / `known shape` / `davinci model` / `sink` | `runtime_version = RT1.0`，加载 `known_shape_runtime.md` |
| `autofuse` / `融合` / `fuse` / `fusion engine` | `pass_requirements.new_passes` 类型为融合 Pass；定位到 `compiler/graph/passes/fusion/` |
| `Tiling` / `tiling sink` / `aicpu tiling` | `tiling_strategy`；加载 `tiling_sink.md` |
| `内存` / `block_mem` / `内存复用` / `零拷贝` / `连续内存` | `memory_requirements`；加载 `memory-constraints.md` |
| `流` / `stream` / `多流` / `event` | `stream_impact = true`；加载 `stream_allocator.md` |
| `图拆分` / `cluster` / `partition` / `动静拆分` | `partition_strategy`；加载 `graph_split.md` 和 `graph_spliter.md` |
| `集合通信` / `allreduce` / `HCCL` | `engines_involved` 含 `HCCL`；`hccl_integration = true` |
| `InferShape` / `符号化推导` / `OriginShape` / `StorageShape` | `shape_inference`；加载 `infer_shape.md` |
| `Format` / `TransData` / `OriginFormat` / `StorageFormat` | `format_inference`；加载 `infer_format.md` |
| `引擎` / `EnginePlacer` / `DNNEngine` / `OpsKernelInfoStore` | `engines_involved`；`engine_placer_impact`；加载 `engine.md` |
| `dump` / `溢出` / `落盘` | 加载 `datadump.md` |
| `external weight` / `FileConstant` / `权重分离` | 加载 `external_weight.md` |
| `多 batch` / `动态分档` / `dynamic_dims` | 加载 `dynamic_gear.md` |
| `常量折叠` | 加载 `constant_folding.md` |
| `Variable` / `变量管理` | 加载 `variable_manager.md` |
| `model cache` / `编译缓存` / `JIT 缓存` | 加载 `model_cache.md` |
| `profiling` / `msprof` / `性能调优` | 加载 `profiling.md` |
| `SO in OM` / `算子打包` | 加载 `so_in_om.md` |
| `TensorMove` / `冗余拷贝` | 加载 `tensormove_delete.md` |
| `zero copy` / `用户内存` | 加载 `zero_copy.md` |
| `concat` / `虚拟算子` / `不生成 Task` | 加载 `concat_no_task.md` |
| `GE Local` / `NoOp` / `PhonyConcat` / `PhonySplit` | 加载 `ge_local_operator.md` |
| `ACL` / `ATC` / `Session` / `对外接口` | `acl_atc_api_change = true`；后续触发 `api-doc-generator` |

## 示例

**输入**：

```json
{
  "requirement_text": "扩展 TensorMove 消除特性，识别更多冗余拷贝场景（跨子图边界、控制流入口），减少端到端 om 大小并降低 Runtime 拷贝开销。目标硬件 Atlas A3 训练系列产品 / Atlas A2 训练系列产品。",
  "depth": "standard"
}
```

**关键输出片段**：

```json
{
  "requirement_analysis": {
    "background": "用户当前训练任务中存在大量冗余张量拷贝，导致端到端模型文件偏大、Runtime 拷贝开销可观。早期 TensorMove 消除只覆盖了简单场景，跨子图边界与控制流入口的冗余拷贝未覆盖。",
    "business_goal": "在不改变模型语义的前提下，让用户的训练任务获得更小的 om 文件与更低的运行时拷贝开销，无需用户改写模型或调整 ATC 选项。",
    "business_scope": {
      "in_scope": [
        "更多场景下的冗余拷贝自动消除（用户感知：om 变小、推理/训练吞吐改善）",
        "对用户完全透明，不引入新的 ATC 选项"
      ],
      "out_of_scope": [
        "用户手动指定哪些 TensorMove 该消除（不引入新接口）",
        "改变模型计算语义的优化"
      ],
      "scope_rationale": "用户期待的是无感优化；新增对外开关会增加调用方负担且违背 TensorMove 消除作为内部优化的定位"
    },
    "feature_type": "extension",
    "extends_what": "在已有的 TensorMove 消除能力基础上扩大识别范围",
    "stakeholders": ["训练任务调用方（无感）", "GE 维护方"],
    "scope_validation_note": "已校验：business_scope 仅描述用户能感知的能力变化，未混入 Pass/IR/内存等内部术语"
  },
  "technical_decomposition": {
    "reusable_components": {
      "components": [
        {
          "component": "TensorMove 消除 Pass",
          "location": "compiler/graph/passes/standard_optimize/",
          "reuse_way": "扩展现有 Pass 的识别范围"
        }
      ]
    },
    "workload_estimation": {
      "total_kloc": "~1.5",
      "breakdown": [
        {"module": "TensorMove 消除扩展", "type": "修改", "lines": "~500", "note": "扩展识别逻辑"},
        {"module": "测试", "type": "新增", "lines": "~400", "note": "UT+ST"}
      ]
    },
    "compilation_target": {
      "deployment_mode": "online (ATC + Runtime)",
      "target_hardware": ["Atlas A3 训练系列产品", "Atlas A2 训练系列产品"],
      "execution_mode": "training",
      "runtime_version": "both"
    },
    "pass_requirements": {
      "new_passes": [],
      "affected_existing_passes": ["compiler/graph/passes/standard_optimize/", "compiler/graph/passes/memory_optimize/"]
    },
    "memory_requirements": {
      "block_mem_impact": "潜在减少 block_mem 申请次数",
      "zero_copy_impact": "需评估对零拷贝路径的影响"
    },
    "compat_requirements": {
      "om_backward_compat": true,
      "acl_atc_api_change": false
    },
    "risk_indicators": [
      "跨子图边界识别可能与图拆分逻辑耦合，需走 graph_split 评审",
      "控制流入口的 TensorMove 消除需保证语义等价"
    ],
    "open_questions": [
      "是否影响 dump / profiling 观测点？"
    ]
  },
  "blockers": []
}
```
