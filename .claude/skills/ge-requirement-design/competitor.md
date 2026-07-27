# 子文档：GE 竞品调研（外部 + 启示双层）

## 描述

对外部图编译器/优化器做技术级对比，**为 GE 设计提供借鉴**。本文档**不是**让用户在外部产品和 GE 之间选型，而是从外部产品中提取对 GE AscendIR / Pass / EnginePartition / Tiling / Codegen / Runtime 设计有启发的做法。

可独立调用，也可被 `SKILL.md` 主入口编排。

standalone: true

## 上下文加载（强制）

调用前必须先读取：
- `references/ge_glossary.md`（用于将外部术语映射回 GE 模块）
- `analysis.md` 的输出（如已有）—— 用于聚焦外部对标维度

## 参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `analysis_output` | 否 | 需求分析输出，用于聚焦对标维度 |
| `target_scenario` | 是 | 目标应用场景（如 "Ascend 上的离线推理 autofuse"、"RT2.0 动态 shape 训练"、"集合通信下的多流编排"） |
| `performance_priority` | 否 | `latency` / `throughput` / `memory` / `compilation_time` / `portability`，默认 `latency` |
| `focus_dimension` | 否 | 重点对比维度：`ir_design` / `pass_pipeline` / `partition` / `tiling_codegen` / `runtime` / `dynamic_shape` / `quantization` |
| `analysis_depth` | 否 | `overview` / `technical` / `benchmark`，默认 `technical` |

## 输出格式（标准 JSON）

```json
{
  "version": "2.0",
  "domain": "GE / Graph Engine on Ascend",
  "analysis_timestamp": "<ISO 日期>",
  "perspective": "外部竞品对 GE 的设计启示",

  "competitors": [
    {
      "name": "PyTorch Inductor",
      "type": "direct",
      "vendor": "Meta / PyTorch 团队",
      "positioning": "PyTorch 原生默认编译器",

      "architecture": {
        "ir_layer": "FX Graph / TorchIR",
        "optimization_passes": ["算子融合", "内存规划", "布局优化"],
        "backend_target": "C++ / OpenMP / Triton / CUDA",
        "compilation_model": "eager-with-compilation / graph-mode"
      },
      "compilation_strategy": {
        "approach": "tracing-based with guards",
        "dynamic_shape_support": "partial (Dynamo + guards)",
        "key_innovation": "Pythonic 编译体验，与 Eager 模式无缝切换"
      },
      "performance_profile": {
        "strengths": ["Python 生态无缝", "动态图友好", "开发体验好"],
        "weaknesses": ["复杂控制流受限", "编译开销较大", "极致性能非最优"],
        "typical_speedup": "1.2x-2.0x vs eager"
      },
      "quantization_support": {"FP32": true, "FP16": true, "BF16": true, "INT8": "QAT via torch.ao.quantization", "INT4": "experimental"},

      "implications_for_ge": {
        "ir_design": "Dynamo guard 机制对 GE guard miss fallback 设计有直接借鉴：guard 应在 Runtime 加载/执行入口校验，失配时切换回原始 om",
        "pass_pipeline": "Inductor 的 Pre-/Post-grad pass 分层可对应到 GE 的 PreRunOptimizeOriginalGraph / PreRunOptimizeSubGraph / PreRunAfterOptimizeSubGraph 三阶段",
        "partition_design": "Inductor 不做后端硬件分区（单一 CUDA 后端），对 GE EnginePartition 借鉴有限",
        "tiling_codegen": "Triton 模板化 kernel 生成对 GE Tiling 自动化有启发，但 GE 的 AICore Tiling 已有专用流程",
        "runtime_design": "tracing + guard 的运行时回退模型可作为 RT2.0 动态 shape guard miss 的参考"
      },
      "ge_equivalent": {
        "frontend_layer": "parser/（GE 不做 PyTorch 直接对接，走 ONNX / MindSpore 中转）",
        "ir_layer": "AscendIR / ComputeGraph / OpDesc",
        "optimization_layer": "compiler/graph/passes/（按 stage 分布）",
        "partition_layer": "compiler/graph/partition/",
        "build_layer": "compiler/graph/build/",
        "runtime_layer": "runtime/v2/（动态场景）"
      }
    },

    {
      "name": "PyTorch AOTInductor",
      "type": "direct",
      "vendor": "Meta / PyTorch 团队",
      "positioning": "Ahead-of-Time 编译，生产级推理",

      "architecture": {
        "ir_layer": "FX Graph → AOTInductor IR",
        "optimization_passes": ["算子融合", "内存优化", "序列化编译产物"],
        "backend_target": "C++ shared library / CUDA",
        "compilation_model": "ahead-of-time compilation"
      },
      "compilation_strategy": {
        "approach": "AOT compilation with serialized artifacts",
        "dynamic_shape_support": "partial (via symbolic shapes)",
        "key_innovation": "编译一次到处部署；脱离 Python 运行时"
      },
      "performance_profile": {
        "strengths": ["启动延迟低", "无 GIL 开销", "部署轻量"],
        "weaknesses": ["编译时确定形状约束", "调试体验下降"],
        "typical_speedup": "1.5x-3.0x vs eager"
      },
      "quantization_support": {"FP32": true, "FP16": true, "BF16": true, "INT8": "via torch.ao.quantization export", "INT4": "not supported"},

      "implications_for_ge": {
        "ir_design": "Symbolic shapes 表达对 GE 符号化 InferShape 直接对应；AOTInductor 把 symbolic 约束序列化到产物中的做法可借鉴到 om",
        "pass_pipeline": "AOT 模式下 Pass 顺序确定性高，对 GE 编译稳定性有借鉴",
        "partition_design": "无后端分区",
        "tiling_codegen": "编译产物自包含（含 weight）思路对 GE external_weight / so_in_om 设计有参考",
        "runtime_design": "脱离 Python 运行时 = GE 的纯 C++ Runtime；产物格式（.so + 元数据）对 om 演进有启示"
      },
      "ge_equivalent": {
        "frontend_layer": "parser/",
        "ir_layer": "AscendIR + symbolic shape",
        "optimization_layer": "compiler/graph/passes/symbolic/",
        "partition_layer": "compiler/graph/partition/",
        "build_layer": "compiler/graph/build/（含 ModelSerializer）",
        "runtime_layer": "runtime/v1/ + runtime/v2/"
      }
    },

    {
      "name": "TensorRT",
      "type": "direct",
      "vendor": "NVIDIA",
      "positioning": "NVIDIA GPU 推理极致优化",

      "architecture": {
        "ir_layer": "TensorRT Network Definition",
        "optimization_passes": ["层融合", "精度校准 (INT8/FP16)", "内核自动调优", "多流执行"],
        "backend_target": "CUDA / cuDNN / Tensor Cores",
        "compilation_model": "static graph compilation"
      },
      "compilation_strategy": {
        "approach": "static graph + builder optimization",
        "dynamic_shape_support": "limited (explicit batch / optimization profile)",
        "key_innovation": "硬件协同设计，极致推理性能"
      },
      "performance_profile": {
        "strengths": ["NVIDIA GPU 推理极致性能", "低精度量化成熟", "生产级稳定性"],
        "weaknesses": ["NVIDIA 硬件锁定", "动态图支持弱", "编译时间长"],
        "typical_speedup": "2x-10x vs framework baseline"
      },
      "quantization_support": {"FP32": true, "FP16": "native AMP", "BF16": "limited", "INT8": "PTQ/QAT full support", "INT4": "experimental"},

      "implications_for_ge": {
        "ir_design": "Network Definition 偏向 builder API 模式，对 GE AscendIR 的对外构造接口（ATC / Session）有借鉴",
        "pass_pipeline": "Builder 内置层融合策略库 + cost model 选择，可对照 GE 融合 Pass + Pattern Matcher 的策略匹配机制完善 cost model",
        "partition_design": "Optimization Profile 机制（一个引擎支持多种 shape profile）对 GE 动态分档 / dynamic_gear 有直接借鉴",
        "tiling_codegen": "Kernel auto-tuning + 多算法实现选择对 GE Tiling 自动化 / AICore kernel 选择有借鉴",
        "runtime_design": "极致 latency 优化的多流并发 + Plan 序列化思路对 GE RT1.0 模型下沉 + StreamAllocator 有参考"
      },
      "ge_equivalent": {
        "frontend_layer": "parser/onnx",
        "ir_layer": "AscendIR / ComputeGraph",
        "optimization_layer": "compiler/graph/passes/fusion/ + compiler/graph/optimize/",
        "partition_layer": "compiler/graph/partition/ + dynamic_gear 分档",
        "build_layer": "compiler/graph/build/（StreamAllocator + TaskBuilder）",
        "runtime_layer": "runtime/v1/"
      }
    },

    {
      "name": "TensorFlow XLA",
      "type": "direct",
      "vendor": "Google",
      "positioning": "TensorFlow / JAX 原生编译优化",

      "architecture": {
        "ir_layer": "HLO (High Level Operations)",
        "optimization_passes": ["算子融合", "布局优化", "并行化", "自动微分优化"],
        "backend_target": "CPU / GPU / TPU via LLVM",
        "compilation_model": "graph-mode with JIT/AOT"
      },
      "compilation_strategy": {
        "approach": "HLO-based compilation with aggressive fusion",
        "dynamic_shape_support": "partial (via tf.function experimental)",
        "key_innovation": "TPU 原生支持，跨硬件统一 IR"
      },
      "performance_profile": {
        "strengths": ["TPU 生态深度集成", "大规模训练优化", "跨硬件可移植"],
        "weaknesses": ["TF 生态萎缩", "调试困难", "灵活性不足"],
        "typical_speedup": "1.5x-5.0x vs TF eager"
      },
      "quantization_support": {"FP32": true, "FP16": true, "BF16": "TPU native", "INT8": "QAT supported", "INT4": "not supported"},

      "implications_for_ge": {
        "ir_design": "HLO 作为稳定中间 IR 的设计原则（保留语义、可序列化、可优化）对 AscendIR 的版本演进治理有参考",
        "pass_pipeline": "XLA 的 HLO Pass Pipeline + Buffer Assignment 对 GE 内存规划 (SymbolToAnchors) 有直接借鉴",
        "partition_design": "Auto-clustering 算法对 GE EnginePartition Cluster-based 算法有同源参考",
        "tiling_codegen": "TPU 编译器的 tile 选择策略对 GE AICore Tiling 有启示",
        "runtime_design": "PJRT 抽象层对 GE Runtime V1/V2 的对外抽象有参考"
      },
      "ge_equivalent": {
        "frontend_layer": "parser/tensorflow",
        "ir_layer": "AscendIR (类比 HLO)",
        "optimization_layer": "compiler/graph/passes/",
        "partition_layer": "compiler/graph/partition/（Cluster-based）",
        "build_layer": "compiler/graph/build/memory/",
        "runtime_layer": "runtime/"
      }
    },

    {
      "name": "OpenXLA / StableHLO",
      "type": "direct",
      "vendor": "Google / OpenXLA 社区",
      "positioning": "开源统一编译器生态（XLA + IREE）",

      "architecture": {
        "ir_layer": "StableHLO / MHLO",
        "optimization_passes": ["HLO 优化", "流式执行", "多设备并行"],
        "backend_target": "CPU / GPU / TPU / 移动设备 via IREE",
        "compilation_model": "unified compilation stack"
      },
      "compilation_strategy": {
        "approach": "StableHLO 作为通用 IR + IREE for edge",
        "dynamic_shape_support": "improving (via dynamism RFCs)",
        "key_innovation": "标准化 IR，跨框架跨硬件编译"
      },
      "performance_profile": {
        "strengths": ["标准化 IR", "TPU/GPU 统一", "IREE 边缘支持"],
        "weaknesses": ["生态建设初期", "PyTorch 集成间接", "工具链复杂"],
        "typical_speedup": "1.5x-3.0x (训练), hardware-dependent (推理)"
      },
      "quantization_support": {"FP32": true, "FP16": true, "BF16": true, "INT8": "IREE support", "INT4": "experimental"},

      "implications_for_ge": {
        "ir_design": "StableHLO 的 IR 版本化（major/minor 兼容承诺）对 AscendIR 对外接口稳定性治理有直接借鉴",
        "pass_pipeline": "MLIR Dialect 分层对 GE Pass 体系按 stage 组织有参考价值（已部分对齐）",
        "partition_design": "IREE 的 stream dialect 对 GE StreamAllocator 设计语言有启发",
        "tiling_codegen": "IREE 的 hal.executable 概念对 GE om 模型自包含设计有参考",
        "runtime_design": "PJRT runtime 抽象对 GE Runtime V1/V2 双轨制有借鉴"
      },
      "ge_equivalent": {
        "frontend_layer": "parser/",
        "ir_layer": "AscendIR",
        "optimization_layer": "compiler/graph/passes/",
        "partition_layer": "compiler/graph/partition/",
        "build_layer": "compiler/graph/build/",
        "runtime_layer": "runtime/"
      }
    },

    {
      "name": "Apache TVM",
      "type": "indirect",
      "vendor": "Apache / 社区",
      "positioning": "开源端到端深度学习编译器，多硬件后端",

      "architecture": {
        "ir_layer": "Relay / TVMScript / Relax",
        "optimization_passes": ["自动算子融合", "张量化", "自动调优 (AutoTVM/AutoScheduler)"],
        "backend_target": "多硬件后端（GPU/CPU/FPGA/ASIC）",
        "compilation_model": "full compilation stack"
      },
      "compilation_strategy": {
        "approach": "learning-based compilation with auto-tuning",
        "dynamic_shape_support": "partial (Relax IR)",
        "key_innovation": "硬件无关优化 + 自动调优"
      },
      "performance_profile": {
        "strengths": ["硬件覆盖最广", "自动调优潜力大", "开源可控"],
        "weaknesses": ["编译/调优时间长", "工程复杂度高", "生态工具链弱"],
        "typical_speedup": "hardware-dependent, up to 5x+"
      },
      "quantization_support": {"FP32": true, "FP16": true, "BF16": "limited", "INT8": "AutoTVM + BYOC", "INT4": "community experimental"},

      "implications_for_ge": {
        "ir_design": "Relax IR 的 first-class symbolic shape 表达对 GE 符号化 InferShape 有直接借鉴",
        "pass_pipeline": "BYOC (Bring Your Own Codegen) 机制对 GE 引擎插件 (OpsKernelInfoStore) 注册有参考",
        "partition_design": "TVM 的图分区到子图 + 后端委托对 GE EnginePartition 是同源思想",
        "tiling_codegen": "AutoScheduler / Ansor 自动 schedule 搜索对 GE Tiling 自动化和 autofuse 设计参考价值高",
        "runtime_design": "TVM Runtime 的 PackedFunc 调度对 GE Task 调度抽象有启发"
      },
      "ge_equivalent": {
        "frontend_layer": "parser/",
        "ir_layer": "AscendIR (类比 Relay/Relax)",
        "optimization_layer": "compiler/graph/passes/fusion/ + autofuse",
        "partition_layer": "compiler/graph/partition/",
        "build_layer": "compiler/graph/build/",
        "runtime_layer": "runtime/"
      }
    },

    {
      "name": "ONNX Runtime",
      "type": "indirect",
      "vendor": "Microsoft",
      "positioning": "跨框架推理引擎",

      "architecture": {
        "ir_layer": "ONNX (Open Neural Network Exchange)",
        "optimization_passes": ["图优化", "算子融合", "常量折叠"],
        "backend_target": "多种 Execution Provider",
        "compilation_model": "runtime graph optimization"
      },
      "compilation_strategy": {
        "approach": "standardized IR + pluggable backends",
        "dynamic_shape_support": "good (ONNX spec support)",
        "key_innovation": "框架无关，一次转换多平台运行"
      },
      "performance_profile": {
        "strengths": ["框架互操作", "部署标准化", "多硬件支持"],
        "weaknesses": ["性能非极致", "算子覆盖有限", "前沿特性滞后"],
        "typical_speedup": "1.0x-1.5x vs framework baseline"
      },
      "quantization_support": {"FP32": true, "FP16": true, "BF16": "limited", "INT8": "ONNX Runtime Quantization", "INT4": "not supported"},

      "implications_for_ge": {
        "ir_design": "ONNX Operator Spec 版本化机制对 AscendIR 算子注册和兼容性管理有参考；GE parser/ 直接对接 ONNX 作为入口",
        "pass_pipeline": "Graph Optimization Level 分级（Basic/Extended/All）对 GE 编译优化等级对外暴露有借鉴",
        "partition_design": "Execution Provider 机制对 GE EnginePlacer 多引擎选择有参考",
        "tiling_codegen": "对 GE Tiling 借鉴有限",
        "runtime_design": "Session API 设计对 GE Session 接口有同源参考"
      },
      "ge_equivalent": {
        "frontend_layer": "parser/onnx",
        "ir_layer": "AscendIR",
        "optimization_layer": "compiler/graph/passes/standard_optimize/ + fusion/",
        "partition_layer": "compiler/graph/partition/（多引擎选择）",
        "build_layer": "compiler/graph/build/",
        "runtime_layer": "runtime/"
      }
    },

    {
      "name": "ExecuTorch",
      "type": "direct",
      "vendor": "Meta / PyTorch 团队",
      "positioning": "PyTorch 原生移动端/边缘推理",

      "architecture": {
        "ir_layer": "Edge Dialect (基于 TorchIR)",
        "optimization_passes": ["算子分解", "内存规划", "量化", "委托 (delegate)"],
        "backend_target": "C++ runtime / 硬件委托后端",
        "compilation_model": "ahead-of-time export + runtime delegation"
      },
      "compilation_strategy": {
        "approach": "export-to-edge + delegate-to-backend",
        "dynamic_shape_support": "limited (static by default, partial dynamic via constraints)",
        "key_innovation": "PyTorch 生态原生移动端部署，委托机制解耦硬件后端"
      },
      "performance_profile": {
        "strengths": ["PyTorch 模型无缝导出", "超轻量 runtime", "委托后端可扩展"],
        "weaknesses": ["生态较新", "复杂模型支持待完善", "调试工具链不成熟"],
        "typical_speedup": "N/A (部署优化为主)"
      },
      "quantization_support": {"FP32": true, "FP16": "delegate-dependent", "BF16": "not supported", "INT8": "XNNPACK/Core ML delegate", "INT4": "not supported"},

      "implications_for_ge": {
        "ir_design": "Edge Dialect 算子分解策略对 GE GE Local Engine（NoOp / PhonyConcat）有同源参考",
        "pass_pipeline": "导出 + 委托的两阶段对 GE 离线编译 + Runtime 委托模式有参考",
        "partition_design": "Delegate 机制对 GE EnginePartition 中委托给特定引擎（如 HCCL / AICore）有直接借鉴",
        "tiling_codegen": "对 GE AICore Tiling 借鉴有限",
        "runtime_design": "超轻量 runtime + 后端委托对 Ascend 310/Lite 边缘 Runtime 设计有参考"
      },
      "ge_equivalent": {
        "frontend_layer": "parser/",
        "ir_layer": "AscendIR + GE Local Engine 抽象",
        "optimization_layer": "compiler/graph/passes/feature/",
        "partition_layer": "compiler/graph/partition/",
        "build_layer": "compiler/graph/build/",
        "runtime_layer": "runtime/（边缘场景）"
      }
    },

    {
      "name": "MLIR",
      "type": "indirect",
      "vendor": "LLVM / 多厂商",
      "positioning": "编译器基础设施（非终端产品）",

      "architecture": {
        "ir_layer": "多层级 IR (Dialect 系统)",
        "optimization_passes": ["方言转换", "Pass Pipeline", "代码生成"],
        "backend_target": "任意后端",
        "compilation_model": "infrastructure for building compilers"
      },
      "compilation_strategy": {
        "approach": "DSL-based multi-level IR transformation",
        "dynamic_shape_support": "supported at IR level (depends on frontend)",
        "key_innovation": "可扩展的编译器框架，多领域复用"
      },
      "performance_profile": {
        "strengths": ["极致灵活性", "工业级基础设施", "多领域复用"],
        "weaknesses": ["非终端产品", "学习曲线陡峭", "需自行构建完整工具链"],
        "typical_speedup": "N/A (基础设施)"
      },
      "quantization_support": {"FP32": true, "FP16": "via dialect", "BF16": "via dialect", "INT8": "via dialect", "INT4": "custom dialect required"},

      "implications_for_ge": {
        "ir_design": "Dialect 分层 + Op Spec / Trait 系统对 GE AscendIR 演进有体系性参考；GE 当前 Op 注册体系可对照 MLIR ODS 现代化",
        "pass_pipeline": "PatternRewriter + Conversion Pass 对 GE Pattern Matcher / Pass 框架有直接借鉴",
        "partition_design": "Dialect 之间的转换可类比 GE 引擎间的 op 转换，但 GE 已通过引擎分区解决",
        "tiling_codegen": "Linalg / GPU dialect 的 tile/distribute 对 GE Tiling 设计有概念参考",
        "runtime_design": "MLIR 不提供 runtime，参考价值有限"
      },
      "ge_equivalent": {
        "frontend_layer": "parser/",
        "ir_layer": "AscendIR (类比 MLIR 顶层 Dialect)",
        "optimization_layer": "compiler/graph/passes/",
        "partition_layer": "compiler/graph/partition/",
        "build_layer": "compiler/graph/build/",
        "runtime_layer": "runtime/"
      }
    }
  ],

  "comparative_analysis": {
    "dynamic_shape_flexibility": {
      "best_external": ["Inductor (Dynamo guards)", "ONNX Runtime"],
      "ge_position": "RT2.0 已支持 Unknown Shape Executor + 符号化推导；guard miss fallback 借鉴 Dynamo 模式",
      "lessons_for_ge": "Dynamo 的 guard 序列化、guard 失配统计、recompile 缓存策略可直接搬到 RT2.0"
    },
    "fusion_strategy": {
      "best_external": ["TensorRT (cost-model)", "TVM (AutoScheduler)", "Inductor (pattern matching)"],
      "ge_position": "Pattern Matcher + 融合 Pass 框架已具备；autofuse 是当前补强方向",
      "lessons_for_ge": "TVM Ansor 的搜索式融合 + TensorRT cost-model + Inductor 的 pre/post-grad pass 分层，三者结合是 autofuse 设计的参考池"
    },
    "memory_planning": {
      "best_external": ["XLA Buffer Assignment", "Inductor (memory planning)"],
      "ge_position": "SymbolToAnchors 等价类 + block_mem 已就位",
      "lessons_for_ge": "XLA 的 buffer reuse 算法、Inductor 的 inplace 决策对 GE memory-constraints 演进有借鉴"
    },
    "compilation_artifact": {
      "best_external": ["AOTInductor (.so + 元数据)", "TensorRT (.plan)", "ExecuTorch (.pte)"],
      "ge_position": "om 模型已是产物标准；external_weight / so_in_om 是演进方向",
      "lessons_for_ge": "AOTInductor 的产物自包含 + ExecuTorch 的轻量 runtime 对 om 长期演进有参考"
    },
    "runtime_abstraction": {
      "best_external": ["OpenXLA PJRT", "ONNX Runtime EP"],
      "ge_position": "RT1.0 / RT2.0 双轨制 + EnginePlacer 多引擎",
      "lessons_for_ge": "PJRT 的统一对外接口对 GE Runtime 对外抽象有借鉴"
    },
    "ascend_npu_portability": {
      "evaluation": "TVM 通过 BYOC 已有 Ascend backend 社区尝试；Inductor / TensorRT 无原生 Ascend 后端；OpenXLA 可通过 IREE Ascend backend 接入（社区方案）。这些路径都不是 GE 项目内的主路径，但对 GE 对外接口治理（如何让外部框架更容易接入）有借鉴。"
    },
    "om_artifact_compat": {
      "evaluation": "外部产物 → om 的转换均需经 ONNX 中转（参考 parser/onnx）；不存在原生兼容方案。"
    }
  },

  "borrow_and_avoid": {
    "borrow": [
      {"from": "PyTorch Inductor + AOTInductor", "what": "Dynamo guard 机制 + symbolic shape 序列化", "apply_to": "RT2.0 + autofuse guard miss fallback"},
      {"from": "TVM AutoScheduler / Ansor", "what": "搜索式自动融合", "apply_to": "autofuse pattern 库扩展"},
      {"from": "TensorRT Optimization Profile", "what": "多 shape profile 共编译", "apply_to": "dynamic_gear 动态分档"},
      {"from": "XLA Buffer Assignment", "what": "buffer 复用算法", "apply_to": "memory-constraints / SymbolToAnchors 优化"},
      {"from": "OpenXLA StableHLO 版本化", "what": "IR 兼容性承诺机制", "apply_to": "AscendIR 对外接口治理"},
      {"from": "MLIR PatternRewriter", "what": "声明式 pattern + benefit 排序", "apply_to": "GE Pattern Matcher / 融合 Pass 现代化"}
    ],
    "avoid": [
      {"from": "TensorRT 静态 profile", "pitfall": "动态场景下 profile 数量爆炸", "ge_implication": "dynamic_gear 不要走过细分档路线，需有 RT2.0 作为兜底"},
      {"from": "ONNX Runtime", "pitfall": "为了通用性牺牲极致性能", "ge_implication": "AscendIR 演进不要为了通用性放弃 Ascend 硬件协同设计"},
      {"from": "TVM 长调优时间", "pitfall": "AutoTVM 编译时间不可控", "ge_implication": "autofuse 搜索式策略需要严格的编译时间预算"},
      {"from": "Inductor Python 运行时耦合", "pitfall": "无法脱离 Python", "ge_implication": "RT2.0 必须保持纯 C++ 可部署"}
    ]
  },

  "recommendation": {
    "for_scenario": "<target_scenario>",
    "primary_lessons": ["核心借鉴 1", "核心借鉴 2", "核心借鉴 3"],
    "anti_patterns": ["要避免的设计陷阱"],
    "open_research": ["需要进一步调研的点（如 Triton MLIR Backend 在 Ascend 上的可行性）"]
  }
}
```

## 执行逻辑

1. 根据 `target_scenario` 和 `analysis_output`（如有）确定本次重点对标的外部产品子集（不需要每次都覆盖全部）
2. 对每个产品填充 `architecture` / `compilation_strategy` / `performance_profile` / `quantization_support` 等通用字段
3. **强制**为每个产品填充 `implications_for_ge` 和 `ge_equivalent` 双字段——这是本 skill 区别于通用竞品分析的核心
4. 填充 `comparative_analysis`：每个维度同时给出"外部最佳实践"和"GE 当前位置"和"对 GE 的启示"
5. 输出 `borrow_and_avoid` 清单：明确指出 GE 应借鉴什么、要避免哪些设计陷阱
6. 按 `analysis_depth` 控制粒度（overview 可省略子字段，benchmark 需附性能数据）

## 注意事项

- **不要**给出"建议用户用 TensorRT 还是 GE"这类选型结论——GE 项目内的人不会切换到外部编译器
- `decision_matrix` / `ranked_recommendations` 在本 skill 中被替换为 `borrow_and_avoid`，更贴合 GE 项目研发的实际诉求
- 涉及 Ascend 硬件型号时使用规范名称（Ascend 950PR/Ascend 950DT、Atlas A3 训练系列产品 等）
- 如果 `analysis_output.compat_requirements.acl_atc_api_change = true`，在 `borrow_and_avoid.borrow` 中补充对外接口治理相关的外部经验
