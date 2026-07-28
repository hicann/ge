# 子文档：GE 实现设计

## 描述

基于 `analysis.md` 和 `competitor.md` 的输出，产出两段式设计方案：**顶层设计**（功能拆分/概念模型/流程设计/模块交互/设计决策）→ **详细设计**（模块改动/Pass/Runtime/接口/风险/测试）。两层之间有明确的代码引用边界。产物对齐 `docs/en/design/design_document_template.md` 模板。

可独立调用，也可被 `SKILL.md` 主入口编排。

standalone: true

## 上下文加载（强制）

调用前必须先读取：
- `references/ge_glossary.md`
- `docs/en/design/design_document_template.md`（设计文档模板，最终 spec 输出对齐）
- `docs/en/design/cross_feature_check.md`（跨特性检查清单）
- 根据 `analysis_output.runtime_requirements`、`memory_requirements`、`pass_requirements` 命中的约束文档（`docs/en/design/constraints/*.md`）

## 设计顺序原则（重要）

本文档遵循"**先顶层设计，再详细设计，两层之间有明确边界**"：

### 顶层设计（`high_level_design`）—— 概念性、流程性设计

必须**先**产出。聚焦"做什么"而非"怎么做"：

1. **功能拆分（`functional_decomposition`）**：从用户需求拆解出独立的功能点，用业务语言描述。每条必须有 `business_scope_ref` 追溯到 `analysis.business_scope.in_scope`

   **约束**：
   - 所有功能点必须处于**同一抽象层级**——每条描述一个完整的、可独立验证的能力。内部算法策略属于设计决策或算法章节
   - 贯穿性属性（降级策略、模式开关、兼容性承诺）不列入本表，以独立段落说明
2. **概念模型（`conceptual_model`）**：核心实体、属性、生命周期、实体间关系。非代码级描述
3. **流程设计（`flow_design`）**：端到端三段流程（编译→加载→执行），列出新增节点和复用节点。最终设计文档中优先使用 mermaid flowchart / sequenceDiagram 表达
4. **模块交互与边界（`module_interaction_and_boundary`）**：子系统间的交互契约和职责边界
5. **设计决策（`design_decisions`）**：关键选择的权衡分析，记录问题→候选方案→决策→理由

**顶层设计强制约束**：本节所有字段**禁止**出现以下内容：
- 模块路径（如 `compiler/graph/passes/xxx`）
- 类名/结构体名
- 文件路径或行号
- Pass 注册细节、Rerun 策略
- 函数签名

### 详细设计（`detailed_design`）—— 代码级承载设计

顶层设计确认后**再**产出。此处才允许引用代码路径、类名、模块名等实现细节。`detailed_design.functional_points[i].id` 必须可追溯到 `high_level_design.functional_decomposition[j].id`。

3. **模块/Pass/Runtime 等垂直切片下沉为承载**。`architecture.modules` / `pass_design` / `engine_and_partition_design` / `memory_and_stream_design` / `runtime_design` 这些字段是 `functional_points.carrier` 的展开附录。

## 最终 spec 输出章节排布约束（重要）

skill 的 JSON 中间产物以"两层设计"组织（见上节）。当 JSON 转 markdown 最终 spec 时，**必须按 `docs/en/design/design_document_template.md` 模板章节排布**——模板中有"接口检查项 / 编码检查项 / 性能评估 / 平台化要求"等检查项必须覆盖。但"顶层设计优先"原则不能丢，因此采用以下混合策略：

| 模板章节 | 填充来源（JSON 字段） |
|---|---|
| 简介 - 目的 / 范围 | `analysis.requirement_analysis.business_scope` |
| 总体概述 - 软件概述 - 项目介绍 | `analysis.requirement_analysis.background` |
| 总体概述 - 软件概述 - 产品环境介绍 | `analysis.technical_decomposition.compilation_target` + 与外部仓/上下游关系 |
| 总体概述 - 软件功能 | `analysis.requirement_analysis.business_goal` + `high_level_design.functional_decomposition` 高层列表 |
| 总体概述 - 设计约束 | `analysis.requirement_analysis.business_scope.scope_rationale` + `high_level_design.design_decisions` 中关键决策的硬约束摘要 |
| 总体概述 - 假设和依赖关系 | `detailed_design.blocking_prerequisites` |
| **特性 N - 整体介绍** | **`existing_mechanism_output.phases` + `high_level_design.flow_design` 三段流程 + `high_level_design.conceptual_model` 核心实体关系合并填入**——这是模板里唯一能承载"顶层设计"内容的位置 |
| 特性 N - 功能需求 - 功能需求 i | `high_level_design.functional_decomposition[i]` 展开为模板要求的"介绍 / 输入 / 处理 / 输出"四件套，每条引用对应 `derived_from_flow_nodes` |
| 特性 N - 非功能需求 | 可维护性 / 可测试性 / 可移植性 / 可靠性 / 平台化（GE onetrack 约定）/ 特性交叉（引用 `detailed_design.cross_feature_impacts`）|
| 特性 N - 性能 | 三段评估：模型编译时长 / OM 大小和加载占用内存 / 执行性能。涉及 `optimizeStage1/2/build/loadmodelonline` 四阶段时必须明确影响评估 |
| 特性 N - 接口设计 | `detailed_design.external_interface_design` + 模板要求的"接口检查项"表格逐项打勾。**每个新增 external 接口类/基类必须附开发者使用示例伪代码**（至少覆盖：最简场景、含可选特性的场景、上层封装场景） |
| 特性 N - 软件设计 | 关键数据结构 / 关键算法 / 流程设计（细化，来源：`detailed_design.functional_points[*].carrier + key_algorithms`）/ 对子模块的修改（`detailed_design.architecture.modules`）/ 错误处理 |
| 特性 N - 安全检查 | 编码军规 + 模板要求的"编码检查项"表格逐项打勾 |
| 特性 N - 兼容性检查 | `detailed_design.cross_feature_impacts` 中老 om/新 om 双向兼容 |
| 特性 N - DT 设计 | 测试边界 / 测试设计（按 功能/性能/精度/兼容性/特性交叉 分类）/ 测试框架设计 |
| 特性 N - 验收标准 | `detailed_design.success_criteria` |
| 附录：竞品分析（如 `completed_steps` 含 `competitor`） | `competitor.competitors[*]` 中与本需求相关的竞品，聚焦 `implications_for_ge` + `ge_equivalent` + 对比总结表 |

**最终 spec 排布检查项**：

- [ ] 模板每个章节均已填充（不允许跳过）
- [ ] "整体介绍"承载顶层设计：三段流程 + 概念模型 + 模块交互
- [ ] 功能需求每条逐项展开四件套（介绍 / 输入 / 处理 / 输出），并标注其反向追溯的流程节点 id
- [ ] 接口检查项 8 个子项、编码检查项 2 个子项的"是否涉及"列必须填实（不允许留空）
- [ ] 每个新增 external 接口类/基类已附开发者使用示例伪代码（至少 3 个场景：最简 / 含可选特性 / 上层封装）
- [ ] 性能章节"涉及 optimizeStage1/2/build/loadmodelonline 时必须给影响评估"已逐项确认
- [ ] 若 `completed_steps` 含 `competitor`，附录"竞品分析"已写入

## 参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `analysis_output` | 是 | `analysis.md` 输出 |
| `existing_mechanism_output` | 视情况 | 当 `analysis_output.requirement_analysis.feature_type == "extension"` 时**必填**，由 `existing_mechanism.md` 产出 |
| `competitor_output` | 否 | `competitor.md` 输出，用于借鉴外部经验 |
| `constraints` | 否 | 显式约束（timeline、team_size、affected_modules 等） |
| `design_depth` | 否 | `overview`（仅顶层设计）/ `architecture`（顶层 + 详细架构）/ `detailed`（全量），默认 `architecture` |

## 输出格式（标准 JSON）

```json
{
  "version": "2.0",
  "domain": "GE / Graph Engine on Ascend",

  "high_level_design": {
    "_note": "顶层设计：功能拆分、概念模型、流程设计、模块交互边界、设计决策。本层禁止引用代码路径/类名/文件路径/Pass注册细节。",

    "functional_decomposition": [
      {
        "id": "F1",
        "name": "功能点名称（业务语言，禁止出现模块路径/类名）",
        "derived_from_flow_nodes": ["C2", "L1"],
        "business_scope_ref": "analysis_output.requirement_analysis.business_scope.in_scope 的哪一条（原文复制）",
        "purpose": "做什么、为什么需要（业务语言）",
        "alternatives_considered": [
          {"option": "被否决方向", "reason_rejected": "为什么不选"}
        ]
      }
    ],

    "conceptual_model": {
      "entities": [
        {
          "name": "实体名（业务语言，禁止用类名直接命名）",
          "responsibility": "职责描述",
          "attributes": ["关键属性（业务描述，非字段名）"],
          "lifecycle": "创建/使用/销毁时机"
        }
      ],
      "relationships": [
        {"from": "实体A", "to": "实体B", "type": "依赖 | 组合 | 聚合 | 关联", "description": "关系说明"}
      ]
    },

    "flow_design": {
      "_note": "端到端流程，只描述做什么不描述怎么做。复用节点列出即可，不引用代码路径。",
      "summary": "一句话说明本特性下端到端流程的整体形态",
      "compile_flow": {
        "summary": "编译期一句话流程",
        "nodes": [
          {
            "id": "C1",
            "name": "节点名（业务语言，与现有代码概念对齐时用现有概念名）",
            "is_new": false,
            "trigger": "什么条件触发本节点",
            "input": "输入数据/状态",
            "processing": "处理逻辑（一两句话，非伪代码）",
            "output": "输出数据/状态",
            "next": ["C2"],
            "reuse_from_existing_mechanism": "若 is_new=false，对齐 existing_mechanism_output.phases.compile_phase.key_steps 的哪一步",
            "rationale": "仅新增节点需填：为什么需要"
          }
        ]
      },
      "load_flow": {
        "summary": "...",
        "nodes": [
          {"id": "L1", "name": "...", "is_new": false, "trigger": "...", "input": "...", "processing": "...", "output": "...", "next": ["L2"]}
        ]
      },
      "execute_flow": {
        "summary": "...",
        "nodes": [
          {"id": "E1", "name": "...", "is_new": false, "trigger": "...", "input": "...", "processing": "...", "output": "...", "next": ["E2"]}
        ]
      },
      "flow_closure_check": {
        "compile_to_load_handoff": "编译产物如何被加载流程接住",
        "load_to_execute_handoff": "加载产物如何被执行流程接住",
        "user_input_to_output_traceable": true
      }
    },

    "module_interaction_and_boundary": {
      "subsystems": [
        {
          "name": "子系统名（架构层概念，如'编译优化层'而非'compiler/graph/passes/'）",
          "responsibility": "职责边界",
          "provides": ["对外提供的能力"],
          "depends_on": ["依赖的其他子系统"],
          "contract": "数据/控制流约定"
        }
      ],
      "interaction_sequence": "子系统间交互时序的文字描述"
    },

    "design_decisions": [
      {
        "id": "D1",
        "problem": "要解决的设计问题",
        "candidates": ["方案A描述", "方案B描述"],
        "decision": "选择的方案",
        "rationale": "选择理由",
        "consequences": ["正面影响", "负面影响/代价"],
        "linked_to": ["F1", "D2"]
      }
    ]
  },

  "detailed_design": {
    "_note": "详细设计：代码级承载。本层可引用模块路径、类名、文件路径、Pass注册细节等。所有字段仅当对应功能涉及时才填充（不需要全量空填）。",

    "functional_points": [
      {
        "id": "F1",
        "_note": "id 必须对齐 high_level_design.functional_decomposition 中同 id 条目",
        "carrier": {
          "modules": ["承载模块/文件路径"],
          "new_passes": ["如涉及，引用 pass_design.new_passes 的 name"],
          "data_structure_changes": ["如涉及，引用 data_model 的字段"],
          "interfaces": ["如涉及，引用 external_interface_design 的字段"]
        },
        "key_algorithms": [
          {"name": "...", "strategy": "...", "complexity": "...", "edge_cases": [...]}
        ]
      }
    ],

    "architecture": {
      "_note": "functional_points 的承载层细节，按 GE 4 阶段编译管线视角组织。孤立的承载项视为冗余。",
      "summary": "一句话方案概要",
      "compilation_pipeline": [
        {
          "stage": "Parse",
          "input": "ONNX / Caffe / TF PB / MindSpore",
          "output": "AscendIR ComputeGraph",
          "key_components": ["parser/onnx", "parser/caffe", "parser/tensorflow", "parser/mindspore"],
          "module_path": "parser/",
          "this_feature_impact": "本特性是否在此阶段有改动"
        },
        {
          "stage": "Preprocess",
          "input": "ComputeGraph",
          "output": "Normalized ComputeGraph",
          "key_components": ["GraphPrepare", "InferShape", "InferFormat", "InsertTransData"],
          "module_path": "compiler/graph/preprocess/",
          "this_feature_impact": "..."
        },
        {
          "stage": "GraphOptimize / Partition",
          "input": "Normalized Graph",
          "output": "Optimized + Partitioned Graph",
          "key_components": ["PreRunOptimizeOriginalGraph", "EnginePartition (EnginePlacer + Cluster)", "PreRunOptimizeSubGraph", "PreRunAfterOptimizeSubGraph", "Pattern Matcher", "Pass Registry"],
          "module_path": "compiler/graph/optimize/ + compiler/graph/partition/ + compiler/graph/passes/",
          "this_feature_impact": "..."
        },
        {
          "stage": "Build",
          "input": "Optimized Graph",
          "output": "om Model",
          "key_components": ["StreamAllocator", "MemoryPlanner (SymbolToAnchors + block_mem)", "TaskBuilder", "ModelSerializer"],
          "module_path": "compiler/graph/build/",
          "this_feature_impact": "..."
        },
        {
          "stage": "Runtime Load & Execute",
          "input": "om Model + 用户输入",
          "output": "Tensor Outputs",
          "key_components": ["RT1.0 Known Shape Executor (DavinciModel / 模型下沉)", "RT2.0 Unknown Shape Executor (Hybrid / Lowering / ModelV2Executor)", "HCCL", "ACL"],
          "module_path": "runtime/v1/ + runtime/v2/",
          "this_feature_impact": "..."
        }
      ],

      "modules": [
        {
          "name": "<新增/修改的模块>",
          "responsibility": "...",
          "module_path": "compiler/graph/.../...",
          "interfaces": ["公共类/函数签名"],
          "dependencies": ["graph_metadef/", "base/", "其他模块"],
          "is_new": false,
          "constraints_to_obey": ["docs/en/design/constraints/<对应约束文档>"]
        }
      ],

      "data_model": {
        "ascend_ir_changes": "本特性是否新增/修改 AscendIR 对象（OpDesc、TensorDesc、Attr）；如改动 graph_metadef/ 必须有明确理由",
        "anchor_usage_changes": "是否新增 Anchor 类型或改变现有 Anchor 语义",
        "om_format_changes": "om 模型格式是否变更；如变更需保证向后兼容",
        "cache_format_changes": "model_cache / JIT 缓存格式变更"
      }
    },

    "key_algorithms": {
      "<algorithm_name>": {
        "purpose": "...",
        "strategy": "...",
        "ge_module": "对应实现位置",
        "borrowed_from": "竞品对应做法（来自 competitor_output.borrow_and_avoid.borrow）",
        "complexity": "时间/空间复杂度",
        "edge_cases": ["边界场景"]
      }
    },

    "pass_design": {
      "new_passes": [
        {
          "name": "...",
          "type": "GraphPass | NodePass",
          "stage": "PreRunOptimizeOriginalGraph | PreRunOptimizeSubGraph | PreRunAfterOptimizeSubGraph",
          "module_path": "compiler/graph/passes/<sub>/",
          "registration": "如何注册到 Pass Registry",
          "pattern": "如使用 Pattern Matcher，列出模式骨架",
          "rerun_policy": "立即重遍历 / 延迟重遍历 / 不重遍历",
          "interaction_with_others": "与现有 Pass 的执行顺序约束"
        }
      ],
      "modified_passes": [
        {"name": "...", "module_path": "...", "change": "..."}
      ]
    },

    "engine_and_partition_design": {
      "engines_affected": ["AICore", "AICPU", "VectorEngine", "HCCL", "GE Local Engine"],
      "engine_placer_changes": "如有",
      "partition_algorithm": "Cluster-based / DynamicShapePartition / 自定义；HasSecondPath 校验",
      "subgraph_strategy": "子图切分策略"
    },

    "memory_and_stream_design": {
      "memory_planner_changes": "是否影响 SymbolToAnchors / block_mem / zero_copy",
      "stream_allocator_changes": "是否影响多流编排、流复用、event 同步",
      "memory_conflict_handling": "新增 Inplace / 连续内存场景的冲突处理",
      "om_size_strategy": "om 大小控制策略（如有上限，列出超限处理：拆分 / 降级 / 失败）"
    },

    "runtime_design": {
      "v1_changes": "Known Shape Executor / DavinciModel / 模型下沉 / 地址刷新 改动",
      "v2_changes": "Unknown Shape Executor / Lowering / ExecuteGraph 改动",
      "guard_design": "guard 注入点、guard 校验时机、guard miss 时的 fallback 策略（必须 fallback 原始 om 时显式声明）",
      "cross_runtime_compat": "RT1.0 / RT2.0 共存策略"
    },

    "external_interface_design": {
      "acl_api_changes": [
        {"name": "新增/修改的 ACL 接口", "header": "对应头文件", "doc_action": "需要 api-doc-generator 生成对外接口文档"}
      ],
      "atc_options": ["新增 ATC 命令行选项及默认值"],
      "session_api_changes": "Session 接口变更",
      "ascend_ir_external_changes": "对外暴露的 AscendIR 结构变更（如有，需 parser/ 同步）"
    },

    "milestones": [
      {
        "phase": "顶层设计",
        "deliverables": ["high_level_design 五个字段完整产出"],
        "exit_criteria": "评审通过"
      },
      {
        "phase": "详细设计",
        "deliverables": ["detailed_design 完整产出", "跨特性影响清单（cross_feature_check.md 逐场景分析）", "对外接口草案"],
        "exit_criteria": "评审通过 + 接口冻结"
      },
      {
        "phase": "编码",
        "deliverables": ["源码 PR", "Pass 注册与单测桩"],
        "exit_criteria": "通过 ge-code-reviewer 检视"
      },
      {
        "phase": "UT",
        "deliverables": ["UT 用例（覆盖率目标 ≥ X%）"],
        "exit_criteria": "ge-dt-runner --ut 全通过"
      },
      {
        "phase": "ST",
        "deliverables": ["ST 用例（关键场景 + 边界场景）"],
        "exit_criteria": "ge-dt-runner --st 全通过"
      },
      {
        "phase": "跨特性回归",
        "deliverables": ["跨特性 ST 回归报告（覆盖 cross_feature_check.md 中影响的特性）"],
        "exit_criteria": "无回归"
      },
      {
        "phase": "验收",
        "deliverables": ["性能数据 vs 基线", "对外文档（api-doc-generator 生成）", "特性文档（docs/en/design/features/ 下新增）"],
        "exit_criteria": "性能达标 + 文档齐备"
      }
    ],

    "risk_mitigation": [
      {
        "risk": "om 产物约束未对齐（大小、向后兼容、产物自包含）",
        "category": "om_constraint",
        "probability": "medium",
        "impact": "high",
        "mitigation": "在详细设计中明确 om 大小上限、超限策略、向后兼容承诺",
        "contingency": "降级或失败返回，并在 release notes 中显式说明"
      },
      {
        "risk": "Runtime 兼容性问题（guard 失配、RT1.0/RT2.0 共存）",
        "category": "runtime_compat",
        "probability": "medium",
        "impact": "high",
        "mitigation": "明确 guard 注入点、校验时机和失配处理路径；走 ST 回归",
        "contingency": "Runtime 显式日志 + 安全降级"
      },
      {
        "risk": "跨特性影响遗漏",
        "category": "cross_feature",
        "probability": "high",
        "impact": "medium",
        "mitigation": "强制按 docs/en/design/cross_feature_check.md 逐场景分析",
        "contingency": "上线前补充跨特性 ST"
      },
      {
        "risk": "graph_metadef/ 未授权改动",
        "category": "metadef_constraint",
        "probability": "low",
        "impact": "high",
        "mitigation": "除新增类型定义外，禁止修改 graph_metadef/（参考 AGENTS.md）",
        "contingency": "如确需改动需走 graph_metadef.md 约束评审"
      }
    ],

    "integration_strategy": {
      "parser_integration": "parser/<format>/ 接入路径（如需要）",
      "runtime_integration": "RT1.0 / RT2.0 适配（runtime/v1/ / runtime/v2/）",
      "acl_integration": "对外 ACL/ATC API 暴露（如非空，触发 api-doc-generator）",
      "session_integration": "Session 层适配（如需要）",
      "cross_feature_integration": "必须填写 docs/en/design/cross_feature_check.md 清单",
      "feature_documentation": "新增 docs/en/design/features/<feature>.md 文档"
    },

    "cross_feature_impacts": {
      "method": "按 docs/en/design/cross_feature_check.md 逐场景分析",
      "scenarios": [
        {"scenario": "动态 shape", "affected": "yes/no", "impact": "...", "mitigation": "..."},
        {"scenario": "内存复用", "affected": "yes/no", "impact": "...", "mitigation": "..."},
        {"scenario": "流分配", "affected": "yes/no", "impact": "...", "mitigation": "..."},
        {"scenario": "零拷贝", "affected": "yes/no", "impact": "...", "mitigation": "..."},
        {"scenario": "external weight", "affected": "yes/no", "impact": "...", "mitigation": "..."},
        {"scenario": "dump / 溢出", "affected": "yes/no", "impact": "...", "mitigation": "..."},
        {"scenario": "model cache", "affected": "yes/no", "impact": "...", "mitigation": "..."},
        {"scenario": "profiling", "affected": "yes/no", "impact": "...", "mitigation": "..."}
      ]
    },

    "test_strategy": {
      "ut_plan": [
        {"target": "tests/ut/<sub>/...", "scenarios": ["..."]}
      ],
      "st_plan": [
        {"target": "tests/st/<sub>/...", "scenarios": ["关键场景", "边界场景", "异常场景"]}
      ],
      "regression_st": ["跨特性回归列表"],
      "performance_benchmark": "性能基线 + 验收阈值"
    },

    "success_criteria": [
      "功能正确性：与基线数值误差 ≤ X",
      "性能：throughput/latency 达到 analysis.performance_requirements.metrics",
      "om 大小：≤ 设定上限",
      "UT 覆盖率：≥ X%",
      "ST 通过率：100%",
      "跨特性回归：无新增失败",
      "对外接口文档：齐备（api-doc-generator 产出）",
      "无 graph_metadef/ 未授权改动"
    ],

    "blocking_prerequisites": [
      {
        "id": "B1",
        "question": "必须在详细设计前回答的问题（来自 analysis_output.blockers 或扩展场景下 existing_mechanism_output.key_abi.extension_impact_note）",
        "why_blocking": "为什么这个问题不回答整个设计无法收敛",
        "owner": "回答人/团队",
        "proposed_action": "下一步动作（如：spike / 与 X 团队对齐 / 读 Y 代码）"
      }
    ],

    "next_actions": [
      "（自动填充）调用 api-doc-generator 生成对外接口文档",
      "（自动填充）对照 docs/en/design/cross_feature_check.md 补齐影响清单",
      "（自动填充）按 docs/en/design/design_document_template.md 模板产出最终设计 spec",
      "（自动填充）在 docs/en/design/features/ 下新增特性文档"
    ]
  }
}
```

## 执行逻辑

### 阶段一：顶层设计（`high_level_design`）

**必须**先产出，在与人确认通过前不得进入阶段二。本阶段**禁止**引用代码路径、类名、文件名。

1. **功能拆分（`functional_decomposition`）**：
   - 从 `analysis_output.requirement_analysis.business_scope.in_scope` 拆出独立功能点，用业务语言描述
   - 所有功能点必须**同层级**：每条是可独立验证的完整能力，内部算法策略放设计决策或算法章节
   - 贯穿性属性（降级策略、模式开关、兼容性约束）不放入此表，以独立段落说明
   - 每条必须有 `business_scope_ref`，无法对齐的视为衍生膨胀
   - 每条至少记录一条 `alternatives_considered`
2. **概念模型（`conceptual_model`）**：
   - 识别核心实体、属性、生命周期、实体间关系
   - 实体名用业务语言，禁止用类名直接命名
3. **流程设计（`flow_design`）**：
   - 若有 `existing_mechanism_output`，先将其 `phases.*.key_steps` 作为复用候选（`is_new=false`）
   - 再根据功能拆分识别新增节点（`is_new=true`），填 `rationale`
   - 校验 `flow_closure_check`：三段流程首尾相接、用户输入到输出可追溯
   - **禁止**在流程节点中引用模块路径
4. **模块交互与边界（`module_interaction_and_boundary`）**：
   - 划分子系统，明确每个子系统的职责边界、提供的能力、依赖关系
   - 用架构层面概念描述（如"编译优化层"而非"compiler/graph/passes/"）
5. **设计决策（`design_decisions`）**：
   - 记录关键权衡：问题 → 候选方案 → 决策 → 理由 → 影响
   - 决策记录可关联到功能点（`linked_to`）

**顶层设计产出后暂停**，展示摘要并等待用户确认，再进入阶段二。

### 阶段二：详细设计（`detailed_design`）

顶层确认后再产出。此处允许代码级引用。

6. **功能点承载（`functional_points`）**：
   - `id` 对齐 `high_level_design.functional_decomposition` 中同 id 条目
   - 填充 `carrier`：modules / new_passes / data_structure_changes / interfaces
   - 每条至少给一条 `key_algorithms`（如适用）
7. **管线映射**：从 `analysis_output.technical_decomposition` 字段反推哪些 GE 阶段受影响：
   - `ir_requirements.metadef_impact` → Parse + graph_metadef/
   - `pass_requirements` → GraphOptimize（按 stage 字段定位）
   - `engine_partition_requirements` → EnginePartition
   - `memory_requirements` / `stream_impact` → Build
   - `runtime_requirements.v1_impact` / `v2_impact` → Runtime
8. **模块改动清单**：将管线映射结果细化为具体模块路径（用真实路径，参考 `references/ge_glossary.md`）
9. **承载层（architecture / pass_design / runtime_design 等）反向填充**：
   - 从 `functional_points.carrier` 反向汇总到 `architecture.modules` / `pass_design` / `engine_and_partition_design` / `memory_and_stream_design` / `runtime_design`
   - 没有 functional_point 引用的 carrier 条目要被质疑
10. **算法选择**：从 `competitor_output.borrow_and_avoid.borrow` 中挑选可借鉴的外部经验
11. 按顺序产出剩余字段：`external_interface_design` → `cross_feature_impacts` → `risk_mitigation` → `test_strategy` → `success_criteria` → `integration_strategy`
12. **里程碑**：固定按 `顶层设计 → 详细设计 → 编码 → UT → ST → 跨特性回归 → 验收` 七阶段产出
13. **风险管控**：必须包含 `om_constraint` / `runtime_compat` / `cross_feature` / `metadef_constraint` 四大固定类别
14. **前置阻塞项**：必须从 `analysis_output.blockers` 与 `existing_mechanism_output.key_abi.extension_impact_note` 中识别，无阻塞项时显式填空数组
15. **跨特性影响**：按 `cross_feature_check.md` 逐场景标记 `cross_feature_impacts.scenarios`
16. **next_actions 自动填充**（参见 SKILL.md 步骤 5 规则）

### design_depth 控制

| depth | 产出范围 |
|-------|---------|
| `overview` | 仅 `high_level_design` 全部五个字段 |
| `architecture` | `high_level_design` + `detailed_design`（含 architecture + cross_feature_impacts + risk_mitigation + milestones，不含 pass/runtime/memory 细节） |
| `detailed` | 全量（`high_level_design` + `detailed_design` 全部） |

## 交付检查清单

实现设计完成后**必须**输出以下格式的检查结果（与 SKILL.md 步骤 6 一致）：

```
### 需求设计检查结果
- [x] 架构文档加载：已加载 <列出加载的 docs/en/design/*.md 文件>
- [x] 两层边界校验：high_level_design 未出现代码路径/类名/文件名/Pass注册细节
- [x] 流程闭环校验：flow_design 三段流程已闭环（compile → load → execute），无悬空节点
- [x] 功能拆分可追溯：所有 functional_decomposition 均能反向追溯到 business_scope.in_scope
- [x] 功能点承载对齐：detailed_design.functional_points[*].id 均能追溯到 high_level_design.functional_decomposition[*].id
- [x] 概念模型完整：核心实体/关系/生命周期已描述（非代码级）
- [x] 设计决策记录：关键权衡已记录（问题→候选→决策→理由→影响）
- [x] 跨特性交叉影响：已按 cross_feature_check.md 逐场景分析，影响清单见 cross_feature_impacts.scenarios
- [x] 设计文档模板对齐：已按 design_document_template.md 章节填充
- [x] om 约束（如触发）：列出 analysis.constraints 中声明的 om 大小上限、向后兼容、guard/fallback 等并逐项校验
- [x] graph_metadef/ 改动：<无 / 仅新增类型定义，理由 ...>
- [x] 前置阻塞项：blocking_prerequisites 已显式列出（含空数组场景）
- [x] 下游串接：已建议调用 api-doc-generator 生成对外接口文档
- [x] 接口示例：每个新增 external 接口类/基类已附开发者使用示例伪代码
```

## 注意事项

- **顶层设计与详细设计之间的边界不可逾越**：顶层设计中出现的任何代码路径引用都是错误，必须在进入详细设计前清理
- `functional_decomposition[*].id` 与 `functional_points[*].id` 必须一一对应
- `architecture.modules` 中所有模块路径**必须**是真实路径；编造路径会导致后续编码阶段卡死
- `pass_design.new_passes[].stage` 字段只能取 `PreRunOptimizeOriginalGraph` / `PreRunOptimizeSubGraph` / `PreRunAfterOptimizeSubGraph` 三个值
- 涉及对外接口时，`integration_strategy.acl_integration` **必须**非空，且 `next_actions` 必须包含调用 `api-doc-generator`
- 涉及 `runtime_requirements.v2_impact` 时，**必须**在 `runtime_design.guard_design` 中明确 guard miss fallback 策略
- 涉及 om 产物的所有特性，**必须**在 `risk_mitigation` 中包含 `om_constraint` 风险条目
- 产品型号命名规范同 SKILL.md（Ascend 950PR/Ascend 950DT、Atlas A3 训练系列产品 等）
- **扩展特性下，`functional_points` 中所有 `carrier` 必须能在 `existing_mechanism_output.design_inputs_for_extension.must_new_in_extension` 找到对应业务条目**；不能对齐说明流程设计阶段对现状理解有偏差，要回去补走读
