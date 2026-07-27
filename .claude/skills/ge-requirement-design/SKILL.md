---
name: ge-requirement-design
description: GE (Graph Engine) 特性需求设计全流程，包含需求分析、竞品调研、实现设计三段式产出。当用户提到需求设计、特性预研、特性方案、写设计文档、设计 spec、design document、写设计、技术方案、TDD、架构设计、竞品分析、对标 TensorRT/Inductor/XLA 等场景时使用此 skill。覆盖 GE 4 阶段编译流程（Parse → Preprocess → GraphOptimize/Partition → Build）、AscendIR 数据模型、Ascend 硬件约束（Ascend 950PR/950DT、Atlas A3、Atlas A2、Ascend 310/Lite）、om 模型产物约束、Pass 体系扩展、RT1.0/RT2.0 运行时选型。
---

# GE 特性需求设计

面向 GE 项目特性研发的三段式需求设计 skill：**需求分析 → 竞品调研 → 实现设计**。产出可直接用于评审的中间 JSON 文档和最终设计草案，并自动衔接下游 `api-doc-generator`、`docs/en/design/cross_feature_check.md` 和 `docs/en/design/design_document_template.md`。

## 触发场景

- 新特性预研（如 autofuse、tiling 下沉、新引擎接入、新执行器）
- 写详细设计文档前的方案调研
- 竞品分析（对标 Inductor / TensorRT / XLA / TVM / OpenXLA / ExecuTorch 等外部图编译器）
- 改造既有 GE 模块（Pass 体系、内存规划、引擎分区、Runtime）前的需求梳理

## 工作流程

### 步骤 1：上下文加载（强制）

触发后**必须**先读取下列文件，再开始分析：

| 文件 | 用途 |
|------|------|
| `AGENTS.md` | 项目级总览、架构文档索引表（"架构文档加载"和"关键特性设计原则和软件约束"两张表） |
| `references/ge_glossary.md` | GE 术语 ↔ 通用图编译器术语映射 |
| `docs/en/design/design_document_template.md` | 设计文档模板（实现设计阶段对齐） |
| `docs/en/design/cross_feature_check.md` | 跨特性检查清单 |

根据 `requirement_text` 命中的触发词，**按需**加载 AGENTS.md 表中对应的架构文档（如涉及内存就加载 `memory-constraints.md`，涉及 RT2.0 就加载 `rt2_runtime.md`），不要一次性全加载。

### 步骤 2：解析输入

输入参数：

| 参数 | 必填 | 说明 |
|------|------|------|
| `requirement_text` | 是 | 原始需求描述（如"做一个 autofuse 离线推理特性，guard miss 必须 fallback 原始 om"） |
| `focus_area` | 否 | `analysis` / `competitor` / `implementation` / `all`，默认 `all` |
| `previous_output` | 否 | 上次执行的部分结果，用于断点续传，含 `completed_steps` 和 `step_outputs` |
| `user_confirmation` | 否 | 每步完成后是否暂停确认，默认 `true` |
| `constraints` | 否 | 显式约束，见下表 |

`constraints` 字段：

| 子字段 | 示例 |
|--------|------|
| `timeline` | "6个月内 GA" |
| `team_size` | 5 |
| `target_hardware` | `["Ascend 950PR/Ascend 950DT", "Atlas A3 训练系列产品", "Atlas A2 训练系列产品"]` |
| `runtime_version` | `RT1.0` / `RT2.0` / `both` |
| `om_size_limit` | "≤ 2 GB" |
| `compat` | `{guard_miss_fallback: true, om_backward_compat: true}` |
| `affected_modules` | `["compiler/graph/optimize/", "runtime/v1/"]` |

### 步骤 3：执行三段式（扩展场景为四段）

根据 `focus_area` 和 `previous_output.completed_steps` 决定执行哪些子流程，依次调用：

1. **需求分析** → `analysis.md`：先产出黑盒 `requirement_analysis`（背景 / 业务目标 / 业务范围 / feature_type），再产出 `technical_decomposition`（设计衍生层），最后产出 `blockers`
2. **现状机制走读**（仅当 `analysis_output.requirement_analysis.feature_type == "extension"` 时执行） → `existing_mechanism.md`：走读被扩展能力在编译/加载/执行三阶段的现状机制，产出 `phases.*.key_steps`、`key_data_structures`、`key_abi`、`current_boundary`，最后给 `design_inputs_for_extension` 作为下游 implementation 的输入
3. **竞品调研** → `competitor.md`：对外部图编译器（Inductor、TensorRT、XLA、TVM 等）做对比，每条产出 `implications_for_ge` 和 `ge_equivalent` 双字段
4. **实现设计** → `implementation.md`：先产出 `high_level_design`（功能拆分/概念模型/流程设计/模块交互边界/设计决策，禁止代码路径引用），确认后再产出 `detailed_design`（代码级承载：模块/Pass/Runtime/接口/风险）。同时产出 `blocking_prerequisites`

每完成一步：
- 如 `user_confirmation = true`，展示该步骤摘要并询问"是否继续下一步"
- 将该步骤输出存入 `step_outputs[step_name]`

### 步骤 4：领域规则触发

在解析 `requirement_text` 时主动匹配关键词，将匹配结果作为隐式约束注入到所有子步骤：

| 关键词 | 触发动作 |
|--------|----------|
| `om` / `离线` / `ATC` | `deployment_mode = offline`；加载 om 大小约束子流程 |
| `autofuse` / `融合 Pass` | 加载 `compiler/graph/passes/fusion/` 模块；提示梳理融合 pattern、cost model、回退策略 |
| `RT2.0` / `动态 shape` / `dynamic` | `runtime_version = RT2.0`；加载 `docs/en/design/constraints/rt2_runtime.md` |
| `静态 shape` / `known shape` | 加载 `docs/en/design/constraints/known_shape_runtime.md` |
| `内存` / `block_mem` / `零拷贝` | 加载 `docs/en/design/constraints/memory-constraints.md` |
| `图拆分` / `cluster` / `partition` | 加载 `docs/en/design/constraints/graph_split.md` |
| `流` / `stream` / `多流` | 加载 `docs/en/design/constraints/stream_allocator.md` |
| `集合通信` / `allreduce` / `HCCL` | 标记 HCCL 引擎 + Stream 编排路径 |
| `Tiling` / `tiling sink` | 加载 `docs/en/design/features/tiling_sink.md` |
| `InferShape` / `符号化` | 加载 `docs/en/design/features/infer_shape.md` |
| `Format` / `TransData` | 加载 `docs/en/design/features/infer_format.md` |
| `引擎` / `EnginePlacer` | 加载 `docs/en/design/features/engine.md` |
| `dump` / `溢出` | 加载 `docs/en/design/features/datadump.md` |

### 步骤 5：聚合与下游串接

所有子步骤完成后产出最终结构：

```json
{
  "status": "completed | partial | need_clarification",
  "completed_steps": ["analysis", "existing_mechanism", "competitor", "implementation"],
  "outputs": {
    "analysis": { ... },
    "existing_mechanism": { ... },   // 仅 extension 场景非空
    "competitor": { ... },
    "implementation": { ... }
  },
  "summary": "本次需求设计的核心结论摘要（不超过 500 字）",
  "next_actions": [],
  "risk_flags": [],
  "blocking_prerequisites": []        // 来自 implementation.detailed_design.blocking_prerequisites
}
```

`next_actions` 的自动填充规则：

| 触发条件 | 自动加入的下一步 |
|----------|------------------|
| `implementation.detailed_design.integration_strategy.acl_integration` 非空 | "调用 `api-doc-generator` 生成对外接口文档" |
| `implementation.detailed_design.risk_mitigation` 含跨特性风险 | "对照 `docs/en/design/cross_feature_check.md` 补齐影响清单" |
| `completed_steps` 包含 `"competitor"` | "将竞品调研结果作为附录写入最终设计 spec（聚焦与本需求相关的竞品方案、对 GE 的启示、GE 等价实现对比）" |
| 任意 | "按 `docs/en/design/design_document_template.md` 模板产出最终设计 spec" |
| 触发了"关键特性设计原则和软件约束"中的文档 | "在设计 spec 中显式声明对应约束文档已加载并对齐" |

### 步骤 6：交付检查清单（实现设计阶段必走）

实现设计完成后，输出 `### 需求设计检查结果` 区块，按以下格式：

```
### 需求设计检查结果
- [x] 需求分析层级校验：requirement_analysis.business_scope 未混入 Pass / IR / SoBin / stream / engine / dispatcher 等内部术语
- [x] 扩展特性现状机制：已产出 existing_mechanism（feature_type=extension） / 不适用（feature_type=new）
- [x] 两层边界校验：high_level_design 未出现代码路径/类名/文件名/Pass注册细节
- [x] 流程闭环校验：high_level_design.flow_design 三段流程已闭环，所有 functional_decomposition 反向可追溯到流程节点与 business_scope.in_scope
- [x] 功能点承载对齐：detailed_design.functional_points[*].id 均能追溯到 high_level_design.functional_decomposition[*].id
- [x] 概念模型完整：high_level_design.conceptual_model 已描述核心实体/关系/生命周期
- [x] 设计决策记录：high_level_design.design_decisions 已记录关键权衡
- [x] 架构文档加载：已加载 rt2_runtime.md / memory-constraints.md
- [x] 跨特性交叉影响：已按 cross_feature_check.md 逐场景分析，影响清单见 detailed_design.cross_feature_impacts
- [x] 设计文档模板对齐：已按 design_document_template.md 章节填充
- [x] 竞品分析体现：若 completed_steps 包含 competitor，竞品调研结果已作为附录写入最终设计 spec
- [x] 前置阻塞项：detailed_design.blocking_prerequisites 显式列出（含空数组场景）
- [x] om 约束（如触发）：列出 constraints 中声明的 om 大小上限、向后兼容、fallback 策略等并逐项校验
- [x] 下游串接：已建议调用 api-doc-generator 生成对外接口文档（如有 ACL/ATC 暴露）
- [x] 文档精简规范：已按"设计文档精简规范"检查，无冗余章节、无重复描述、无全篇"不涉及"的章节
```

### 步骤 7：设计文档精简规范（强制）

产出最终设计文档时**必须**遵循以下精简原则，避免文档冗长：

#### 7.1 结构精简

| 规则 | 说明 | 反面案例 |
|------|------|----------|
| **合并重叠章节** | "简介"+"总体概述"合并为单一"概述"章节 | 两个章节分别描述背景和目标 |
| **删除全篇"不涉及"的章节** | 如"接口设计"全篇"不涉及"，用一行带过 | 保留完整章节结构，每行写"不涉及" |
| **合并薄章节** | 安全检查、兼容性、平台化等薄章节合并到"非功能需求" | 每个薄点独立成章节 |
| **降级/回退机制权威定义** | 选择一处权威描述（如独立"降级回退"章节），其他章节仅引用 | 在 3-4 个章节重复描述同一机制 |

#### 7.2 内容去重

| 规则 | 说明 | 检查方法 |
|------|------|----------|
| **单一权威定义** | 每个核心概念（如回退机制、复用组件、约束条件）只在一处完整描述 | 全文搜索关键词，检查是否多处完整描述 |
| **引用而非重复** | 其他章节需要时写"见第 X 章"而非重复内容 | 检查是否有 3+ 处描述同一机制 |
| **约束/依赖集中** | 约束和依赖集中在"概述"或独立章节，不在功能需求中重复 | 检查每个功能需求是否重复提及约束 |

#### 7.3 代码示例规范

| 层级 | 适用场景 | 示例 |
|------|----------|------|
| **签名级** | 接口定义、函数原型 | `Status GenerateTask(const NodePtr &, TaskDef &)` |
| **伪代码级** | 关键算法流程 | 用缩进文本描述步骤，不写完整语法 |
| **完整实现级** | 仅用于关键数据结构定义 | `struct FusedHostCpuArgs { ... }` |

**禁止**：在非关键路径上写完整实现代码（如完整的 kernel 执行函数、完整的 Builder 实现）。

#### 7.4 表达优化

| 场景 | 推荐 | 避免 |
|------|------|------|
| 多个并列属性/字段 | 表格 | 散文列表 |
| 多个子系统/模块 | 表格（模块/路径/说明） | 散文段落 |
| 编译流水线位置 | mermaid flowchart | 展开嵌套的 ASCII 树 |
| 端到端流程 | mermaid flowchart / sequenceDiagram | 纯文字列表 |
| 模块交互时序 | mermaid sequenceDiagram | ASCII 箭头图 |

#### 7.5 精简检查清单

设计文档产出前逐项检查：

- [ ] 全文搜索核心关键词（如"回退"、"降级"、"复用"），检查是否 3+ 处完整描述同一机制
- [ ] 检查是否有章节全篇"不涉及"，如有则删除或合并为一行
- [ ] 检查代码示例是否超出签名级/伪代码级（关键数据结构除外）
- [ ] 检查是否有 2+ 个章节描述相同背景/目标
- [ ] 检查薄章节（<10 行）是否可合并
- [ ] 检查约束/依赖是否在功能需求中重复提及

## 断点续传

当 `previous_output.completed_steps` 非空时，跳过已完成步骤，仅执行剩余步骤。`step_outputs` 中已有的中间产物作为下游步骤的输入。

示例：
- 输入 `completed_steps: ["analysis", "competitor"]`，`focus_area: "implementation"` → 直接基于已有 analysis/competitor 输出执行 implementation
- 输入 `focus_area: "competitor"` → 仅执行竞品分析

## 子文档独立调用

`analysis.md` / `existing_mechanism.md` / `competitor.md` / `implementation.md` 四个文档均可独立调用（每个文档顶部都标记了 `standalone: true`）。但独立调用时**仍需先读取步骤 1 列出的强制上下文文件**。

`existing_mechanism.md` 独立调用时需手动提供 `existing_capability` 参数（业务语言描述要走读的能力）。

## 输出格式约定

- 中间产物：JSON（便于断点续传和机器后处理）
- 最终设计 spec：Markdown，按 `docs/en/design/design_document_template.md` 模板章节填充
- 摘要：中文，简洁，不超过 500 字

## Agent 编排与 Skills 优化

### 三阶段交互模型 → 四阶段设计模型

原三阶段模型升级为四阶段，**阶段三（顶层设计）必须与人充分交互确认后才能进入阶段四（详细设计）**：

```
阶段一：需求分析（交互密集）
  ├─ [并行] explore agent × N：代码调研，搜索现有机制、接口、执行路径
  ├─ [主 agent] 综合分析 → 生成问题清单
  ├─ [交互] 批量提问，与人确认每个需求点
  ├─ [交互] 补充提问（如有遗漏）
  └─ 产出：需求确认清单 + 派生需求 + 约束条件 + 风险项

阶段二：竞品调研 + 现状走读（按需并行）
  ├─ [可选] 竞品调研：外部编译器对标，产出 borrow_and_avoid
  ├─ [可选] 现状机制走读（extension 场景）：编译/加载/执行三阶段现状
  └─ 产出：competitor_output / existing_mechanism_output

阶段三：顶层设计（概念性、流程性，禁止代码路径）
  ├─ [主 agent] 功能拆分 + 概念模型 + 流程设计 + 模块交互边界 + 设计决策
  ├─ [交互] 确认关键设计决策、流程闭环、模块边界
  └─ 产出：high_level_design（全部五个字段）

阶段四：详细设计（代码级承载，收敛产出）
  ├─ [主 agent] 模块/Pass/Runtime/接口设计 + 风险管控 + 测试策略
  ├─ [可选] general agent 编写 PoC 验证高风险点
  └─ 产出：detailed_design（全量字段）+ 最终设计 spec
```

**关键约束**：阶段三产出 `high_level_design` 时，所有字段禁止出现代码路径/类名/文件路径/Pass注册细节。阶段四才允许进入代码级设计。

### Agent 编排规则

| 规则 | 说明 |
|------|------|
| **阶段一并行搜索** | 多个 explore agent 并行执行，按主题分工（如：接口层 / 执行路径层 / 资源管理层），信息收集时间减半 |
| **问题驱动交互** | 先穷举所有待确认问题，再批量提问，避免反复打断。每批问题不超过 6 个 |
| **阶段三顶层设计** | 不引代码、不引路径、不引类名。产出功能拆分/概念模型/流程/模块边界/设计决策。确认后再进入阶段四 |
| **阶段四 PoC 验证** | 对高风险设计点（如 ABI 兼容性、内存布局约束）用代码验证 |
| **复用 task_id** | 阶段二的补充调研可复用阶段一的 explore agent task_id，避免重复搜索 |

### 阶段一问题清单模板

需求分析阶段应覆盖以下维度的问题：

| 维度 | 示例问题 |
|------|---------|
| 数量/规模 | 单个实例可能需要多少个？固定还是动态？ |
| 同步机制 | 由框架自动同步还是使用者手动？触发时机？ |
| 生命周期 | 资源何时创建/销毁？跨 step 复用还是每次重建？ |
| 执行路径覆盖 | 需要支持哪些执行路径（V1/V2/both）？ |
| 能力范围 | 资源具备哪些能力？与已有资源是否等价？ |
| API 语义 | 多次调用的幂等性？默认参数？错误处理？ |
| 跨 step 一致性 | 跨 step 复用时是否保证一致性语义？ |
| 实现策略 | 多条路径是共用实现还是独立实现？ |
| 资源保护 | 是否需要上限保护？ |
| 降级策略 | 新机制失败时如何回退？回退粒度（整体/部分/逐节点）？回退时机（编译/加载/执行）？ |
| 复用深度 | 复用现有机制的哪些部分？仅接口/仅数据结构/完整流程？缺失部分如何处理？ |

### Extension 场景推荐搜索分工

当 `feature_type == "extension"` 时，阶段一并行搜索推荐按以下分工启动 explore agent：

| Agent | 搜索主题 | 典型搜索范围 |
|-------|---------|-------------|
| Agent 1 | 被扩展能力的完整机制 | 现有 Pass/模块的实现、注册、调用链 |
| Agent 2 | 可复用的基础设施 | 符号系统、codegen、公共工具类等可复用组件 |
| Agent 3 | 运行时执行路径 | RT1.0/RT2.0 中相关能力的 lowering + kernel 执行 |

三个 agent 并行执行，信息收集时间减半。搜索完成后综合分析，识别可复用组件和必须新增的能力。

### 工作量估算模板

需求分析阶段应产出 KLOC 估算，按以下模板：

| 模块 | 新增/修改 | 估算（行） | 说明 |
|------|----------|-----------|------|
| 模块 A | 新增 | ~N | 简要说明 |
| 模块 B | 修改 | ~N | 简要说明 |
| 测试（UT+ST） | 新增 | ~N | 简要说明 |
| **总计** | | **~N 行 ≈ X KLOC** | |

估算粒度为模块级别，不需要精确到函数。目的是帮助评估需求规模和排期。

### 优化检查清单

每次需求设计完成后回顾：

- [ ] 阶段一的 explore agent 是否并行执行？
- [ ] 问题清单是否一次穷举（而非分多批）？
- [ ] 阶段二的补充调研是否复用了阶段一的 task_id？
- [ ] 阶段三（顶层设计）产出是否完全未出现代码路径/类名/文件名/Pass注册细节？
- [ ] 阶段三（顶层设计）是否与人确认后才进入阶段四？
- [ ] 阶段四是否对高风险点做了 PoC 验证？
- [ ] `functional_decomposition[*].id` 与 `functional_points[*].id` 是否一一对应？
- [ ] 设计文档中是否标注了 external 接口（`【external】`标记）？
- [ ] 每个新增 external 接口类/基类是否附了开发者使用示例伪代码（至少 3 个场景）？
- [ ] extension 场景是否产出了可复用组件识别（`reusable_components`）？
- [ ] 是否产出了工作量估算（KLOC）？
- [ ] 概念模型是否完整（实体/关系/生命周期）？
- [ ] 设计决策是否记录了关键权衡（问题→候选→决策→理由→影响）？

### 经验教训

| 类别 | 发现的问题 | 固化改进 |
|------|-----------|----------|
| **交互模式** | 问题清单分多批提问，增加交互轮次 | 先穷举所有待确认问题，再批量提问，每批不超过 6 个 |
| **调研覆盖** | 阶段一只搜接口层，执行路径未全覆盖 | 阶段一并行搜索就必须覆盖所有执行路径（V1+V2） |
| **调研覆盖** | extension 场景缺少可复用组件识别的结构化产出 | `analysis.md` 增加 `reusable_components` 字段 |
| **问题清单** | 降级策略和复用深度未在模板中覆盖 | 模板增加"降级策略"和"复用深度"维度 |
| **问题清单** | 降级方案经历多轮迭代才收敛 | "降级策略"维度应引导一次性穷举所有可选方案 |
| **问题清单** | 需要 KLOC 估算但 skill 没有输出模板 | 增加模块级工作量估算模板 |
| **文档结构** | Skills 编排内容误写入设计文档 | 设计文档中不得出现 agent 编排描述 |
| **文档结构** | 设计文档冗长，同一机制多章节重复 | 增加精简规范：单一权威定义、引用而非重复、全篇"不涉及"的章节删除 |
| **文档层次** | 顶层设计与详细设计混杂，读者无法建立心智模型 | 重构为两层结构：`high_level_design` → `detailed_design` |
| **功能拆分** | 功能点粒度不一，算法策略、质量属性混入能力列表 | `functional_decomposition` 增加同层级约束 + 贯穿性属性独立描述 |
| **接口质量** | external 接口缺少开发者使用示例，评审无法验证可用性 | 每个新增 external 接口必须附至少 3 个场景的伪代码 |
| **下游串接** | 竞品调研做了但未写入最终设计文档 | `next_actions` 增加竞品串接规则 + 检查清单 |
| **下游串接** | 对外接口变更后未触发 docs 生成 | `next_actions` 增加 ACL/ATC 接口 doc 串接规则 |

## 注意事项

- 触发本 skill 时**不要**自动调用 `superpower brainstorming skill`（AGENTS.md 中的"需求开发"触发词）。两个 skill 定位不同：brainstorming 偏发散，本 skill 偏收敛和结构化产出。如果用户需要发散讨论，先用 brainstorming，再用本 skill 收敛。
- 涉及 `graph_metadef/` 的需求要特别警示：除非显式新增类型定义，否则不要修改 `graph_metadef/`（参考 AGENTS.md）
- 产品型号命名必须使用规范名称（Ascend 950PR/Ascend 950DT、Atlas A3 训练系列产品 等），参考 `api-doc-generator` SKILL.md 中的规则 #2
- 输出 GE 内部模块路径时使用真实路径（`compiler/graph/build/`、`compiler/graph/passes/`、`runtime/v1/`、`runtime/v2/`），不要使用通用术语（"frontend"、"middle-end"）作为主语
- 设计文档中**不要**包含 Skills 编排内容，Skills 编排属于本 skill 文件，不属于设计产出
- 涉及 external 接口变更的设计文档，必须以 `【external】` 标记所有 external 接口，并在文档开头汇总 external 变更清单
- 设计文档中每个新增的 external 接口类/基类，**必须**在接口设计章节附开发者使用示例伪代码（至少覆盖：最简场景、含可选特性的场景、上层封装场景），让评审者能从开发者视角验证接口的可用性和语义清晰度
