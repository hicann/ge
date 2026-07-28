# 子文档：现状机制走读（扩展特性专用）

## 描述

当 `analysis.md` 输出中 `requirement_analysis.feature_type == "extension"` 时，**强制调用本子文档**。

扩展特性的设计若不先把"被扩展的能力"在编译/加载/执行三阶段的现状机制讲清楚，下游设计会落到零碎引用（`SoBinType::kAutofuse @ op_so_bin.h:30` 这种），读者拼不出完整心智模型，评审与编码都会卡。

本文档的产物是后续 `implementation.flow_design` 的**输入**，自身不产出设计决策，只产出"现状是什么"。

standalone: true

## 上下文加载（强制）

调用前必须先读取：
- `analysis.md` 的输出（`requirement_analysis.extends_what` 字段确定要走读的能力）
- `references/ge_glossary.md`
- 根据被扩展能力涉及的目录，加载 `AGENTS.md` "架构文档加载"表中对应的特性文档

## 参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `existing_capability` | 是 | 要走读的现有能力名（业务语言，对齐 `analysis_output.requirement_analysis.extends_what`） |
| `entry_points` | 否 | 用户已知的入口提示（API / Session / ATC option / 文件路径）。**用于加快定位，不限制走读范围** |
| `analysis_output` | 否 | `analysis.md` 的输出，用于校验走读范围与需求层一致 |
| `depth` | 否 | `overview` / `standard` / `deep`，默认 `standard` |

`depth` 含义：
- `overview`：只走读用户接口 + 三阶段触发条件，不展开关键步骤
- `standard`：完整走读 `phases` + 关键数据结构 + 当前边界
- `deep`：增加 ABI 细节、关键 file:line 索引、调用链跟踪

## 工作流程

1. **接口走读**：用 grep / read 找到现有能力的用户入口
   - API 入口：通常在 `api/`、`base/common/` 下
   - Session 入口：`api/session/`
   - ATC 入口：`api/atc/main_impl.cc` 的命令行解析
   - 框架适配入口：`parser/` 下相关解析器
2. **编译期走读**：从入口追到编译产物
   - 触发条件：什么场景下走这条路径
   - 关键步骤：Pass 列表 / Lowering / 数据结构变换
   - 产物：落到磁盘还是内存数据结构
3. **加载期走读**：从产物到 Runtime 数据结构
   - 触发条件：什么 API 触发加载
   - 关键步骤：反序列化 / 注册 / 检查
   - 产物：Runtime 可执行对象
4. **执行期走读**：从用户调用到推理输出
   - 触发条件：什么 API 触发执行
   - 关键步骤：调度 / 内核执行 / 同步
   - 产物：用户输出 Tensor
5. **关键数据结构与 ABI**：列出在三阶段流转的核心数据结构（类名 + file:line）
6. **当前能力边界**：已支持 / 不支持 / 已知遗留

## 输出格式（标准 JSON）

```json
{
  "version": "1.0",
  "existing_capability": "<对齐 analysis_output.requirement_analysis.extends_what>",

  "user_interfaces": [
    {
      "name": "入口名（业务语言，如 'Session.RunGraph'）",
      "entry": "file:line",
      "purpose": "用户用它来做什么",
      "key_inputs": ["关键输入参数"],
      "key_outputs": ["关键输出"]
    }
  ],

  "phases": {
    "compile_phase": {
      "trigger": "什么条件触发本能力的编译路径",
      "key_steps": [
        {
          "step": "步骤名",
          "location": "file:line 或模块路径",
          "what_it_does": "做什么（一两句话）",
          "produces": "产物"
        }
      ],
      "produces": "编译期最终产物（数据结构 / 文件）"
    },
    "load_phase": {
      "trigger": "什么 API 触发加载",
      "key_steps": [...],
      "produces": "加载完成后 Runtime 持有的对象"
    },
    "execute_phase": {
      "trigger": "什么 API 触发执行",
      "key_steps": [...],
      "produces": "用户拿到的输出"
    }
  },

  "key_data_structures": [
    {
      "name": "类/结构体名",
      "location": "file:line",
      "purpose": "在本能力中承担什么角色",
      "carries_across_phases": ["compile_phase", "load_phase", "execute_phase"]
    }
  ],

  "key_abi": [
    {
      "name": "接口/函数名",
      "signature": "返回类型 + 参数列表（简化）",
      "location": "file:line",
      "context_dependencies": ["aclrtStream", "session_id", "gert::Tensor", "..."],
      "extension_impact_note": "扩展场景下这些上下文依赖是否仍然成立"
    }
  ],

  "current_boundary": {
    "supported": ["已支持场景列表（业务语言）"],
    "unsupported": ["明确不支持的场景"],
    "known_gaps": ["已知遗留 / TODO（带出处）"]
  },

  "design_inputs_for_extension": {
    "_note": "本节是写给 implementation.flow_design 看的：哪些现有节点可以复用、哪些必须新增、哪些边界要打破",
    "reusable_compile_nodes": ["流程节点 id + 说明"],
    "reusable_load_nodes": [...],
    "reusable_execute_nodes": [...],
    "must_new_in_extension": ["扩展场景下必须新增的能力（业务语言）"],
    "abi_to_break_or_extend": ["现有 ABI 在扩展场景下需要剥离/重新设计的清单"]
  }
}
```

## 执行逻辑

1. 从 `analysis_output.requirement_analysis.extends_what` 推断被走读的能力
2. 用 `entry_points`（如给出）作为起点；否则用关键词 grep 找入口
3. 按"接口 → 编译 → 加载 → 执行"顺序走读，**禁止跳过中间阶段**
4. 每个 `key_steps` 必须有 `location`（file:line 或精确模块路径），不允许只写概念
5. `design_inputs_for_extension` 是本文档**唯一的"对设计阶段输出"**，其余字段都是事实描述
6. 按 `depth` 控制粒度

## 与下游 implementation.md 的衔接

- `phases.*` 提供给 `implementation.flow_design` 作为流程节点的"基线参考"
- `key_data_structures` 提供给 `implementation.functional_points` 的 `carrier_modules` 字段
- `key_abi.context_dependencies` 与 `extension_impact_note` 直接驱动 `flow_design` 中需要"新增节点"的判断
- `current_boundary.known_gaps` 进入 `implementation` 的 `blockers` 或前置阻塞清单
- `design_inputs_for_extension.must_new_in_extension` 必须能在 `flow_design.nodes[*].is_new=true` 找到对应节点

## 注意事项

- 本文档**不做设计决策**。即使走读过程中发现"明显该这么改"，也只写在 `design_inputs_for_extension` 里作为输入，不写在其它字段
- 所有 `location` 字段必须经实际 grep / read 验证，**不允许编造行号**
- 若 `entry_points` 给出的路径与实际入口不一致，以实际为准，并在 `assumptions` 中说明
- 涉及 Ascend 硬件型号时使用规范名称（Ascend 950PR/Ascend 950DT、Atlas A3 训练系列产品 等）
- 走读完成后，若发现 `requirement_analysis.business_scope` 与现状机制存在不可调和的冲突，必须停下来用 `blockers` 反馈给上游 `analysis.md`，不要自行修正
