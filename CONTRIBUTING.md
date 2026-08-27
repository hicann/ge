# 贡献指南

本项目欢迎广大开发者体验并参与贡献，在参与社区贡献之前。请参见[cann-community](https://gitcode.com/cann/community)了解行为准则，进行CLA协议签署，了解源码仓的贡献流程，该仓详细介绍了如何参与CANN开源项目的贡献的前置条件，包括但不限于：

1. 如何提交PR
2. gitcode工作流
3. 流水线触发命令（[线上流水线问题指导](https://gitcode.com/cann/ge/wiki/%E7%BA%BF%E4%B8%8A%E6%B5%81%E6%B0%B4%E7%BA%BF%E9%97%AE%E9%A2%98%E6%8C%87%E5%AF%BC.md)）
4. 代码检视
5. 其他注意事项
   详情可以参考[cann-community](https://gitcode.com/cann/community)。

除此之外，开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。
2. 使用git进行代码提交前，可以参考[pre-commit工具使用说明](docs/zh/contributions/precommit_guide.md)来使您的代码提交更合规高效。
3. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。
4. 提交pr时，请确保您的代码符合项目的代码规范，具体参考google的[开源代码规范](https://google.github.io/styleguide/)，包括但不限于：
   - 代码格式化
   - 注释规范
   - 变量命名规范
   - 函数命名规范
   - 类命名规范
   - 接口命名规范
   - 配置参数命名规范
   - 代码流程规范
5. 提交pr时，如果存在多个无效commit，建议您在提交pr前先进行rebase操作，合并多个commit为一个，以保持代码的简洁性和可读性，具体参考[git rebase](https://git-scm.com/docs/git-rebase)，同时，commit message也需要符合项目的代码规范，能够清晰地描述本次变更的意图和内容，格式为：<类型>: <简短描述>。 例如:

| 类型     | 说明                       | 示例                         |
| -------- | -------------------------- | ---------------------------- |
| feat     | 新功能                     | feat: 添加用户注册功能       |
| fix      | 修复 bug                   | fix: 修复登录态过期问题      |
| docs     | 文档更新                   | docs: 更新 API 使用说明      |
| style    | 代码格式调整（不影响逻辑） | style: 调整代码缩进          |
| refactor | 重构（非功能新增/修复）    | refactor: 优化用户服务类结构 |
| perf     | 性能优化                   | perf: 减少数据库查询次数     |
| test     | 测试相关                   | test: 添加登录功能单元测试   |
| chore    | 构建/工具链变更            | chore: 更新 webpack 配置     |
| ci       | CI 配置相关                | ci: 添加自动化测试流程       |

开发者贡献场景主要包括：

- Bug修复

  如果您在本项目中发现了某些Bug，希望对其进行修复，欢迎您新建Issue进行反馈和跟踪处理。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Bug-Report|缺陷反馈` 类Issue对Bug进行描述，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行处理。

- 贡献新功能

  如果您在本项目中发现了某些功能缺失，希望对其进行新增，欢迎您新建Issue进行反馈和跟踪处理。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Requirement|需求建议` 类Issue对新增功能进行说明，并提供您的设计方案，
  然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行跟踪实现。

- 文档纠错

  如果您在本项目中发现某些文档描述错误，欢迎您新建Issue进行反馈和修复。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Documentation|文档反馈` 类Issue指出对应文档的问题，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您纠正对应文档描述。

- 帮助解决他人Issue

  如果社区中他人遇到的问题您有合适的解决方法，欢迎您在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

  如果对应Issue需要进行代码修改，您可以在Issue评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您，跟踪协助解决问题。

## 提交前自检清单

提交 PR 前，请逐项确认以下内容。本地通过自检可避免绝大部分 CI 失败。

### 代码格式

- 安装并运行 pre-commit：`pip3 install pre-commit && pre-commit run`
  - 详见 [pre-commit 使用指南](docs/zh/contributions/precommit_guide.md)
- 所有新增文件包含完整的 CANN OSL 版权头，年份和格式参考仓库根目录 [OAT.xml](OAT.xml) 及现有源文件。版权头必须包含完整的许可证说明，不能只保留 Copyright 行：
  - `.cpp`/`.h`：
    ```cpp
    /**
     * Copyright (c) <year> Huawei Technologies Co., Ltd.
     * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
     * CANN Open Software License Agreement Version 2.0 (the "License").
     * Please refer to the License for details. You may not use this file except in compliance with the License.
     * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
     * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
     * See LICENSE in the root of the software repository for the full text of the License.
     */
    ```
  - `.sh`：
    ```bash
    #!/bin/bash
    # -----------------------------------------------------------------------------------------------------------
    # Copyright (c) <year> Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.
    # -----------------------------------------------------------------------------------------------------------
    ```
  - `.py`：
    ```python
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    # -----------------------------------------------------------------------------------------------------------
    # Copyright (c) <year> Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.
    # -----------------------------------------------------------------------------------------------------------
    ```
- Shell 脚本应根据错误处理需求使用 `set -e`；如果脚本需要允许命令失败或无匹配结果继续执行，应显式处理返回值，并在脚本中说明例外原因。
- `cd` 命令检查返回值：`cd path || { echo "error"; exit 1; }`
- 函数体不超过 50 行（超大函数会被 CI 拦截）
- 循环处理多文件时，若全部文件都失败则正确返回 `FAILED`

### 提交规范

- 如果分支包含多个无效 commit，建议在提交 PR 前通过 rebase 进行 squash，以保持提交历史简洁：
  ```bash
  git rebase -i upstream/develop   # 多余的 pick → s (squash)
  ```
- PR 的目标仓库和目标分支：提交到 [cann/ge](https://gitcode.com/cann/ge) 仓库的 `develop` 分支。
- 基于最新 develop 分支：
  ```bash
  git fetch upstream develop && git rebase upstream/develop
  ```
- commit message 格式：`<type>: <描述>`（如 `feat: add xxx`, `fix: fix xxx`）

### 文档

- 样例需包含中英文 README（`README.md` + `README_en.md`），含功能描述、目录结构、环境准备、实现步骤、构建验证、预期输出
- 在父级 README 中注册新样例/模块条目（如 `examples/README.md`、`examples/acl/README.md` 等）

### CI 门禁说明

提交 PR 后，需要在 PR 评论区发送 `compile`（也支持 `/compile`）触发 CI 流水线；PR 更新后如需重新执行流水线，请再次发送该评论。流水线触发后会运行以下检查项及其本地验证方式：

| CI 检查项 | 作用 | 本地验证 |
|---|---|---|
| `codecheck_precommit` | 代码格式合规（pre-commit hooks） | `pre-commit run` |
| `codecheck_dt` | DT 测试用例规范检查 | 参考 [DT 用例开发指导](docs/zh/contributions/dev_test_guide/) |
| `clang-format` | C++ 代码格式化 | `clang-format -i <file>` |
| `ruff` | Python 代码格式化与 lint | `python3 -m ruff format <file> && python3 -m ruff check --fix <file>` |
| `codespell` | 拼写检查 | `codespell <file>` |
| `OAT` | 开源合规（许可证头、二进制文件检查） | `pre-commit run oat-check` |

---

## 贡献分级指引

GE 的贡献按复杂度分为三级，请根据您的贡献类型选择对应流程。

### 初级贡献（Bug 修复 / 文档纠错）

适用范围：修复 Bug、修正文档错误、小的代码格式调整等不改变代码流程的修改。

**流程**：

1. Fork 仓库并创建分支
2. 修改代码，本地运行 `pre-commit run` 确保格式合规
3. 提交 PR，按 [PR 模板](.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md) 填写变更描述
4. 确保 CI 门禁全部通过（见上方 [CI 门禁说明](#ci-门禁说明)）
5. 在 PR 评论区 `@committer_gitcode_id` 提请检视
6. Committer 检视通过后标注 `/lgtm`，Maintainer 标注 `/approve` 后合入

> **合入标签**：`/lgtm`（Looks Good To Me）由 Committer 添加，表示代码检视通过；`/approve` 由 Maintainer 添加，表示批准合入。完整机器人命令参见[社区评论命令指南](https://gitcode.com/cann/infrastructure/blob/main/docs/robot/robot%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)。

### 高级贡献（功能开发 / 模块修改）

适用范围：新增特性、新增接口、修改代码流程、性能优化等涉及核心模块的改动。

**流程**：

1. **先提交 Issue**：新建 `Requirement|需求建议` 类 Issue，说明背景、价值、设计方案
2. **等待 Committer/Maintainer 同意**：获得方案认可后再开始编码，避免代码被拒绝合入
3. **编码并提交 PR**：遵守下方约束速查表和测试要求
4. CI 门禁通过 + Committer 检视通过（`/lgtm`）+ Maintainer 批准（`/approve`）

#### 代码目录导航

| 目录 | 职责 |
|---|---|
| `compiler/graph/passes/` | 图优化 pass（融合、常量折叠、格式推导） |
| `compiler/graph/build/` | 编译构建（内存分配、流分配、task 生成） |
| `compiler/graph/partition/` | 图拆分（动静分离、引擎分区） |
| `runtime/v2/` | 动态执行器 RT2.0（Lowering、ExecuteGraph） |
| `runtime/v1/` | 静态执行器（DavinciModel、Task Sink） |
| `api/acl/` | ACL 对外 API 实现 |
| `base/` | 基础图结构、IR 定义 |
| `graph_metadef/` | 算子定义与注册 |
| `parser/` | 模型格式解析器（ONNX、TF、Caffe） |

#### 贡献场景约束速查

| 改动场景 | 必须遵守的约束 | 参考文档 |
|---|---|---|
| 改图优化 pass | 图等价变换、控制边处理、节点名约束 | `docs/zh/contributions/coding_red_lines.md` 规则 4、7、8 |
| 改内存分配 | 内存复用约束、跨 so 析构安全 | `docs/zh/design/constraints/memory-constraints.md` |
| 改 runtime 执行 | RT2 动态 shape 约束、hybrid 执行流程 | `docs/zh/design/constraints/rt2_runtime.md` |
| 改图拆分逻辑 | 切图不改变语义、子图通信隔离 | `docs/zh/design/constraints/graph_split.md` |
| 改流分配 | 多流并行 event 同步、流复用 | `docs/zh/design/constraints/stream_allocator.md` |
| 新增特性/feature | 必须做跨特性交叉影响分析 | `docs/zh/design/cross_feature_check.md` |
| 改对外接口 | API/ABI 兼容性、禁止改已有接口签名 | `docs/zh/contributions/coding_red_lines.md` 规则 3 |

#### 测试要求

| 改动类型 | 最低测试要求 |
|---|---|
| 新增 pass | UT + ST；UT 覆盖率 > 90%，ST 覆盖率 > 80% |
| 改 runtime | UT + ST + 回归测试 |
| 改 ACL API | UT + API 接口用例验证 |
| 新增样例 | 至少跑通一次完整推理流程 |
| Bug 修复 | 根据被测行为，补充能复现该 Bug 的 UT 用例、ST 用例，或同时补充两者 |

测试开发指南详见 [`docs/zh/contributions/dev_test_guide/`](docs/zh/contributions/dev_test_guide/)。

### 架构级贡献（跨模块 / 架构变更）

适用范围：涉及多个模块协同变更、运行时架构调整、编译流程重构等影响面较大的改动。

**流程**：

1. **在 sig-ge 例会上讨论**：先在 GE SIG 例会上申报议题，讨论方案可行性和影响范围
   - 会议看板：[CANN 社区会议](https://meeting.osinfra.cn/cann)（搜索 sig-ge 相关会议）
   - 会议指南：[社区会议指南](https://gitcode.com/cann/infrastructure/blob/main/docs/meeting/CANN%E7%A4%BE%E5%8C%BA%E4%BC%9A%E8%AE%AE%E6%8C%87%E5%8D%97.md)
2. **撰写设计文档**：按[设计文档模板](docs/zh/design/design_document_template.md)编写，必须覆盖：
   - 功能需求与非功能需求（性能、内存）
   - 对已有特性（21 个 features，见[架构总览](docs/zh/design/architecture.md)）的交叉影响分析
   - DT 测试方案
   - 兼容性检查（API/ABI、模型格式、芯片版本）
3. **跨特性交叉影响分析**：参考 [`docs/zh/design/cross_feature_check.md`](docs/zh/design/cross_feature_check.md) 逐场景评估
4. **提交 PR**：需附设计文档链接，CI + Committer + Maintainer 全部通过

#### 通用编码红线

所有代码修改必须遵守 [`docs/zh/contributions/coding_red_lines.md`](docs/zh/contributions/coding_red_lines.md)，关键规则摘要：

- 禁止硬编码敏感信息（密钥、密码）
- 外部数据输入必须校验
- 资源申请后必须释放
- 禁止使用 `std::unordered_map` 等无序容器；涉及图增删改和遍历时，优先使用 `std::map` 等有序容器；禁止使用 Node 指针作为 key 依赖地址顺序
- 图优化 pass 不得改变图的计算语义
- 禁止在代码中硬编码芯片型号
