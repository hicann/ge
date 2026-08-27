# Contributing Guide

This project welcomes developers to experience and participate in contributions. Before participating in community contributions, please see [cann-community](https://gitcode.com/cann/community) to understand the code of conduct, sign the CLA agreement, and understand the contribution process of source repositories. This repository details the prerequisites for participating in CANN open source project contributions, including but not limited to:

1. How to submit Pull Requests
2. gitcode workflow
3. Pipeline trigger commands ([Online Pipeline Issue Guide](https://gitcode.com/cann/ge/wiki/%E7%BA%BF%E4%B8%8A%E6%B5%81%E6%B0%B4%E7%BA%BF%E9%97%AE%E9%A2%98%E6%8C%87%E5%AF%BC.md))
4. Code review
5. Other Precautions
   For details, please refer to [cann-community](https://gitcode.com/cann/community).

In addition, developers need to pay attention to the following points when preparing local code and submitting Pull Requests:

1. When submitting a Pull Request, please carefully fill in the business background, purpose, solution, and other information according to the Pull Request template.
2. Before committing code using git, you can refer to [pre-commit tool usage instructions](docs/en/contributions/precommit_guide.md) to make your code submission more compliant and efficient.
3. If your modification is not a simple bug fix, but involves adding new features, new interfaces, new configuration parameters, or modifying code flows, please be sure to discuss the solution through an Issue first to avoid your code being rejected for merge. If you are unsure whether this modification can be classified as a "simple bug fix", you can also discuss the solution by submitting an Issue.
4. When submitting a Pull Request, please ensure your code complies with the project's code standards. Please refer to Google's [Open Source Code Standards](https://google.github.io/styleguide/), including but not limited to:
   - Code formatting
   - Comment standards
   - Variable naming standards
   - Function naming standards
   - Class naming standards
   - Interface naming standards
   - Configuration parameter naming standards
   - Code flow standards
5. When submitting a Pull Request, if there are multiple invalid commits, we recommend performing a rebase operation before submitting the Pull Request to merge multiple commits into one to maintain code simplicity and readability. Please refer to [git rebase](https://git-scm.com/docs/git-rebase). Also, commit messages need to comply with the project's code standards and clearly describe the intent and content of this change. The format is: <type>: <brief description>. For example:

| Type     | Description                       | Example                         |
| -------- | --------------------------------- | ------------------------------- |
| feat     | New feature                       | feat: Add user registration function |
| fix      | Bug fix                           | fix: Fix login state expiration issue |
| docs     | Documentation update              | docs: Update API usage instructions |
| style    | Code format adjustment (does not affect logic) | style: Adjust code indentation |
| refactor | Refactoring (non-feature addition/fix) | refactor: Optimize user service class structure |
| perf     | Performance optimization          | perf: Reduce database query count |
| test     | Test related                      | test: Add login function unit test |
| chore    | Build/toolchain change            | chore: Update webpack configuration |
| ci       | CI configuration related          | ci: Add automated test flow |

Developer contribution scenarios mainly include:

- Bug Fix

  If you discover certain bugs in this project and wish to fix them, you are welcome to create a new Issue for feedback and tracking.

  You can follow the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create a `Bug-Report|Bug Feedback` type Issue to describe the bug, then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for handling.

- Contribute New Features

  If you discover certain feature gaps in this project and wish to add them, you are welcome to create a new Issue for feedback and tracking.

  You can follow the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create a `Requirement|Feature Suggestion` type Issue to explain the new feature and provide your design solution,
  then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for tracking implementation.

- Documentation Correction

  If you discover certain documentation description errors in this project, you are welcome to create a new Issue for feedback and correction.

  You can follow the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create a `Documentation|Documentation Feedback` type Issue to point out the corresponding documentation problem, then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself to correct the corresponding documentation description.

- Help Resolve Others' Issues

  If you have appropriate solutions for problems encountered by others in the community, you are welcome to comment and discuss in the Issue to help others solve problems and pain points, and jointly optimize usability.

  If the corresponding Issue requires code modification, you can enter "/assign" or "/assign @yourself" in the Issue comment box to assign the Issue to yourself to track and assist in solving the problem.

## Pre-Submission Checklist

Before submitting a Pull Request, verify each of the following items. Passing these checks locally can avoid most CI failures.

### Code Formatting

- Install and run pre-commit: `pip3 install pre-commit && pre-commit run`
  - See the [pre-commit usage guide](docs/en/contributions/precommit_guide.md) for details.
- All new files must contain a complete CANN OSL license header. Refer to the root [OAT.xml](OAT.xml) and existing source files for the year and format. The header must include the complete license text, not only the Copyright line:
  - `.cpp`/`.h`:
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
  - `.sh`:
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
  - `.py`:
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
- Shell scripts should use `set -e` according to their error-handling requirements. If a script needs to continue after a command fails or returns no matches, it must handle the return value explicitly and document the exception in the script.
- Check the return value of `cd`: `cd path || { echo "error"; exit 1; }`
- Keep function bodies no longer than 50 lines (oversized functions may be rejected by CI).
- When processing multiple files in a loop, return `FAILED` correctly if all files fail.

### Submission Requirements

- If the branch contains multiple invalid commits, we recommend squashing them before submitting the PR to keep the commit history concise:
  ```bash
  git rebase -i upstream/develop   # change extra `pick` entries to `s` (squash)
  ```
- The PR target repository and branch must be the `develop` branch of the [cann/ge](https://gitcode.com/cann/ge) repository.
- Rebase onto the latest `develop` branch:
  ```bash
  git fetch upstream develop && git rebase upstream/develop
  ```
- Commit message format: `<type>: <description>` (for example, `feat: add xxx` or `fix: fix xxx`).

### Documentation

- Examples must include Chinese and English README files (`README.md` and `README_en.md`) covering the feature description, directory structure, environment setup, implementation steps, build verification, and expected output.
- Register new examples or modules in the parent README (for example, `examples/README.md` or `examples/acl/README.md`).

### CI Gate Requirements

After submitting a PR, post `compile` in the PR comment area (`/compile` is also supported) to trigger the CI pipeline. If the PR is updated and the pipeline needs to be run again, post the comment again. The triggered pipeline runs the following checks, with the corresponding local verification methods:

| CI check | Purpose | Local verification |
|---|---|---|
| `codecheck_precommit` | Code formatting compliance (pre-commit hooks) | `pre-commit run` |
| `codecheck_dt` | DT test case specification checks | See the [DT test development guide](docs/en/contributions/dev_test_guide/) |
| `clang-format` | C++ code formatting | `clang-format -i <file>` |
| `ruff` | Python formatting and linting | `python3 -m ruff format <file> && python3 -m ruff check --fix <file>` |
| `codespell` | Spelling check | `codespell <file>` |
| `OAT` | Open-source compliance (license headers and binary-file checks) | `pre-commit run oat-check` |

---

## Contribution Levels

GE contributions are divided into three levels by complexity. Select the process that matches your contribution type.

### Basic Contributions (Bug Fixes / Documentation Corrections)

Applicable to bug fixes, documentation corrections, and small code-format changes that do not change the code flow.

**Process**:

1. Fork the repository and create a branch.
2. Modify the code and run `pre-commit run` locally to ensure formatting compliance.
3. Submit a PR and complete the change description according to the [English PR template](.gitcode/PULL_REQUEST_TEMPLATE.en-US.md).
4. Ensure that all CI gates pass (see [CI Gate Requirements](#ci-gate-requirements)).
5. Mention `@committer_gitcode_id` in the PR comments to request a review.
6. After the Committer approves the review, the Committer adds `/lgtm`; the Maintainer adds `/approve` before merging.

> **Merge labels**: `/lgtm` (Looks Good To Me) is added by a Committer to indicate that the code review has passed; `/approve` is added by a Maintainer to approve the merge. For the complete bot command list, see the [community comment command guide](https://gitcode.com/cann/infrastructure/blob/main/docs/robot/robot%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md).

### Advanced Contributions (Feature Development / Module Changes)

Applicable to new features, new interfaces, changes to code flows, performance optimizations, and other changes involving core modules.

**Process**:

1. **Submit an Issue first**: Create a `Requirement|Feature Suggestion` Issue describing the background, value, and design solution.
2. **Obtain Committer/Maintainer agreement**: Obtain approval of the solution before coding to avoid rejection of the implementation.
3. **Implement and submit a PR**: Follow the constraints and test requirements below.
4. Pass the CI gates, obtain Committer review (`/lgtm`), and obtain Maintainer approval (`/approve`).

#### Code Directory Guide

| Directory | Responsibility |
|---|---|
| `compiler/graph/passes/` | Graph optimization passes (fusion, constant folding, format inference) |
| `compiler/graph/build/` | Compilation and build (memory allocation, stream allocation, task generation) |
| `compiler/graph/partition/` | Graph partitioning (dynamic/static separation, engine partitioning) |
| `runtime/v2/` | Dynamic-shape executor RT2.0 (Lowering, ExecuteGraph) |
| `runtime/v1/` | Static-shape executor (DavinciModel, Task Sink) |
| `api/acl/` | ACL public API implementation |
| `base/` | Basic graph structures and IR definitions |
| `graph_metadef/` | Operator definitions and registration |
| `parser/` | Model-format parsers (ONNX, TF, Caffe) |

#### Contribution Scenario Constraints

| Change scenario | Required constraints | Reference |
|---|---|---|
| Modify graph optimization passes | Graph-equivalent transformations, control-edge handling, and node-name constraints | Rules 4, 7, and 8 in `docs/en/contributions/coding_red_lines.md` |
| Modify memory allocation | Memory reuse constraints and safe destruction across shared objects | `docs/en/design/constraints/memory-constraints.md` |
| Modify runtime execution | RT2 dynamic-shape constraints and the hybrid execution flow | `docs/en/design/constraints/rt2_runtime.md` |
| Modify graph partitioning | Partitioning must preserve semantics and isolate subgraph communication | `docs/en/design/constraints/graph_split.md` |
| Modify stream allocation | Multi-stream event synchronization and stream reuse | `docs/en/design/constraints/stream_allocator.md` |
| Add a feature | Cross-feature impact analysis is required | `docs/en/design/cross_feature_check.md` |
| Modify a public interface | API/ABI compatibility; existing interface signatures must not be changed | Rule 3 in `docs/en/contributions/coding_red_lines.md` |

#### Test Requirements

| Change type | Minimum test requirements |
|---|---|
| New pass | UT + ST; UT coverage > 90% and ST coverage > 80% |
| Runtime change | UT + ST + regression tests |
| ACL API change | UT + API interface test verification |
| New example | Run through at least one complete inference flow |
| Bug fix | Add a UT case, an ST case, or both, according to the behavior under test, to reproduce the bug |

See the [`docs/en/contributions/dev_test_guide/`](docs/en/contributions/dev_test_guide/) for the test development guide.

### Architectural Contributions (Cross-Module / Architectural Changes)

Applicable to changes involving collaboration across multiple modules, runtime architecture adjustments, compiler-flow refactoring, and other changes with a broad impact.

**Process**:

1. **Discuss the proposal at a sig-ge meeting**: Submit an agenda item to the GE SIG meeting to discuss feasibility and impact.
   - Meeting board: [CANN Community Meetings](https://meeting.osinfra.cn/cann) (search for sig-ge meetings)
   - Meeting guide: [Community Meeting Guide](https://gitcode.com/cann/infrastructure/blob/main/docs/meeting/CANN%E7%A4%BE%E5%8C%BA%E4%BC%9A%E8%AE%AE%E6%8C%87%E5%8D%97.md)
2. **Write a design document**: Use the [design document template](docs/en/design/design_document_template.md) and cover:
   - Functional and non-functional requirements (performance and memory)
   - Cross-feature impact analysis for existing features (see the [architecture overview](docs/en/design/architecture.md))
   - DT test plan
   - Compatibility checks (API/ABI, model formats, and chip versions)
3. **Perform cross-feature impact analysis**: Evaluate each applicable scenario according to [`docs/en/design/cross_feature_check.md`](docs/en/design/cross_feature_check.md).
4. **Submit the PR**: Include a link to the design document; CI, Committer, and Maintainer approval are all required.

#### General Coding Red Lines

All code changes must comply with [`docs/en/contributions/coding_red_lines.md`](docs/en/contributions/coding_red_lines.md). Key rules include:

- Do not hard-code sensitive information (keys or passwords).
- Validate external data input.
- Release resources after they are acquired.
- Do not use unordered containers such as `std::unordered_map`; for graph insertion, deletion, modification, and traversal, prefer ordered containers such as `std::map`, and do not rely on Node-pointer addresses as key ordering.
- Graph optimization passes must not change graph computation semantics.
- Do not hard-code chip models in code.
