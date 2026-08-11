---
name: acl-mdl-api-creator
description: Use when adding new public APIs to inc/external/acl/ headers (acl_mdl.h, acl_base_mdl.h, acl_op.h, acl_op_compiler.h), or adding new types (enums/macros/structs) to those headers. Covers three-layer architecture for acl_mdl.h, naming conventions, Doxygen comments, stub generation, UT patterns, and binary compatibility rules.
---

# ACL Public API Creator

## Overview

GE 在 `inc/external/acl/` 下对外暴露 4 个头文件：

| 头文件 | 内容 | 新增频率 |
|--------|------|----------|
| `acl_mdl.h` | 模型加载/执行/查询/Dataset/AIPP/Config 等 | 高（主要新增入口） |
| `acl_base_mdl.h` | TensorDesc 等基础类型接口 | 低 |
| `acl_op.h` | 算子相关接口 | 低 |
| `acl_op_compiler.h` | 算子编译相关接口 | 低 |

- **acl_mdl.h 新增接口**：需要完整走三层架构（头文件 → 路由层 → Impl 层）+ 测试 + 兼容性看护
- **其他头文件新增接口**：主要关注兼容性规范（命名、编码规范、兼容性看护），实现链路参考各自模块现有代码

## When to Use

- 新增 ACL 对外接口、对外 API、acl_mdl.h 接口、acl_op.h 接口
- 新增对外枚举、宏、结构体
- 为已有接口添加 `V2`/`V3` 版本
- 修改 `inc/external/acl/` 下头文件（仅允许兼容性追加，不允许删除或改变已有定义）

## Architecture

```
acl_mdl.h (对外声明，C 风格)
    ↓
acl_model.cpp (路由层 → libacl_mdl.so，仅 1 个源文件)
    ├── model_om2.cpp (OM2 Impl → libacl_mdl_impl_om2.so)
    └── model.cpp (Legacy/OM1 Impl → libacl_mdl_impl.so)
              └── 反向依赖 libacl_mdl_impl_om2.so（复用共用代码）
```

**编写代码前，先读取以下文件学习现有模式**：
- 路由层：`api/acl/acl_model/model/acl_model.cpp`
- OM2 Impl 声明：`api/acl/acl_model/model/acl_model_impl_om2.h`
- OM2 Impl 实现：`api/acl/acl_model/model/model_om2.cpp`
- Legacy Impl 声明：`api/acl/acl_model/model/acl_model_impl.h`
- Legacy Impl 实现：`api/acl/acl_model/model/model.cpp`
- UT 测试：`tests/test_c/ut/testcase/acl/acl_mdl_test.cc`
- 兼容性看护：`tests/acl_ut/ut/acl/testcase/compatibility/`
- 内部结构体：`api/acl/acl_model/model/model_desc_internal.h`
- 共用函数：`api/acl/acl_model/model/model_common.h`

### 头文件结构 (acl_mdl.h, 1725 行)

| 区域 | 行号 | 内容 |
|------|------|------|
| Include Guard | 11-12 | `INC_EXTERNAL_ACL_ACL_MODEL_H_` |
| Include 依赖 | 14-18 | `<stddef.h>`, `<stdint.h>`, `acl_base.h`, `acl_rt.h` |
| extern "C" | 20-22 | C 兼容声明 |
| 宏定义 | 24-49 | `ACL_MAX_DIM_CNT`(128), 加载类型(1-6), 流标志等 |
| 前向声明 | 51-58 | `aclmdlDataset`, `aclmdlDesc`, `aclmdlAIPP`, `aclmdlConfigHandle` 等 |
| 枚举定义 | 60-137 | `aclmdlConfigAttr`, `aclmdlAttr`, `aclmdlExecConfigAttr` 等 |
| 结构体定义 | 139-233 | `aclmdlIODims`, `aclAippDims`, `aclmdlBatch`, `aclmdlHW`, `aclAippInfo` 等 |
| 函数声明 | 235-1719 | 所有对外 API（带 Doxygen 注释） |

**新增代码插入位置**：
- 新增前向声明 → 追加到前向声明区域末尾
- 新增枚举值 → 追加到对应枚举定义末尾（不要插入中间，避免改变已有取值）
- 新增枚举类型 → 追加到枚举定义区域末尾
- 新增结构体 → 追加到结构体定义区域末尾
- 新增函数声明 → 追加到函数声明区域末尾，按功能分组

## 开发流程

新增 ACL 对外接口分三个阶段：**设计分析 → 代码实现 → 验证**。

### 阶段一：设计分析（编码前必须完成）

**1. 接口必要性**
- 是否已有类似接口可以满足需求？（优先复用，避免接口膨胀）
- 如果是修改已有接口的行为，必须新增 V2/V3 版本，不能改原接口

**2. 接口签名设计**

| 决策项 | 确认内容 |
|--------|----------|
| 函数名 | 符合 `aclmdl` + 操作动词 + 对象 规则？无形容词？ |
| 返回类型 | 操作型→`aclError`，创建型→指针，查询型→`size_t`？ |
| 异步版本 | 是否需要同步接口 + `Async` 后缀的异步版本？ |
| 参数列表 | 每个参数的 `[IN]`/`[OUT]`/`[IN][OUT]` 方向明确？ |
| `const` 修饰 | 指针入参是否加 `const`？未来是否需要修改该参数（如动态 shape 场景）？ |
| 返回码 | 是否需要新增 `ACL_ERROR_*` 返回码？与已有返回码是否冲突？ |
| 新增类型 | 是否需要新增前向声明/枚举/结构体？放在头文件的哪个区域？ |
| 结构体扩展 | 新增结构体是否需要预留指针参数扩展字段？ |

**3. 路由模式选择** — 按下方"路由模式决策"确定

**4. 兼容性影响评估**

| 检查项 | 确认 |
|--------|------|
| 是否修改了已有对外接口？ | 不允许，只能新增 |
| 是否修改了已有枚举值？ | 不允许，只能追加 |
| 是否修改了已有结构体字段？ | 不允许，只能追加且不影响已有偏移 |
| 新增枚举取值是否与已有冲突？ | 不允许冲突 |
| 新增返回码是否与已有冲突？ | 不允许冲突 |

**5. 实现方案确认**
- Impl 层需要调用哪些内部 API？
- 是否可以复用 `model_common.h` 中的共用函数？
- 是否需要修改 `model_desc_internal.h` 中的内部结构体？

### 阶段二：代码实现

按 Checklist 逐文件修改，编写时先读取对应现有文件学习写法。

### 阶段三：验证

| # | 验证项 | 方式 |
|---|--------|------|
| 1 | 编译通过 | `bash build.sh --ge_executor` |
| 2 | UT 通过 | 使用 `ge-dt-runner` skill 编译运行 acl_utest |
| 3 | 兼容性看护通过 | 确认 compatibility 测试全部通过 |
| 4 | Stub 自动生成 | 构建日志中确认 stub 生成无报错 |

---

## Checklist

| # | 文件 | 修改内容 |
|---|------|----------|
| 1 | `inc/external/acl/acl_mdl.h` | 函数声明 + Doxygen 注释 |
| 2 | `api/acl/acl_model/model/acl_model_impl_om2.h` | `ImplOm2` 声明 |
| 3 | `api/acl/acl_model/model/model_om2.cpp` | OM2 实现 |
| 4 | `api/acl/acl_model/model/acl_model.cpp` | 路由函数（选择路由模式） |
| 5 | `api/acl/acl_model/model/acl_model_impl.h` | （如需 Legacy）`Impl` 声明 |
| 6 | `api/acl/acl_model/model/model.cpp` | （如需 Legacy）实现 |
| 7 | `tests/test_c/ut/testcase/acl/acl_mdl_test.cc` | UT 测试用例 |
| 8 | `tests/acl_ut/ut/acl/testcase/compatibility/enum_check.cpp` | （如有新增枚举）追加枚举值看护 |
| 9 | `tests/acl_ut/ut/acl/testcase/compatibility/const_check.cpp` | （如有新增宏/常量）追加宏值看护 |
| 10 | `tests/acl_ut/ut/acl/testcase/compatibility/struct_check.cpp` | （如有新增/修改结构体）追加偏移量+大小看护 |

**自动处理（无需手动修改）：**
- Stub 生成：两个脚本分工生成 stub，均位于 `api/acl/stub/`：
  - `gen_stubapi.py`：从对外头文件（`acl_mdl.h` 等）解析**对外接口**（无后缀），生成同名 stub，返回 `ACL_ERROR_COMPILING_STUB_MODE`
  - `gen_stubapi_acl_mdl_impl.py`：从 `acl_model_impl.h`/`acl_model_impl_om2.h` 解析，为 `*Impl`（Legacy）和 `*ImplOm2`（OM2）函数生成 stub，函数名自动追加 `Impl` 后缀，返回 `ACL_ERROR_API_NOT_SUPPORT`
- 符号导出：使用 `ACL_FUNC_VISIBILITY` 宏即自动导出。该宏定义在外部 CANN runtime SDK 头文件 `acl/acl_base_rt.h` 中，不在 GE 仓库内，构建时通过 `${TOP_DIR}/runtime/include/external` 引入

## 路由模式决策

```
接口参数中是否包含 modelId / modelPath / modelData / bundleId / ConfigHandle / Desc？
  ├── 否 → 模式 A：纯 OM2 直连（直接调用 ImplOm2）
  └── 是 → 是否需要 Legacy(OM1) 支持？
              ├── 否 → 模式 A：纯 OM2 直连
              └── 是 → 按参数类型选择路由判断函数：
                        modelId → ById | modelPath → ByPath | modelData → ByData
                        Desc → ByDesc | ConfigHandle → ByConfig | bundleId → BundleById
```

### 模式 A：纯 OM2 直连

路由层直接调用 `ImplOm2`，不做 OM1/OM2 判断。适用于 Desc/Dataset/AIPP/ConfigHandle 的创建销毁和属性查询，或 OM1 从未支持的全新功能。参考 `acl_model.cpp` 中 `aclmdlGetNumInputs` 等接口的写法。

### 模式 B：OM1/OM2 分发

通过路由判断函数区分 OM 格式，分发到 `ImplOm2` 或 `Impl`。适用于模型加载、执行、卸载、动态 shape 设置、AIPP 绑定等。参考 `acl_model.cpp` 中 `aclmdlExecute`、`aclmdlLoadFromFile` 等接口的写法。

**6 个路由判断函数**（定义在 `acl_model_router.h`）：

| 函数 | 判断依据 | 实现原理 |
|------|----------|----------|
| `AclIsOm2ModelById(modelId, &isOm2)` | `uint32_t modelId` | 查 `AclResourceManagerOm2` |
| `AclIsOm2ModelByPath(modelPath, &isOm2)` | `const char* path` | 读文件头魔数 |
| `AclIsOm2ModelByData(model, size, &isOm2)` | `const void*, size_t` | 读内存头魔数 |
| `AclIsOm2ModelByDesc(desc, &isOm2)` | `aclmdlDesc*` | 通过 `desc->modelId` 查 |
| `AclIsOm2ModelByConfig(handle, &isOm2)` | `aclmdlConfigHandle*` | 从 `handle->loadPath` 或 `handle->mdlAddr` 判断 |
| `AclIsOm2BundleById(bundleId, &isOm2)` | `uint32_t bundleId` | 查 `AclResourceManagerOm2` |

### 模式 C/D/E：特殊路由

- **C（OM2 优先回退 OM1）**：先调 `ImplOm2`，特定错误码时回退 `Impl`。参考 `aclmdlCreateAndGetOpDesc`
- **D（先 OM1 后 OM2）**：两者都需执行。参考 `aclRecoverAllHcclTasks`
- **E（只用 Legacy）**：直接调 `Impl`。参考 `aclTransTensorDescFormat`

### 何时需要 Legacy Impl

| 场景 | 是否需要 Legacy |
|------|----------------|
| Desc/Dataset/AIPP/ConfigHandle 的创建销毁和属性查询 | 不需要（纯 OM2） |
| 模型加载/执行/卸载/动态 shape/AIPP 绑定 | 需要（OM1+OM2 分发） |
| 全新功能（OM1 从未支持） | 不需要（纯 OM2） |

## ACL 编码规范

### 命名

| # | 规则 | 示例 |
|---|------|------|
| 1 | 驼峰风格。模块名全小写放最前 | `aclmdlLoadFromFile` |
| 2 | 对外：`acl` + 模块类别 + 操作动词 + 对象。内部：大驼峰，无需 `acl` 前缀 | 对外：`aclmdlCreateDesc`；内部：`CreateModelDesc` |
| 3 | 模块名与操作对象重叠时，对象省略 | — |
| 4 | 接口名原则上不允许有形容词 | — |
| 5 | 不暴露实现细节，使用前向声明 | `typedef struct aclmdlDesc aclmdlDesc;` |
| 6 | 对外头文件为 C 风格，不使用 C++ 标识符 | — |
| 7 | 用宏定义常量，避免 `const int` | `#define ACL_MAX_DIM_CNT 8` |
| 8 | 避免 `bool` 类型，用 `uint8_t` 替代 | `uint8_t enable;` |
| 9 | 枚举值全大写 + `ACL_` 前缀 | `ACL_MDL_LOAD_FROM_FILE` |
| 10 | 整型优先 `<cstdint>`；长度用 `size_t` | `uint32_t modelId` |
| 11 | 异步接口名末尾加 `Async`，`stream` 参数放最后 | `aclmdlExecuteAsync` |
| 12 | 类成员变量 `_` 后缀，结构体成员小驼峰 | 类：`modelId_`；结构体：`modelId` |

### Impl 函数命名

| 层级 | 命名规则 | 示例 |
|------|----------|------|
| 对外接口 | `aclmdl` + PascalCase | `aclmdlLoadFromFile` |
| OM2 Impl | 对外名 + `ImplOm2` | `aclmdlLoadFromFileImplOm2` |
| Legacy Impl | 对外名 + `Impl` | `aclmdlLoadFromFileImpl` |

Impl 函数签名与对外接口参数完全一致，仅函数名不同。

### 格式

- 对外接口必须使用 `ACL_FUNC_VISIBILITY` 标记（该宏定义在外部 CANN runtime SDK 头文件 `acl/acl_base_rt.h` 中，不在 GE 仓库内）
- 对外头文件必须使用 `extern "C"` 包裹，确保 C++ 代码能正确调用 C 语言接口，指示编译器按 C 语言方式编译和链接：
  ```c
  #ifdef __cplusplus
  extern "C" {
  #endif
  // 所有对外函数声明和类型定义
  #ifdef __cplusplus
  }
  #endif
  ```

### 注释

- 函数头注释按 Doxygen 格式，必须包含 `@ingroup AscendCL`、`@brief`、`@param`（标注方向）、`@retval`：
  ```c
  /**
   * @ingroup AscendCL
   * @brief 简短描述接口功能
   *
   * @param modelId [IN]   model id
   * @param result [OUT]   query result
   *
   * @retval ACL_SUCCESS The function is successfully executed.
   * @retval OtherValues Failure
   */
  ```
- 指针入参且函数体不修改时，加 `const` 修饰，注释标 `[IN]`
- 既作入参又作出参时，注释标 `[IN][OUT]`
- 新建接口先读取 `acl_mdl.h` 中相邻接口的注释学习格式

### 其他注意事项

- 枚举无效值需足够大，或不定义无效值
- 高频接口中不增加非必要耗时逻辑
- 接口参数设为 `const` 需从扩展性思考（`aclopExecute` 的 `inputDesc`/`outputDesc` 定义为 `const` 导致动态 shape 场景无法使用，不得不新增 `aclopExecuteV2`）

## 兼容性规范（强制）

### 对外接口兼容性

| 规则 | 说明 |
|------|------|
| 不允许删除 | 已发布的对外接口永远不能删除 |
| 不允许改名 | 包括大小写 |
| 参数不可修改 | 对象类型不可改变，已有含义不可改变。例外：变更后相同输入必须有相同输出 |
| 返回码不可变 | 原有含义及取值不可改变，新增特性可增加返回码 |
| 存在即合理 | 即使旧接口定义不合理，只要已在现网运行，不允许擅自修改 |

### 实现和逻辑兼容性

- 新增/增强功能不允许丢失原有特性
- 新增功能默认情况下必须和基础版本完全兼容
- 修正老代码错误不能造成特性改变

### 数据兼容性

- 已定义的对外数据对象不能删除，名称不能修改
- 枚举类型取值不能减少，非枚举类型取值含义不能修改
- 新增枚举取值不能与已有取值冲突
- 数据对象需考虑可扩展性，扩展属性不能影响兼容性

### 结构体扩展性

- 对外结构体原则上需要稳定，不存在后续参数扩展
- 如果不确定结构体后续是否会扩展，**预留一个指针参数**方便后续扩展

### 兼容性看护（必须执行）

新增或修改对外头文件中的枚举、宏、结构体时，**必须**在 `tests/acl_ut/ut/acl/testcase/compatibility/` 对应文件中追加断言：

| 新增类型 | 看护文件 | 断言方式 |
|----------|----------|----------|
| 枚举类型/枚举值 | `enum_check.cpp` | `(枚举类型)固定整数值` vs `枚举常量`，测试类 `UTEST_ACL_compatibility_enum_check` |
| 宏/常量 | `const_check.cpp` | `宏名` vs `固定字面量`，字符串宏用 `std::string` 包装，测试类 `UTEST_ACL_compatibility_const_check` |
| 结构体/结构体字段 | `struct_check.cpp` | `OFFSET_OF_MEMBER` 算偏移 + `sizeof` 算总大小，测试类 `UTEST_ACL_compatibility_struct_check` |

编写时先读取对应看护文件学习现有断言写法。

## API 参考

### 参数校验宏

定义在 `api/acl/common/log_inner.h`：

| 宏 | 用途 |
|----|------|
| `ACL_REQUIRES_NOT_NULL(ptr)` | 非空校验 |
| `ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(ptr)` | 非空校验 + 报错上报（推荐） |
| `ACL_REQUIRES_NOT_NULL_RET_NULL(ptr)` | 非空校验，失败返回 `nullptr` |
| `ACL_REQUIRES_NOT_NULL_RET_VOID(ptr)` | 非空校验，失败返回 `void` |
| `ACL_REQUIRES_OK(expr)` | 表达式成功校验（返回 aclError） |
| `ACL_REQUIRES_OK_WITH_INNER_MESSAGE(expr, ...)` | 同上，附加内部错误日志 |
| `ACL_REQUIRES_TRUE(expr, errCode, errDesc)` | 条件校验，失败返回指定错误码 |
| `ACL_REQUIRES_CALL_GE_OK(expr, ...)` | GE `Status` 成功校验 |
| `ACL_REQUIRES_CALL_RTS_OK(expr, fn)` | Runtime 调用成功校验 |
| `ACL_REQUIRES_NON_NEGATIVE(val)` | 非负校验 |
| `ACL_REQUIRES_POSITIVE(val)` | 正值校验 |
| `ACL_REQUIRES_EQ(a, b)` | 相等校验 |
| `ACL_REQUIRES_LE(a, b)` | 小于等于校验 |
| `ACL_CHECK_MALLOC_RESULT(val)` | malloc 结果校验，失败返回 `ACL_ERROR_BAD_ALLOC` |
| `ACL_CHECK_RANGE_INT(val, min, max)` | int 范围校验 |

### 错误码

| 宏/值 | 用途 |
|-------|------|
| `ACL_GET_ERRCODE_GE(ret)` | GE 错误码转换 |
| `ACL_GET_ERRCODE_RTS(ret)` | Runtime 错误码转换 |
| `ACL_ERROR_INVALID_PARAM` | 参数无效 |
| `ACL_ERROR_FAILURE` | 通用失败 |
| `ACL_ERROR_STORAGE_OVER_LIMIT` | 存储超限 |

### 日志宏

| 宏 | 级别 |
|----|------|
| `ACL_LOG_ERROR("[Tag][SubTag] ...")` | 错误 |
| `ACL_LOG_INNER_ERROR("[Check][Param] ...")` | 内部错误 |
| `ACL_LOG_CALL_ERROR("[Model][FromData] ...")` | 调用失败 |
| `ACL_LOG_WARN(...)` | 警告 |
| `ACL_LOG_INFO(...)` | 信息 |
| `ACL_LOG_DEBUG(...)` | Debug |

日志标签格式: `[功能标签][子标签]`

### OM1 vs OM2 关键差异

| 维度 | Legacy (OM1) | OM2 |
|------|-------------|-----|
| 执行器 | `ge::GeExecutor` + `gert::ModelV2Executor` | `gert::Om2ModelExecutor` |
| 资源管理 | `acl::AclResourceManager` | `acl::AclResourceManagerOm2` |
| 模型加载 | `executor.LoadDataFromFile` + `LoadModelFromDataWithArgs` | `gert::LoadOm2DataFromFile` + `LoadOm2ExecutorFromData` |
| 模型数据 | `ge::ModelData` + `ge::ModelLoadArg` | `ge::ModelData` + `gert::Om2ModelLoadArg` |

Legacy 存在两条加载路径：传统路径（`ge::GeExecutor`）和 RT2.0 路径（`gert::ModelV2Executor`），根据模型是否支持 RT2.0 选择。

## Common Mistakes

| 错误 | 正确做法 |
|------|----------|
| 路由层直接写业务逻辑 | 路由层只做 OM1/OM2 分发，逻辑放 Impl 层 |
| 忘记 `ACL_FUNC_VISIBILITY` 宏 | 对外接口和 Impl 声明都必须加 |
| Impl 函数名不加后缀 | OM2 加 `ImplOm2`，Legacy 加 `Impl` |
| Impl 签名与对外接口不一致 | Impl 参数必须与对外接口完全一致 |
| 手动修改 stub 文件 | stub 由 `gen_stubapi.py` / `gen_stubapi_acl_mdl_impl.py` 自动生成，不要手动改 |
| 头文件中使用 `bool`/`const int` | 用 `uint8_t` 替代 `bool`，用 `#define` 替代 `const int` |
| 对外结构体不考虑扩展 | 预留指针参数方便后续扩展 |
| 参数加 `const` 不考虑未来扩展 | 评估动态 shape 等场景是否需要修改参数 |
| 修改/删除已有对外接口 | 存在即合理，只能新增，不能删改 |
| 新增枚举/宏/结构体不加兼容性看护 | 必须在 `compatibility/` 对应文件中追加断言 |
| 纯数据结构操作加 OM1/OM2 路由判断 | Desc/Dataset/AIPP 操作用模式 A 直连 OM2 |
| 新功能加 Legacy Impl | OM1 从未支持的新功能只需 OM2 Impl |
