# ACL GEMM 接口迁移样例

## 功能描述

演示如何使用 **ACLNN 接口替代已废弃的 ACLBLAS 和 ACLOP 接口**。通过 GEMM（通用矩阵乘法）算子，对比三种 ACL 接口的等效实现，帮助理解接口迁移流程。

| 接口 | 状态 | 调用方式 |
|------|------|----------|
| `aclblasGemmEx` | 已废弃 | 通过 ACLBLAS 接口直接执行 GEMM |
| `aclopExecuteV2("GEMM")` | 已废弃 | 通过 ACLOP 接口调用 GEMM 算子 |
| `aclnnMatmul` + `aclnnMuls` + `aclnnAdd` | 推荐 | ACLNN 原子操作组合 |

样例使用相同输入数据依次执行三种接口，自动比较精度并输出 `max_error`。

### GEMM 公式

```
C = α·A·B + β·C
```

| 参数 | 值 | 说明 |
|------|----|------|
| M | 64 | 矩阵 A 行数 / 矩阵 C 行数 |
| N | 64 | 矩阵 B 列数 / 矩阵 C 列数 |
| K | 64 | 矩阵 A 列数 / 矩阵 B 行数 |
| alpha | 2.0 | 矩阵乘法结果缩放系数 |
| beta | 2.0 | 原始矩阵 C 缩放系数 |
| 数据类型 | FP16 | 所有矩阵和标量使用 FP16 |

**ACLNN 接口组合说明**：ACLNN 更接近底层算子原子操作，需将 GEMM 拆解为：
1. `aclnnMatmul` → `A @ B`
2. `aclnnMuls` → `alpha × (A @ B)` 和 `beta × C`
3. `aclnnAdd` → 两部分相加

## 快速开始

```bash
bash scripts/build.sh && bash scripts/run.sh
```

## 目录结构

```
4_sample_acl_gemm/
├── CMakeLists.txt              # 顶层编译脚本
├── scripts/
│   ├── build.sh                # 编译脚本
│   └── run.sh                  # 运行脚本
└── src/
    ├── CMakeLists.txt          # 编译配置
    └── sample_acl_gemm.cpp     # 主程序（三种接口对比实现）
```

## 环境准备

- [ ] 按 [环境准备](../../../docs/zh/quick_install.md) 安装 `toolkit` 和 `ops` 包
- [ ] 设置环境变量：`source /usr/local/Ascend/cann/set_env.sh`（路径按实际安装位置调整）
- [ ] （可选）指定芯片版本：`export SOC_VERSION=Ascend910B3`（未设置时默认 `Ascend910B3`）

## 详细步骤

### Step 1：编译

```bash
bash scripts/build.sh
```

### Step 2：运行

```bash
bash scripts/run.sh
```

## 预期输出

```
[INFO] ACL GEMM sample starts
[INFO] Running ACLBLAS GEMM...
[INFO] Running ACLOP GEMM...
[INFO] Running ACLNN GEMM...
[INFO] Comparing results...
max_error: 0
[INFO] VERIFICATION PASSED
[INFO] SAMPLE PASSED
```

## 常见问题

**Q: 编译时报找不到 ACLNN 头文件**

确认已执行 `source /usr/local/Ascend/cann/set_env.sh`，并确保安装的 toolkit 版本包含 ACLNN 接口。

**Q: max_error 不为 0**

FP16 精度下可能存在微小误差。若 max_error 较小（如 < 1e-3），属于正常浮点误差；若较大，请检查 `SOC_VERSION` 是否与当前芯片匹配。

**Q: 如何迁移自己的 ACLOP 代码到 ACLNN？**

参考本样例 `sample_acl_gemm.cpp` 中 ACLNN 部分的实现模式：先调用 `aclnnXxxGetWorkspaceSize` 获取 workspace，再调用 `aclnnXxx` 执行计算。
