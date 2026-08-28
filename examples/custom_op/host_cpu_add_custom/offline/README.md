# HostCpu AddCustom 自定义算子离线 OM 样例

## 样例概述

本离线样例演示 `AddCustom` 自定义算子的离线编译和部署流程：使用 ES API 构图生成 `.air`，通过 ATC 转换为 `.om`，再用 ACL API 加载执行。

## 前置依赖

- 参考[安装指导](../../../../docs/zh/quick_install.md)完成 `toolkit` 和 `ops` 包安装。
- 设置环境变量（假设包安装在 `/usr/local/Ascend/`）：
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

## 快速运行

在 `examples/custom_op/host_cpu_add_custom/offline` 目录执行：

```bash
bash run.sh
```

脚本会完成以下步骤：

1. 构建 `output/op_graph/lib/<os>/<arch>/libcust_opapi.so`。
2. 运行 `single_add_graph_build` 生成 `output/single_add.air`。
3. 调用 `atc` 生成 `output/single_add_<os>_<arch>.om`。
4. 运行 `single_add_model_exec` 加载并执行 OM。

运行成功时，终端应打印：

```text
[HostCpuExecuteOp] Execute for AddCustom
[INFO] Model executed successfully!
output values: 6 8 10 12
[INFO] Output verification passed!
```

## 关键文件

```text
offline
├── CMakeLists.txt
├── run.sh
├── ge
│   ├── add_custom_ir.h       // AddCustom 原型定义
│   ├── add_custom_ir.cc      // 编译 AddCustom 原型
│   └── custom_op.cpp         // HostCpuExecuteOp / ShapeInferOp / PortableOp 实现
├── graph_build
│   └── main.cc               // ES 构图并导出 AIR
└── model_exec
    └── main.cc               // ACL 加载并执行 OM
```

## 实现步骤

`ge/custom_op.cpp` 中 `AddCustom` 的实现是本样例的核心：

- `HostCpuExecuteOp::Execute` 在 host 侧完成 float 向量加法。
- `ShapeInferOp` 将输出 shape 和 dtype 设为与输入一致。
- `PortableOp::Serialize/Deserialize` 提供离线 OM 所需的实例数据持久化实现。
- 通过 `REG_OP_BACKEND(AddCustom, "AddCustom", ge::OpBackend::kHostCPU)` 注册 kHostCPU backend。

## 注意事项

- `run.sh` 默认使用 `--soc_version=Ascend910B1`，如需适配其他环境请按实际硬件修改。
- 图输入 shape 固定为 `[4]` float32，输入数据为 `[1,2,3,4]` + `[5,6,7,8]`，期望输出 `[6,8,10,12]`。
- `ASCEND_CUSTOM_OPP_PATH` 会在 `run.sh` 中自动追加当前样例的 `output/`。
