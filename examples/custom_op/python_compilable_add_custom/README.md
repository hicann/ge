# Python CompilableAddCustom 在线与离线编译样例

本样例对应 GE Python `CompilableOp` Python 化接口，使用同一个 Python
`compile`/`declare_launch_args` 实现覆盖两条编译链路：

- 在线：GE `Session` 构图时由 `CustomGraphOptimizer` 调用 `compile`，随后直接执行图。
- 离线：C++ 构图程序生成 AIR，ATC 调用同一个 Python `compile` 回调生成 OM；随后只保留
  C++ OPP 交付件加载 OM，验证执行阶段不再导入 Python 插件。

## 样例链路

```text
Python plugin 加载
    -> register_op_impl 识别 compile / declare_launch_args
    -> CustomGraphOptimizer 调用 compile(x, y, z)
    -> get_compile_platform_info() 查询 NpuArch/SoC
    -> BiSheng + llvm-objcopy 生成并拥有 kernel bytes
    -> declare_launch_args 发布 kernel launch 描述
    ├── run_online.sh  -> GE Session 在线执行
    └── run_offline.sh -> AIR -> ATC -> OM -> 脱离 Python 插件的 ACL 执行
```

## 前置条件

- 已安装并配置与 GE 版本匹配的 CANN，先执行
  `source /path/to/cann/set_env.sh`。
- `cmake`、`atc`、`bisheng`、`llvm-objcopy`、Python 3；离线执行还需要 ACL 开发库。
- 在线脚本和离线 OM 执行都需要 NPU；AIR→OM 的 ATC 编译本身可以在 host 上完成。
- 当前 kernel 固定使用 float32，输入元素个数必须是 1024 的整数倍。

## 运行

### 在线

```bash
source /path/to/cann/set_env.sh
cd examples/custom_op/python_compilable_add_custom
bash run_online.sh
```

脚本构建 proto/ES wrapper，设置同时包含 OPP 包根和 Python plugin 目录的
`ASCEND_CUSTOM_OPP_PATH`，然后运行 Python `Session`。成功时会看到：

```text
PY_COMPILE_MODULE_LOADED=1
PY_COMPILE_CALLBACK_ENTER=1 mode=online ...
PY_COMPILE_ONLINE_NPU=PASS
```

### 离线

```bash
source /path/to/cann/set_env.sh
cd examples/custom_op/python_compilable_add_custom
bash run_offline.sh
```

脚本默认使用 `Ascend910B1` 生成 OM；其他芯片可先设置
`PYTHON_COMPILABLE_ADD_SOC_VERSION` 为目标 `soc_version`。

脚本依次执行：

1. 构建 C++ `REG_OP` 交付件、AIR 构图程序和 ACL OM 执行程序；
2. 设置 `ASCEND_CUSTOM_OPP_PATH=<OPP根目录>:<Python插件目录>`，生成 AIR；
3. 调用 ATC。ATC 日志必须包含 `PY_COMPILE_CALLBACK_ENTER=1 mode=offline`；
4. 清除 Python 插件路径，仅保留 OPP 根目录加载 OM，并检查 `x + y = 3`。

最后一步由 C++ 程序打印：

```text
PY_COMPILE_OFFLINE_OM=PASS
```

这一步证明 Python compile 只参与模型编译，执行 OM 时不依赖 Python callback。

## 目录结构

```text
python_compilable_add_custom
├── CMakeLists.txt                              # proto、ES wrapper、离线工具
├── run_online.sh                               # GE Session 在线编译与执行
├── run_offline.sh                              # AIR -> ATC -> OM -> ACL
├── kernel/add_custom.asc                       # compile callback 使用的 Ascend C 源码
├── python/es_custom/__init__.py              # 生成 ES wrapper 的包入口模板
├── proto/add_custom.h                          # PythonCompilableAddCustom 原型
├── proto/add_custom.cc                         # shape/data type 推导
├── src/ge/python_compilable_add_custom.py     # compile + declare_launch_args
├── src/run.py                                  # 在线构图和执行
├── src/offline_graph_build.cc                  # 离线 AIR 构图
└── src/offline_model_exec.cc                   # 脱离 Python 的 OM 执行
```

## 关键实现说明

- `compile` 使用 `get_compile_platform_info()` 查询
  `get_platform_resource("version", "NpuArch")` 和 `get_soc_version()`；编译失败直接
  传播为图编译失败。
- 生成的 `.aicore.o` 只在 Python holder 实例内按 shape/dtype key 缓存；
  当编译目标的 SoC 或 NPU 架构变化时会清空该缓存，避免复用错误平台的二进制。
  本地文件缓存同时包含 SoC、源码内容和 Ascend C 头文件路径，源码或目标平台更新后会重新编译。
  `declare_launch_args` 发生 cache miss 时显式报错，不会偷偷重复编译。
- `ASCEND_CUSTOM_OPP_PATH` 必须同时包含 OPP 包根目录和 Python plugin 目录：前者供
  C++ `REG_OP` 原型发现，后者供 Python custom-op loader 发现。
- 在线和离线都复用 GE 已有 `AnnotatedArgs` 下发路径；不新增执行期 Python callback，
  也不把 Python 状态写入 OM 或模型缓存。

## 验证

真实链路分别使用 `bash run_online.sh` 和 `bash run_offline.sh` 验证。
