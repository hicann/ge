# Autofuse 适配 ExtendConv2D：LLT 与仓间依赖

本文记录 ExtendConv2D 联调后的 LLT（UT + ST）怎么跑、覆盖率怎么看，以及 **GE 仓测试是否依赖 graph-autofusion 仓的编译和安装**。

特性流程见同目录 [autofuse_extend_conv2d_cv_fusion.md](./autofuse_extend_conv2d_cv_fusion.md)。

---

## 1. 结论先说：GE 的 UT 要不要编、装 autofuse 仓？

**日常写/跑 GE 仓 UT，不需要编译 graph-autofusion 仓，也不需要把该仓的产物再装进 CANN。**

2026-08-25 在干净重装的 cann-9.2.0 上复核过：官方 toolkit **自带** `graph_autofusion` 组件（`share/info/graph_autofusion/version.info`，`lib64/libaihac_codegen.so` 等）。这不是过去手工叠加的仓，而是当前 CANN 包的一部分。官方包里**还没有** `ExtendConv2D*` ASCIR 注册。

| 你在做什么 | 要不要编 autofuse 仓 | 要不要把 autofuse 仓装进 CANN | 实际用到什么 |
|------------|----------------------|-------------------------------|--------------|
| GE infer-shape / tiling UT | 否 | 否 | GE 源码 + 测试 stub |
| GE autofuse frontend UT（`autofusion_ut`） | 否 | 否 | GE 源码编 `libautofuse.so`；codegen 链接 **官方 toolkit 自带的** `libaihac_codegen.so` |
| GE lowering Realize 出新 ASCIR 类型（`ExtendConv2D*`） | 否 | **官方包还没有这些类型时才需要** | toolkit 里已注册的 ASCIR |
| Autofuse 仓自己的 UT/ST | **要编 autofuse 仓** | 否（测的是本仓 `build/` 产物） | graph-autofusion 源码 |
| GE ST / 真 kernel / 二次 tiling | 要 | **要** | 把本仓新编的 autofuse 覆盖进 CANN |

干净包上复跑 `ExtendConv2DLoweringUT.*` + `StoreIgnoredOutputIsNotExtern`：**5/5 通过**。日志仍有 `unknown op type 'ExtendConv2D'`，说明 Realize 用的是官方包里的旧 ASCIR，不影响本次 UT 断言。

---

## 2. 两仓各自测什么

```text
GE 仓（frontend）
  InferSymbolShape     pad_mode 在 ExtendConv2D 属性下标 7（round_mode 占 6）
  Lowering             按 bias/scale0 选 4 种 ASCIR 变体；未使用 y1 走 StoreIgnoredOutput
  Lifting / tiling     conv_subgraph + ExtendConv2D tiling 属性

graph-autofusion 仓（backend）
  ParseConv2DAttr      is_extend_conv2d / has_bias / has_scale0 / round_mode
  build_conv_args      10 个逻辑输入槽、双输出占位
  codegen / e2e ST     ExtendConv2D、ExtendConv2DBiasScale 出 tiling key
```

GE 仓 frontend 源码在 `compiler/graph/optimize/autofuse/`，ASCIR 头文件是 vendored 的 `temporary_dependencies/ascir/ascir_ops.h`。
kernel 模板、tiling key、Python codegen 在 graph-autofusion 仓。

---

## 3. 环境

本机参考：

- toolkit：`/usr/local/Ascend/ascend-toolkit/latest` → cann-9.2.0
- 核数少时编译用 `-j 3`（默认过大容易 OOM）
- 跑 GE 测试前：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
unset LD_LIBRARY_PATH
unset ASCEND_OPP_PATH
```

`unset` 是为了避免旧 `LD_LIBRARY_PATH` / `ASCEND_OPP_PATH` 把测试链到错误的包。

---

## 4. Autofuse 仓怎么跑 LLT、怎么看覆盖率

统一入口：`graph-autofusion/scripts/test/run_autofuse_test.sh`。

```bash
cd /root/workspace/cann-open/graph-autofusion

# C++ common UT + 覆盖率
bash scripts/test/run_autofuse_test.sh -u -m common -c -j 3 \
  --ascend_install_path=/usr/local/Ascend/ascend-toolkit/latest

# Python codegen UT（模块名 codegen）
bash scripts/test/run_autofuse_test.sh -u -m codegen -j 3 \
  --ascend_install_path=/usr/local/Ascend/ascend-toolkit/latest

# backend ST（含 Conv2D / ExtendConv2D codegen）
bash scripts/test/run_autofuse_test.sh -s -m backend -j 3 \
  --ascend_install_path=/usr/local/Ascend/ascend-toolkit/latest
```

UT 模块：`common` / `att` / `optimize` / `codegen` / `framework` / `all`。
ST 模块：`backend` / `e2e` / `codegen` / `framework` / `all`。

也可以增量编指定二进制：

```bash
cmake --build build --target test_common -j 3
./build/autofuse/tests/ut/common/test_common --gtest_filter='CommonUtilsTest.ExtendConv2D*'
```

覆盖率：加 `-c` 后打开

```text
graph-autofusion/cov/coverage_report/index.html
```

注意：官方脚本默认写 `build/`。若已有 Release/pkg 的 `build/`，再跑 `-u -c` 会改 cmake 配置。本机曾用独立目录 `build_ut` 编 UT，避免冲掉打包目录。

---

## 5. GE 仓怎么跑 LLT、怎么看覆盖率

GE 有两套构建，不要混用。

### 5.1 Autofuse frontend UT（不依赖编 autofuse 仓）

构建目录是 `ge/build`（`RUN_TEST=1`），二进制 `autofusion_ut`。

```bash
cd /root/workspace/cann-open/ge
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
unset LD_LIBRARY_PATH
unset ASCEND_OPP_PATH

# 脚本一条龙（含覆盖率）
bash scripts/test/run_autofuse_test.sh -u -m autofuse -c -j 3 \
  --ascend_install_path=/usr/local/Ascend/ascend-toolkit/latest

# 或增量编、单跑
cmake --build build --target autofusion_ut -j 3
./build/tests/autofuse/ut/autofuse/autofusion_ut \
  --gtest_filter='ExtendConv2DLoweringUT.*:LoopApiUT.StoreIgnoredOutputIsNotExtern'
```

`v35/` 下那批 lowering 用例**没有**挂进 `autofusion_ut`（整目录缺头文件，不能 `add_subdirectory(v35)`）。ExtendConv2D lowering 用例放在 `tests/autofuse/ut/autofuse/extend_conv2d_lowering_unittest.cpp`。

### 5.2 GE infer-shape / tiling UT

构建目录必须是 `cmake-build-gcov`（`CMAKE_BUILD_TYPE=GCOV -DENABLE_GE_UT=ON`）。

```bash
make -C cmake-build-gcov ut_libge_symbol_infer_utest -j 3
./cmake-build-gcov/tests/ge/ut/ge/ut_libge_symbol_infer_utest \
  --gtest_filter='SymbolicShapeInferFuncUT.InferSymbolicShapeForExtendConv2D'

make -C cmake-build-gcov ut_libge_common_utest -j 3
./cmake-build-gcov/tests/ge/ut/ge/ut_libge_common_utest \
  --gtest_filter='RegisterOpTilingRT2UT.AutofuseNodeWithExtendConvTilingAttrsSuccess'
```

这两个 target 都链 `ge_compiler`。toolkit 头文件若比代码旧（缺 `aclskScopeVerifyGraphInfo` 等），会编不过，与 autofuse 仓无关。

覆盖率：

```bash
bash scripts/coverage/ge_cov.sh
# 或 autofuse 脚本 -c
# 报告：ge/cov/coverage_report/index.html
```

只跑部分用例时，整文件覆盖率会偏低（例如只跑 ExtendConv2D lowering，`lowering_impl.cpp` 行覆盖大约十几个百分点）。看覆盖率时要进 HTML，点到具体函数，不要只看整文件百分比。

---

## 6. 本次为 ExtendConv2D 补的用例

### Autofuse 仓

| 文件 | 用例 | 断言 |
|------|------|------|
| `autofuse/tests/ut/python/test_asc_codegen_compile_conv2d.py` | `test_build_conv_args_extend_conv2d_*` | 10 个输入槽、双输出；可选槽可空 |
| `autofuse/tests/ut/common/test_common_utils.cpp` | `CommonUtilsTest.ExtendConv2D*` | `IsConv2DGraphType` / `ParseConv2DAttr` 的 bias、scale0、round_mode |
| `autofuse/tests/st/backend_e2e/conv2d_elemwise_test/conv2d_backend_generate.cpp` | `ExtendConv2DE2eCodegen`、`ExtendConv2DBiasScaleE2eCodegen` | codegen 能出 `GenCVFusionTilingKey` |

`tests/st/backend_e2e/CMakeLists.txt` 里原先注释掉的 `add_subdirectory(conv2d_elemwise_test)` 已打开，否则 ST 编不进来。

### GE 仓

| 文件 | 用例 | 断言 |
|------|------|------|
| `tests/autofuse/ut/autofuse/extend_conv2d_lowering_unittest.cpp` | 四种 bias/scale0 组合 | Lowering 成功；y0 Cube；未用 y1 非 Extern |
| `tests/autofuse/ut/autofuse/loop_api_unittest.cpp` | `StoreIgnoredOutputIsNotExtern` | 占位不是 Extern、是 Cube |
| `tests/ge/ut/ge/graph/optimize/symbolic/symbolic_shape_infer_func_unittest.cc` | `InferSymbolicShapeForExtendConv2D` | SPECIFIC / SAME_LOWER 成功，非法 pad_mode 失败 |
| `tests/ge/ut/ge/common/op_tiling_rt2_unittest.cc` | `AutofuseNodeWithExtendConvTilingAttrsSuccess` | ExtendConv2D tiling 属性 |

构图辅助：`all_ops.cpp` / `all_ops.h` / `all_ops_cpp.h`、`op_creator_register.h` 增加了 `ExtendConv2D`。y1 在 IR 上是 required，图里可以不连边。

---

## 7. 什么时候才需要编、装 autofuse 仓

只有这些场景需要：

1. 改了 graph-autofusion 源码，要跑 **autofuse 仓自己的** UT/ST。
2. GE lowering 之后要 **Realize 出新 ASCIR 节点**（toolkit 里还没有 `ExtendConv2D*`）。
3. GE ST / 真机编译融合 kernel / 二次 tiling。

安装步骤（本机 toolkit 是 9.2.0 时，不要复用指向 9.1.0 的旧 `build/`）：

```bash
cd /root/workspace/cann-open/graph-autofusion
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
# 旧 build/ 若 cmake 缓存仍是 cann-9.1.0，先清掉
rm -rf build
sh build.sh --pkg -j 3
# 把生成的 cann-graph-autofusion_*.run 装到 toolkit
```

不装包、只想让本机 GE 进程优先加载刚编的 so：

```bash
export LD_LIBRARY_PATH=<autofuse构建目录>/autofuse:$LD_LIBRARY_PATH
```

ABI 要和 toolkit 一致（编译器、Build Type）。

---

## 8. 本机踩过的坑

1. **不要**把整个 `tests/autofuse/v35` 加进 `autofusion_ut`，缺 `base/att_const_values.h` 等，整目录编不过。
2. `es::Tensor` 没有默认构造，可选输入写成 `es::Tensor bias(nullptr)`。
3. optional 输入连上以后，节点上对应 `InputDesc` 可能仍为空；lowering 前要把对端 output desc 同步到 input desc，否则 `IsStaticShape` 断言失败。
4. GE `cmake-build-gcov` 与 toolkit 头文件版本不一致时，`ge_compiler` 编不过，infer-shape / tiling UT 会一起挂，这和 autofuse 仓无关。
5. gcov 报 checksum mismatch，多半是旧 `.gcda` 和当前 `.o` 不一致，删对应 `.gcda` 或全量重编即可。
6. 覆盖率脚本依赖本机 `lcov`/`gcov` 版本；lcov 2.x 常要加 `--ignore-errors mismatch,empty`。
