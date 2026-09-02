# Autofuse Adaptation for ExtendConv2D: LLT and Cross-Repository Dependencies

This document describes how to run the LLT (UT + ST) after ExtendConv2D integration, how to inspect coverage, and **whether tests in the GE repository depend on building and installing the graph-autofusion repository**.

For the feature flow, see [autofuse_extend_conv2d_cv_fusion.md](./autofuse_extend_conv2d_cv_fusion.md) in the same directory.

---

## 1. Conclusion First: Do GE UTs Require Building and Installing the Autofuse Repository?

**For routine development and execution of UTs in the GE repository, you do not need to build the graph-autofusion repository or install its artifacts into CANN.**

This was verified on August 25, 2026, using a clean reinstallation of cann-9.2.0: the official toolkit **includes** the `graph_autofusion` component (`share/info/graph_autofusion/version.info`, `lib64/libaihac_codegen.so`, and others). It is not a repository manually overlaid in the past, but part of the current CANN package. The official package **does not yet contain** AscendC IR registrations for `ExtendConv2D*`.

| What you are doing | Build the autofuse repository? | Install the autofuse repository into CANN? | What is actually used |
|--------------------|--------------------------------|--------------------------------------------|-----------------------|
| GE infer-shape / tiling UT | No | No | GE source code + test stubs |
| GE autofuse frontend UT (`autofusion_ut`) | No | No | `libautofuse.so` built from GE source code; codegen links against `libaihac_codegen.so` **included in the official toolkit** |
| GE lowering Realize produces a new AscendC IR type (`ExtendConv2D*`) | No | **Only required when the official package does not yet contain these types** | AscendC IR registered in the toolkit |
| UT/ST of the Autofuse repository itself | **Build the autofuse repository** | No (tests use artifacts from this repository's `build/`) | graph-autofusion source code |
| GE ST / real kernel / secondary tiling | Yes | **Yes** | Overlay the newly built autofuse repository into CANN |

Rerunning `ExtendConv2DLoweringUT.*` + `StoreIgnoredOutputIsNotExtern` on the clean package: **5/5 passed**. The log still contains `unknown op type 'ExtendConv2D'`, indicating that Realize uses the old AscendC IR in the official package; this does not affect the UT assertions here.

---

## 2. What Each Repository Tests

```text
GE repository (frontend)
  InferSymbolShape     pad_mode is at ExtendConv2D attribute index 7 (round_mode occupies 6)
  Lowering             selects among 4 AscendC IR variants based on bias/scale0; unused y1 uses StoreIgnoredOutput
  Lifting / tiling     conv_subgraph + ExtendConv2D tiling attributes

graph-autofusion repository (backend)
  ParseConv2DAttr      is_extend_conv2d / has_bias / has_scale0 / round_mode
  build_conv_args      10 logical input slots, dual-output placeholders
  codegen / e2e ST     ExtendConv2D and ExtendConv2DBiasScale produce a tiling key
```

The GE repository's frontend source code is under `compiler/graph/optimize/autofuse/`, and its AscendC IR header is the vendored `temporary_dependencies/ascir/ascir_ops.h`.
Kernel templates, tiling keys, and Python codegen are in the graph-autofusion repository.

---

## 3. Environment

Local reference environment:

- toolkit: `/usr/local/Ascend/ascend-toolkit/latest` → cann-9.2.0
- On machines with few cores, build with `-j 3` (the default parallelism is too high and can easily cause OOM)
- Before running GE tests:

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
unset LD_LIBRARY_PATH
unset ASCEND_OPP_PATH
```

The variables are unset to prevent stale `LD_LIBRARY_PATH` / `ASCEND_OPP_PATH` values from linking tests against the wrong package.

---

## 4. How to Run LLT and Inspect Coverage in the Autofuse Repository

Unified entry point: `graph-autofusion/scripts/test/run_autofuse_test.sh`.

```bash
cd /root/workspace/cann-open/graph-autofusion

# C++ common UT + coverage
bash scripts/test/run_autofuse_test.sh -u -m common -c -j 3 \
  --ascend_install_path=/usr/local/Ascend/ascend-toolkit/latest

# Python codegen UT (module name: codegen)
bash scripts/test/run_autofuse_test.sh -u -m codegen -j 3 \
  --ascend_install_path=/usr/local/Ascend/ascend-toolkit/latest

# Backend ST (including Conv2D / ExtendConv2D codegen)
bash scripts/test/run_autofuse_test.sh -s -m backend -j 3 \
  --ascend_install_path=/usr/local/Ascend/ascend-toolkit/latest
```

UT modules: `common` / `att` / `optimize` / `codegen` / `framework` / `all`.
ST modules: `backend` / `e2e` / `codegen` / `framework` / `all`.

You can also incrementally build a specific binary:

```bash
cmake --build build --target test_common -j 3
./build/autofuse/tests/ut/common/test_common --gtest_filter='CommonUtilsTest.ExtendConv2D*'
```

Coverage is enabled by adding `-c`:

```text
graph-autofusion/cov/coverage_report/index.html
```

Note: The official script writes to `build/` by default. If `build/` already contains a Release/package build, running `-u -c` changes its CMake configuration. A separate `build_ut` directory was used locally to build UTs and avoid overwriting the package build directory.

---

## 5. How to Run LLT and Inspect Coverage in the GE Repository

GE has two build configurations; do not mix them.

### 5.1 Autofuse Frontend UT (Does Not Depend on Building the Autofuse Repository)

The build directory is `ge/build` (`RUN_TEST=1`), and the binary is `autofusion_ut`.

```bash
cd /root/workspace/cann-open/ge
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
unset LD_LIBRARY_PATH
unset ASCEND_OPP_PATH

# Run the complete script (including coverage)
bash scripts/test/run_autofuse_test.sh -u -m autofuse -c -j 3 \
  --ascend_install_path=/usr/local/Ascend/ascend-toolkit/latest

# Alternatively, build incrementally and run selected tests
cmake --build build --target autofusion_ut -j 3
./build/tests/autofuse/ut/autofuse/autofusion_ut \
  --gtest_filter='ExtendConv2DLoweringUT.*:LoopApiUT.StoreIgnoredOutputIsNotExtern'
```

The lowering tests under `v35/` are **not** included in `autofusion_ut` (the entire directory is missing headers and cannot be added with `add_subdirectory(v35)`). The ExtendConv2D lowering tests are in `tests/autofuse/ut/autofuse/extend_conv2d_lowering_unittest.cpp`.

### 5.2 GE Infer-Shape / Tiling UT

The build directory must be `cmake-build-gcov` (`CMAKE_BUILD_TYPE=GCOV -DENABLE_GE_UT=ON`).

```bash
make -C cmake-build-gcov ut_libge_symbol_infer_utest -j 3
./cmake-build-gcov/tests/ge/ut/ge/ut_libge_symbol_infer_utest \
  --gtest_filter='SymbolicShapeInferFuncUT.InferSymbolicShapeForExtendConv2D'

make -C cmake-build-gcov ut_libge_common_utest -j 3
./cmake-build-gcov/tests/ge/ut/ge/ut_libge_common_utest \
  --gtest_filter='RegisterOpTilingRT2UT.AutofuseNodeWithExtendConvTilingAttrsSuccess'
```

Both targets link against `ge_compiler`. If the toolkit headers are older than the code (for example, if `aclskScopeVerifyGraphInfo` is missing), compilation fails; this is unrelated to the autofuse repository.

Coverage:

```bash
bash scripts/coverage/ge_cov.sh
# Or use -c with the autofuse script
# Report: ge/cov/coverage_report/index.html
```

When only some tests are run, whole-file coverage is low (for example, when only ExtendConv2D lowering tests are run, line coverage for `lowering_impl.cpp` is approximately a dozen percent). When inspecting coverage, open the HTML report and navigate to the specific functions instead of looking only at the whole-file percentage.

---

## 6. Tests Added for ExtendConv2D

### Autofuse Repository

| File | Test | Assertion |
|------|------|-----------|
| `autofuse/tests/ut/python/test_asc_codegen_compile_conv2d.py` | `test_build_conv_args_extend_conv2d_*` | 10 input slots and dual outputs; optional slots may be empty |
| `autofuse/tests/ut/common/test_common_utils.cpp` | `CommonUtilsTest.ExtendConv2D*` | bias, scale0, and round_mode handling in `IsConv2DGraphType` / `ParseConv2DAttr` |
| `autofuse/tests/st/backend_e2e/conv2d_elemwise_test/conv2d_backend_generate.cpp` | `ExtendConv2DE2eCodegen`, `ExtendConv2DBiasScaleE2eCodegen` | codegen can produce `GenCVFusionTilingKey` |

The previously commented-out `add_subdirectory(conv2d_elemwise_test)` in `tests/st/backend_e2e/CMakeLists.txt` has been enabled; otherwise, the ST is not included in the build.

### GE Repository

| File | Test | Assertion |
|------|------|-----------|
| `tests/autofuse/ut/autofuse/extend_conv2d_lowering_unittest.cpp` | Four bias/scale0 combinations | Lowering succeeds; y0 is Cube; unused y1 is not Extern |
| `tests/autofuse/ut/autofuse/loop_api_unittest.cpp` | `StoreIgnoredOutputIsNotExtern` | The placeholder is not Extern and is Cube |
| `tests/ge/ut/ge/graph/optimize/symbolic/symbolic_shape_infer_func_unittest.cc` | `InferSymbolicShapeForExtendConv2D` | SPECIFIC / SAME_LOWER succeed; invalid pad_mode fails |
| `tests/ge/ut/ge/common/op_tiling_rt2_unittest.cc` | `AutofuseNodeWithExtendConvTilingAttrsSuccess` | ExtendConv2D tiling attributes |

Graph construction helpers `all_ops.cpp` / `all_ops.h` / `all_ops_cpp.h` and `op_creator_register.h` were extended with `ExtendConv2D`. y1 is required in the IR but may remain disconnected in the graph.

---

## 7. When the Autofuse Repository Must Be Built and Installed

It is required only in these scenarios:

1. You modified graph-autofusion source code and need to run the **autofuse repository's own** UT/ST.
2. GE lowering must **Realize a new AscendC IR node** (the toolkit does not yet contain `ExtendConv2D*`).
3. GE ST / compilation of a fused kernel on real hardware / secondary tiling.

Installation steps (when the local toolkit is 9.2.0, do not reuse an old `build/` that points to 9.1.0):

```bash
cd /root/workspace/cann-open/graph-autofusion
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
# If the CMake cache in the old build/ still uses cann-9.1.0, remove it first
rm -rf build
sh build.sh --pkg -j 3
# Install the generated cann-graph-autofusion_*.run into the toolkit
```

To avoid installing the package and only make local GE processes load the newly built shared object first:

```bash
export LD_LIBRARY_PATH=<autofuse-build-directory>/autofuse:$LD_LIBRARY_PATH
```

The ABI must match the toolkit (compiler and Build Type).

---

## 8. Issues Encountered Locally

1. **Do not** add the entire `tests/autofuse/v35` directory to `autofusion_ut`; headers such as `base/att_const_values.h` are missing, so the directory does not compile.
2. `es::Tensor` has no default constructor; write an optional input as `es::Tensor bias(nullptr)`.
3. After an optional input is connected, the corresponding `InputDesc` on the node may still be empty. Before lowering, synchronize the peer output desc to the input desc; otherwise, the `IsStaticShape` assertion fails.
4. When the GE `cmake-build-gcov` configuration and toolkit headers are from different versions, `ge_compiler` does not compile and the infer-shape / tiling UTs both fail. This is unrelated to the autofuse repository.
5. A gcov checksum mismatch is usually caused by stale `.gcda` files that do not match the current `.o` files. Delete the corresponding `.gcda` files or perform a full rebuild.
6. The coverage script depends on the locally installed `lcov`/`gcov` versions; lcov 2.x often requires `--ignore-errors mismatch,empty`.
