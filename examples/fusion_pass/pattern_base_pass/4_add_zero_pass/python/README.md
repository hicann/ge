## Python Pass[v1]

该目录提供了两个纯 Python 的 `PatternFusionPass` 版本：

- [src/test_python_pattern_pass.py](./src/test_python_pattern_pass.py)
- [src/test_python_pattern_pass_cpp_equivalent.py](./src/test_python_pattern_pass_cpp_equivalent.py)

两个 sample 的定位不同：

- `test_python_pattern_pass.py`
  - 演示 Python V1 的 `PatternMatcherConfigBuilder.enable_const_value_match()`
  - 通过 strict const-value-match 直接把 `Add(x, 0.0f)` 前置到 matcher 阶段
  - 需要注意：当前 `ConstantMatcher::IsMatch` 的值匹配是严格二进制匹配，不带浮点容差，也不做跨 dtype 归一化
  - 因此它更适合作为 matcher_config 示例，而不是 C++ sample 的完全等价版本

- `test_python_pattern_pass_cpp_equivalent.py`
  - 语义上对齐 [C++ pass样例](../cpp/src/add_zero_pass.cpp)
  - `patterns()` 只描述 `Data + Const` 拓扑
  - `meet_requirements()` 中再显式读取匹配到的 `Const.value`，按 C++ sample 同样的规则判断零值
  - 当前支持与 C++ sample 一致的 `DT_FLOAT` / `DT_DOUBLE` / `DT_INT32`

两者共同演示了以下 Python V1 能力：

- 使用 `GraphBuilder` 构造 pattern / replacement graph
- 使用 `capture_tensor()` / `MatchResult.get_captured_tensor()` 获取捕获输入
- 在 `meet_requirements()` 中读取 `MatchResult.get_matched_nodes()` 返回的 `Const` 节点属性
- 通过 `@register_fusion_pass` 以 `PatternFusionPass` 形式注册到 GE

使用方式如下：

1. 设置 Python pass 插件路径
   ```bash
   export ASCEND_GE_PY_PASS_PATH=$(pwd)/src/test_python_pattern_pass.py
   ```
   如果需要运行与 C++ 语义等价的版本，可改为：
   ```bash
   export ASCEND_GE_PY_PASS_PATH=$(pwd)/src/test_python_pattern_pass_cpp_equivalent.py
   ```
2. 复用[C++ pass样例](../cpp/README.md#程序运行)的 ATC 或在线推理步骤执行模型编译

run包已包含 GE Python 运行时所需的 ge-py wheel，本节不需要再单独安装 `ge_py-0.0.1-py3-none-any.whl`。

预期日志中会出现类似输出：

```text
[PythonAddZeroPass] matched=add_zero_pattern captured=input_0:0
[PythonAddZeroPassCppEquivalent] matched=add_zero_pattern_cpp_equivalent captured=input_0:0 const_dtype=DT_FLOAT zero=True
```
