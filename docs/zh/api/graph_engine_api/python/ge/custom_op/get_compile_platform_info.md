# get\_compile\_platform\_info

## 产品支持情况

全量芯片支持。

## 功能说明

获取当前schema-bound`compile`回调的平台信息只读视图。

## 函数原型

```python
get_compile_platform_info() -> CompilePlatformInfo
```

## 参数说明

无

## 约束说明

- 只能在当前同步`compile`回调内调用；回调外调用时抛出`RuntimeError`。
- 返回对象在回调返回或抛出异常后失效。

## 调用示例

```python
from ge.custom_op import get_compile_platform_info


def compile(self, x, y) -> None:
    platform = get_compile_platform_info()
    soc_version = platform.get_soc_version()
```
