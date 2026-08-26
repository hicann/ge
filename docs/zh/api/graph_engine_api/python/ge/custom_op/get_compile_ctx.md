# get\_compile\_ctx

## 产品支持情况

全量芯片支持。

## 功能说明

获取当前[compile](compile.md)回调的只读编译上下文。

## 函数原型

```python
get_compile_ctx() -> OpCompileContext
```

## 参数说明

无

## 约束说明

- 只能在当前同步`compile`回调内调用；回调外调用时抛出`RuntimeError`。
- 返回对象及由schema参数取得的`Tensor`、Tensor属性均为借用视图，回调返回或抛出异常后失效。
- 字符串、整数和`dict`等查询结果是Python值副本，可以在回调结束后继续使用。
- 该接口只用于图编译阶段，不在模型加载或执行阶段调用。

## 调用示例

```python
from ge.custom_op import get_compile_ctx


def compile(self, x, y, *, alpha: int) -> None:
    ctx = get_compile_ctx()
    option = ctx.get_option("custom.compile.option")
```
