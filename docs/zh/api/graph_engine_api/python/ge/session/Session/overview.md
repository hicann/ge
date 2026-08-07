# 简介

Session类用于管理图的执行会话。

会话执行调用示例：

```python
from ge.session import Session
from ge.graph import Tensor
from ge.ge_global import GeApi

# 初始化 GE
config = {
    "ge.execDeviceId": "0",
    "ge.graphRunMode": "0"
}
GeApi.ge_initialize(config)


# 创建会话
session = Session()

# 添加图
session.add_graph(0, graph)

# 准备输入
input_tensor = Tensor(data=[1.0, 2.0, 3.0, 4.0], data_type=DataType.DT_FLOAT, format=Format.FORMAT_ND, shape=[2, 2])

# 运行图
outputs = session.run_graph(0, [input_tensor])

# 终结 GE
GeApi.ge_finalize()
```

**Session接口产品支持情况如下：**

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- Atlas 200I/500 A2 推理产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- Atlas 训练系列产品：支持
<!-- end id6 -->
<!-- @ref: ge/res/docs/zh/api/graph_engine_api/python/ge/session/Session/overview_res.md#id1 -->
