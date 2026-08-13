# E12029 Invalid\_Argument\_TensorFlow\_Model\_Data

## 错误信息

```text
Failed to find the subgraph library.
```

## 可能原因

要转换的模型包含function subgraphs，但未找到graph\_def\_library.pbtxt文件。

## 解决方法

1. 要使用function subgraphs转换TensorFlow模型，需将子图的.proto描述文件与模型文件放在同一目录下，并将其命名为graph\_def\_library.pbtxt。
2. 然后在ATC工具安装目录下运行func2graph.py脚本，将子图保存到graph\_def\_library.pbtxt中。
