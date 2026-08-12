# E12013 Invalid\_Argument\_TensorFlow\_Model\_Data

## 错误信息

报错格式如下，占位符%s表示图名称：

```text
Failed to find a subgraph by the name %s.
```

报错示例如下：

```text
Failed to find a subgraph by the name tf_subgraph.
```

## 解决方法

1. 要使用function subgraphs转换TensorFlow模型，需将子图的.proto描述文件与模型文件放在同一目录下，并将其命名为graph\_def\_library.pbtxt。
2. 然后在ATC工具安装目录下运行func2graph.py脚本，将子图保存到graph\_def\_library.pbtxt中。
