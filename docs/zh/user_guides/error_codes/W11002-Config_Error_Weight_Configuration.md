# W11002 Config\_Error\_Weight\_Configuration

## 错误信息

报错格式如下，占位符%s的含义依次为文件名、算子名称：

```text
In the compression weight configuration file %s, some nodes do not exist in graph: %s.
```

报错示例如下：

```text
In the compression weight configuration file xxx, some nodes do not exist in graph: graph_name.
```

## 解决方法

检查权重文件是否与模型文件匹配。
