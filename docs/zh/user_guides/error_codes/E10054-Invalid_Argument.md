# E10054 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s表示参数名：

```text
The required parameter %s for ATC is empty. Another possible reason is that the values of some parameters are not enclosed by quotation marks ("").
```

报错示例如下：

```text
The required parameter --soc_version for ATC is empty. Another possible reason is that the values of some parameters are not enclosed by quotation marks ("").
```

## 解决方法

检查命令行参数格式是否正确。
