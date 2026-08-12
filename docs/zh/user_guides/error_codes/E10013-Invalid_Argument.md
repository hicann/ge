# E10013 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数值、参数名：

```text
Value %s for --%s is out of range.
```

报错示例如下：

```text
Value 99999999999999999999 for --input_shape is out of range.
```

## 解决方法

运行“atc -h”命令查看相关参数的使用方法，详情请参见官方文档中的ATC工具使用说明。
