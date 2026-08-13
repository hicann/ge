# E10011 Invalid\_Argument\_Tensor\_Input\_Shape

## 错误信息

报错格式如下，占位符%s的含义依次为参数值、参数名、报错原因：

```text
Value %s for parameter --input_shape is invalid. Shape values must be positive integers. The error value in the shape is %s.
```

报错示例如下：

```text
Value [-1,2,3,4] for parameter --input_shape is invalid. Shape values must be positive integers. The error value in the shape is -1.
```

## 解决方法

- 在静态Shape场景下，将--input\_shape中的shape值设置为正整数。
- 在动态shape场景下，请在命令行中添加相关的动态输入选项，例如--dynamic\_batch\_size、--dynamic\_image\_size或--dynamic\_dims。
