# E10008 Invalid\_Argument

## 错误信息

```text
--weight must not be empty when --framework is set to 0 (Caffe).
```

## 解决方法

- 如果源模型框架是Caffe，请尝试使用有效的--weight参数重新运行。
- 如果源模型框架不是Caffe，请尝试使用有效的--framework参数重新运行。
