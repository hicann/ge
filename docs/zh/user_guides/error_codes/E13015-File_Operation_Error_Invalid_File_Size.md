# E13015 File\_Operation\_Error\_Invalid\_File\_Size

## 错误信息

报错格式如下，占位符%s的含义依次为文件名、文件大小、最大值：

```text
File %s has a size of %s, which is out of valid range (0, %s].
```

报错示例如下：

```text
File /home/file.txt has a size of 2147483649, which is out of valid range (0, 2147483647].
```

## 解决方法

需按照提示提供有效大小的文件。
