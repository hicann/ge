# E13004 File\_Operation\_Error\_Write

## 错误信息

报错格式如下，占位符%s的含义依次为文件名称、报错原因：

```text
Failed to write file %s. Reason: %s.
```

报错示例如下：

```text
Failed to write file /home/output. Reason: [Error 13] Permission denied.
```

## 解决方法

读取文件失败，需按照Reason中的提示定位问题，提供正确的文件。
