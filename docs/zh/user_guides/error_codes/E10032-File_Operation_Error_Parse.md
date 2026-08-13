# E10032 File\_Operation\_Error\_Parse

## 错误信息

报错格式如下，占位符%s的含义依次为文件名、报错原因：

```text
Failed to parse JSON file %s. Reason: %s.
```

报错示例如下：

```text
Failed to parse JSON file /home/singleop.json. Reason: [json.exception.out_of_range.401] array index 5 is out of range.
```

## 解决方法

检查JSON文件是否有效。
