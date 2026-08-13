# E13027 Invalid\_Argument\_Network\_Connection

## 错误信息

报错格式如下，占位符%s表示ip address：

```text
Failed to connect to the peer address %s.
```

报错示例如下：

```text
Failed to connect to the peer address 192.0.0.1.
```

## 可能原因

IP地址、端口或token无效。

## 解决方法

请检查配置文件中的ipaddr、port或token配置是否正确。
