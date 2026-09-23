# OpBackend

Python自定义算子执行后端（backend）枚举，[register_kernel](register_kernel.md) 通过该枚举选择 execute 的执行后端。

| 枚举值 | 说明 |
| --- | --- |
| DEVICE | Device后端。 |
| HOST | Host CPU后端。 |
