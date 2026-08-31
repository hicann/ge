# OpBackend

自定义算子后端类型枚举，头文件位于CANN软件安装后文件存储路径下的include/graph/custom\_op.h。

```c++
enum class OpBackend : uint32_t {
  kDevice = 0,
  kHostCPU = 1,
};
```

各枚举项说明如下：

- kDevice：Device后端。
- kHostCPU：Host CPU后端。
