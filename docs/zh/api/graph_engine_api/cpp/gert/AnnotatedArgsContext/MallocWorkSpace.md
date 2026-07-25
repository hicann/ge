# MallocWorkSpace

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/annotated\_args\_context.h\>
- 库文件：liblowering.so

## 功能说明

编译时框架有[内存复用优化技术](https://www.hiascend.com/developer/techArticles/202407005-1?envFlag=1)。该接口描述所需workspace大小，框架将该信息记录在模型内，返回在该workspace在模型内整块逻辑内存的偏移量。与真实的物理地址概念不同，通常称之为逻辑地址。

## 函数原型

```c++
WorkspaceAddr MallocWorkSpace(size_t size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| size | 输入 | 申请的workspace大小，单位为字节，必须大于0。 |

## 返回值说明

返回`WorkspaceAddr`描述符。成功时`addr`非空，`index`对应该workspace在本次申请序列中的0-based序号；失败时`addr`为`nullptr`，此时不得继续使用该描述符构造launch参数。

`addr`是逻辑地址，并非真实地址，无需管理其生命周期。

## 约束说明

无

## 调用示例

以下片段位于`DeclareLaunchArgs`内。申请失败时应终止声明；成功后直接将返回的描述符作为`AnnotatedKernelArgs`参数，最终交给`AddLaunch`。

```c++
const auto workspace0 = ctx.MallocWorkSpace(1024U);
const auto workspace1 = ctx.MallocWorkSpace(2048U);
if ((workspace0.addr == nullptr) || (workspace1.addr == nullptr)) {
  return ge::GRAPH_FAILED;
}

// workspace0.index == 0，workspace1.index == 1。
gert::AnnotatedKernelArgs args(workspace0, workspace1);
// 继续构造其他参数并调用ctx.AddLaunch(..., std::move(args))。
```
