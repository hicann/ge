# aclGetCustomOpLibPath

## 产品支持情况

请参见[Session接口产品支持情况](../../cpp/ge/Session/overview.md)。

## 头文件

\#include <register/register\_base.h\>

## 功能说明

获取自定义算子库的路径。

该接口为内部框架会调用的接口，称之为**内部关联接口**。开发者不会直接调用内部关联接口，无需关注。

## 函数原型

```c
const char *aclGetCustomOpLibPath()
```

## 参数说明

无

## 返回值说明

返回自定义算子库的路径。多个路径之间用英文冒号分隔，路径已按照用户设置的算子优先级顺序排列好，优先级越高的位置越靠前。

关于自定义算子的优先级设置请参考《[Ascend C算子开发](https://gitcode.com/cann/asc-devkit/blob/9.2.0-beta.2/docs/zh/guide/index.md)》中的“编程指南 \> 附录 \> 工程化算子开发 \>  算子动态库编译”章节。

## 约束说明

无

## 调用示例

```c
const char* path = aclGetCustomOpLibPath();
```
