# 简介

## 产品支持情况

全量芯片支持。

## 功能说明

`OpCompileContext`是当前schema-bound`compile`回调的只读编译上下文，由[get_compile_ctx](../get_compile_ctx.md)返回，用户不能直接构造。

该对象提供编译option查询。它是仅在当前回调内有效的借用视图；回调返回或抛出异常后，再调用任何方法都会抛出`RuntimeError`。平台资源、核数和SoC信息请通过[get_compile_platform_info()](../get_compile_platform_info.md)获取。

`OpCompileContext`只查询编译环境，不保存用户的编译结果。用户可以在实现实例中暂存结果，但GE不会序列化、恢复或回滚这些Python状态。

## 函数原型

无

## 参数说明

无

## 约束说明

该对象只能在当前同步`compile`回调内使用。

## 调用示例

无
