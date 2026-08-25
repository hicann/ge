# 简介

## 产品支持情况

全量芯片支持。

## 功能说明

`CompilePlatformInfo`是当前schema-bound`compile`回调的平台信息只读视图，由`get_compile_platform_info()`返回，用户不能直接构造。

该对象仅在当前`compile`回调内有效；回调返回或抛出异常后，再调用任何方法都会抛出`RuntimeError`。

## 函数原型

无

## 参数说明

无

## 约束说明

该对象只能在当前同步`compile`回调内使用。

## 调用示例

无
