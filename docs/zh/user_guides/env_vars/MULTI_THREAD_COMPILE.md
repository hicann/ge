# MULTI\_THREAD\_COMPILE

## 功能描述

此环境变量用于控制模型转换时是否使用单线程编译。

<!-- @ref: ge/res/docs/zh/user_guides/env_vars/MULTI_THREAD_COMPILE_res.md#id1 -->

- 0：开启单线程编译
- 1：（默认值）开启多线程编译

## 配置示例

```bash
export MULTI_THREAD_COMPILE=1
```

## 使用约束

- 该环境变量仅适用于使用ATC工具进行模型转换，生成离线om文件的场景。
- 如果此环境变量设置了其他非法值，可能会导致未定义的行为发生。

## 产品支持情况

全量芯片支持
