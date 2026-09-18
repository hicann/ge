# DefaultSwitch

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <register/register\_custom\_pass.h\>
- 库文件：libregister.so

## 功能说明

设置自定义Pass的默认开关状态。声明为`kOff`的pass默认不执行，仅当用户通过运行时配置显式开启时才执行。

## 函数原型

```c++
PassRegistrationData &DefaultSwitch(PassSwitch pass_switch)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| pass_switch | 输入 | 表示Pass的默认开关状态。详情请参见[PassSwitch](../PassSwitch.md)。 |

## 返回值说明

返回PassRegistrationData对象，支持链式调用。

## 约束说明

- 不调用`DefaultSwitch`时，默认为`PassSwitch::kOn`，与历史行为一致。
- 若`.so`需要在旧版GE（< 9.3.0）上运行，需使用`COMPILER_VERSION_NUM`编译宏保护`DefaultSwitch`调用。

## 调用示例

```c++
// 注册自定义Pass，声明默认关闭
REGISTER_CUSTOM_PASS("RiskyCustomPass")
    .DefaultSwitch(PassSwitch::kOff)
    .CustomPassFn(MyCustomPass)
    .Stage(CustomPassStage::kBeforeInferShape);

// 不调用DefaultSwitch时默认为kOn（向后兼容）
REGISTER_CUSTOM_PASS("SafeCustomPass")
    .CustomPassFn(MyCustomPass)
    .Stage(CustomPassStage::kBeforeInferShape);
```
