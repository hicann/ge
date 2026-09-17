# GetDefaultSwitch

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <register/register\_custom\_pass.h\>
- 库文件：libregister.so

## 功能说明

获取自定义Pass的默认开关状态。

## 函数原型

```c++
PassSwitch GetDefaultSwitch() const
```

## 参数说明

无

## 返回值说明

返回Pass的默认开关状态，类型为[PassSwitch](../PassSwitch.md)。impl为null时返回`PassSwitch::kOn`。

## 约束说明

无

## 调用示例

```c++
PassRegistrationData pass_data("MyCustomPass");
pass_data.DefaultSwitch(PassSwitch::kOff);

// 读取默认开关状态
PassSwitch switch_status = pass_data.GetDefaultSwitch();
// switch_status == PassSwitch::kOff
```
