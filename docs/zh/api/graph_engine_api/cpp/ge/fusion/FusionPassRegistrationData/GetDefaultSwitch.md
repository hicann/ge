# GetDefaultSwitch

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <ge/fusion/pass/fusion\_pass\_reg.h\>
- 库文件：libge\_compiler.so

## 功能说明

获取融合Pass的默认开关状态。

## 函数原型

```c++
PassSwitch GetDefaultSwitch() const
```

## 参数说明

无

## 返回值说明

返回Pass的默认开关状态，类型为[PassSwitch](../../PassSwitch.md)。impl为null时返回`PassSwitch::kOn`。

## 约束说明

无

## 调用示例

```c++
FusionPassRegistrationData pass_data("MyFusionPass");
pass_data.DefaultSwitch(PassSwitch::kOff);

// 读取默认开关状态
PassSwitch switch_status = pass_data.GetDefaultSwitch();
// switch_status == PassSwitch::kOff
```
