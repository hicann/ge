# PassSwitch

Pass默认开关状态枚举，头文件：\#include <register/register\_custom\_pass.h\>

```c++
enum class PassSwitch : uint32_t {
  kOn = 0,
  kOff = 1
};
```

- kOn：pass默认开启（与历史行为一致）。不调用`DefaultSwitch`时的默认值。
- kOff：pass默认关闭。仅当用户通过`fusion_switch_file` JSON配置或`--optimization_switch`命令行参数显式开启时才执行。

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <register/register\_custom\_pass.h\>
- 库文件：libregister.so

## 约束说明

无

## 注意事项

`kOff`适用于存在精度风险或性能回退的pass。声明为`kOff`后，pass在注册时默认不执行，用户可通过运行时配置（graph option > JSON精确匹配 > JSON ALL通配）覆盖注册默认值，显式开启该pass。

## 调用示例

```c++
// 声明pass默认关闭
REGISTER_CUSTOM_PASS("RiskyCustomPass")
    .DefaultSwitch(PassSwitch::kOff)
    .Stage(CustomPassStage::kBeforeInferShape);

// 不调用DefaultSwitch时默认为kOn
REGISTER_CUSTOM_PASS("SafeCustomPass")
    .Stage(CustomPassStage::kBeforeInferShape);
```
