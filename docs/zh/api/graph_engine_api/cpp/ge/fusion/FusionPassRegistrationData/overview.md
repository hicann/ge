# 简介

融合Pass注册信息类。

## 需要包含的头文件

```c++
#include <ge/fusion/pass/fusion_pass_reg.h>
```

## Public成员函数

```c++
FusionPassRegistrationData() = default
explicit FusionPassRegistrationData(const AscendString &pass_name)
AscendString GetPassName() const
FusionPassRegistrationData &Stage(CustomPassStage stage)
CustomPassStage GetStage() const
FusionPassRegistrationData &CreatePassFn(const CreateFusionPassFn &create_fusion_pass_fn)
CreateFusionPassFn GetCreatePassFn() const
FusionPassRegistrationData &DefaultSwitch(PassSwitch pass_switch)
PassSwitch GetDefaultSwitch() const
AscendString ToString() const
```
