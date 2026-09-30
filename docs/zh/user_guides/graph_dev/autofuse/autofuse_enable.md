# 启用AutoFuse

根据融合实现路径的不同，AutoFuse有两种启用方式：

- **GE路径**：介绍基于GE路径启用AutoFuse自动融合的方法，以及依赖版本、环境变量等。
- **Inductor路径**：介绍基于Inductor路径（PyTorch框架）启用AutoFuse自动融合的方法，以及依赖版本、`torch.compile`配置、环境变量等。

下面分别介绍两种场景的启用方式。

## GE路径启用AutoFuse

### 前提条件

#### 搭建运行环境

| 依赖项                        | 要求                                                         |
| :---------------------------- | :----------------------------------------------------------- |
| 硬件与基础软件                | 准备搭载昇腾AI处理器的硬件环境，并安装匹配的驱动固件和CANN软件包。安装步骤请参见《[CANN软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)》。 |
| TensorFlow和TF Adapter插件 | 参见[配套版本](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/latest/releasenote/releasenote_01.html)选择与CANN匹配的版本。<br>**前端框架为TensorFlow时，需要安装该依赖项。** |
| PyTorch和TorchNPU插件      | 参见[配套版本](https://www.hiascend.com/document/detail/zh/Pytorch/latest/releasenote/docs/zh/release_notes/release_notes.md)选择与CANN匹配的版本。<br> **PyTorch模型经TorchAir编译后进入GE路径时，需要安装该依赖项。** |
| TorchAir                      | 使用与PyTorch/TorchNPU配套的版本，安装及源码编译要求请参见[TorchAir 官方仓库](https://gitcode.com/Ascend/torchair)。<br>**PyTorch模型经TorchAir编译后进入GE路径时，需要安装该依赖项。** |
| GCC                           | 9.5.0及以上，建议9.5.0。                                   |
| CMake                         | 3.20.0及以上，建议3.20.0。                                 |

#### 设置环境变量

安装CANN软件后，使用CANN运行用户进行编译和运行时，需以CANN运行用户登录环境，执行如下命令设置环境变量：

```bash
source ${INSTALL_DIR}/set_env.sh
```

$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。

### 启用AutoFuse

GE路径下，通过环境变量启用AutoFuse：

```bash
export AUTOFUSE_FLAGS="--enable_autofuse=true"
```

配置`--enable_autofuse=true`后，即可开启基础AutoFuse融合功能（最简配置），支持Elemwise算子与Broadcast算子之间的自动融合。

更多配置请参见[环境变量参考](../appendix/autofuse_env_vars.md)。

## Inductor路径启用AutoFuse

### 前提条件

#### 搭建运行环境

| 依赖项                   | 要求                                                         |
| :----------------------- | :----------------------------------------------------------- |
| 硬件与基础软件           | 准备搭载昇腾AI处理器的硬件环境，并安装匹配的驱动固件和CANN软件包。安装步骤请参见《[CANN软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)》。 |
| PyTorch和TorchNPU插件 | 参见[配套版本](https://www.hiascend.com/document/detail/zh/Pytorch/latest/releasenote/docs/zh/release_notes/release_notes.md)选择与CANN匹配的版本。 |
| GCC                      | 9.5.0及以上，建议9.5.0。                                   |
| CMake                    | 3.20.0及以上，建议3.20.0。                                 |

#### 设置环境变量

安装CANN软件后，使用CANN运行用户进行编译和运行时，需以CANN运行用户登录环境，执行如下命令设置环境变量：

```bash
source ${INSTALL_DIR}/set_env.sh
```

$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。

### 启用AutoFuse

- Inductor路径下，在`torch.compile`中指定Ascend C后端启用AutoFuse：

  ```python
  model = torch.compile(
     model,
     options={"npu_backend": "ascendc"},
  )
  ```

  其中，`options={"npu_backend": "ascendc"}`用于选择Ascend C后端并启用AutoFuse。

- 使用装饰器形式启用AutoFuse：

  ```python
  @torch.compile(options={"npu_backend": "ascendc"})
  def test_add_ge(x, y, z):
      return torch.ge(torch.add(x, y), z)
  ```

- 通过设置环境变量`TORCHINDUCTOR_NPU_BACKEND`启用AutoFuse：

  ```bash
  export TORCHINDUCTOR_NPU_BACKEND="ascendc"
  ```

  设置该环境变量后，直接使用`torch.compile`编译模型即可，无需再通过`options`指定后端：

  ```python
  model = torch.compile(model)
  ```

​ 更多环境变量请参见[环境变量参考](../appendix/autofuse_env_vars.md)。
