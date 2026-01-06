# 源码构建

### 1. 安装依赖

GE支持源码编译。在源码编译前，请根据实际情况选择 **方式一（手动安装）** 或 **方式二（Docker容器）** 完成基础环境准备，然后进行 **CANN Toolkit** 的安装。

#### 方式一：手动安装依赖

   以下所列为GE源码编译用到的依赖，请注意版本要求。

   - GCC >= 7.3.x

   - Python3 >= 3.9.x

     除了CANN开发套件包需要的Python依赖外，还需要额外安装coverage，并将Python3的bin路径添加到PATH环境变量中，命令示例如下：

     ```bash
     pip3 install coverage
     # 修改下面的PYTHON3_HOME为实际的PYTHON安装目录
     export PATH=$PATH:$PYTHON3_HOME/bin
     ```

   - CMake >= 3.16.0 （建议使用3.20.0版本）
   - bash >= 5.1.16 
   - ccache/asan/autoconf/automake/libtool/gperf/graph-easy(其中graph-easy可选)

     ```bash
     # Ubuntu/Debian操作系统安装命令示例如下，其他操作系统请自行安装
     # asan以gcc 7.5.0版本为例安装的是libasan4，其他版本请安装对应版本asan
     sudo apt-get install cmake ccache bash lcov libasan4 autoconf automake libtool gperf libgraph-easy-perl
     ```

   - python三方库依赖。

     ```bash
     # 编译依赖的python三方库在源码根目录下的requirements.txt
     pip3 install -r requirements.txt
     ```

#### 方式二：使用 Docker 镜像

  **配套 X86 构建镜像地址**：`swr.cn-north-4.myhuaweicloud.com/ci_cann/ubuntu20.04.05_x86:lv1_latest`
  
  **配套 ARM 构建镜像地址**：`swr.cn-north-4.myhuaweicloud.com/ci_cann/ubuntu20.04.05_arm:lv1_latest`

  以下是推荐的使用方式，可供参考:

  ```shell
  image=${根据本地机器架构类型从上面选择配套的构建镜像地址}

  # 1. 拉取配套构建镜像
  docker pull ${image}
  # 2. 创建容器
  docker run --name env_for_ge_build --cap-add SYS_PTRACE -d -it ${image} /bin/bash
  # 3. 启动容器
  docker start env_for_ge_build
  # 4. 进入容器
  docker exec -it env_for_ge_build /bin/bash
  ```
  完成后可以进入[安装软件包](#2-安装软件包)章节。

  > [!NOTE]说明
  > - `--cap-add SYS_PTRACE`：创建Docker容器时添加`SYS_PTRACE`权限，以支持[本地验证](#5-本地验证utst)时的内存泄漏检测功能。
  > - 更多 docker 选项介绍请通过 `docker --help` 查询。

### 2. 安装软件包

#### 步骤一：安装社区版CANN Toolkit包

根据实际环境，下载对应`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`包，下载链接为[toolkit x86_64包](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/x86_64/Ascend-cann-toolkit_8.5.0-beta.1_linux-x86_64.run)、[toolkit aarch64包](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/aarch64/Ascend-cann-toolkit_8.5.0-beta.1_linux-aarch64.run)。

  ```bash
  # 确保安装包具有可执行权限
  chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run 
  # 安装命令(其中--install-path为可选)
  ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
  ```

-   `${cann_version}`：表示CANN包版本号。
-   `${arch}`：表示CPU架构，如`aarch64`、`x86_64`。
-   `${install_path}`：表示指定安装路径，可选，默认安装在/usr/local/Ascend目录，指定路径安装时，指定的路径权限需设置为755。

#### 步骤二：安装社区版CANN ops包（运行样例依赖）

运行样例时必须安装该软件包，若仅编译源码，可跳过本操作。

根据实际产品型号和环境架构，获取对应的`Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run`包，本操作以获取Atlas A2系列产品软件包为例，下载链接为[ops_x86_64](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/x86_64/Ascend-cann-910b-ops_8.5.0-beta.1_linux-x86_64.run)、[ops_aarch64](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/aarch64/Ascend-cann-910b-ops_8.5.0-beta.1_linux-aarch64.run)。

  ```bash
  # 确保安装包具有可执行权限
  chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
  # 安装命令
  ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
  ```

- `${soc_name}`：表示NPU型号名称，即$\{soc\_version\}删除“ascend”后剩余的内容。
- `${install_path}`：表示指定安装路径，需要与Toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

#### 步骤三：配置环境变量

根据实际场景，选择合适的命令：

  ```bash
  # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}） 
  source /usr/local/Ascend/cann/set_env.sh
  # 指定路径安装
  source ${install_path}/cann/set_env.sh
  ```

### 3. 下载源码

开发者可通过如下命令下载本仓源码：

  ```bash
  # 下载项目源码，以master分支为例 
  git clone https://gitcode.com/cann/ge.git
  ```

### 4. 编译

`GE`提供一键式编译能力，进入代码仓根目录，可通过如下命令进行编译：

  ```bash
  bash build.sh --<pkg_type>
  ```

--<pkg_type>（可选）：表示子包类型，取值包括`ge_compiler`、`ge_executor`与`dflow`，不同编译参数编译不同子包，不设置则同时编译三种子包。

更多编译参数可以通过`bash build.sh -h`查看；编译成功后，会在`build_out`目录下生成`cann-<component>_<version>_<arch>.run`软件包。

- `<component>`表示子包名称，取值包括ge-compiler、ge-executor与dflow-executor。
- `<version>`表示版本号。
- `<arch>`表示操作系统架构，取值包括x86_64与aarch64。

#### 4.1 关于签名的补充说明
* 编译产生`cann-dflow-executor_<version>_<arch>.run`软件包中含有`cann-udf-compat.tar.gz`(UDF兼容升级包)。
* `cann-udf-compat.tar.gz`会在业务启动时加载至Device，加载过程中默认会由驱动进行安全验签，确保包可信。
* 开发者下载本仓源码自行编译产生`cann-udf-compat.tar.gz` 并不含签名头，为此需要关闭驱动安全验签的机制。
* 关闭验签方式：
  配套使用HDK 25.5.T2.B001或以上版本，并通过该HDK配套的npu-smi工具关闭验签。参考如下命令，以root用户在物理机上执行。  
  以device 0为例：  
  npu-smi set -t custom-op-secverify-enable -i ***0*** -d 1     # 使能验签配置  
  npu-smi set -t custom-op-secverify-mode -i ***0*** -d 0      # 关闭客户自定义验签

### 5. 本地验证（UT/ST）

编译完成后，用户可以进行开发者测试。

- 编译执行`UT`测试用例：

  ```bash
  #编译执行所有的UT测试用例
  bash tests/run_test.sh --ut
  #编译执行特定的UT测试用例（推荐）
  bash tests/run_test.sh --ut=${TARGET}
  ```
  --ut（必选）：可以指定`${TARGET}`编译特定对象的ut测试用例，取值可通过`bash tests/run_test.sh -h`查看。

- 编译执行`ST`测试用例：

  ```bash
  #编译执行所有的ST测试用例
  bash tests/run_test.sh --st
  #编译执行特定的ST测试用例（推荐）
  bash tests/run_test.sh --st=${TARGET}
  ```
  --st（必选）：可以指定`${TARGET}`编译特定对象的st测试用例，取值可通过`bash tests/run_test.sh -h`查看。


- 统计代码覆盖率:

  使用 `tests/run_test.sh` 脚本的 `-c` 参数可以在测试用例运行过程中生成代码覆盖率统计文件。

  **前置条件**：
  - 确保 `lcov` 工具已正确安装
  - 编译运行环境上的 `gcc` 和 `gcov` 必须是配套版本

  **使用方法**：
  ```bash
  bash tests/run_test.sh -c [其他参数]
  ```

  **输出位置**：生成的覆盖率文件位于代码根目录下的 `cov/` 目录中。


- 清理产物：

`UT/ST`测试用例编译输出目录为`build_ut`和`build_st`，如果想清除历史编译记录，可执行如下操作：

```bash
rm -rf build_ut/ build_st/ output/ build/ build_out/ cov/
```

> [!NOTE]说明
> `tests/run_test.sh`脚本支持的详细命令参数可通过`bash tests/run_test.sh -h`查看。


### 6. 安装与卸载

- 安装

  本地验证完成后，可执行如下命令安装编译生成的`GE`软件包，执行安装命令时，请确保安装用户对软件包具有可执行权限。

  ```shell
  ./cann-<component>_<version>_<arch>.run --full --install-path=${install_path}
  ```

  > [!CAUTION]注意
  > 此处的安装路径（无论默认还是指定）需与前面安装toolkit包时的路径保持一致。安装完成后，用户编译生成的`GE`软件包会替换已安装CANN开发套件包中的`GE`相关软件。
 

- 卸载

  若您想卸载安装的`cann-<component>_<version>_<arch>.run`软件包，可执行如下命令。

  ```shell
  ./cann-<component>_<version>_<arch>.run --uninstall --install-path=${install_path}
  ```

  执行时需要将上述命令中的软件包名称替换为实际的自定义`cann-<component>_<version>`软件包名称。

安装完成后，可以参考[样例执行](../examples/README.md)运行样例。
