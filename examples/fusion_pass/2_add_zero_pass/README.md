# 样例使用指导

## 功能描述

本样例为删除加零操作的自定义pass样例，
提供在线推理与atc工具离线编译模型两种方式演示框架如何调用自定义pass完成图优化。
本样例使用eager style api和融合接口实现。

## 目录结构

```
├── src
│   ├──add_zero_pass.cpp         // pass实现文件 
├── CMakeLists.txt               // 编译脚本
├── data         
|   ├──torch_gen_onnx.py         // torch脚本用于导出onnx
|   ├──torch_forward.py          // torch脚本用于在线推理
|—— gen_es_api
|   |——CMakeLists.txt            // 生成eager style api的编译脚本
```

## 环境要求

- 使用python及其依赖库版本：python>=3.8 、pytorch>=2.1
- 已完成[昇腾AI软件栈在开发环境上的部署](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)

## 实现步骤

1. 定义类`AddZeroPass`继承`PatternFusionPass`。
2. 重写基类`PatternFusionPass`中的3个函数：
   - `Patterns`定义匹配模板，用于在整图中获取与该模板相同的拓扑。
   - `MeetRequirements`对模板匹配到的拓扑进行筛选。
   - `Replacement`定义替换部分。
3. 注册`AddZeroPass`为自定义融合pass，执行阶段为BeforeInferShape。


## 程序编译

假设CANN软件包的安装目录为INSTALL_PATH, 例如`/home/HwHiAiUser/Ascend/`。

1. 配置环境变量。

   运行软件包中设置环境变量脚本，命令如下：

   ```
   source ${ASCEND_PATH}/set_env.sh
   ```

   `${ASCEND_PATH}`为CANN软件包安装目录下的cann路径。请替换相关软件包的实际安装路径，例如`${INSTALL_PATH}/cann`。

2. 根据实际情况修改**CMakeLists.txt**文件中的如下信息。

   - ASCEND_PATH：可以设置默认的软件包路径，如果通过set_env.sh设置了`$ASCEND_HOME_PATH`，无需修改。

   - target_include_directories：需要包含的头文件，对于本示例，无需修改。如果是用户自行开发的代码，当需要添加头文件时，在示例下方直接增加行即可，注意不要删除原有项目。如果网络中有自定义算子，请增加自定义算子的原型定义头文件。

   - target_link_libraries：需要链接的库，对于本示例，无需修改。如果是用户自行开发的代码，当需要添加链接库时，在示例下方直接增加行即可，注意不要删除原有项目。

     > 禁止链接软件包中的其他so，否则后续升级可能会导致兼容性问题。

3. 执行如下命令 生成eager style api

   依次执行:

   ```
   mkdir build && cd build
   cmake ..
   make generate_es_api
   ```
   执行后，在**build**目录下产生generated_ops目录，内含es构图api的头文件及源码

4. 临时步骤，执行如下命令生成es api对应的so，并拷贝到run包安装路径下。可以在make后增加参数`-j<N>`用于并行执行构建任务，`N`推荐选择CPU核心数或CPU核心数+1。
   ```
   make esb_generated_shared
   ```
   编译完成后，在**build**/gen_es_api目录下生成libesb_generated_shared.so
   将so拷贝到ASCEND_PATH下lib64目录下。
   ```
   cp gen_es_api/libesb_generated_shared.so ${ASCEND_PATH}/x86_64-linux/lib64
   ```
   若为arm环境，目标路径调整为
   ```
   cp gen_es_api/libesb_generated_shared.so ${ASCEND_PATH}/aarch64-linux/lib64
   ```

5. 执行如下命令编译自定义pass so，并将编译后的动态库文件libadd_zero_pass.so拷贝到自定义融合pass目录下，其中“xxx”为用户自定义目录。
   ```
   make
   cp ./libadd_zero_pass.so ${ASCEND_PATH}/opp/vendors/xxx/custom_fusion_passes/libadd_zero_pass.so
   ```

## 程序运行

1. 配置环境变量(如已执行，跳过)。

   - 运行软件包中设置环境变量脚本，命令如下：

      ```
      source ${ASCEND_PATH}/set_env.sh
      ```

   `${ASCEND_PATH}`请替换相关软件包的实际安装路径。

2. 使用ATC离线推理。

   - 设置环境变量，dump出编译过程中的模型图：
     ```
     export DUMP_GE_GRAPH=1
     ```
   - 进入data目录执行.py文件导出onnx（文件中使用了torch的onnx导出器，依赖额外的Python包onnx，运行前确保安装）：
     ```
     python torch_gen_onnx.py
     ```
   - 执行结束后，在data目录下生成.onnx格式的模型文件，名称为model.onnx。
   - 执行ATC工具命令(关于ATC工具的详细说明，请前往[昇腾社区](www.hiascend.com)搜索ATC离线模型编译工具)，`soc_version`请根据实际环境修改：
     ```
     atc --model=./model.onnx --framework=5 --soc_version=xxx --output=./model
     ```
   - 日志中出现如下打印：
     ```
     Define pattern for AddZeroPass
     Define MeetRequirements for AddZeroPass
     Define replacement for AddZeroPass
     ```

3. 在线推理
   - 设置环境变量，dump出编译过程中的模型图：
      ```
      export DUMP_GE_GRAPH=1
      ```
   - 进入data目录执行.py文件进行在线推理（在线推理请确保已安装torch_npu插件）：
      ```
      python torch_forward.py
      ```  
   - 日志中出现如下打印：
     ```
     Define pattern for AddZeroPass
     Define MeetRequirements for AddZeroPass
     Define replacement for AddZeroPass
     ```

4. 查看运行结果

   - 执行完成后，目录下生成一系列.pbtxt文件。
     对比以下dump图：
      - `ge_onnx_xxxxx_PreRunBegin.pbtxt`执行前dump图
      - `ge_onnx_xxxxx_RunCustomPassBeforeInferShape.pbtxt`执行InferShape前的自定义pass dump图

   可以发现模型已按预期优化，即加零节点被删除。

