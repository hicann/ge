# ResNet-50 动态 Batch 图片分类样例

## 功能描述

基于 ONNX ResNet-50 网络，使用**动态 Batch** 特性实现多 Batch 图片分类。与样例 1（固定单 Batch）不同，本样例将 2 张图片作为一个 Batch 同时送入模型推理，输出各自的 Top-5 分类结果。

## 快速开始

如果环境已就绪，可按以下命令快速复现（请根据实际情况替换 `soc_version`）：

```bash
# 1. 创建目录并下载模型和图片
mkdir -p model data
wget -O model/resnet50_Opset16.onnx "https://github.com/onnx/models/raw/main/Computer_Vision/resnet50_Opset16_timm/resnet50_Opset16.onnx"
wget -O data/dog1_1024_683.jpg "https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog1_1024_683.jpg"
wget -O data/dog2_1024_683.jpg "https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog2_1024_683.jpg"

# 2. 转换模型（动态 Batch，需指定 input_shape 和 dynamic_batch_size）
cd model && atc --model=resnet50_Opset16.onnx --framework=5 --output=resnet50_dynamic_batch \
  --soc_version=Ascend910B1 --input_format=NCHW --output_type=FP32 \
  --input_shape="x:-1,3,224,224" --dynamic_batch_size="1,2,4,8" && cd ..

# 3. 转换图片
cd data && python3 ../scripts/transfer_pic.py && cd ..

# 4. 编译运行
bash scripts/build.sh && bash scripts/run.sh
```

## 目录结构

```
2_sample_resnet50_imagenet_classification_dynamic_batch/
├── CMakeLists.txt                          # 顶层编译脚本
├── scripts/
│   ├── build.sh                            # 编译脚本
│   ├── run.sh                              # 运行脚本
│   └── transfer_pic.py                     # 图片预处理（jpg → bin，缩放至 224×224）
├── src/
│   ├── acl.json                            # ACL 初始化配置
│   ├── CMakeLists.txt                      # 编译配置
│   └── sample_resnet50_imagenet_classification_dynamic_batch.cpp  # 主程序
├── model/        [需手动创建]
│   └── resnet50_dynamic_batch.om           # atc 转换后的动态 Batch 离线模型
└── data/         [需手动创建]
    ├── dog1_1024_683.jpg                   # 测试图片 1
    ├── dog2_1024_683.jpg                   # 测试图片 2
    ├── dog1_1024_683.bin                   # 预处理后的 bin（由 transfer_pic.py 生成）
    └── dog2_1024_683.bin                   # 预处理后的 bin（由 transfer_pic.py 生成）
```

## 环境准备

- [ ] 按 [环境准备](../../../docs/zh/quick_install.md#1-环境准备) 安装 `toolkit` 和 `ops` 包
- [ ] 设置环境变量：`source /usr/local/Ascend/cann/set_env.sh`（路径按实际安装位置调整）
- [ ] 安装 Python 依赖：`pip3 install Pillow numpy --user`

## 详细步骤

### Step 1：准备模型

1. 下载 ONNX 模型文件到 `model/` 目录：

    ```bash
    mkdir -p model
    wget -O model/resnet50_Opset16.onnx \
      "https://github.com/onnx/models/raw/main/Computer_Vision/resnet50_Opset16_timm/resnet50_Opset16.onnx"
    ```

2. 转换动态 Batch 离线模型（在 `model/` 目录下执行 atc，以 Atlas A2 系列为例）：

    ```bash
    cd model
    atc --model=resnet50_Opset16.onnx --framework=5 --output=resnet50_dynamic_batch \
      --soc_version=Ascend910B1 --input_format=NCHW --output_type=FP32 \
      --input_shape="x:-1,3,224,224" --dynamic_batch_size="1,2,4,8"
    cd ..
    ```

    | 参数 | 说明 |
    |------|------|
    | `--input_shape` | 指定输入 shape，Batch 维度设为 `-1` 表示可变 |
    | `--dynamic_batch_size` | 支持的 Batch size 列表，运行时自动适配 |
    | `--framework` | 框架类型：0=Caffe, 1=MindSpore, 3=TensorFlow, 5=ONNX |
    | `--soc_version` | 昇腾 AI 处理器版本，参考 [版本列表](https://hiascend.com/document/redirect/CannCommunityAtcSocVersion) |

    > 若修改 `--output` 路径，需同步修改源码中的 `omModelPath`：
    > ```cpp
    > const char* omModelPath = "../model/resnet50_dynamic_batch.om";
    > ```

### Step 2：准备测试图片

1. 下载测试图片到 `data/` 目录：

    ```bash
    mkdir -p data
    wget -O data/dog1_1024_683.jpg "https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog1_1024_683.jpg"
    wget -O data/dog2_1024_683.jpg "https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog2_1024_683.jpg"
    ```

2. 在 `data/` 目录下执行图片预处理（脚本扫描当前目录的 `*.jpg` 文件）：

    ```bash
    cd data
    python3 ../scripts/transfer_pic.py
    cd ..
    ```

    执行后在 `data/` 下生成 2 个 `*.bin` 文件。

### Step 3：编译运行

```bash
bash scripts/build.sh
bash scripts/run.sh
```

## 预期输出

```
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] load model ../model/resnet50_dynamic_batch.om success
[INFO] start to process file:../data/dog1_1024_683.bin
[INFO] start to process file:../data/dog2_1024_683.bin
[INFO] model execute success
[INFO] Result of picture 1:
[INFO] top 1: index[162] value[xxxxxx]
[INFO] top 2: index[161] value[xxxxxx]
[INFO] top 3: index[166] value[xxxxxx]
[INFO] top 4: index[167] value[xxxxxx]
[INFO] top 5: index[163] value[xxxxxx]
[INFO] Result of picture 2:
[INFO] top 1: index[267] value[xxxxxx]
[INFO] top 2: index[266] value[xxxxxx]
[INFO] top 3: index[265] value[xxxxxx]
[INFO] top 4: index[153] value[xxxxxx]
[INFO] top 5: index[99] value[xxxxxx]
[INFO] output data success
[INFO] SAMPLE PASSED
```

| 类别标识 | 对应类别（ImageNet） |
|---------|---------------------|
| 161 | basset, basset hound |
| 267 | standard poodle |

> 具体数值可能因版本和环境不同而有差异。标签与类别的对应关系基于 ImageNet 数据集。

## 常见问题

**Q: 与样例 1 的区别是什么？**

样例 1 是固定单 Batch，每次只能处理 1 张图片；本样例使用 `--dynamic_batch_size` 参数转换模型，运行时可将多张图片打包为一个 Batch 同时推理，提高吞吐。

**Q: 如何修改运行时 Batch size？**

修改源码中将多张图片送入模型的逻辑，Batch size 需在 `--dynamic_batch_size` 指定的列表中（1/2/4/8）。

**Q: 执行 transfer_pic.py 报错 `ModuleNotFoundError: No module named 'PIL'`**

```bash
pip3 install Pillow --user
```

**Q: atc 转换报错找不到模型文件**

确保在 `model/` 目录下执行 atc 命令，或在 `--model` 参数中使用正确的相对/绝对路径。
