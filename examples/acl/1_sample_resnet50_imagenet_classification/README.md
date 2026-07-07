# ResNet-50 图片分类样例

## 功能描述

基于 ONNX ResNet-50 网络（单输入、单 Batch）实现图片分类。加载离线模型对 2 张测试图片进行推理，输出 Top-5 置信度的类别标识。

## 快速开始

如果环境已就绪，可按以下命令快速复现（请根据实际情况替换 `soc_version`）：

```bash
# 1. 创建目录并下载模型和图片
mkdir -p model data
wget -O model/resnet50_Opset16.onnx "https://github.com/onnx/models/raw/main/Computer_Vision/resnet50_Opset16_timm/resnet50_Opset16.onnx"
wget -O data/dog1_1024_683.jpg "https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog1_1024_683.jpg"
wget -O data/dog2_1024_683.jpg "https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog2_1024_683.jpg"

# 2. 转换模型
cd model && atc --model=resnet50_Opset16.onnx --framework=5 --output=resnet50 --soc_version=Ascend910B1 --input_format=NCHW --output_type=FP32 && cd ..

# 3. 转换图片
cd data && python3 ../scripts/transfer_pic.py && cd ..

# 4. 编译运行
bash scripts/build.sh && bash scripts/run.sh
```

## 目录结构

```
1_sample_resnet50_imagenet_classification/
├── CMakeLists.txt                          # 顶层编译脚本
├── scripts/
│   ├── build.sh                            # 编译脚本
│   ├── run.sh                              # 运行脚本
│   └── transfer_pic.py                     # 图片预处理（jpg → bin，缩放至 224×224）
├── src/
│   ├── acl.json                            # ACL 初始化配置
│   ├── CMakeLists.txt                      # 编译配置
│   └── sample_resnet50_imagenet_classification.cpp  # 主程序
├── model/        [需手动创建]
│   └── resnet50.om                         # atc 转换后的离线模型
└── data/         [需手动创建]
    ├── dog1_1024_683.jpg                   # 测试图片 1
    ├── dog2_1024_683.jpg                   # 测试图片 2
    ├── dog1_1024_683.bin                   # 预处理后的 bin 文件（由 transfer_pic.py 生成）
    └── dog2_1024_683.bin                   # 预处理后的 bin 文件（由 transfer_pic.py 生成）
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

2. 转换离线模型（在 `model/` 目录下执行 atc，以 Atlas A2 系列为例）：

    ```bash
    cd model
    atc --model=resnet50_Opset16.onnx --framework=5 --output=resnet50 \
      --soc_version=Ascend910B1 --input_format=NCHW --output_type=FP32
    cd ..
    ```

    | 参数 | 说明 |
    |------|------|
    | `--framework` | 框架类型：0=Caffe, 1=MindSpore, 3=TensorFlow, 5=ONNX |
    | `--soc_version` | 昇腾 AI 处理器版本，参考 [版本列表](https://hiascend.com/document/redirect/CannCommunityAtcSocVersion) |
    | `--output_type` | 输出数据类型，此处为 FP32 |

    > 若修改 `--output` 路径，需同步修改源码中的 `omModelPath`：
    > ```cpp
    > const char* omModelPath = "../model/resnet50.om";
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
[INFO] load model ../model/resnet50.om success
[INFO] start to process file:../data/dog1_1024_683.bin
[INFO] model execute success
[INFO] top 1: index[161] value[xxxxxx]
[INFO] top 2: index[xxx] value[xxxxxx]
...
[INFO] top 5: index[xxx] value[xxxxxx]
[INFO] output data success
[INFO] start to process file:../data/dog2_1024_683.bin
[INFO] model execute success
[INFO] top 1: index[267] value[xxxxxx]
...
[INFO] output data success
[INFO] SAMPLE PASSED
```

| 类别标识 | 对应类别（ImageNet） |
|---------|---------------------|
| 161 | basset, basset hound |
| 267 | standard poodle |

> 具体数值可能因版本和环境不同而有差异。标签与类别的对应关系基于 ImageNet 数据集。

## 常见问题

**Q: 执行 transfer_pic.py 报错 `ModuleNotFoundError: No module named 'PIL'`**

```bash
pip3 install Pillow --user
```

**Q: atc 转换报错找不到模型文件**

确保在 `model/` 目录下执行 atc 命令，或在 `--model` 参数中使用正确的相对/绝对路径。

**Q: 编译报错找不到头文件**

确认已执行 `source /usr/local/Ascend/cann/set_env.sh`，环境变量 `ASCEND_HOME_PATH` 已正确设置。
