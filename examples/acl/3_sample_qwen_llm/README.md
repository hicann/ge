# Qwen LLM 推理样例

## 功能描述

基于 ONNX 格式的 Qwen 大语言模型，演示 LLM 的加载、推理和结果获取。给定输入 token 序列，执行一次前向推理，输出预测的下一个 token ID 及 KV Cache。

> 本样例不涉及 ONNX 模型的导出。如需导出 ONNX 模型，参考：[Qwen 离线模型导出示例](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/ACL_PyTorch/built-in/nlp/Qwen2_for_Pytorch/readme.md)

## 快速开始

如果环境已就绪，可按以下命令快速复现（请根据实际情况替换 `soc_version`）：

```bash
# 1. 创建目录并下载模型（文件较大，请耐心等待）
mkdir -p model
wget -O model/qwen.onnx "https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/cann_test/qwen.onnx"

# 2. 转换模型（在 model/ 下执行，注意 soc_version 需与实际芯片匹配）
cd model && atc --model=qwen.onnx --output=qwen --framework=5 \
  --soc_version=Ascend910B4-1 --precision_mode=must_keep_origin_dtype \
  --op_select_implmode=high_precision --external_weight=0 --output_type=FP32 \
  --input_shape='input_ids:1,512;past_key_0.key:1,2,512,64;past_key_0.value:1,2,512,64;past_key_1.key:1,2,512,64;past_key_1.value:1,2,512,64;past_key_2.key:1,2,512,64;past_key_2.value:1,2,512,64;past_key_3.key:1,2,512,64;past_key_3.value:1,2,512,64;past_key_4.key:1,2,512,64;past_key_4.value:1,2,512,64;past_key_5.key:1,2,512,64;past_key_5.value:1,2,512,64;past_key_6.key:1,2,512,64;past_key_6.value:1,2,512,64;past_key_7.key:1,2,512,64;past_key_7.value:1,2,512,64;past_key_8.key:1,2,512,64;past_key_8.value:1,2,512,64;past_key_9.key:1,2,512,64;past_key_9.value:1,2,512,64;past_key_10.key:1,2,512,64;past_key_10.value:1,2,512,64;past_key_11.key:1,2,512,64;past_key_11.value:1,2,512,64;past_key_12.key:1,2,512,64;past_key_12.value:1,2,512,64;past_key_13.key:1,2,512,64;past_key_13.value:1,2,512,64;past_key_14.key:1,2,512,64;past_key_14.value:1,2,512,64;past_key_15.key:1,2,512,64;past_key_15.value:1,2,512,64;past_key_16.key:1,2,512,64;past_key_16.value:1,2,512,64;past_key_17.key:1,2,512,64;past_key_17.value:1,2,512,64;past_key_18.key:1,2,512,64;past_key_18.value:1,2,512,64;past_key_19.key:1,2,512,64;past_key_19.value:1,2,512,64;past_key_20.key:1,2,512,64;past_key_20.value:1,2,512,64;past_key_21.key:1,2,512,64;past_key_21.value:1,2,512,64;past_key_22.key:1,2,512,64;past_key_22.value:1,2,512,64;past_key_23.key:1,2,512,64;past_key_23.value:1,2,512,64' \
  --log=error && cd ..

# 3. 编译运行
bash scripts/build.sh && bash scripts/run.sh
```

## 目录结构

```
3_sample_qwen_llm/
├── CMakeLists.txt              # 顶层编译脚本
├── scripts/
│   ├── build.sh                # 编译脚本
│   └── run.sh                  # 运行脚本
├── src/
│   ├── acl.json                # ACL 初始化配置
│   ├── CMakeLists.txt          # 编译配置
│   └── sample_qwen_llm.cpp    # 主程序
└── model/    [需手动创建]
    └── qwen.om                 # atc 转换后的 Qwen 离线模型
```

## 环境准备

- [ ] 按 [环境准备](../../../docs/zh/quick_install.md) 安装 `toolkit` 和 `ops` 包
- [ ] 设置环境变量：`source /usr/local/Ascend/cann/set_env.sh`（路径按实际安装位置调整）

## 详细步骤

### Step 1：准备模型

1. 下载 Qwen ONNX 模型到 `model/` 目录：

    ```bash
    mkdir -p model
    wget -O model/qwen.onnx "https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/cann_test/qwen.onnx"
    ```

    > 如果 wget 失败，可在浏览器中打开上述链接下载后手动放入 `model/` 目录。

2. 转换离线模型（在 `model/` 目录下执行 atc）：

    ```bash
    cd model
    atc \
      --model=./qwen.onnx \
      --output=./qwen \
      --soc_version=Ascend910B4-1 \
      --framework=5 \
      --precision_mode=must_keep_origin_dtype \
      --op_select_implmode=high_precision \
      --external_weight=0 \
      --output_type=FP32 \
      --input_shape='input_ids:1,512;past_key_0.key:1,2,512,64;past_key_0.value:1,2,512,64;past_key_1.key:1,2,512,64;past_key_1.value:1,2,512,64;past_key_2.key:1,2,512,64;past_key_2.value:1,2,512,64;past_key_3.key:1,2,512,64;past_key_3.value:1,2,512,64;past_key_4.key:1,2,512,64;past_key_4.value:1,2,512,64;past_key_5.key:1,2,512,64;past_key_5.value:1,2,512,64;past_key_6.key:1,2,512,64;past_key_6.value:1,2,512,64;past_key_7.key:1,2,512,64;past_key_7.value:1,2,512,64;past_key_8.key:1,2,512,64;past_key_8.value:1,2,512,64;past_key_9.key:1,2,512,64;past_key_9.value:1,2,512,64;past_key_10.key:1,2,512,64;past_key_10.value:1,2,512,64;past_key_11.key:1,2,512,64;past_key_11.value:1,2,512,64;past_key_12.key:1,2,512,64;past_key_12.value:1,2,512,64;past_key_13.key:1,2,512,64;past_key_13.value:1,2,512,64;past_key_14.key:1,2,512,64;past_key_14.value:1,2,512,64;past_key_15.key:1,2,512,64;past_key_15.value:1,2,512,64;past_key_16.key:1,2,512,64;past_key_16.value:1,2,512,64;past_key_17.key:1,2,512,64;past_key_17.value:1,2,512,64;past_key_18.key:1,2,512,64;past_key_18.value:1,2,512,64;past_key_19.key:1,2,512,64;past_key_19.value:1,2,512,64;past_key_20.key:1,2,512,64;past_key_20.value:1,2,512,64;past_key_21.key:1,2,512,64;past_key_21.value:1,2,512,64;past_key_22.key:1,2,512,64;past_key_22.value:1,2,512,64;past_key_23.key:1,2,512,64;past_key_23.value:1,2,512,64' \
      --log=error
    cd ..
    ```

    | 参数 | 说明 |
    |------|------|
    | `--precision_mode` | `must_keep_origin_dtype`：强制保持原始数据类型，避免精度损失 |
    | `--op_select_implmode` | `high_precision`：优先保证计算精度 |
    | `--external_weight` | `0`：权重内嵌在 om 文件中，不分离存储 |
    | `--soc_version` | 昇腾 AI 处理器版本，参考 [版本列表](https://hiascend.com/document/redirect/CannCommunityAtcSocVersion) |

    > 若生成的 om 文件带有架构后缀（如 `qwen_linux_aarch64.om`），需重命名为 `qwen.om`。
    >
    > 若修改 `--output` 路径，需同步修改源码中的模型路径：
    > ```cpp
    > ret = sampleQwen.PrepareModel("../model/qwen.om");
    > ```

### Step 2：编译运行

```bash
bash scripts/build.sh
bash scripts/run.sh
```

## 预期输出

```
[INFO] The sample starts to run
[INFO]  SAMPLE start to execute.
[INFO]  acl init success
[INFO]  set device success
[INFO]  create context success
[INFO]  create stream success
[INFO]  load model ../model/qwen.om success.
[INFO]  Start to Process.
[INFO]  The first five inputs information:
[INFO]    Input[0], tensorName=input_ids, size=4096 bytes, dtype=9, format=0, dims=1 512
[INFO]    Input[1], tensorName=past_key_0.key, size=131072 bytes, dtype=1, format=0, dims=1 2 512 64
[INFO]    Input[2], tensorName=past_key_0.value, size=131072 bytes, dtype=1, format=0, dims=1 2 512 64
[INFO]    Input[3], tensorName=past_key_1.key, size=131072 bytes, dtype=1, format=0, dims=1 2 512 64
[INFO]    Input[4], tensorName=past_key_1.value, size=131072 bytes, dtype=1, format=0, dims=1 2 512 64
[INFO]  Start to execute model.
[INFO]  The first five outputs information and the predicted token id:
[INFO]    Output[0], tensorName=/lm_head/MatMul:0:logits, size=311164928 bytes, dtype=0, format=0, dims=1 512 151936
[INFO]    Output[1], tensorName=/model/self_attn/Concat_5:0:present_0.key, size=524288 bytes, dtype=0, format=0, dims=1 2 1024 64
[INFO]    Output[2], tensorName=/model/self_attn/Concat_6:0:present_0.value, size=524288 bytes, dtype=0, format=0, dims=1 2 1024 64
[INFO]    Output[3], tensorName=/model/self_attn_1/Concat_5:0:present_1.key, size=524288 bytes, dtype=0, format=0, dims=1 2 1024 64
[INFO]    Output[4], tensorName=/model/self_attn_1/Concat_6:0:present_1.value, size=524288 bytes, dtype=0, format=0, dims=1 2 1024 64
[INFO]  predicted_token_id: 33975
[INFO]  SAMPLE PASSED.
```

## 常见问题

**Q: 生成的 om 文件名称带架构后缀（如 `qwen_linux_aarch64.om`）**

```bash
mv model/qwen_linux_aarch64.om model/qwen.om
```

**Q: wget 下载 Qwen 模型失败**

文件较大（约 4GB+），网络不稳定时可用浏览器下载后手动放入 `model/` 目录。

**Q: atc 转换报错 `soc_version` 不匹配**

本样例默认使用 `Ascend910B4-1`，请根据实际芯片型号调整，参考 [版本列表](https://hiascend.com/document/redirect/CannCommunityAtcSocVersion)。
