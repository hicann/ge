# Qwen LLM Inference Sample

## Function Description

Demonstrates LLM loading, inference, and result retrieval based on an ONNX-format Qwen model. Given an input token sequence, performs one forward pass to output the predicted next token ID and updated KV Cache.

> This sample does not cover ONNX model export. To export an ONNX model, refer to: [Qwen Offline Model Export Example](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/ACL_PyTorch/built-in/nlp/Qwen2_for_Pytorch/readme.md)

## Quick Start

If your environment is ready, run the following commands to reproduce (replace `soc_version` as needed):

```bash
# 1. Create directory and download model (large file, please wait)
mkdir -p model
wget -O model/qwen.onnx "https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/cann_test/qwen.onnx"

# 2. Convert model (run in model/, adjust soc_version to match your chip)
cd model && atc --model=qwen.onnx --output=qwen --framework=5 \
  --soc_version=Ascend910B4-1 --precision_mode=must_keep_origin_dtype \
  --op_select_implmode=high_precision --external_weight=0 --output_type=FP32 \
  --input_shape='input_ids:1,512;past_key_0.key:1,2,512,64;past_key_0.value:1,2,512,64;past_key_1.key:1,2,512,64;past_key_1.value:1,2,512,64;past_key_2.key:1,2,512,64;past_key_2.value:1,2,512,64;past_key_3.key:1,2,512,64;past_key_3.value:1,2,512,64;past_key_4.key:1,2,512,64;past_key_4.value:1,2,512,64;past_key_5.key:1,2,512,64;past_key_5.value:1,2,512,64;past_key_6.key:1,2,512,64;past_key_6.value:1,2,512,64;past_key_7.key:1,2,512,64;past_key_7.value:1,2,512,64;past_key_8.key:1,2,512,64;past_key_8.value:1,2,512,64;past_key_9.key:1,2,512,64;past_key_9.value:1,2,512,64;past_key_10.key:1,2,512,64;past_key_10.value:1,2,512,64;past_key_11.key:1,2,512,64;past_key_11.value:1,2,512,64;past_key_12.key:1,2,512,64;past_key_12.value:1,2,512,64;past_key_13.key:1,2,512,64;past_key_13.value:1,2,512,64;past_key_14.key:1,2,512,64;past_key_14.value:1,2,512,64;past_key_15.key:1,2,512,64;past_key_15.value:1,2,512,64;past_key_16.key:1,2,512,64;past_key_16.value:1,2,512,64;past_key_17.key:1,2,512,64;past_key_17.value:1,2,512,64;past_key_18.key:1,2,512,64;past_key_18.value:1,2,512,64;past_key_19.key:1,2,512,64;past_key_19.value:1,2,512,64;past_key_20.key:1,2,512,64;past_key_20.value:1,2,512,64;past_key_21.key:1,2,512,64;past_key_21.value:1,2,512,64;past_key_22.key:1,2,512,64;past_key_22.value:1,2,512,64;past_key_23.key:1,2,512,64;past_key_23.value:1,2,512,64' \
  --log=error && cd ..

# 3. Build and run
bash scripts/build.sh && bash scripts/run.sh
```

## Directory Structure

```
3_sample_qwen_llm/
├── CMakeLists.txt              # Top-level build script
├── scripts/
│   ├── build.sh                # Build script
│   └── run.sh                  # Run script
├── src/
│   ├── acl.json                # ACL initialization config
│   ├── CMakeLists.txt          # Build configuration
│   └── sample_qwen_llm.cpp    # Main program
└── model/    [create manually]
    └── qwen.om                 # Qwen offline model converted by atc
```

## Environment Requirements

- [ ] Install `toolkit` and `ops` packages per [Environment Preparation](../../../docs/en/quick_install.md)
- [ ] Set environment variables: `source /usr/local/Ascend/cann/set_env.sh` (adjust path to your installation)

## Detailed Steps

### Step 1: Prepare Model

1. Download the Qwen ONNX model to the `model/` directory:

    ```bash
    mkdir -p model
    wget -O model/qwen.onnx "https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/cann_test/qwen.onnx"
    ```

    > If wget fails, open the URL in a browser to download and manually place the file in `model/`.

2. Convert the offline model (run atc in the `model/` directory):

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

    | Parameter | Description |
    |-----------|-------------|
    | `--precision_mode` | `must_keep_origin_dtype`: force keeping original data type to avoid precision loss |
    | `--op_select_implmode` | `high_precision`: prioritize computation precision |
    | `--external_weight` | `0`: weights are embedded in the om file, not stored separately |
    | `--soc_version` | Ascend AI processor version, see [Version List](https://hiascend.com/document/redirect/CannCommunityAtcSocVersion) |

    > If the generated om file has an architecture suffix (e.g., `qwen_linux_aarch64.om`), rename it to `qwen.om`.
    >
    > If you change the `--output` path, update the model path in the source code accordingly:
    > ```cpp
    > ret = sampleQwen.PrepareModel("../model/qwen.om");
    > ```

### Step 2: Build and Run

```bash
bash scripts/build.sh
bash scripts/run.sh
```

## Expected Output

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

## FAQ

**Q: Generated om file has an architecture suffix (e.g., `qwen_linux_aarch64.om`)**

```bash
mv model/qwen_linux_aarch64.om model/qwen.om
```

**Q: wget fails to download the Qwen model**

The file is large (~4GB+). If the network is unstable, download it via browser and manually place it in `model/`.

**Q: atc reports `soc_version` mismatch**

This sample defaults to `Ascend910B4-1`. Adjust according to your actual chip model, see [Version List](https://hiascend.com/document/redirect/CannCommunityAtcSocVersion).
