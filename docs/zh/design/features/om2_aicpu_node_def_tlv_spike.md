# OM2 普通 AICPU NodeDef TLV 穿刺说明

本文只记录本次普通 AICPU 静态图穿刺样例。以下 AICPU tag 仍是评审草稿中的暂定值，不能视为已冻结公开接口。

## 已确认旧载荷布局

`libcpu_kernels.so` / `RunCpuKernel` 的 `TaskDef.kernel.args` 布局为：

```text
AicpuParamHead
uint64_t io_addrs[head.ioAddrNum]
uint32_t node_def_len
uint8_t node_def[node_def_len]
```

GE 生产证据在 `compiler/engines/cpu_engine/cpu_engine/common/kernel_builder/cpu_kernel_builder.cpp`：

- `BuildArgs` 和 `BuildMemCopyInfo` 按上述布局拼接 `args`。
- `BuildAndLaunchKernel` / `BuildMemCopyInfo` 设置 `kernel_def.args`、`args_size`、`so_name`、`kernel_name`。
- 默认 `so_name` 为 `libcpu_kernels.so`，默认 `kernel_name` 为 `RunCpuKernel`。

opbase 消费证据在 `aicpu_common/context/common/cpu_kernel_cache.cc`：

- `ParseIoAddr` 消费 `AicpuParamHead + ioAddr[] + uint32_t NodeDefLen + NodeDef`。
- `GetCpuKernelContext` 和 `GetCpuKernelContextWithBlock` 原先对 NodeDef 后缀执行 protobuf `ParseFromString`。

## 暂定 TLV 字段表

Frame 沿用 DumpWire v1：`magic:u32=0x544C5601`、`version:u16=1`、`header_size:u16=16`、`total_size:u32`、`record_count:u32`。record 为 `tag:u16 + payload_len:u32 + payload`，均为小端。

| tag | 名称 | payload |
| --- | --- | --- |
| `0x1000` | NodeDef root | 嵌套 record |
| `0x1001` | op | UTF-8 bytes |
| `0x1002` | input tensor | 嵌套 tensor record，可重复 |
| `0x1003` | output tensor | 嵌套 tensor record，可重复 |
| `0x1004` | attr | 嵌套 attr record，可重复 |
| `0x1011` | tensor.shape | 嵌套 shape record |
| `0x1012` | tensor.tensor_type | `i32` |
| `0x1013` | tensor.name | UTF-8 bytes |
| `0x1014` | tensor.data_ptr | `u64` |
| `0x1015` | tensor.data_size | `u64` |
| `0x1021` | shape.dims | `u32 count + i64[count]` |
| `0x1022` | shape.unknown_rank | `u8` |
| `0x1023` | shape.data_format | `i32` |
| `0x1031` | attr.name | UTF-8 bytes |
| `0x1032` | attr.value | `u16 value_type + typed payload` |

`attr.value` 的暂定 `value_type`：`0 empty`，`1 string/bytes`，`2 int64`，`3 float32`，`4 bool`，`5 data_type(i32)`，`6 shape`，`7 tensor`，`8 list_string`，`9 list_int64`，`10 list_float32`，`11 list_bool`，`12 list_data_type`，`13 list_shape`，`14 list_tensor`，`15 list_list_int64`。

`list_shape` 和 `list_list_int64` 需要同时保留。当前 GE 普通 AICPU builder 对 `VT_LIST_LIST_INT` 的历史实现写入 protobuf `AttrValue.array.shape`，即每个内层整数列表被表示为一个 `TensorShape`；因此本次穿刺将该现有语义映射到 `list_shape`，而不是强制改写成 `list_list_int64`。`list_list_int64` 仅表示 protobuf `AttrValue.list_list_int` oneof 的直接语义。空 `AttrValue` 用于保留当前“不支持的 GE 属性类型仍插入空 AttrValue 键”的语义。

## Golden

最小 NodeDef：`op = "Relu"`，无 input/output/attr。

```text
01 56 4C 54 01 00 10 00 20 00 00 00 01 00 00 00
00 10 0A 00 00 00 01 10 04 00 00 00 52 65 6C 75
```

解释：

- frame header：magic/version/header_size/total_size=32/record_count=1。
- root record：tag `0x1000`，payload 长度 10。
- op record：tag `0x1001`，payload `"Relu"`。
