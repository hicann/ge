# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
TileLang Add Kernel + 编译产出 .so

本文件用 TileLang 实现 element-wise Add kernel，并编译产出 Ascend .so 交付件。
产出的 .so 供 GE 自定义算子 (EagerExecuteOp) 通过 dlopen + dlsym("call") 加载并调用。

TileLang-Ascend 编译后的 .so 导出函数签名为:
  extern "C" void call(uint8_t* A_handle, uint8_t* B_handle, uint8_t* C_handle, aclrtStream stream)
内部封装了 main_kernel<<<>>> 的 launch 逻辑。
"""

import os
import shutil

import tilelang
import tilelang.language as T

N = 4096
BLOCK_SIZE = 1024


@tilelang.jit(out_idx=[-1])
def vec_add(n, block_size, dtype="float"):
    m_num = n // block_size
    vec_num = 2

    @T.prim_func
    def main(
        a: T.Tensor((n,), dtype),
        b: T.Tensor((n,), dtype),
        c: T.Tensor((n,), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_size // vec_num,), dtype)
            b_ub = T.alloc_ub((block_size // vec_num,), dtype)
            c_ub = T.alloc_ub((block_size // vec_num,), dtype)
            with T.Scope("V"):
                T.copy(a[cid * block_size + vid * block_size // vec_num], a_ub)
                T.copy(b[cid * block_size + vid * block_size // vec_num], b_ub)

                T.barrier_all()
                T.tile.add(c_ub, a_ub, b_ub)
                T.barrier_all()

                T.copy(c_ub, c[cid * block_size + vid * block_size // vec_num])

    return main


func = vec_add(N, BLOCK_SIZE)

adapter = func.adapter
so_path = adapter.lib._name

output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "add_kernel.so")
shutil.copy2(so_path, output_path)

print(f"Kernel .so saved to: {output_path}")
print(f"File size: {os.path.getsize(output_path)} bytes")
print("Export function: call(uint8_t* A, uint8_t* B, uint8_t* C, aclrtStream stream)")
