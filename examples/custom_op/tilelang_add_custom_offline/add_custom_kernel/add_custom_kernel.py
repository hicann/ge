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
供 GE 自定义算子 (CompilableOp::Compile) 在 GE/ATC 编译阶段通过 subprocess 调用本脚本完成在线编译。

用法:
  python3 add_custom_kernel.py [N] [output_path]

参数:
  N            - 元素总数（默认 4096，需为 BLOCK_SIZE 的整数倍）
  output_path  - 产出 .so 的路径（默认脚本目录下 add_kernel.so）

TileLang-Ascend 编译后的 .so 导出函数签名为:
  extern "C" void call(uint8_t* A_handle, uint8_t* B_handle, uint8_t* C_handle, aclrtStream stream)
内部封装了 main_kernel<<<>>> 的 launch 逻辑。
"""

import os
import shutil
import sys

import tilelang
import tilelang.language as T

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


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if n % BLOCK_SIZE != 0:
        print(
            f"Error: N={n} must be a multiple of BLOCK_SIZE={BLOCK_SIZE}",
            file=sys.stderr,
        )
        sys.exit(1)

    func = vec_add(n, BLOCK_SIZE)

    adapter = func.adapter
    so_path = getattr(getattr(adapter, "lib", None), "_name", None)
    if so_path is None:
        print(
            "Error: cannot locate compiled .so path from tilelang adapter",
            file=sys.stderr,
        )
        sys.exit(1)

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "add_kernel.so"
        )

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(so_path, output_path)

    print(f"Kernel .so saved to: {output_path}")
    print(f"N={n}, BLOCK_SIZE={BLOCK_SIZE}")
    print(f"File size: {os.path.getsize(output_path)} bytes")
    print(
        "Export function: call(uint8_t* A, uint8_t* B, uint8_t* C, aclrtStream stream)"
    )


if __name__ == "__main__":
    main()
