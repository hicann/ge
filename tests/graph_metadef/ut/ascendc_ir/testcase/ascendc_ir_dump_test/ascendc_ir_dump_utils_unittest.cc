/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/ascendc_ir/utils/ascendc_ir_dump_utils.h"
#include "stub_graph.h"
#include <iostream>
#include <fstream>
class UtestAscirDump : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};
using namespace ge;

TEST_F(UtestAscirDump, DumpAscirGraphTest) {
  AscGraph graph("test_graph");
  FaBeforeAutoFuse(graph);
  FaAfterScheduler(graph);
  FaAfterQueBufAlloc(graph);
  std::string res = R"(TilingKey: 1
Graph Name: test_graph
Axis:
    axis1: 
        name: b
        id: 0
        type: ORIGINAL
        bind_block: false
        size: B
        align: 1
        split_pair_other_id: 0
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis2: 
        name: n
        id: 1
        type: ORIGINAL
        bind_block: false
        size: N
        align: 1
        split_pair_other_id: 0
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis3: 
        name: g
        id: 2
        type: ORIGINAL
        bind_block: false
        size: G
        align: 1
        split_pair_other_id: 0
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis4: 
        name: s1
        id: 3
        type: ORIGINAL
        bind_block: false
        size: S1
        align: 1
        split_pair_other_id: 0
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis5: 
        name: s2
        id: 4
        type: ORIGINAL
        bind_block: false
        size: S2
        align: 1
        split_pair_other_id: 0
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis6: 
        name: d
        id: 5
        type: ORIGINAL
        bind_block: false
        size: D
        align: 1
        split_pair_other_id: 0
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis7: 
        name: l
        id: 6
        type: ORIGINAL
        bind_block: false
        size: 8
        align: 1
        split_pair_other_id: 0
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis8: 
        name: s1T
        id: 7
        type: TILE_OUTER
        bind_block: false
        size: Ceiling((S1 / (s1t_size)))
        align: 1
        from: {3, }
        split_pair_other_id: 8
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis9: 
        name: s1t
        id: 8
        type: TILE_INNER
        bind_block: false
        size: s1t_size
        align: 128
        from: {3, }
        split_pair_other_id: 7
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis10: 
        name: bngs1T
        id: 9
        type: MERGED
        bind_block: false
        size: (B * Ceiling((S1 / (s1t_size))) * G * N)
        align: 1
        from: {0, 1, 2, 7, }
        split_pair_other_id: 0
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis11: 
        name: bngs1TB
        id: 10
        type: BLOCK_OUTER
        bind_block: false
        size: Ceiling((B * Ceiling((S1 / (s1t_size))) * G * N / (bngs1Tb_size)))
        align: 1
        from: {9, }
        split_pair_other_id: 11
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis12: 
        name: bngs1Tb
        id: 11
        type: BLOCK_INNER
        bind_block: false
        size: bngs1Tb_size
        align: 1
        from: {9, }
        split_pair_other_id: 10
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis13: 
        name: s2T
        id: 12
        type: TILE_OUTER
        bind_block: false
        size: Ceiling((S2 / (s2t_size)))
        align: 1
        from: {4, }
        split_pair_other_id: 13
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis14: 
        name: s2t
        id: 13
        type: TILE_INNER
        bind_block: false
        size: s2t_size
        align: 256
        from: {4, }
        split_pair_other_id: 12
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis15: 
        name: s1tT
        id: 14
        type: TILE_OUTER
        bind_block: false
        size: Ceiling((s1t_size / (s1tt_size)))
        align: 1
        from: {8, }
        split_pair_other_id: 15
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis16: 
        name: s1tt
        id: 15
        type: TILE_INNER
        bind_block: false
        size: s1tt_size
        align: 1
        from: {8, }
        split_pair_other_id: 14
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis17: 
        name: s1tT2
        id: 16
        type: TILE_OUTER
        bind_block: false
        size: Ceiling((s1t_size / (s1tt2_size)))
        align: 1
        from: {8, }
        split_pair_other_id: 17
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
    axis18: 
        name: s1tt2
        id: 17
        type: TILE_INNER
        bind_block: false
        size: s1tt2_size
        align: 1
        from: {8, }
        split_pair_other_id: 16
        allow_oversize_axis: 0
        allow_unaligned_tail: 0
nodes:
    node1 info: 
        node name: query
        inputs: 
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 10, 11, 12, 8, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, s1t_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, D, 0, 1, 0, 
                vectorized_axis: 8, 13, 5, 
                vectorized_strides: X,Y,Z,
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 0
                    axis: 10, 11, 12, 8, 13, 5, 6, 
                    loop_axis: 12

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node2 info: 
        node name: key
        inputs: 
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 12, 8, 5, 13, 6, 
                repeats: (S2 / (s2t_size)), 1, D, s2t_size, 1, 
                strides: (D * s2t_size), 0, 1, D, 0, 
                vectorized_axis: 8, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 1
                    axis: 10, 11, 12, 8, 5, 13, 6, 
                    loop_axis: 12

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node3 info: 
        node name: bmm1
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 10, 11, 12, 8, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, s1t_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, D, 0, 1, 0, 
                vectorized_axis: 8, 13, 5, 
                vectorized_strides: X,Y,Z,
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 12, 8, 5, 13, 6, 
                repeats: (S2 / (s2t_size)), 1, D, s2t_size, 1, 
                strides: (D * s2t_size), 0, 1, D, 0, 
                vectorized_axis: 8, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 8, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), s1t_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, S2, 1, 0, 0, 
                vectorized_axis: 8, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 0
                    alloc_type: QUEUE
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 0
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 2
                    axis: 10, 11, 12, 8, 13, 5, 6, 
                    loop_axis: 12

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node4 info: 
        node name: load1
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 8, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), s1t_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, S2, 1, 0, 0, 
                vectorized_axis: 8, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 0
                    alloc_type: QUEUE
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 0
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 1
                    alloc_type: QUEUE
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 1
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 3
                    axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                    loop_axis: 14

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node5 info: 
        node name: pse
        inputs: 
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 0, 1, 2, 3, 4, 5, 6, 
                repeats: B, N, G, S1, S2, 1, 1, 
                strides: (G * N * S1 * S2), (G * S1 * S2), (S1 * S2), S2, 1, 0, 0, 
                vectorized_axis: 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 4
                    axis: 
                    loop_axis: -1

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node6 info: 
        node name: loadPse
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 0, 1, 2, 3, 4, 5, 6, 
                repeats: B, N, G, S1, S2, 1, 1, 
                strides: (G * N * S1 * S2), (G * S1 * S2), (S1 * S2), S2, 1, 0, 0, 
                vectorized_axis: 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 2
                    alloc_type: BUFFER
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 0
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 5
                    axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                    loop_axis: 14

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node7 info: 
        node name: castPse
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 2
                    alloc_type: BUFFER
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 0
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 3
                    alloc_type: BUFFER
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 6
                    axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                    loop_axis: 14

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node8 info: 
        node name: add1
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 1
                    alloc_type: QUEUE
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 1
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 3
                    alloc_type: BUFFER
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 4
                    alloc_type: QUEUE
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 1
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 7
                    axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                    loop_axis: 14

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node9 info: 
        node name: scaleValue
        inputs: 
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 
                repeats: 
                strides: 
                vectorized_axis: 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 8
                    axis: 
                    loop_axis: -1

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node10 info: 
        node name: mul1
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 4
                    alloc_type: QUEUE
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 1
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
            AscTensor: 
                DataType: DT_FLOAT
                axis: 
                repeats: 
                strides: 
                vectorized_axis: 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 5
                    alloc_type: QUEUE
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 1
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 9
                    axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                    loop_axis: 14

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node11 info: 
        node name: attenMask
        inputs: 
        outputs: 
            AscTensor: 
                DataType: DT_UINT8
                axis: 0, 1, 2, 3, 4, 5, 6, 
                repeats: B, 1, 1, S1, S2, 1, 1, 
                strides: (S1 * S2), (S1 * S2), (S1 * S2), S2, 1, 0, 0, 
                vectorized_axis: 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 10
                    axis: 
                    loop_axis: -1

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node12 info: 
        node name: loadAttenMask
        inputs: 
            AscTensor: 
                DataType: DT_UINT8
                axis: 0, 1, 2, 3, 4, 5, 6, 
                repeats: B, 1, 1, S1, S2, 1, 1, 
                strides: (S1 * S2), (S1 * S2), (S1 * S2), S2, 1, 0, 0, 
                vectorized_axis: 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_UINT8
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 12
                    alloc_type: BUFFER
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 2
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 11
                    axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                    loop_axis: 14

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node13 info: 
        node name: select
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 5
                    alloc_type: QUEUE
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 1
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
            AscTensor: 
                DataType: DT_UINT8
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 12
                    alloc_type: BUFFER
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 2
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 6
                    alloc_type: QUEUE
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 1
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 12
                    axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                    loop_axis: 14

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node14 info: 
        node name: softmaxExp
        inputs: 
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 12, 14, 15, 13, 5, 6, 
                repeats: 1, (s1t_size / (s1tt_size)), s1tt_size, 1, 1, 8, 
                strides: 0, (8 * s1tt_size), 8, 0, 0, 1, 
                vectorized_axis: 14, 15, 13, 5, 6, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 7
                    alloc_type: QUEUE
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 3
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 13
                    axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                    loop_axis: 14

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node15 info: 
        node name: softmaxApiTmpBuf
        inputs: 
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 12, 14, 15, 13, 5, 6, 
                repeats: (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 8
                    alloc_type: BUFFER
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 14
                    axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                    loop_axis: 14

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node16 info: 
        node name: flashSoftmax
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 6
                    alloc_type: QUEUE
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 1
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
            AscTensor: 
                DataType: DT_FLOAT
                axis: 12, 14, 15, 13, 5, 6, 
                repeats: 1, (s1t_size / (s1tt_size)), s1tt_size, 1, 1, 8, 
                strides: 0, (8 * s1tt_size), 8, 0, 0, 1, 
                vectorized_axis: 14, 15, 13, 5, 6, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 7
                    alloc_type: QUEUE
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 3
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
            AscTensor: 
                DataType: DT_FLOAT
                axis: 12, 14, 15, 13, 5, 6, 
                repeats: (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 8
                    alloc_type: BUFFER
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 9
                    alloc_type: QUEUE
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 1
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 8, 
                strides: (8 * bngs1Tb_size * s1t_size), (8 * s1t_size), 0, (8 * s1tt_size), 8, 0, 0, 1, 
                vectorized_axis: 14, 15, 13, 5, 6, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 10
                    alloc_type: BUFFER
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 4
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 8, 
                strides: (8 * bngs1Tb_size * s1t_size), (8 * s1t_size), 0, (8 * s1tt_size), 8, 0, 0, 1, 
                vectorized_axis: 14, 15, 13, 5, 6, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 11
                    alloc_type: QUEUE
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 2
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 15
                    axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                    loop_axis: 14

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node17 info: 
        node name: storeSoftmaxMax
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 8, 
                strides: (8 * bngs1Tb_size * s1t_size), (8 * s1t_size), 0, (8 * s1tt_size), 8, 0, 0, 1, 
                vectorized_axis: 14, 15, 13, 5, 6, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 11
                    alloc_type: QUEUE
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 2
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 8, 
                strides: (8 * bngs1Tb_size * s1t_size), (8 * s1t_size), 0, (8 * s1tt_size), 8, 0, 0, 1, 
                vectorized_axis: 14, 15, 13, 5, 6, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 26
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: 0
                    merge_scope: 0
        attr: 
            AscNode: 
                sched: 
                    exec_order: 16
                    axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                    loop_axis: 14

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node18 info: 
        node name: softmaxMax
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 8, 
                strides: (8 * bngs1Tb_size * s1t_size), (8 * s1t_size), 0, (8 * s1tt_size), 8, 0, 0, 1, 
                vectorized_axis: 14, 15, 13, 5, 6, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 26
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: 0
                    merge_scope: 0
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 
                repeats: 
                strides: 
                vectorized_axis: 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 17
                    axis: 
                    loop_axis: -1

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node19 info: 
        node name: dropMask
        inputs: 
        outputs: 
            AscTensor: 
                DataType: DT_UINT8
                axis: 0, 1, 2, 3, 4, 5, 6, 
                repeats: B, N, G, S1, S2, 1, 1, 
                strides: (G * N * S1 * S2), (G * S1 * S2), (S1 * S2), S2, 1, 0, 0, 
                vectorized_axis: 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 18
                    axis: 
                    loop_axis: -1

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node20 info: 
        node name: loadDropMask
        inputs: 
            AscTensor: 
                DataType: DT_UINT8
                axis: 0, 1, 2, 3, 4, 5, 6, 
                repeats: B, N, G, S1, S2, 1, 1, 
                strides: (G * N * S1 * S2), (G * S1 * S2), (S1 * S2), S2, 1, 0, 0, 
                vectorized_axis: 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_UINT8
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 13
                    alloc_type: BUFFER
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 3
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 19
                    axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                    loop_axis: 14

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node21 info: 
        node name: dropout
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 9
                    alloc_type: QUEUE
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 1
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
            AscTensor: 
                DataType: DT_UINT8
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 13
                    alloc_type: BUFFER
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 3
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 14
                    alloc_type: QUEUE
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 1
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 20
                    axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                    loop_axis: 14

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node22 info: 
        node name: castVec1Res
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 14
                    alloc_type: QUEUE
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 1
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 15
                    alloc_type: BUFFER
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 0
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 21
                    axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                    loop_axis: 14

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node23 info: 
        node name: storeVec1Res
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 15
                    alloc_type: BUFFER
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 0
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 14, 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 16
                    alloc_type: QUEUE
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 4
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 22
                    axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                    loop_axis: 14

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node24 info: 
        node name: value
        inputs: 
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 12, 8, 13, 5, 6, 
                repeats: (S2 / (s2t_size)), 1, s2t_size, D, 1, 
                strides: (D * s2t_size), 0, D, 1, 0, 
                vectorized_axis: 8, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 23
                    axis: 10, 11, 12, 8, 13, 5, 6, 
                    loop_axis: 12

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node25 info: 
        node name: bmm2
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 1, 
                strides: (S2 * bngs1Tb_size * s1t_size), (S2 * s1t_size), s2t_size, (S2 * s1tt_size), S2, 1, 0, 0, 
                vectorized_axis: 14, 15, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 16
                    alloc_type: QUEUE
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 4
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 12, 8, 13, 5, 6, 
                repeats: (S2 / (s2t_size)), 1, s2t_size, D, 1, 
                strides: (D * s2t_size), 0, D, 1, 0, 
                vectorized_axis: 8, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 8, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, s1t_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, D, 0, 1, 0, 
                vectorized_axis: 8, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 17
                    alloc_type: QUEUE
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 5
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 24
                    axis: 10, 11, 12, 8, 13, 5, 6, 
                    loop_axis: 12

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node26 info: 
        node name: load2
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 8, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, s1t_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, D, 0, 1, 0, 
                vectorized_axis: 8, 13, 5, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 17
                    alloc_type: QUEUE
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 5
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 18
                    alloc_type: BUFFER
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 25
                    axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                    loop_axis: 16

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node27 info: 
        node name: addResOut
        inputs: 
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 19
                    alloc_type: QUEUE
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 6
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 26
                    axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                    loop_axis: 16

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node28 info: 
        node name: loadAddResOut
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 19
                    alloc_type: QUEUE
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 6
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 20
                    alloc_type: BUFFER
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 5
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 27
                    axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                    loop_axis: 16

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node29 info: 
        node name: mulRes
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 20
                    alloc_type: BUFFER
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 5
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
            AscTensor: 
                DataType: DT_FLOAT
                axis: 12, 14, 15, 13, 5, 6, 
                repeats: 1, (s1t_size / (s1tt_size)), s1tt_size, 1, 1, 8, 
                strides: 0, (8 * s1tt_size), 8, 0, 0, 1, 
                vectorized_axis: 14, 15, 13, 5, 6, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 7
                    alloc_type: QUEUE
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 3
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 21
                    alloc_type: BUFFER
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 5
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 28
                    axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                    loop_axis: 16

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node30 info: 
        node name: addRes
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 18
                    alloc_type: BUFFER
                    position: VECIN
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 21
                    alloc_type: BUFFER
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 5
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 22
                    alloc_type: BUFFER
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 5
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 29
                    axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                    loop_axis: 16

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node31 info: 
        node name: div
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 22
                    alloc_type: BUFFER
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 5
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 14, 15, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, (S2 / (s2t_size)), (s1t_size / (s1tt_size)), s1tt_size, s2t_size, 1, 8, 
                strides: (8 * bngs1Tb_size * s1t_size), (8 * s1t_size), 0, (8 * s1tt_size), 8, 0, 0, 1, 
                vectorized_axis: 14, 15, 13, 5, 6, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 11
                    alloc_type: QUEUE
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: 2
                    depth: 2
                    buf_num: 2
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 23
                    alloc_type: BUFFER
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 5
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 30
                    axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                    loop_axis: 16

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node32 info: 
        node name: castBmm2Res
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 23
                    alloc_type: BUFFER
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 5
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 24
                    alloc_type: BUFFER
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 5
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: 24
        attr: 
            AscNode: 
                sched: 
                    exec_order: 31
                    axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                    loop_axis: 16

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node33 info: 
        node name: store
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 24
                    alloc_type: BUFFER
                    position: VECOUT
                    hardware: UB
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: 5
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: 24
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 25
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: 0
                    merge_scope: 0
        attr: 
            AscNode: 
                sched: 
                    exec_order: 32
                    axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                    loop_axis: 16

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node34 info: 
        node name: buf
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 25
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: 0
                    merge_scope: 0
        outputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 0, 1, 2, 3, 4, 5, 6, 
                repeats: B, N, G, S1, 1, D, 1, 
                strides: (D * G * N * S1), (D * G * S1), (D * S1), D, 0, 1, 0, 
                vectorized_axis: 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: -1
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 33
                    axis: 
                    loop_axis: -1

                Api: 
                    Api type: INVALID
                    Compute unit: INVALID
                    Compute type: INVALID

    node35 info: 
        node name: buf_
        inputs: 
            AscTensor: 
                DataType: DT_FLOAT16
                axis: 10, 11, 12, 16, 17, 13, 5, 6, 
                repeats: (B * G * N * S1 / (bngs1Tb_size * s1t_size)), bngs1Tb_size, 1, (s1t_size / (s1tt2_size)), s1tt2_size, 1, D, 1, 
                strides: (D * bngs1Tb_size * s1t_size), (D * s1t_size), 0, (D * s1tt2_size), D, 0, 1, 0, 
                vectorized_axis: 17, 5, 13, 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 25
                    alloc_type: GLOBAL
                    position: GM
                    hardware: GM
                    buf_ids: 
                    name: 
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: 0
                    merge_scope: 0
        outputs: 
            AscTensor: 
                DataType: DT_DUAL_SUB_UINT8
                axis: 0, 1, 2, 3, 4, 5, 6, 
                repeats: B, N, G, S1, 1, D, 1, 
                strides: (D * G * N * S1), (D * G * S1), (D * S1), D, 0, 1, 0, 
                vectorized_axis: 
                vectorized_strides: 
                MemAttr: 
                    tensor_id: 1
                    alloc_type: L1
                    position: GM
                    hardware: UB
                    buf_ids: 1, 2, 3, 4, 5, 
                    name: Mem_
                MemQueAttr: 
                    id: -1
                    depth: -1
                    buf_num: -1
                    name: 
                MemBufAttr: 
                    id: -1
                    name: 
                MemOptAttr: 
                    reuse_id: -1
                    ref_tensor: -1
                    merge_scope: -1
        attr: 
            AscNode: 
                sched: 
                    exec_order: 34
                    axis: 1, 2, 3, 4, 5, 
                    loop_axis: 3

                Api: 
                    Api type: BUFFER
                    Compute unit: MTE1
                    Compute type: REDUCE

)";
  //EXPECT_EQ(res, ge::DumpAscirGraph::DumpGraph(graph));
  ge::DumpAscirGraph::WriteOutToFile("../ascendc_ir_dump_test/dump_graph.txt", graph);
}
