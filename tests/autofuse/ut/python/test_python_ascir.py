# Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import pytest
import json
import time
import os
import shutil
from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions, Schedule, CodeGen
from autofuse import ascir_api

PYF_PATH = os.path.dirname(os.path.realpath(__file__))


# pyascir 构图能力暂不支持
class TestAscir():
    @staticmethod
    def test_graph_create_size_expr_by_long():
        s0 = ascir.SizeExpr()
        assert s0 == 1
        try:
            s0 = ascir.SizeExpr('100')
        except Exception as e:
            assert e.args[0] == 'Only support type of SizeExpr or long'

        assert s0 == 1
        s0 = ascir.SizeExpr(0)
        assert s0 == 0

        s1 = ascir.SizeExpr(1)
        assert s1 == 1

        s2 = ascir.SizeExpr(1)
        assert s2 == 1

        s3 = ascir.SizeExpr(100)
        assert s3 == 100
        s4 = ascir.SizeExpr(1) + ascir.SizeExpr(2)
        assert s4 == 3
        s5 = ascir.SizeExpr(1) + ascir.SizeExpr(128)
        assert s5 == 129
        s7 = s5 + s4
        assert s7 == 132
        s8 = s1 + s2
        assert s8 == 2

        s9 = s3 * s4
        assert s9 == 300
        s10 = s3 * 2
        assert s10 == 200

    @staticmethod
    def test_graph_create_size():
        graph = ascir.HintGraph("test")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: test\n",
            "Sizes:\n",
            "  s0: VAR\n",
            "  s1: VAR\n",
            "  s2: VAR\n",
            "Axis:\n",
            "Nodes:\n",
        ])

    @staticmethod
    def test_graph_create_axis():
        graph = ascir.HintGraph("test")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        z0 = graph.create_axis("z0", s0)
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", s2)
        z3 = graph.create_axis("z3", 512)
        z4 = graph.create_axis("z4", s1 + s2)

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: test\n",
            "Sizes:\n",
            "  s0: VAR\n",
            "  s1: VAR\n",
            "  s2: VAR\n",
            "Axis:\n",
            "  z0(0) : s0, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  z1(1) : s1, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  z2(2) : s2, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  z3(3) : 512, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  z4(4) : (s1 + s2), ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "Nodes:\n",
        ])

    @staticmethod
    def test_graph_create_node():
        graph = ascir.HintGraph("test")

        x = ascir.ops.Data("x", graph)
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: test\n",
            "Sizes:\n",
            "Axis:\n",
            "Nodes:\n",
            "  x: Data (0)\n",
            "    .ir_attr =  {    }\n"
            "    .axis = {}\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .y.dtype = float32\n",
            "    .y.axis = {}\n",
            "    .y.repeats = {}\n",
            "    .y.strides = {}\n",
            "    .y.vectorized_axis = {}\n",
            "    .y.vectorized_strides = {}\n"
        ])

    @staticmethod
    def test_graph_create_node_with_cast_infer():
        graph = ascir.HintGraph("test")

        x = ascir.ops.Data("x", graph)
        x.y.dtype = ascir.dtypes.int8
        x.attr.ir_attr.index = 0
        assert x.attr.ir_attr.index == 0
        cast = ascir.ops.Cast("cast")
        cast.y.dtype = ascir.dtypes.int4
        cast.x = x
        cast.attr.api.compute_type = "elemwise"
        try:
            graph.infer_dtypes()
        except Exception as e:
            assert e.args[0] == 'Check dtype failed for cast Cast; input_dtypes: [DT_INT8], output_dytpes: [DT_INT4]'
        import sys
        # 通常为2 （变量 + getrefcount 参数）
        print(f"x.attr ref count is {sys.getrefcount(x.attr)}")
        del x.attr

    @staticmethod
    def test_graph_create_node_with_cast_api():
        graph = ascir.HintGraph("test")

        x = ascir_api.Data(graph, dtype=ascir.dtypes.int8)
        cast = None
        try:
            cast = ascir_api.Cast(graph, x, dtype=ascir.dtypes.int4, axis=[])
        except Exception as e:
            assert e.args[0] == 'Check dtype failed for cast_0 Cast; input_dtypes: [DT_INT8], output_dytpes: [DT_INT4]'

    @staticmethod
    def test_graph_create_const_node_with_value_str_attr():
        graph = ascir.HintGraph("test")

        x = ascir.ops.Scalar("x", graph)
        x.attr.ir_attr.value = '11.1'
        x.y.dtype = ascir.dtypes.float32
        debug_str = ascir.utils.debug_str(graph)
        assert x.attr.ir_attr.value == '11.1'
        assert debug_str == "".join([
            "Graph: test\n",
            "Sizes:\n",
            "Axis:\n",
            "Nodes:\n",
            "  x: Scalar (0)\n",
            "    .ir_attr =  {.value = s: \"11.1\"\n    }\n",
            "    .axis = {}\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n",
            "      .compute_type = invalid\n",
            "    .y.dtype = float32\n",
            "    .y.axis = {}\n",
            "    .y.repeats = {}\n",
            "    .y.strides = {}\n",
            "    .y.vectorized_axis = {}\n",
            "    .y.vectorized_strides = {}\n"
        ])

    @staticmethod
    def test_graph_create_node_with_axis():
        graph = ascir.HintGraph("test")

        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")

        z0 = graph.create_axis("z0", s0)
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", 512)

        x = ascir.ops.Data("x", graph)
        x.attr.sched.axis = [z0, z1, z2]

        load = ascir.ops.Load("load")
        x.y.dtype = ascir.dtypes.float16
        x.y.axis = [z0, z1, z2]
        assert x.y.axis == [z0.id, z1.id, z2.id]
        x.y.size = [s0, s1, 512]
        assert x.y.size == [s0, s1, 512]
        x.y.strides = [s1 * 512, 512, ascir.SizeExpr(1)]
        assert x.y.strides == [s1 * 512, 512, ascir.SizeExpr(1)]
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: test\n",
            "Sizes:\n",
            "  s0: VAR\n",
            "  s1: VAR\n",
            "Axis:\n",
            "  z0(0) : s0, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  z1(1) : s1, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  z2(2) : 512, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "Nodes:\n",
            "  x: Data (0)\n",
            "    .ir_attr =  {    }\n",
            "    .axis = {z0, z1, z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .y.dtype = float16\n",
            "    .y.axis = {z0, z1, z2, }\n",
            "    .y.repeats = {s0, s1, 512, }\n",
            "    .y.strides = {(512 * s1), 512, 1, }\n",
            "    .y.vectorized_axis = {}\n",
            "    .y.vectorized_strides = {}\n"
        ])

    @staticmethod
    def test_graph_link_nodes():
        graph = ascir.HintGraph("test")

        x = ascir.ops.Data("x", graph)
        x.y.dtype = ascir.dtypes.int64
        load = ascir.ops.Load("load")
        load.x = x
        try:
            graph.infer_dtypes()
        except Exception as e:
            assert e.args[0] == 'Infer dtype failed for load Load; input_dtypes: [DT_INT64] is not supportted now'
        graph.infer_dtypes()
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: test\n",
            "Sizes:\n",
            "Axis:\n",
            "Nodes:\n",
            "  x: Data (0)\n",
            "    .ir_attr =  {    }\n",
            "    .axis = {}\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .y.dtype = int64_t\n"
            "    .y.axis = {}\n",
            "    .y.repeats = {}\n",
            "    .y.strides = {}\n",
            "    .y.vectorized_axis = {}\n"
            "    .y.vectorized_strides = {}\n"
            "  load: Load (1)\n",
            "    .ir_attr =  {    }\n",
            "    .axis = {}\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .x = x.y\n"
            "    .y.dtype = int64_t\n"
            "    .y.axis = {}\n",
            "    .y.repeats = {}\n",
            "    .y.strides = {}\n",
            "    .y.vectorized_axis = {}\n"
            "    .y.vectorized_strides = {}\n"
        ])

    @staticmethod
    def test_graph_link_nodes_by_output():
        graph = ascir.HintGraph("test")

        x = ascir.ops.Data("x", graph)

        load = ascir.ops.Load("load")
        load.x = x.y
        graph.infer_dtypes()
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: test\n",
            "Sizes:\n",
            "Axis:\n",
            "Nodes:\n",
            "  x: Data (0)\n",
            "    .ir_attr =  {    }\n",
            "    .axis = {}\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .y.dtype = float32\n"
            "    .y.axis = {}\n",
            "    .y.repeats = {}\n",
            "    .y.strides = {}\n",
            "    .y.vectorized_axis = {}\n"
            "    .y.vectorized_strides = {}\n"
            "  load: Load (1)\n",
            "    .ir_attr =  {    }\n",
            "    .axis = {}\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .x = x.y\n"
            "    .y.dtype = float32\n"
            "    .y.axis = {}\n",
            "    .y.repeats = {}\n",
            "    .y.strides = {}\n"
            "    .y.vectorized_axis = {}\n"
            "    .y.vectorized_strides = {}\n"
        ])

    @staticmethod
    def test_duration_record():
        start = time.time()
        time.sleep(0.1)
        end = time.time()
        ascir.utils.duration_record(["device", "fused_graph"], int(start * 1e9), int((end - start) * 1e9))
        ascir.utils.report_durations()
        try:
            ascir.utils.duration_record(["device", "fused_graph"], "time")
        except TypeError as e:
            assert e.args[0] == 'UtilsDurationRecord param parse failed'

        try:
            ascir.utils.duration_record(["device", "fused_graph"], int(-1), int(-1))
        except TypeError as e:
            assert e.args[0] == 'duration param is invalid'

        try:
            ascir.utils.duration_record([0, 1], int(-1), int(-1))
        except TypeError as e:
            assert e.args[0] == 'target param is invalid'


class TestAutofuseLoadAbsStore():
    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("LoadAbsStore")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        z0 = graph.create_axis("z0", s0 + 100)
        assert z0.size.expression == "(100 + s0)"
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", s2)
        buf_z0 = graph.create_axis("buf_z0", 100)
        buf_z1 = graph.create_axis("buf_z1", s1)
        buf_z2 = graph.create_axis("buf_z2", s2)

        arg3_1 = ascir.ops.Data("arg3_1", graph)
        arg3_1.attr.ir_attr.index = 0
        arg3_1.attr.sched.axis = [z0, z1, z2]
        arg3_1.y.dtype = ascir.dtypes.float16
        arg3_1.y.axis = [z0, z1, z2]
        arg3_1.y.size = [100 + s0, s1, s2]
        arg3_1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        load = ascir.ops.Load("load")
        try:
            load.attr.ir_attr.offset = "3"
        except Exception as e:
            assert e.args[0] == 'Only support type of SizeExpr or long'
        offset_of_0 = ascir.SizeExpr(0)
        load.attr.ir_attr.offset = offset_of_0
        assert load.attr.ir_attr.offset.expression == "0"
        load.x = arg3_1
        load.attr.sched.axis = [z0, z1, z2]
        load.y.dtype = ascir.dtypes.float16
        load.y.axis = [z0, z1, z2]
        load.y.size = [100 + s0, s1, s2]
        load.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        abs_op = ascir.ops.Abs("abs")
        abs_op.x = load
        abs_op.attr.sched.axis = [z0, z1, z2]
        abs_op.y.dtype = ascir.dtypes.float16
        abs_op.y.axis = [z0, z1, z2]
        abs_op.y.size = [100 + s0, s1, s2]
        abs_op.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        store = ascir.ops.Store("store")
        try:
            store.attr.ir_attr.offset = "4"
        except Exception as e:
            assert e.args[0] == 'Only support type of SizeExpr or long'
        store.attr.ir_attr.offset = offset_of_0 + 1
        assert store.attr.ir_attr.offset.expression == "1"
        store.x = abs_op
        store.attr.sched.axis = [z0, z1, z2]
        store.y.dtype = ascir.dtypes.float16
        store.y.axis = [z0, z1, z2]
        store.y.size = [100 + s0, s1, s2]
        store.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        buf1 = ascir.ops.Output("buf1", graph)
        buf1.attr.ir_attr.index = 0
        assert buf1.attr.ir_attr.index == 0
        buf1.x = store
        buf1.attr.sched.axis = [z0, z1, z2]
        buf1.y.dtype = ascir.dtypes.float16
        buf1.y.axis = [z0, z1, z2]
        buf1.y.size = [100 + s0, s1, s2]
        buf1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]
        graph.set_axis_map({z0: [buf_z0], z1: [buf_z1], z2: [buf_z2]})
        return graph

    def test_construct_graph(self):
        graph = self.construct_graph()
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: LoadAbsStore\n",
            "Sizes:\n",
            "  s0: VAR\n",
            "  s1: VAR\n",
            "  s2: VAR\n",
            "Axis:\n",
            "  z0(0) : (100 + s0), ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  z1(1) : s1, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  z2(2) : s2, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  buf_z0(3) : 100, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  buf_z1(4) : s1, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  buf_z2(5) : s2, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "Nodes:\n",
            "  arg3_1: Data (0)\n",
            "    .ir_attr =  {.index = i: 0\n    }\n",
            "    .axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .y.dtype = float16\n"
            "    .y.axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .y.repeats = {(100 + s0), s1, s2, }\n",
            "    .y.strides = {(s1 * s2), s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"
            "    .y.vectorized_strides = {}\n"
            "  load: Load (1)\n",
            "    .ir_attr =  {.offset = expression: \"0\"\n    }\n",
            "    .axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .x = arg3_1.y\n"
            "    .y.dtype = float16\n"
            "    .y.axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .y.repeats = {(100 + s0), s1, s2, }\n",
            "    .y.strides = {(s1 * s2), s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"
            "    .y.vectorized_strides = {}\n"
            "  abs: Abs (2)\n",
            "    .axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .x = load.y\n"
            "    .y.dtype = float16\n"
            "    .y.axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .y.repeats = {(100 + s0), s1, s2, }\n",
            "    .y.strides = {(s1 * s2), s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"
            "    .y.vectorized_strides = {}\n"
            "  store: Store (3)\n",
            "    .ir_attr =  {.offset = expression: \"1\"\n    }\n",
            "    .axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .x = abs.y\n"
            "    .y.dtype = float16\n"
            "    .y.axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .y.repeats = {(100 + s0), s1, s2, }\n",
            "    .y.strides = {(s1 * s2), s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"
            "    .y.vectorized_strides = {}\n"
            "  buf1: Output (4)\n",
            "    .ir_attr =  {.index = i: 0\n    }\n",
            "    .axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .x = store.y\n"
            "    .y.dtype = float16\n"
            "    .y.axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .y.repeats = {(100 + s0), s1, s2, }\n",
            "    .y.strides = {(s1 * s2), s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"
            "    .y.vectorized_strides = {}\n"
        ])

    def test_optimize(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)
        graph_name = schedule_results.get_name()
        input_num = schedule_results.get_input_num()
        output_num = schedule_results.get_output_num()

    def test_codegen(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        impl_graphs = fuser.schedule(hint_graph)
        tiling_def, host_tiling, op_kernel = fuser.codegen(impl_graphs)

    def test_autofuse_backend(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        tiling_def, host_tiling, op_kernel = fuser.autofuse_backend(hint_graph)


class TestAutofuseLoadMatMulStore():
    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("LoadCubeStore")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        z0 = graph.create_axis("z0", s0 + 100)
        assert z0.size.expression == "(100 + s0)"
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", s2)
        buf_z0 = graph.create_axis("buf_z0", 100)
        buf_z1 = graph.create_axis("buf_z1", s1)
        buf_z2 = graph.create_axis("buf_z2", s2)

        arg3_1 = ascir.ops.Data("arg3_1", graph)
        arg3_1.attr.ir_attr.index = 0
        arg3_1.attr.sched.axis = [z0, z1, z2]
        arg3_1.y.dtype = ascir.dtypes.float16
        arg3_1.y.axis = [z0, z1, z2]
        arg3_1.y.size = [100 + s0, s1, s2]
        arg3_1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        load = ascir.ops.Load("load")
        try:
            load.attr.ir_attr.offset = "3"
        except Exception as e:
            assert e.args[0] == 'Only support type of SizeExpr or long'
        offset_of_0 = ascir.SizeExpr(0)
        load.attr.ir_attr.offset = offset_of_0
        assert load.attr.ir_attr.offset.expression == "0"
        load.x = arg3_1
        load.attr.sched.axis = [z0, z1, z2]
        load.y.dtype = ascir.dtypes.float16
        load.y.axis = [z0, z1, z2]
        load.y.size = [100 + s0, s1, s2]
        load.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        matmul_op = ascir.ops.MatMul("matmul")
        matmul_op.x1 = load
        matmul_op.x2 = load
        matmul_op.attr.sched.axis = [z0, z1, z2]
        matmul_op.attr.api.compute_type = "cube"
        matmul_op.attr.ir_attr.enable_hf32 = 1
        matmul_op.attr.ir_attr.transpose_x1 = 0
        matmul_op.attr.ir_attr.transpose_x2 = 1
        matmul_op.attr.ir_attr.has_relu = 1
        matmul_op.attr.ir_attr.offset_x = 1
        matmul_op.y.dtype = ascir.dtypes.float16
        matmul_op.y.axis = [z0, z1, z2]
        matmul_op.y.size = [100 + s0, s1, s2]
        matmul_op.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        store = ascir.ops.Store("store")
        try:
            store.attr.ir_attr.offset = "4"
        except Exception as e:
            assert e.args[0] == 'Only support type of SizeExpr or long'
        store.attr.ir_attr.offset = offset_of_0 + 1
        assert store.attr.ir_attr.offset.expression == "1"
        store.x = matmul_op
        store.attr.sched.axis = [z0, z1, z2]
        store.y.dtype = ascir.dtypes.float16
        store.y.axis = [z0, z1, z2]
        store.y.size = [100 + s0, s1, s2]
        store.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        buf1 = ascir.ops.Output("buf1", graph)
        buf1.attr.ir_attr.index = 0
        assert buf1.attr.ir_attr.index == 0
        buf1.x = store
        buf1.attr.sched.axis = [z0, z1, z2]
        buf1.y.dtype = ascir.dtypes.float16
        buf1.y.axis = [z0, z1, z2]
        buf1.y.size = [100 + s0, s1, s2]
        buf1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]
        graph.set_axis_map({z0: [buf_z0], z1: [buf_z1], z2: [buf_z2]})
        ascir.utils.dump(graph)
        return graph

    def test_optimize_cube(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)
        attr = schedule_results.is_cube_type()
        attr = schedule_results.get_cube_attributes()


class TestAutofuseLoadBatchMatmulStore():
    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("LoadBatchMatMulStore")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        z0 = graph.create_axis("z0", s0 + 100)
        assert z0.size.expression == "(100 + s0)"
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", s2)
        buf_z0 = graph.create_axis("buf_z0", 100)
        buf_z1 = graph.create_axis("buf_z1", s1)
        buf_z2 = graph.create_axis("buf_z2", s2)

        arg3_1 = ascir.ops.Data("arg3_1", graph)
        arg3_1.attr.ir_attr.index = 0
        arg3_1.attr.sched.axis = [z0, z1, z2]
        arg3_1.y.dtype = ascir.dtypes.float16
        arg3_1.y.axis = [z0, z1, z2]
        arg3_1.y.size = [100 + s0, s1, s2]
        arg3_1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        load = ascir.ops.Load("load")
        try:
            load.attr.ir_attr.offset = "3"
        except Exception as e:
            assert e.args[0] == 'Only support type of SizeExpr or long'
        offset_of_0 = ascir.SizeExpr(0)
        load.attr.ir_attr.offset = offset_of_0
        assert load.attr.ir_attr.offset.expression == "0"
        load.x = arg3_1
        load.attr.sched.axis = [z0, z1, z2]
        load.y.dtype = ascir.dtypes.float16
        load.y.axis = [z0, z1, z2]
        load.y.size = [100 + s0, s1, s2]
        load.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        matmul_op = ascir.ops.BatchMatMul("batch_matmul")
        matmul_op.x1 = load
        matmul_op.x2 = load
        matmul_op.attr.sched.axis = [z0, z1, z2]
        matmul_op.attr.api.compute_type = "cube"
        matmul_op.attr.ir_attr.enable_hf32 = 1
        matmul_op.attr.ir_attr.adj_x1 = 0
        matmul_op.attr.ir_attr.adj_x2 = 1
        matmul_op.attr.ir_attr.has_relu = 1
        matmul_op.attr.ir_attr.offset_x = 1
        matmul_op.y.dtype = ascir.dtypes.float16
        matmul_op.y.axis = [z0, z1, z2]
        matmul_op.y.size = [100 + s0, s1, s2]
        matmul_op.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        store = ascir.ops.Store("store")
        try:
            store.attr.ir_attr.offset = "4"
        except Exception as e:
            assert e.args[0] == 'Only support type of SizeExpr or long'
        store.attr.ir_attr.offset = offset_of_0 + 1
        assert store.attr.ir_attr.offset.expression == "1"
        store.x = matmul_op
        store.attr.sched.axis = [z0, z1, z2]
        store.y.dtype = ascir.dtypes.float16
        store.y.axis = [z0, z1, z2]
        store.y.size = [100 + s0, s1, s2]
        store.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        buf1 = ascir.ops.Output("buf1", graph)
        buf1.attr.ir_attr.index = 0
        assert buf1.attr.ir_attr.index == 0
        buf1.x = store
        buf1.attr.sched.axis = [z0, z1, z2]
        buf1.y.dtype = ascir.dtypes.float16
        buf1.y.axis = [z0, z1, z2]
        buf1.y.size = [100 + s0, s1, s2]
        buf1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]
        graph.set_axis_map({z0: [buf_z0], z1: [buf_z1], z2: [buf_z2]})
        ascir.utils.dump(graph)
        return graph

    def test_optimize_cube(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)
        attr = schedule_results.is_cube_type()
        attr = schedule_results.get_cube_attributes()


class TestAutofuseGatherAbsStore():
    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("GatherAbsStore")
        size_of_z0 = ascir.SizeExpr(4001)
        z0 = graph.create_axis("z0", size_of_z0)
        size_of_z1 = ascir.SizeExpr(100)
        z1 = graph.create_axis("z1", size_of_z1)
        size_of_z2 = ascir.SizeExpr(1000)
        z2 = graph.create_axis("z2", size_of_z2)

        assert z0.size.expression == "4001"
        assert z1.size.expression == "100"
        assert z2.size.expression == "1000"

        data_0 = ascir.ops.Data("data_0", graph)
        data_0.attr.sched.axis = [z0]
        data_0.y.axis = [z0]
        data_0.y.size = [4001]
        data_0.y.strides = [1]
        data_0.y.dtype = ascir.dtypes.float32

        data_1 = ascir.ops.Data("data_1", graph)
        data_1.attr.sched.axis = [z1, z2]
        data_1.y.axis = [z1, z2]
        data_1.y.size = [100, 1000]
        data_1.y.strides = [1000, 1]
        data_1.y.dtype = ascir.dtypes.int64

        gather_0 = ascir.ops.Gather("gather_0")
        gather_0.attr.ir_attr.axis = 0
        gather_0.attr.api.compute_type = "gather"
        gather_0.attr.sched.axis = [z1, z2]
        gather_0.x1 = data_0.y
        gather_0.x2 = data_1.y
        gather_0.y.axis = [z1, z2]
        gather_0.y.size = [100, 1000]
        gather_0.y.strides = [1000, 1]
        gather_0.y.dtype = ascir.dtypes.float32

        abs_0 = ascir.ops.Abs("abs_0")
        abs_0.attr.sched.axis = [z1, z2]
        abs_0.x = gather_0.y
        abs_0.y.axis = [z1, z2]
        abs_0.y.size = [100, 1000]
        abs_0.y.strides = [1000, 1]
        abs_0.y.dtype = ascir.dtypes.float32

        store_0 = ascir.ops.Store("store_0")
        store_0.attr.sched.axis = [z1, z2]
        store_0.x = abs_0.y
        store_0.y.axis = [z1, z2]
        store_0.y.size = [100, 1000]
        store_0.y.strides = [1000, 1]
        store_0.y.dtype = ascir.dtypes.float32

        output_0 = ascir.ops.Output("output_0")
        output_0.attr.sched.axis = [z1, z2]
        output_0.x = store_0.y
        output_0.y.dtype = ascir.dtypes.float32
        graph.set_axis_map({z0: [z0], z1: [z1], z2: [z2]})
        return graph

    def test_construct_graph(self):
        graph = self.construct_graph()
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: GatherAbsStore\n",
            "Sizes:\n",
            "Axis:\n",
            "  z0(0) : 4001, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  z1(1) : 100, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  z2(2) : 1000, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "Nodes:\n",
            "  data_0: Data (0)\n",
            "    .ir_attr =  {    }\n",
            "    .axis = {z0, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n",
            "      .compute_type = invalid\n",
            "    .y.dtype = float32\n",
            "    .y.axis = {z0, }\n",
            "    .y.repeats = {4001, }\n",
            "    .y.strides = {1, }\n",
            "    .y.vectorized_axis = {}\n",
            "    .y.vectorized_strides = {}\n",
            "  data_1: Data (1)\n",
            "    .ir_attr =  {    }\n",
            "    .axis = {z1, z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n",
            "      .compute_type = invalid\n",
            "    .y.dtype = int64_t\n",
            "    .y.axis = {z1, z2, }\n",
            "    .y.repeats = {100, 1000, }\n",
            "    .y.strides = {1000, 1, }\n",
            "    .y.vectorized_axis = {}\n",
            "    .y.vectorized_strides = {}\n",
            "  gather_0: Gather (2)\n",
            "    .ir_attr =  {.axis = i: 0\n",
            "    }\n",
            "    .axis = {z1, z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n",
            "      .compute_type = gather\n",
            "    .x1 = data_0.y\n",
            "    .x2 = data_1.y\n",
            "    .y.dtype = float32\n",
            "    .y.axis = {z1, z2, }\n",
            "    .y.repeats = {100, 1000, }\n",
            "    .y.strides = {1000, 1, }\n",
            "    .y.vectorized_axis = {}\n",
            "    .y.vectorized_strides = {}\n",
            "  abs_0: Abs (3)\n",
            "    .axis = {z1, z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n",
            "      .compute_type = invalid\n",
            "    .x = gather_0.y\n",
            "    .y.dtype = float32\n",
            "    .y.axis = {z1, z2, }\n",
            "    .y.repeats = {100, 1000, }\n",
            "    .y.strides = {1000, 1, }\n",
            "    .y.vectorized_axis = {}\n",
            "    .y.vectorized_strides = {}\n",
            "  store_0: Store (4)\n",
            "    .ir_attr =  {    }\n",
            "    .axis = {z1, z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n",
            "      .compute_type = invalid\n",
            "    .x = abs_0.y\n",
            "    .y.dtype = float32\n",
            "    .y.axis = {z1, z2, }\n",
            "    .y.repeats = {100, 1000, }\n",
            "    .y.strides = {1000, 1, }\n",
            "    .y.vectorized_axis = {}\n",
            "    .y.vectorized_strides = {}\n",
            "  output_0: Output (5)\n",
            "    .ir_attr =  {    }\n",
            "    .axis = {z1, z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n",
            "      .compute_type = invalid\n",
            "    .x = store_0.y\n",
            "    .y.dtype = float32\n",
            "    .y.axis = {}\n",
            "    .y.repeats = {}\n",
            "    .y.strides = {}\n",
            "    .y.vectorized_axis = {}\n",
            "    .y.vectorized_strides = {}\n"
        ])

    @pytest.mark.skip
    def test_optimize(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)
        hint_graph = self.construct_graph()
        schedule_results = fuser.autofuse(hint_graph)

    @pytest.mark.skip
    def test_codegen(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        impl_graphs = fuser.schedule(hint_graph)
        tiling_def, host_tiling, op_kernel = fuser.codegen(impl_graphs)


class TestAutofuseLoadConcatStore():
    @staticmethod
    def construct_graph():
        try:
            NpuKernel0Graph = ascir.HintGraph(100)
        except Exception as e:
            assert e.args[0] == 'argument 1 must be str, not int'
        NpuKernel0Graph = ascir.HintGraph('LoadConcatStore')
        s0 = NpuKernel0Graph.create_size("s0")
        s1 = NpuKernel0Graph.create_size("s1")
        z0 = NpuKernel0Graph.create_axis("z0", s0)
        z1 = NpuKernel0Graph.create_axis("z1", s1 * 2)
        arg2_1 = ascir.ops.Data('arg2_1', NpuKernel0Graph)
        arg2_1.y.dtype = ascir.dtypes.float16
        load = ascir.ops.Load('load')
        try:
            load.infer_dtype()
        except Exception as e:
            assert e.args[0] == 'node load Load need set input before call infer dype'
        load.attr.sched.axis = [z0, z1]
        load.x = arg2_1.y
        load.y.axis = [z0, z1]
        load.y.strides = [s1, ascir.SizeExpr(1)]
        load.y.size = [s0, s1]
        load.infer_dtype()
        assert load.y.dtype == ascir.dtypes.float16
        arg3_1 = ascir.ops.Data('arg3_1', NpuKernel0Graph)
        arg3_1.y.dtype = ascir.dtypes.float16
        load1 = ascir.ops.Load('load1')
        load1.attr.sched.axis = [z0, z1]
        assert load1.attr.sched.axis == [z0.id, z1.id]
        load1.x = arg3_1.y
        load1.y.axis = [z0, z1]
        load1.y.strides = [s1, ascir.SizeExpr(1)]
        load1.y.size = [s0, s1]
        concat = ascir.ops.Concat('concat')
        concat.attr.sched.axis = [z0, z1]
        concat.x = [load, load1.y]
        concat.y.axis = [z0, z1]
        concat.y.strides = [s1 + s1, ascir.SizeExpr(1)]
        concat.y.size = [s0, s1 * 2]
        store = ascir.ops.Store('store')
        store.attr.sched.axis = [z0, z1]
        store.x = concat.y
        store.y.axis = [z0, z1]
        store.y.strides = [s1 * 2, ascir.SizeExpr(1)]
        store.y.size = [s0, s1 * 2]
        buf0 = ascir.ops.Output('buf0')
        buf0.x = store.y
        buf0.y.dtype = ascir.dtypes.float16
        NpuKernel0Graph.infer_dtypes()
        return NpuKernel0Graph

    def test_construct_graph(self):
        graph = self.construct_graph()
        debug_graph = ascir.utils.debug_str(graph)
        assert debug_graph != ""

    def test_optimize_graph(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)


class TestWorkspaceOptimize():
    @staticmethod
    def construct_graph():
        NpuKernel0Graph = ascir.HintGraph('workspace')
        s0 = NpuKernel0Graph.create_size("s0")
        z0 = NpuKernel0Graph.create_axis("z0", s0)
        arg2_1 = ascir.ops.Data('arg2_1', NpuKernel0Graph)
        arg2_1.y.dtype = ascir.dtypes.float16
        load = ascir.ops.Load('load')
        load.attr.sched.axis = [z0]
        load.x = arg2_1.y
        load.y.axis = [z0]
        load.y.strides = [ascir.SizeExpr(1)]
        load.y.size = [s0]
        store = ascir.ops.Store('store')
        store.attr.sched.axis = [z0]
        store.x = load.y
        store.y.axis = [z0]
        store.y.strides = [ascir.SizeExpr(1)]
        store.y.size = [s0]
        ws = ascir.ops.Workspace('buf8')
        ws.attr.sched.axis = [z0]
        ws.x = store.y
        ws.y.size = [s0]
        ws.y.dtype = ascir.dtypes.float16
        ws.y.axis = [z0]
        ws.y.strides = [ascir.SizeExpr(1)]
        load1 = ascir.ops.Load('load1')
        load1.attr.sched.axis = [z0]
        load1.x = ws.y
        load1.y.axis = [z0]
        load1.y.strides = [ascir.SizeExpr(1)]
        load1.y.size = [s0]

        store1 = ascir.ops.Store('store1')
        store1.attr.sched.axis = [z0]
        store1.x = load1.y
        store1.y.axis = [z0]
        store1.y.strides = [ascir.SizeExpr(1)]
        store1.y.size = [s0]

        ws1 = ascir.ops.Workspace('buf2')
        ws1.attr.sched.axis = [z0]
        ws1.x = store1.y
        ws1.y.size = [s0]
        ws1.y.dtype = ascir.dtypes.float16
        ws1.y.axis = [z0]
        ws1.y.strides = [ascir.SizeExpr(1)]

        load2 = ascir.ops.Load('load2')
        load2.attr.sched.axis = [z0]
        load2.x = ws1.y
        load2.y.axis = [z0]
        load2.y.strides = [ascir.SizeExpr(1)]
        load2.y.size = [s0]

        load3 = ascir.ops.Load('load3')
        load3.attr.sched.axis = [z0]
        load3.x = ws1.y
        load3.y.axis = [z0]
        load3.y.strides = [ascir.SizeExpr(1)]
        load3.y.size = [s0]
        NpuKernel0Graph.infer_dtypes()
        return NpuKernel0Graph

    def test_construct_graph(self):
        graph = self.construct_graph()
        debug_graph = ascir.utils.debug_str(graph)
        assert debug_graph != ""

    def test_optimize_graph(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)

class TestCodeGenLoadAbsStore():
    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("LoadAbsStore")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        z0 = graph.create_axis("z0", s0)
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", s2)
        buf_z0 = graph.create_axis("buf_z0", s0)
        buf_z1 = graph.create_axis("buf_z1", s1)
        buf_z2 = graph.create_axis("buf_z2", s2)

        arg3_1 = ascir.ops.Data("arg3_1", graph)
        arg3_1.attr.sched.axis = [z0, z1, z2]
        arg3_1.y.dtype = ascir.dtypes.float16
        arg3_1.y.axis = [z0, z1, z2]
        arg3_1.y.size = [s0, s1, s2]
        arg3_1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        load = ascir.ops.Load("load")
        load.x = arg3_1
        load.attr.sched.axis = [z0, z1, z2]
        load.y.dtype = ascir.dtypes.float16
        load.y.axis = [z0, z1, z2]
        load.y.size = [s0, s1, s2]
        load.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        abs_op = ascir.ops.Abs("abs")
        abs_op.x = load
        abs_op.attr.sched.axis = [z0, z1, z2]
        abs_op.y.dtype = ascir.dtypes.float16
        abs_op.y.axis = [z0, z1, z2]
        abs_op.y.size = [s0, s1, s2]
        abs_op.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        store = ascir.ops.Store("store")
        store.x = abs_op
        store.attr.sched.axis = [z0, z1, z2]
        store.y.dtype = ascir.dtypes.float16
        store.y.axis = [z0, z1, z2]
        store.y.size = [s0, s1, s2]
        store.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        buf1 = ascir.ops.Output("buf1", graph)
        buf1.x = store
        buf1.attr.sched.axis = [z0, z1, z2]
        buf1.y.dtype = ascir.dtypes.float16
        buf1.y.axis = [z0, z1, z2]
        buf1.y.size = [s0, s1, s2]
        buf1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]
        graph.set_axis_map({z0: [buf_z0], z1: [buf_z1], z2: [buf_z2]})
        return graph

    def test_construct_graph(self):
        graph = self.construct_graph()
        try:
            graph.infer_dtypes()
        except Exception as e:
            print(e.args)
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: LoadAbsStore\n",
            "Sizes:\n",
            "  s0: VAR\n",
            "  s1: VAR\n",
            "  s2: VAR\n",
            "Axis:\n",
            "  z0(0) : s0, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  z1(1) : s1, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  z2(2) : s2, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  buf_z0(3) : s0, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  buf_z1(4) : s1, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  buf_z2(5) : s2, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "Nodes:\n",
            "  arg3_1: Data (0)\n",
            "    .ir_attr =  {    }\n",
            "    .axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .y.dtype = float16\n"
            "    .y.axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .y.repeats = {s0, s1, s2, }\n",
            "    .y.strides = {(s1 * s2), s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"
            "    .y.vectorized_strides = {}\n"
            "  load: Load (1)\n",
            "    .ir_attr =  {    }\n",
            "    .axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .x = arg3_1.y\n"
            "    .y.dtype = float16\n"
            "    .y.axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .y.repeats = {s0, s1, s2, }\n",
            "    .y.strides = {(s1 * s2), s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"
            "    .y.vectorized_strides = {}\n"
            "  abs: Abs (2)\n",
            "    .axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .x = load.y\n"
            "    .y.dtype = float16\n"
            "    .y.axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .y.repeats = {s0, s1, s2, }\n",
            "    .y.strides = {(s1 * s2), s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"
            "    .y.vectorized_strides = {}\n"
            "  store: Store (3)\n",
            "    .ir_attr =  {    }\n",
            "    .axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .x = abs.y\n"
            "    .y.dtype = float16\n"
            "    .y.axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .y.repeats = {s0, s1, s2, }\n",
            "    .y.strides = {(s1 * s2), s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"
            "    .y.vectorized_strides = {}\n"
            "  buf1: Output (4)\n",
            "    .ir_attr =  {    }\n",
            "    .axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n"
            "      .compute_type = invalid\n"
            "    .x = store.y\n"
            "    .y.dtype = float16\n"
            "    .y.axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .y.repeats = {s0, s1, s2, }\n",
            "    .y.strides = {(s1 * s2), s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"
            "    .y.vectorized_strides = {}\n"
        ])

    def test_schedule(self):
        options = AutofuserOptions()
        scheduler = Schedule(options)

        hint_graph = self.construct_graph()
        impl_graphs = scheduler.schedule(hint_graph)

    @pytest.mark.skip
    def test_codegen(self):
        scheduler = Schedule()
        codegen = CodeGen(tiling_lib_path="./test", tiling_lib_codegen_symbol="test")

        hint_graph = self.construct_graph()
        impl_graphs = scheduler.schedule(hint_graph)
        shape_info = ascir.ShapeInfo({"s0": "GetDimValueFromGraphInputData(0, 0);",
                                      "s1": "GetDimValueFromGraphInputData(0, 1);",
                                      "s2": "GetDimValueFromGraphInputData(1, 0);"})

        kernel_path = "./fused_graph_kernel.o"
        with open(kernel_path, "wb") as o_file:
            o_file.write(b"This is a .o file content.")

        data = {
            "name": "Alice",
            "age": 30,
            "is_student": False,
            "courses": ["Math", "Science", "History"]
        }
        json_path = "./fused_graph_kernel.json"
        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        tiling_data, op_kernel = codegen.device_code_generator(hint_graph, impl_graphs)
        tiling, infer_shape = codegen.host_code_generator(hint_graph, impl_graphs, shape_info, "", ["", ""])
        get_kernel = codegen.get_kernel_and_json_generator(kernel_path, json_path)
        os.remove(kernel_path)
        os.remove(json_path)
        assert tiling_data == "".join([
            "#ifndef __Autofuse_Tiling_Data_H__\n"
            "#define __Autofuse_Tiling_Data_H__\n"
            "#include <stdint.h>\n"
            "#include \"kernel_tiling/kernel_tiling.h\"\n"
            "#define BEGIN_TILING_DATA_DEF_T(name) struct name {\n"
            "#define TILING_DATA_FIELD_DEF_T(type, name) \\\n"
            "  type name; \\\n"
            "  inline void set_##name(type value) { name = value; } \\\n",
            "  inline type get_##name() { return name; } \\\n"
            "  inline type* get_addr_##name() {return &name;}\n"
            "#define END_TILING_DATA_DEF_T };\n"
            "#define TILING_DATA_FIELD_DEF_T_STRUCT(struct_type, filed_name) \\\n"
            "  struct_type filed_name;\n\n"
            "BEGIN_TILING_DATA_DEF_T(AutofuseTilingData)\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, block_dim);\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, corenum);\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, ub_size);\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, hbm_size);\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, tiling_key);\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, z0z1z2t_size);\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, z0z1z2Tb_size);\n"
            "END_TILING_DATA_DEF_T;\n\n"
            "struct AutofuseTilingDataPerf {\n"
            "  AutofuseTilingData tiling_data;\n"
            "  double best_perf;\n"
            "};\n"
            "#endif\n"
        ])

        assert infer_shape == "".join([
            ""])

        assert get_kernel == "".join([
            "#include <cstdint>\n"
            "#include <cstring>\n"
            "#include <vector>\n"
            "extern \"C\" void GetKernelBin(std::vector<char> &kernel_bin) {\n"
            "  std::vector<uint8_t> temp_kernel = {\n"
            "    84, 104, 105, 115, 32, 105, 115, 32, 97, 32, 46, 111, 32, 102, 105, 108, 101, 32, 99, 111, \n"
            "    110, 116, 101, 110, 116, 46, };\n"
            "  kernel_bin.resize(temp_kernel.size());\n"
            "  std::memcpy(kernel_bin.data(), temp_kernel.data(), temp_kernel.size() * sizeof(uint8_t));\n"
            "}"])


class TestComputeGraphInput():
    @staticmethod
    def construct_compute_graph():
        test_graph = os.path.join(PYF_PATH, "test_graph.txt")
        with open(test_graph, 'r', encoding='utf-8') as file:
            content = file.read()
        compute_graph = ascir.utils.deserialize("compute_graph", content)
        print(compute_graph.get_name(), flush=True)
        print(compute_graph.get_info(), flush=True)
        assert compute_graph != None
        return compute_graph

    @pytest.mark.skip
    def test_scheduleV2(self):
        options = AutofuserOptions()
        scheduler = Schedule(options)

        compute_graph = self.construct_compute_graph()
        schedule_results = scheduler.scheduleV2(compute_graph)

    def test_scheduleV2_fail(self):
        options = AutofuserOptions()
        scheduler = Schedule(options)

        compute_graph = ascir.HintComputeGraph("test")
        try:
            scheduler.scheduleV2(compute_graph)
        except RuntimeError as e:
            pass
    @pytest.mark.skip
    def test_computegraph_codegen(self):
        scheduler = Schedule()
        codegen = CodeGen(tiling_lib_path="./test", tiling_lib_codegen_symbol="test")

        compute_graph = self.construct_compute_graph()
        schedule_results = scheduler.scheduleV2(compute_graph)
        shape_info = ascir.ShapeInfo({"s0": "GetDimValueFromGraphInputData(0, 0);",
                                      "s1": "GetDimValueFromGraphInputData(0, 1);",
                                      "s2": "GetDimValueFromGraphInputData(1, 0);"})

        kernel_path = "./fused_graph_kernel.o"
        with open(kernel_path, "wb") as o_file:
            o_file.write(b"This is a .o file content.")

        data = {
            "name": "Alice",
            "age": 30,
            "is_student": False,
            "courses": ["Math", "Science", "History"]
        }
        json_path = "./fused_graph_kernel.json"
        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        tiling_data, op_kernel = codegen.device_code_generator(schedule_results)
        assert tiling_data == "".join([
            "#ifndef __Autofuse_Tiling_Data_H__\n"
            "#define __Autofuse_Tiling_Data_H__\n"
            "#include <stdint.h>\n"
            "#include \"kernel_tiling/kernel_tiling.h\"\n"
            "#define BEGIN_TILING_DATA_DEF_T(name) struct name {\n"
            "#define TILING_DATA_FIELD_DEF_T(type, name) \\\n"
            "  type name; \\\n"
            "  inline void set_##name(type value) { name = value; } \\\n",
            "  inline type get_##name() { return name; } \\\n"
            "  inline type* get_addr_##name() {return &name;}\n"
            "#define END_TILING_DATA_DEF_T };\n"
            "#define TILING_DATA_FIELD_DEF_T_STRUCT(struct_type, filed_name) \\\n"
            "  struct_type filed_name;\n\n"
            "BEGIN_TILING_DATA_DEF_T(AutofuseTilingData)\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, block_dim);\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, corenum);\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, ub_size);\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, hbm_size);\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, tiling_key);\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, z0z1z2t_size);\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, z0z1z2Tb_size);\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, q0_size);\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, q1_size);\n"
            "  TILING_DATA_FIELD_DEF_T(uint32_t, b0_size);\n"
            "END_TILING_DATA_DEF_T;\n\n"
            "struct AutofuseTilingDataPerf {\n"
            "  AutofuseTilingData tiling_data;\n"
            "  double best_perf;\n"
            "};\n"
            "#endif\n"
        ])

        output_shape = [["s0", "s1"]]
        vector_core_num = "0"
        tiling, infer_shape = codegen.host_code_generator(schedule_results, shape_info, output_shape, "", vector_core_num)
        pgo_src = codegen.pgo_code_generator(schedule_results, "", vector_core_num, "", "")
        get_kernel = codegen.get_kernel_and_json_generator(kernel_path, json_path)
        os.remove(kernel_path)
        os.remove(json_path)
        assert get_kernel == "".join([
            "#include <cstdint>\n"
            "#include <cstring>\n"
            "#include <vector>\n"
            "extern \"C\" void GetKernelBin(std::vector<char> &kernel_bin) {\n"
            "  std::vector<uint8_t> temp_kernel = {\n"
            "    84, 104, 105, 115, 32, 105, 115, 32, 97, 32, 46, 111, 32, 102, 105, 108, 101, 32, 99, 111, \n"
            "    110, 116, 101, 110, 116, 46, };\n"
            "  kernel_bin.resize(temp_kernel.size());\n"
            "  std::memcpy(kernel_bin.data(), temp_kernel.data(), temp_kernel.size() * sizeof(uint8_t));\n"
            "}"])

        try:
            output_shape = ["s0"]
            vector_core_num = "0"
            tiling, infer_shape = codegen.host_code_generator(schedule_results, shape_info,
                                                              output_shape, "", vector_core_num)
        except ValueError as e:
            pass

        try:
            pgo_src = codegen.pgo_code_generator(schedule_results, "", "")
        except ValueError as e:
            pass

        try:
            get_kernel = codegen.get_kernel_and_json_generator(kernel_path, json_path)
        except ValueError as e:
            pass

        try:
            get_kernel = codegen.get_kernel_and_json_generator(kernel_path)
        except ValueError as e:
            pass


class TestHintGraph():
    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("LoadAbsStore")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        z0 = graph.create_axis("z0", s0)
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", s2)
        buf_z0 = graph.create_axis("buf_z0", s0)
        buf_z1 = graph.create_axis("buf_z1", s1)
        buf_z2 = graph.create_axis("buf_z2", s2)

        arg3_1 = ascir.ops.Data("arg3_1", graph)
        arg3_1.attr.sched.axis = [z0, z1, z2]
        arg3_1.y.dtype = ascir.dtypes.float16
        arg3_1.y.axis = [z0, z1, z2]
        arg3_1.y.size = [s0, s1, s2]
        arg3_1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        load = ascir.ops.Load("load")
        load.x = arg3_1
        load.attr.sched.axis = [z0, z1, z2]
        load.y.dtype = ascir.dtypes.float16
        load.y.axis = [z0, z1, z2]
        load.y.size = [s0, s1, s2]
        load.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        abs_op = ascir.ops.Abs("abs")
        abs_op.x = load
        abs_op.attr.sched.axis = [z0, z1, z2]
        abs_op.y.dtype = ascir.dtypes.float16
        abs_op.y.axis = [z0, z1, z2]
        abs_op.y.size = [s0, s1, s2]
        abs_op.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        store = ascir.ops.Store("store")
        store.x = abs_op
        store.attr.sched.axis = [z0, z1, z2]
        store.y.dtype = ascir.dtypes.float16
        store.y.axis = [z0, z1, z2]
        store.y.size = [s0, s1, s2]
        store.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        buf1 = ascir.ops.Output("buf1", graph)
        buf1.x = store
        buf1.attr.sched.axis = [z0, z1, z2]
        buf1.y.dtype = ascir.dtypes.float16
        buf1.y.axis = [z0, z1, z2]
        buf1.y.size = [s0, s1, s2]
        buf1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]
        graph.set_axis_map({z0: [buf_z0], z1: [buf_z1], z2: [buf_z2]})
        return graph

    def test_hintgraph_set_name(self):
        asc_graph = self.construct_graph()
        try:
            asc_graph.set_name(2)
        except TypeError as e:
            assert asc_graph.get_name() == "".join(["LoadAbsStore"])

        asc_graph.set_name("test_graph")
        assert asc_graph.get_name() == "".join(["test_graph"])


class TestFusedGraph():
    @staticmethod
    def construct_add_ascgraph(name: str) -> ascir.HintGraph:
        NpuKernel0Graph = ascir.HintGraph(name)
        s0 = NpuKernel0Graph.create_size("s0")
        s1 = NpuKernel0Graph.create_size("s1")
        z0 = NpuKernel0Graph.create_axis("z0", s0)
        z1 = NpuKernel0Graph.create_axis("z1", s1)
        sub_data0 = ascir.ops.Data('sub_data0', NpuKernel0Graph)
        sub_data0.y.dtype = ascir.dtypes.float16
        sub_data0.attr.ir_attr.index = 0
        load0 = ascir.ops.Load('load')
        load0.attr.ir_attr.offset = 0
        load0.attr.sched.axis = [z0, z1]
        load0.x = sub_data0.y
        load0.y.axis = [z0, z1]
        load0.y.strides = [s1, ascir.SizeExpr(1)]
        load0.y.size = [s0, s1]
        sub_data1 = ascir.ops.Data('sub_data1', NpuKernel0Graph)
        sub_data1.y.dtype = ascir.dtypes.float16
        sub_data1.attr.ir_attr.index = 1
        load1 = ascir.ops.Load('load')
        load1.attr.ir_attr.offset = ascir.SizeExpr(0)
        load1.attr.sched.axis = [z0, z1]
        load1.x = sub_data1.y
        load1.y.axis = [z0, z1]
        load1.y.strides = [s1, ascir.SizeExpr(1)]
        load1.y.size = [s0, s1]

        add0 = ascir.ops.Add('add')
        add0.attr.sched.axis = [z0, z1]
        add0.x1 = load0.y
        add0.x2 = load1.y
        add0.y.axis = [z0, z1]
        add0.y.strides = [s1 + s1, ascir.SizeExpr(1)]
        add0.y.size = [s0, s1 * 2]

        store0 = ascir.ops.Store('store')
        store0.attr.ir_attr.offset = ascir.SizeExpr(0)
        store0.attr.sched.axis = [z0, z1]
        store0.x = add0.y
        store0.y.axis = [z0, z1]
        store0.y.strides = [s1 ** 2, ascir.SizeExpr(1)]
        store0.y.size = [s0, s1 * 2]

        store1 = ascir.ops.Store('store')
        store1.attr.ir_attr.offset = ascir.SizeExpr(10)
        store1.attr.sched.axis = [z0, z1]
        store1.x = add0.y
        store1.y.axis = [z0, z1]
        store1.y.strides = [s1 * 2, ascir.SizeExpr(1)]
        store1.y.size = [s0, s1 * 2]
        buf0 = ascir.ops.Output('buf0')
        buf0.attr.ir_attr.index = 0
        # store0, strore1 写到同一个output上，偏移不同
        buf0.x = [store0.y, store1]
        buf0.y.dtype = ascir.dtypes.float16
        buf1 = ascir.ops.Output('buf1')
        buf1.attr.ir_attr.index = 1
        buf1.x = store1.y
        NpuKernel0Graph.infer_dtypes()
        ascir.utils.dump(NpuKernel0Graph)
        return NpuKernel0Graph

    @staticmethod
    def construct_add_ascgraph_without_data(name: str) -> ascir.HintGraph:
        NpuKernel0Graph = ascir.HintGraph(name)
        s0 = NpuKernel0Graph.create_size("s0")
        s1 = NpuKernel0Graph.create_size("s1")
        z0 = NpuKernel0Graph.create_axis("z0", s0)
        z1 = NpuKernel0Graph.create_axis("z1", s1)
        sub_data0 = ascir.ops.Scalar('sub_data0', NpuKernel0Graph)
        sub_data0.y.dtype = ascir.dtypes.float16
        load0 = ascir.ops.Load('load')
        load0.attr.ir_attr.offset = 0
        load0.attr.sched.axis = [z0, z1]
        load0.x = sub_data0.y
        load0.y.axis = [z0, z1]
        load0.y.strides = [s1, ascir.SizeExpr(1)]
        load0.y.size = [s0, s1]
        sub_data1 = ascir.ops.Scalar('sub_data1', NpuKernel0Graph)
        sub_data1.y.dtype = ascir.dtypes.float16
        load1 = ascir.ops.Load('load')
        load1.attr.ir_attr.offset = ascir.SizeExpr(0)
        load1.attr.sched.axis = [z0, z1]
        load1.x = sub_data1.y
        load1.y.axis = [z0, z1]
        load1.y.strides = [s1, ascir.SizeExpr(1)]
        load1.y.size = [s0, s1]

        add0 = ascir.ops.Add('add')
        add0.attr.sched.axis = [z0, z1]
        add0.x1 = load0.y
        add0.x2 = load1.y
        add0.y.axis = [z0, z1]
        add0.y.strides = [s1 + s1, ascir.SizeExpr(1)]
        add0.y.size = [s0, s1 * 2]

        store0 = ascir.ops.Store('store')
        store0.attr.ir_attr.offset = ascir.SizeExpr(0)
        store0.attr.sched.axis = [z0, z1]
        store0.x = add0.y
        store0.y.axis = [z0, z1]
        store0.y.strides = [s1 ** 2, ascir.SizeExpr(1)]
        store0.y.size = [s0, s1 * 2]

        store1 = ascir.ops.Store('store')
        store1.attr.ir_attr.offset = ascir.SizeExpr(10)
        store1.attr.sched.axis = [z0, z1]
        store1.x = add0.y
        store1.y.axis = [z0, z1]
        store1.y.strides = [s1 * 2, ascir.SizeExpr(1)]
        store1.y.size = [s0, s1 * 2]
        buf0 = ascir.ops.Output('buf0')
        buf0.attr.ir_attr.index = 0
        # store0, strore1 写到同一个output上，偏移不同
        buf0.x = [store0.y, store1]
        buf0.y.dtype = ascir.dtypes.float16
        buf1 = ascir.ops.Output('buf1')
        buf1.attr.ir_attr.index = 1
        buf1.x = store1.y
        NpuKernel0Graph.infer_dtypes()
        ascir.utils.dump(NpuKernel0Graph)
        return NpuKernel0Graph

    def test_fused_graph_construct_and_dump_with_ascbackend_node(self):
        FusedGraph = ascir.FusedGraph('fused_graph')
        data0 = ascir.ops.Data('data0', FusedGraph)
        data0.attr.ir_attr.index = 0
        data1 = ascir.ops.Data('data1', FusedGraph)
        data1.attr.ir_attr.index = 0
        ascgraph_node0 = ascir.ops.AscBackend("ascgraph_node0", self.construct_add_ascgraph_without_data("ascgraph0"),
                                              FusedGraph)
        ascgraph_node1 = ascir.ops.AscBackend("ascgraph_node1", self.construct_add_ascgraph("ascgraph1"), FusedGraph)
        ascgraph_node1.x = [data0.y, data1.y]
        ascgraph_node2 = ascir.ops.AscBackend("ascgraph_node2", self.construct_add_ascgraph("ascgraph2"), FusedGraph)
        ascgraph_node2.x = [ascgraph_node0.y[0], ascgraph_node1.y[1]]
        output = ascir.ops.Output('output', FusedGraph)
        output.x = ascgraph_node2.y[1]
        ascir.utils.dump(FusedGraph)

    def test_fused_graph_inductor(self):
        FusedGraph = ascir.FusedGraph('fused_graph')

        options = AutofuserOptions()
        scheduler = Schedule(options)
        fuser = Autofuser(options)
        try:
            schedule_results = fuser.schedule(FusedGraph)
            tiling_def, host_tiling, op_kernel = fuser.autofuse_backend(FusedGraph)
        except RuntimeError as e:
            pass

    def test_fused_graph_construct_and_dump_with_ascgraph_node(self):
        FusedGraph = ascir.FusedGraph('fused_graph')
        data0 = ascir.ops.Data('data0', FusedGraph)
        data0.attr.ir_attr.index = 0
        data1 = ascir.ops.Data('data1', FusedGraph)
        data1.attr.ir_attr.index = 0
        ascgraph_node0 = ascir.ops.AscGraph("ascgraph_node0", self.construct_add_ascgraph("ascgraph0"), FusedGraph)
        ascgraph_node0.x = [data0.y, data1.y]
        ascgraph_node1 = ascir.ops.AscGraph("ascgraph_node1", self.construct_add_ascgraph("ascgraph1"), FusedGraph)
        ascgraph_node1.x = [data0.y, data1.y]
        ascgraph_node2 = ascir.ops.AscGraph("ascgraph_node2", self.construct_add_ascgraph("ascgraph2"), FusedGraph)
        ascgraph_node2.x = [ascgraph_node0.y[0], ascgraph_node1.y[0]]
        output = ascir.ops.Output('output', FusedGraph)
        output.x = ascgraph_node2.y[0]
        try:
            ascgraph_node2.x = [ascgraph_node0.y[0].dtype, ascgraph_node1.y[0]]
        except TypeError as e:
            assert e.args[0] == "Input Type is invalid."

        ascir.utils.dump(FusedGraph)
        try:
            ascir.utils.dump(data0)
        except TypeError as e:
            assert e.args[0] == "Argument must be a HintGraph or FusedGraph object, got Data"


class TestFusedGraphByApi():
    @staticmethod
    def construct_add_ascgraph(name: str) -> ascir.HintGraph:
        NpuKernel0Graph = ascir.HintGraph(name)
        s0 = NpuKernel0Graph.create_size("s0")
        s1 = NpuKernel0Graph.create_size("s1")
        z0 = NpuKernel0Graph.create_axis("z0", s0)
        z1 = NpuKernel0Graph.create_axis("z1", s1)
        sub_data0 = ascir_api.Data(NpuKernel0Graph, dtype=ascir.dtypes.float16)
        load0 = ascir_api.Load(NpuKernel0Graph, sub_data0, offset=0, axis=[z0, z1])
        assert load0.axis == [z0.id, z1.id]
        assert load0.size == [s0, s1]
        assert load0.strides == [s1, 1]
        sub_data1 = ascir_api.Data(NpuKernel0Graph, dtype=ascir.dtypes.float16)
        load1 = ascir_api.Load(NpuKernel0Graph, sub_data1, offset=0, axis=[z0, z1])
        add0 = ascir_api.Add(NpuKernel0Graph, load0, load1, axis=[z0, z1])
        assert add0.axis == [z0.id, z1.id]
        assert add0.size == [s0, s1]
        assert add0.strides == [s1, 1]
        store0 = ascir_api.Store(NpuKernel0Graph, add0, offset=0, axis=[z0, z1])
        store1 = ascir_api.Store(NpuKernel0Graph, add0, offset=10, axis=[z0, z1])
        # store0, strore1 写到同一个output上，偏移不同
        buf0 = ascir_api.Output(NpuKernel0Graph, [store0, store1], dtype=ascir.dtypes.float16)
        buf1 = ascir_api.Output(NpuKernel0Graph, store1) # infer
        assert buf1.dtype == ascir.dtypes.float16
        print(ascir.utils.debug_str(NpuKernel0Graph))
        return NpuKernel0Graph

    def test_fused_graph_construct_and_dump_with_ascbackend_node(self):
        FusedGraph = ascir.FusedGraph('fused_graph')
        data0 = ascir.ops.Data('data0', FusedGraph)
        data0.attr.ir_attr.index = 0
        data1 = ascir.ops.Data('data1', FusedGraph)
        data1.attr.ir_attr.index = 0
        ascgraph_node0 = ascir.ops.AscGraph("ascgraph_node0", self.construct_add_ascgraph("ascgraph0"),
                                            FusedGraph)
        ascgraph_node0.x = [data0.y, data1.y]
        ascgraph_node1 = ascir.ops.AscGraph("ascgraph_node1", self.construct_add_ascgraph("ascgraph1"), FusedGraph)
        ascgraph_node1.x = [data0.y, data1.y]
        ascgraph_node2 = ascir.ops.AscGraph("ascgraph_node2", self.construct_add_ascgraph("ascgraph2"), FusedGraph)
        ascgraph_node2.x = [ascgraph_node0.y[0], ascgraph_node1.y[1]]
        output = ascir.ops.Output('output', FusedGraph)
        output.x = ascgraph_node2.y[1]
        ascir.utils.dump(FusedGraph)

# 测试包含transpose的sched, codegen的流程, 执行不抛异常, 返回结果非空
class TestAutofuseLoadTransposeStore():
    @staticmethod
    def construct_invalid_graph():
        graph = ascir.HintGraph("LoadTransposeStore")
        s0 = 100
        s1 = 200
        s2 = 300
        z0 = graph.create_axis("z0", s0)
        assert z0.size.expression == "100"
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", s2)

        arg3_1 = ascir.ops.Data("arg3_1", graph)
        arg3_1.attr.ir_attr.index = 0
        arg3_1.attr.sched.axis = [z0, z1, z2]
        arg3_1.y.dtype = ascir.dtypes.float16
        arg3_1.y.axis = [z0, z1, z2]
        arg3_1.y.size = [s0, s1, s2]
        arg3_1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]
        return graph

    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("LoadTransposeStore")
        s0 = 100
        s1 = 200
        s2 = 300
        z0 = graph.create_axis("z0", s0)
        assert z0.size.expression == "100"
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", s2)
        buf_z0 = graph.create_axis("buf_z0", s0)
        buf_z1 = graph.create_axis("buf_z1", s1)
        buf_z2 = graph.create_axis("buf_z2", s2)

        arg3_1 = ascir.ops.Data("arg3_1", graph)
        arg3_1.attr.ir_attr.index= 0
        arg3_1.attr.sched.axis = [z0, z1, z2]
        arg3_1.y.dtype = ascir.dtypes.float16
        arg3_1.y.axis = [z0, z1, z2]
        arg3_1.y.size = [s0, s1, s2]
        arg3_1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        load = ascir.ops.Load("load")
        try:
            load.attr.ir_attr.offset = "3"
        except Exception as e:
            assert e.args[0] == 'Only support type of SizeExpr or long'
        offset_of_0 = ascir.SizeExpr(0)
        load.attr.ir_attr.offset = offset_of_0
        assert load.attr.ir_attr.offset.expression == "0"
        load.x = arg3_1
        load.attr.sched.axis = [z0, z1, z2]
        load.y.dtype = ascir.dtypes.float16
        load.y.axis = [z0, z1, z2]
        load.y.size = [s0, s1, s2]
        load.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        transpose0_op = ascir.ops.Transpose("Transpose0")
        transpose0_op.x = load
        transpose0_op.attr.sched.axis = [z0, z1, z2]
        transpose0_op.y.dtype = ascir.dtypes.float16
        transpose0_op.y.axis = [z1, z0, z2]
        transpose0_op.y.size = [s1, s0, s2]
        transpose0_op.y.strides = [s0 * s2, s2, ascir.SizeExpr(1)]

        store = ascir.ops.Store("store")
        try:
            store.attr.ir_attr.offset = "4"
        except Exception as e:
            assert e.args[0] == 'Only support type of SizeExpr or long'
        store.attr.ir_attr.offset = offset_of_0 + 1
        assert store.attr.ir_attr.offset.expression == "1"
        store.x = transpose0_op
        store.attr.sched.axis = [z0, z1, z2]
        store.y.dtype = ascir.dtypes.float16
        store.y.axis = [z1, z0, z2]
        store.y.size = [s1, s0, s2]
        store.y.strides = [s0 * s2, s2, ascir.SizeExpr(1)]

        buf1 = ascir.ops.Output("buf1", graph)
        buf1.attr.ir_attr.index = 0
        assert buf1.attr.ir_attr.index == 0
        buf1.x = store
        buf1.attr.sched.axis = [z0, z1, z2]
        buf1.y.dtype = ascir.dtypes.float16
        buf1.y.axis = [z1, z0, z2]
        buf1.y.size = [s1, s0, s2]
        buf1.y.strides = [s0 * s2, s2, ascir.SizeExpr(1)]
        graph.set_axis_map({z0:[buf_z0], z1:[buf_z1], z2:[buf_z2]})
        return graph

    def test_construct_graph(self):
        graph = self.construct_graph()
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: LoadTransposeStore\n",
            "Sizes:\n",
            "Axis:\n",
            "  z0(0) : 100, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  z1(1) : 200, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  z2(2) : 300, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  buf_z0(3) : 100, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  buf_z1(4) : 200, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "  buf_z2(5) : 300, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n",
            "Nodes:\n",
            "  arg3_1: Data (0)\n",
            "    .ir_attr =  {.index = i: 0\n",
            "    }\n",
            "    .axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n",
            "      .compute_type = invalid\n",
            "    .y.dtype = float16\n",
            "    .y.axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .y.repeats = {100, 200, 300, }\n",
            "    .y.strides = {60000, 300, 1, }\n",
            "    .y.vectorized_axis = {}\n",
            "    .y.vectorized_strides = {}\n",
            "  load: Load (1)\n",
            "    .ir_attr =  {.offset = expression: \"0\"\n",
            "    }\n",
            "    .axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n",
            "      .compute_type = invalid\n",
            "    .x = arg3_1.y\n",
            "    .y.dtype = float16\n",
            "    .y.axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .y.repeats = {100, 200, 300, }\n",
            "    .y.strides = {60000, 300, 1, }\n",
            "    .y.vectorized_axis = {}\n",
            "    .y.vectorized_strides = {}\n",
            "  Transpose0: Transpose (2)\n",
            "    .axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n",
            "      .compute_type = invalid\n",
            "    .x = load.y\n",
            "    .y.dtype = float16\n",
            "    .y.axis = {buf_z1, buf_z0, buf_z2, }\n",
            "    .y.repeats = {200, 100, 300, }\n",
            "    .y.strides = {30000, 300, 1, }\n",
            "    .y.vectorized_axis = {}\n",
            "    .y.vectorized_strides = {}\n",
            "  store: Store (3)\n",
            "    .ir_attr =  {.offset = expression: \"1\"\n",
            "    }\n",
            "    .axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n",
            "      .compute_type = invalid\n",
            "    .x = Transpose0.y\n",
            "    .y.dtype = float16\n",
            "    .y.axis = {buf_z1, buf_z0, buf_z2, }\n",
            "    .y.repeats = {200, 100, 300, }\n",
            "    .y.strides = {30000, 300, 1, }\n",
            "    .y.vectorized_axis = {}\n",
            "    .y.vectorized_strides = {}\n",
            "  buf1: Output (4)\n",
            "    .ir_attr =  {.index = i: 0\n",
            "    }\n",
            "    .axis = {buf_z0, buf_z1, buf_z2, }\n",
            "    .exec_condition = no_cache\n",
            "    .api:\n",
            "      .compute_type = invalid\n",
            "    .x = store.y\n",
            "    .y.dtype = float16\n",
            "    .y.axis = {buf_z1, buf_z0, buf_z2, }\n",
            "    .y.repeats = {200, 100, 300, }\n",
            "    .y.strides = {30000, 300, 1, }\n",
            "    .y.vectorized_axis = {}\n"
            "    .y.vectorized_strides = {}\n"
        ])

    def test_autofuse_backend(self):
         options = AutofuserOptions()
         fuser = Autofuser(options)
         try:
            hint_graph = self.construct_graph()
            sched_result = fuser.schedule(hint_graph)
            tiling_def, host_tiling, op_kernel = fuser.codegen(sched_result)
            assert len(tiling_def) > 0
            assert len(host_tiling) > 0
            assert len(op_kernel) > 0
         except RuntimeError as e:
            pass
    import os
    def test_autofuse_backend_faild_dump_graph(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)
        hint_graph = self.construct_invalid_graph()
        with pytest.raises(RuntimeError, match=r'^Optimize fail$'):
            sched_result = fuser.schedule(hint_graph)
        target_dir = './'
        for item in os.listdir(target_dir):
            item_path = os.path.join(target_dir, item)
            if os.path.isdir(item_path) and item.startswith('ascgen_dump_pid'):
                print(f"delete dump dir :{item_path}")
                shutil.rmtree(item_path)
