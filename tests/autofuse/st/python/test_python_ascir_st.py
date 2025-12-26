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
import os
from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions, Schedule, CodeGen
PYF_PATH = os.path.dirname(os.path.realpath(__file__))

class TestComputeGraphInput():
    @staticmethod
    def construct_compute_graph():
        test_graph = os.path.join(PYF_PATH, "test_seri_compute_graph.txt")
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
            print(f"Caught a RuntimeError: {e}", flush=True)
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

        output_shape = [["s0","s1"]]
        vector_core_num = "0"
        tiling, infer_shape = codegen.host_code_generator(schedule_results, shape_info,
                                                          output_shape, "", vector_core_num)
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
            output_shape = output_shape = [["s0",1]]
            vector_core_num = "20"
            tiling, infer_shape = codegen.host_code_generator(schedule_results, shape_info,
                                                              output_shape, "", vector_core_num)
        except ValueError as e:
            pass

        try:
            get_kernel = codegen.get_kernel_and_json_generator(kernel_path, json_path)
        except ValueError as e:
            pass

class TestUtilsDeserialize():
    @staticmethod
    def get_serialize_asc_graph():
        test_graph = os.path.join(PYF_PATH, "test_seri_asc_graph.txt")
        with open(test_graph, 'r', encoding='utf-8') as file:
            content = file.read()
        return content

    def test_symbol_source_info(self):
        symbol_source = '{"s0":"GetDimValueFromGraphInputData(0, 0);","s1":"GetDimValueFromGraphInputData(0,1)"}'
        symbol_obj = ascir.utils.deserialize("symbol_source_info", symbol_source)
        assert symbol_obj is not None

        try:
            symbol_source = '{"s0":"GetDimValueFromGraphInputData(0, 0);","s1":0}'
            symbol_obj = ascir.utils.deserialize("symbol_source_info", symbol_source)
        except TypeError as e:
            pass

        try:
            symbol_source = 'test'
            symbol_obj = ascir.utils.deserialize("symbol_source_info", symbol_source)
        except TypeError as e:
            pass

    @pytest.mark.skip
    def test_asc_graph(self):
        asc_graph = self.get_serialize_asc_graph()
        asc_graph_obj = ascir.utils.deserialize("asc_graph", asc_graph)
        assert asc_graph_obj is not None

        try:
            asc_graph_obj = ascir.utils.deserialize("asc_graph", "error")
        except TypeError as e:
            pass
