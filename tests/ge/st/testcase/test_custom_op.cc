/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <unistd.h>
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "faker/fake_value.h"
#include "rt_external_base.h"
#include "ge/ge_api.h"
#include "ge/ge_api_error_codes.h"
#include "ge/ge_graph_compile_summary.h"
#include "graph/execute/model_executor.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/task_info/ge/custom_task_info.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "utils/mock_ops_kernel_builder.h"
#include "utils/taskdef_builder.h"
#include "stub/gert_runtime_stub.h"
#include "easy_graph/builder/graph_dsl.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "utils/taskdef_builder.h"
#include "common/args_checker.h"
#include "graph/load/model_manager/model_manager.h"
#include "init_ge.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "graph/ge_local_context.h"
#include "graph/ge_global_options.h"
#include "utils/synchronizer.h"
#include "common/global_variables/diagnose_switch.h"
#include "hcom/hcom_topo_info.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "engines/custom_engine/custom_graph_optimizer.h"
#include "engines/custom_engine/custom_ops_kernel_builder.h"
#include "graph/compute_graph.h"
#include "graph/custom_op/cast.h"
#include "graph/custom_op/infer_meta.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op.h"
#include "graph/operator_factory.h"
#include "graph/ge_tensor.h"
#include "graph/operator_reg.h"
#include "graph/op_desc.h"
#include "graph/utils/args_format_desc_utils.h"
#include "common/python_runtime/ge_python_runtime_manager.h"
#include "runtime/custom_op/custom_op_loader.h"
#include "runtime/custom_op/python_custom_op_bridge_loader.h"
#include "exe_graph/runtime/storage_shape.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "runtime/v2/kernel/common_kernel_impl/infer_shape.h"

namespace ge {
REG_OP(StPythonAnnotatedArgsCustomOp)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(z, TensorType::ALL())
    .REQUIRED_ATTR(alpha, Int)
    .OP_END_FACTORY_REG(StPythonAnnotatedArgsCustomOp);

REG_OP(StPythonAnnotatedArgsBadAttrCustomOp)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(z, TensorType::ALL())
    .REQUIRED_ATTR(alpha, Int)
    .OP_END_FACTORY_REG(StPythonAnnotatedArgsBadAttrCustomOp);

REG_OP(StPythonCompilableCustomOp)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(z, TensorType::ALL())
    .REQUIRED_ATTR(bias, Int)
    .OP_END_FACTORY_REG(StPythonCompilableCustomOp);
}  // namespace ge

namespace ge {
using namespace gert;
namespace {
Status GenerateTaskForCustomOp(const Node &node, RunContext &run_context, std::vector<domi::TaskDef> &tasks) {
  (void)node;
  (void)run_context;
  domi::TaskDef task_def = {};
  task_def.set_stream_id(node.GetOpDesc()->GetStreamId());
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_sqe_num(5);

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_op_index(node.GetOpDesc()->GetId());
  tasks.push_back(task_def);
  return SUCCESS;
}

Status GenerateTaskForMemCopyAync(const Node &node, RunContext &run_context, std::vector<domi::TaskDef> &tasks) {
  if ((node.GetType() != MEMCPYASYNC) && (node.GetType() != IDENTITY)) {
    return SUCCESS;
  }
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
  auto kernel_def = task_def.mutable_memcpy_async();
  kernel_def->set_op_index(node.GetOpDesc()->GetId());
  kernel_def->set_kind(RT_MEMCPY_ADDR_DEVICE_TO_DEVICE);
  uint8_t *membase = run_context.dataMemBase;
  kernel_def->set_src((uintptr_t)membase + node.GetOpDesc()->GetInputOffset()[0]);
  kernel_def->set_dst((uintptr_t)membase + node.GetOpDesc()->GetOutputOffset()[0]);
  tasks.emplace_back(task_def);
  return SUCCESS;
}
void ConstructCustomInputOutputTensor(size_t input_num, size_t output_num, std::vector<ge::Tensor> &inputs,
                                      std::vector<ge::Tensor> &outputs) {
  for (size_t i = 0; i < input_num; i++) {
    std::vector<float32_t> input_data(2 * 2 * 2, 0);
    TensorDesc desc(Shape({2, 2, 2}));
    ge::Tensor input_tensor{desc};
    input_tensor.SetData(reinterpret_cast<uint8_t *>(input_data.data()), input_data.size() * sizeof(float32_t));
    inputs.emplace_back(input_tensor);
  }

  for (size_t i = 0; i < output_num; ++i) {
    std::vector<uint8_t> output_data_1(32, 0xff);
    TensorDesc output_desc_1(Shape({2, 2, 2}));
    ge::Tensor output_tensor_1{output_desc_1};
    output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
    outputs.emplace_back(output_tensor_1);
  }
  return;
}
void MockGenerateTask() {
  auto aicore_func = [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    if (node.GetType() == CONSTANT) {
      return SUCCESS;
    }

    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AiCoreLib");
    ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_KERNEL_BIN_ID, op_desc->GetName() + "_fake_id");
    const char kernel_bin[] = "kernel_bin";
    vector<char> buffer(kernel_bin, kernel_bin + strlen(kernel_bin));
    ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>("test", std::move(buffer));
    op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin_ptr);
    size_t arg_size = 100;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_kernel_name(node.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    tasks.emplace_back(task_def);
    return SUCCESS;
  };

  auto rts_func = [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    return SUCCESS;
  };

  MockForGenerateTask("AiCoreLib", aicore_func);
  MockForGenerateTask("RTSLib", rts_func);
}
void *output_addr = nullptr;
void **args_table = nullptr;
constexpr const char *kPythonCustomOpTypeForSt = "StPythonPybindRemoveCoverageCustomOp";
constexpr const char *kPythonAnnotatedArgsOpTypeForSt = "StPythonAnnotatedArgsCustomOp";
constexpr const char *kPythonAnnotatedArgsBadAttrOpTypeForSt = "StPythonAnnotatedArgsBadAttrCustomOp";
constexpr const char *kPythonCompilableOpTypeForSt = "StPythonCompilableCustomOp";
constexpr const char *kPythonRt2InferMetaOpTypeForSt = "StPythonRt2InferMetaCustomOp";
constexpr const char *kInferMetaCoverageOpTypeForSt = "StInferMetaCoverageCustomOp";
constexpr const char *kEnvPythonCustomOpPath = "ASCEND_CUSTOM_OPP_PATH";
constexpr const char *kEnvPythonPath = "PYTHONPATH";
constexpr char kSharedPybindCustomOpPreambleForSt[] = R"PY(from pathlib import Path
from ge.custom_op import (
    AnnotatedKernelLaunchInfo,
    get_compile_ctx,
    get_compile_platform_info,
    get_declare_launch_args_ctx,
    register_op,
    register_op_impl,
)
from ge.runtime import Tensor, TensorDesc

MARKER_FILE = r')PY";
constexpr char kSharedPybindEagerCustomOpForSt[] = R"PY('

@register_op(op_type=')PY";
constexpr char kSharedPybindEagerCustomOpImplForSt[] = R"PY(')
def infer_meta(x: TensorDesc, *, axis: int = 0) -> TensorDesc:
    return x

@register_op_impl(op_type=')PY";
constexpr char kSharedPybindAnnotatedArgsPrefixForSt[] = R"PY(')
class StPythonPybindRemoveCoverageCustomOp:
    def execute(self, x: Tensor, *, axis: int) -> None:
        Path(MARKER_FILE).write_text('executed', encoding='utf-8')

@register_op_impl(op_type=')PY";
constexpr char kSharedPybindAnnotatedArgsBodyForSt[] = R"PY(')
class StPythonAnnotatedArgsCustomOp:
    def __init__(self):
        self.saved = None

    def declare_launch_args(self, x: Tensor, z: Tensor, *, alpha: int) -> None:
        ctx = get_declare_launch_args_ctx()
        args = ctx.create_kernel_args()
        if alpha == 1:
            args.append_input(2, x)
        if alpha == 2:
            args.append_input(-1, x)
        if alpha == 3:
            args.append_scalar(-1)
        if alpha == 9:
            try:
                args.append_input(2, x)
            except IndexError as error:
                if 'index 2' not in str(error):
                    raise AssertionError('input index is missing from error')
            else:
                raise AssertionError('invalid input index was accepted')
            try:
                args.append_output(3, z)
            except IndexError as error:
                if 'index 3' not in str(error):
                    raise AssertionError('output index is missing from error')
            else:
                raise AssertionError('invalid output index was accepted')
        if alpha == 7:
            saved_tensor, saved_workspace, saved_args = self.saved
            for access in (
                lambda: saved_tensor.addr,
                lambda: saved_workspace.index,
                lambda: saved_args.append_scalar(0),
            ):
                try:
                    access()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('borrowed DLA object did not expire')
        args.append_input(0, x)
        args.append_output(0, z)
        args.append_scalar(alpha)
        workspace = None
        if alpha == 6:
            workspace = ctx.malloc_workspace(64)
            for name in ('index', 'addr'):
                try:
                    setattr(workspace, name, 0)
                except AttributeError:
                    pass
                else:
                    raise AssertionError('workspace property is writable')
            args.append_workspace(workspace)
        stream_id = ctx.get_stream_id()
        if alpha == 4:
            stream_id += 1
        ctx.add_launch(
            AnnotatedKernelLaunchInfo(
                kernel_name='st_python_dla',
                kernel_bin=b'\x01\x02',
                block_dim=1,
                stream_id=stream_id,
            ),
            args,
        )
        if alpha == 5:
            args.append_scalar(0)
        if alpha == 6:
            self.saved = (x, workspace, args)
)PY";
constexpr char kSharedPybindBadAttrCustomOpForSt[] = R"PY(')
class StPythonAnnotatedArgsBadAttrCustomOp:
    def declare_launch_args(self, x: Tensor, z: Tensor, *, beta: int) -> None:
        pass
)PY";
constexpr char kSharedPybindCompilablePreambleForSt[] = R"PY(
COMPILE_MARKER_FILE = r')PY";
constexpr char kSharedPybindCompilablePrefixForSt[] = R"PY('

PREVIOUS_COMPILE_OBJECTS = None

@register_op_impl(op_type=')PY";
constexpr char kSharedPybindCompilableBodyForSt[] = R"PY(')
class StPythonCompilableCustomOp:
    def compile(self, x: Tensor, z: Tensor, *, bias: int) -> None:
        global PREVIOUS_COMPILE_OBJECTS
        if PREVIOUS_COMPILE_OBJECTS is not None:
            previous_ctx, previous_platform, previous_tensor, previous_attrs = PREVIOUS_COMPILE_OBJECTS
            for access in (
                previous_platform.get_soc_version,
                lambda: previous_tensor.storage_shape.dims,
                lambda: previous_attrs.get_int(0),
            ):
                try:
                    access()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('compile borrowed object did not expire')
        ctx = get_compile_ctx()
        platform = get_compile_platform_info()
        if ctx._get_required_input_tensor(0).storage_shape.dims != x.storage_shape.dims:
            raise AssertionError('compile input tensor metadata mismatch')
        if ctx._get_attrs().get_int(0) != bias:
            raise AssertionError('compile attribute mismatch')
        for mutate in (
            lambda: x.shape.origin_shape.set_dim(0, 2),
            lambda: x.shape.set_storage_shape(x.shape.storage_shape),
            lambda: x.format.set_storage_format(2),
            lambda: x.expand_dims_type.set_expand_index(0),
        ):
            try:
                mutate()
            except RuntimeError:
                pass
            else:
                raise AssertionError('compile tensor metadata must be read-only')
        if ctx.get_option('st.python.compile.option') != 'enabled':
            raise AssertionError('compile option mismatch')
        if platform.get_platform_resource('version', 'NpuArch') != '2201':
            raise AssertionError('platform resource mismatch')
        if platform.get_platform_resource_group('SoCInfo')['ai_core_cnt'] != '24':
            raise AssertionError('platform resource group mismatch')
        if platform.get_core_num() != 8:
            raise AssertionError('core number mismatch')
        if platform.get_core_num('AiCore') != 8:
            raise AssertionError('AiCore number mismatch')
        if platform.get_soc_version() != 'Ascend910B':
            raise AssertionError('SoC version mismatch')
        if platform.get_ai_core_num() != 32:
            raise AssertionError('AI core number mismatch')
        PREVIOUS_COMPILE_OBJECTS = (ctx, platform, x, ctx._get_attrs())
        Path(COMPILE_MARKER_FILE).write_text('compiled', encoding='utf-8')
)PY";
constexpr char kInvalidSignaturePybindPreambleForSt[] = R"PY(from ge.custom_op import register_op_impl
from ge.runtime import Tensor

@register_op_impl(op_type=')PY";
constexpr char kValidBeforeInvalidPybindCustomOpForSt[] = R"PY(')
class StPythonValidBeforeInvalidCustomOp:
    def execute(self, x: Tensor) -> None:
        pass

@register_op_impl(op_type=')PY";
constexpr char kRt2InferMetaPreambleForSt[] = R"PY(from pathlib import Path
from typing import List

from ge.custom_op import register_op
from ge.graph import DataType
from ge.runtime import StorageShape, Tensor, TensorDesc

MARKER_FILE = r')PY";
constexpr char kRt2InferMetaOpTypePrefixForSt[] = R"PY('

@register_op(op_type=')PY";
constexpr char kRt2InferMetaFunctionForSt[] = R"PY(')
def infer_meta(x: TensorDesc, *, attr_int: int, attr_float: float, attr_bool: bool, attr_str: str,
               attr_dtype: DataType, attr_tensor: Tensor, attr_list_int: List[int],
               attr_list_float: List[float], attr_list_bool: List[bool], attr_list_str: List[str],
               attr_list_dtype: List[DataType], attr_list_list_int: List[List[int]]) -> TensorDesc:
    if (attr_int != 7 or abs(attr_float - 1.5) > 1e-6 or not attr_bool or attr_str != 'native' or
            attr_dtype != DataType.DT_INT32 or attr_tensor.data_type != DataType.DT_FLOAT16 or
            attr_list_int != [1, 2] or attr_list_float != [2.5, 3.5] or attr_list_bool != [True, False] or
            attr_list_str != ['a', 'b'] or attr_list_dtype != [DataType.DT_FLOAT, DataType.DT_INT32] or
            attr_list_list_int != [[3, 4], [5]]):
        raise AssertionError('native RuntimeAttrs values were not decoded correctly')
    dims = list(x.shape.storage_shape.dims)
    Path(MARKER_FILE).write_text(str(dims), encoding='utf-8')
    return TensorDesc(StorageShape([dims[0], attr_int], [dims[0], attr_int]), attr_dtype)
)PY";

class ScopedTempDirForCustomOpSt {
 public:
  ScopedTempDirForCustomOpSt() {
    char dir_template[] = "/tmp/ge_python_custom_op_st_XXXXXX";
    const auto *created_dir = mkdtemp(dir_template);
    dir_path_ = (created_dir == nullptr) ? std::string() : std::string(created_dir);
  }

  ~ScopedTempDirForCustomOpSt() {
    if (dir_path_.empty()) {
      return;
    }
    for (const auto &file_path : created_files_) {
      (void)remove(file_path.c_str());
    }
    (void)rmdir(dir_path_.c_str());
  }

  std::string FilePath(const std::string &file_name) const {
    return dir_path_ + "/" + file_name;
  }

  std::string CreateFilePath(const std::string &file_name) {
    const auto file_path = FilePath(file_name);
    created_files_.push_back(file_path);
    return file_path;
  }

 private:
  std::string dir_path_;
  std::vector<std::string> created_files_;
};

class ScopedEnvVarForCustomOpSt {
 public:
  ScopedEnvVarForCustomOpSt(const char *name, const std::string &value) : name_(name) {
    const char *old_value = getenv(name);
    if (old_value != nullptr) {
      old_value_ = old_value;
      has_old_value_ = true;
    }
    (void)setenv(name, value.c_str(), 1);
  }

  ~ScopedEnvVarForCustomOpSt() {
    if (has_old_value_) {
      (void)setenv(name_.c_str(), old_value_.c_str(), 1);
      return;
    }
    (void)unsetenv(name_.c_str());
  }

 private:
  std::string name_;
  std::string old_value_;
  bool has_old_value_{false};
};

class ScopedGraphOptionsForCustomOpSt {
 public:
  explicit ScopedGraphOptionsForCustomOpSt(const std::map<std::string, std::string> &options)
      : old_options_(GetThreadLocalContext().GetAllGraphOptions()) {
    GetThreadLocalContext().SetGraphOption(options);
  }

  ~ScopedGraphOptionsForCustomOpSt() {
    GetThreadLocalContext().SetGraphOption(old_options_);
  }

 private:
  std::map<std::string, std::string> old_options_;
};

Status GeneratePythonAnnotatedArgsTaskForSt(const char *const op_type, const int64_t alpha,
                                            const std::string &soc_version, std::vector<domi::TaskDef> &tasks,
                                            const bool is_unknown_shape = false) {
  const std::map<std::string, std::string> graph_options = {{SOC_VERSION, soc_version}};
  ScopedGraphOptionsForCustomOpSt scoped_graph_options(graph_options);
  auto graph = std::make_shared<ComputeGraph>(std::string("st_python_annotated_args_") + std::to_string(alpha));
  GE_ASSERT_NOTNULL(graph);
  graph->SetGraphUnknownFlag(is_unknown_shape);
  auto op_desc = std::make_shared<OpDesc>("st_python_annotated_args_node", op_type);
  GE_ASSERT_NOTNULL(op_desc);
  op_desc->SetId(7);
  op_desc->SetStreamId(3);
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  op_desc->AppendIrAttrName("alpha");
  GE_ASSERT_TRUE(AttrUtils::SetInt(op_desc, "alpha", alpha));
  GeTensorDesc input_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  input_desc.SetOriginShape(GeShape({1, 16}));
  GeTensorDesc output_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  output_desc.SetOriginShape(GeShape({1, 16}));
  GE_ASSERT_GRAPH_SUCCESS(op_desc->AddInputDesc("x", input_desc));
  GE_ASSERT_GRAPH_SUCCESS(op_desc->AddOutputDesc("z", output_desc));
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({2048});
  const auto node = graph->AddNode(op_desc);
  GE_ASSERT_NOTNULL(node);

  RunContext run_context = {};
  run_context.dataMemBase = reinterpret_cast<uint8_t *>(0x80000000UL);
  run_context.dataMemSize = 4096U;
  custom::CustomOpsKernelBuilder builder;
  return builder.GenerateTask(*node, run_context, tasks);
}

Status AddPythonAnnotatedArgsOpToModelForSt(DavinciModel &model, const uint32_t op_index, OpDescPtr &op_desc) {
  op_desc = std::make_shared<OpDesc>("st_python_annotated_args_loaded", kPythonAnnotatedArgsOpTypeForSt);
  GE_ASSERT_NOTNULL(op_desc);
  op_desc->SetId(op_index);
  op_desc->SetStreamId(3);
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  GeTensorDesc input_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  input_desc.SetOriginShape(GeShape({1, 16}));
  GeTensorDesc output_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  output_desc.SetOriginShape(GeShape({1, 16}));
  GE_ASSERT_GRAPH_SUCCESS(op_desc->AddInputDesc("x", input_desc));
  GE_ASSERT_GRAPH_SUCCESS(op_desc->AddOutputDesc("z", output_desc));
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({2048});
  GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC"));
  model.op_list_[op_index] = op_desc;
  model.SetCustomOpRegistry(CustomOpFactory::GetGlobalRegistryPtr());
  return SUCCESS;
}

void WriteTextFileForCustomOpSt(const std::string &file_path, const std::string &content) {
  std::ofstream file(file_path, std::ios::out | std::ios::trunc);
  ASSERT_TRUE(file.is_open());
  file << content;
}

std::string ReadTextFileForCustomOpSt(const std::string &file_path) {
  std::ifstream file(file_path);
  std::string content;
  std::getline(file, content);
  return content;
}

const std::string &GetSharedPybindCustomOpFilePathForSt() {
  static ScopedTempDirForCustomOpSt dir;
  static const std::string path = dir.CreateFilePath("pybind_custom_ops.py");
  return path;
}

const std::string &GetSharedPybindCustomOpMarkerFilePathForSt() {
  static ScopedTempDirForCustomOpSt dir;
  static const std::string path = dir.CreateFilePath("pybind_custom_op_marker.txt");
  return path;
}

const std::string &GetSharedPybindCustomOpCompileMarkerFilePathForSt() {
  static ScopedTempDirForCustomOpSt dir;
  static const std::string path = dir.CreateFilePath("pybind_custom_op_compile_marker.txt");
  return path;
}

const std::string &GetRt2InferMetaCustomOpFilePathForSt() {
  static ScopedTempDirForCustomOpSt dir;
  static const std::string path = dir.CreateFilePath("rt2_infer_meta_custom_op.py");
  return path;
}

const std::string &GetRt2InferMetaMarkerFilePathForSt() {
  static ScopedTempDirForCustomOpSt dir;
  static const std::string path = dir.CreateFilePath("rt2_infer_meta_marker.txt");
  return path;
}

const std::string &GetInvalidSignaturePybindCustomOpFilePathForSt() {
  static ScopedTempDirForCustomOpSt dir;
  static const std::string path = dir.CreateFilePath("pybind_invalid_signature_custom_op.py");
  return path;
}

void EnsureSharedPybindCustomOpFileForSt() {
  static std::once_flag once;
  std::call_once(once, []() {
    const auto python_file = std::string(kSharedPybindCustomOpPreambleForSt) +
                             GetSharedPybindCustomOpMarkerFilePathForSt() + kSharedPybindEagerCustomOpForSt +
                             kPythonCustomOpTypeForSt + kSharedPybindEagerCustomOpImplForSt + kPythonCustomOpTypeForSt +
                             kSharedPybindAnnotatedArgsPrefixForSt + kPythonAnnotatedArgsOpTypeForSt +
                             kSharedPybindAnnotatedArgsBodyForSt + kSharedPybindCompilablePreambleForSt +
                             GetSharedPybindCustomOpCompileMarkerFilePathForSt() + kSharedPybindCompilablePrefixForSt +
                             kPythonCompilableOpTypeForSt + kSharedPybindCompilableBodyForSt;
    WriteTextFileForCustomOpSt(GetSharedPybindCustomOpFilePathForSt(), python_file);
  });
}

void EnsureInvalidSignaturePybindCustomOpFileForSt() {
  static std::once_flag once;
  std::call_once(once, []() {
    const auto python_file = std::string(kInvalidSignaturePybindPreambleForSt) + kPythonCustomOpTypeForSt +
                             kValidBeforeInvalidPybindCustomOpForSt + kPythonAnnotatedArgsBadAttrOpTypeForSt +
                             kSharedPybindBadAttrCustomOpForSt;
    WriteTextFileForCustomOpSt(GetInvalidSignaturePybindCustomOpFilePathForSt(), python_file);
  });
}

void EnsureRt2InferMetaCustomOpFileForSt() {
  static std::once_flag once;
  std::call_once(once, []() {
    const auto python_file = std::string(kRt2InferMetaPreambleForSt) + GetRt2InferMetaMarkerFilePathForSt() +
                             kRt2InferMetaOpTypePrefixForSt + kPythonRt2InferMetaOpTypeForSt +
                             kRt2InferMetaFunctionForSt;
    WriteTextFileForCustomOpSt(GetRt2InferMetaCustomOpFilePathForSt(), python_file);
  });
}

class ScopedLoadedPythonCustomOpsForSt {
 public:
  ~ScopedLoadedPythonCustomOpsForSt() {
    if (active_) {
      custom_op::UnloadPythonCustomOps();
    }
  }

  void Dismiss() {
    active_ = false;
  }

 private:
  bool active_{true};
};

void RemovePythonPathForCustomOpSt(const std::string &path) {
  using PyIsInitializedFn = int (*)();
  using PyGILStateEnsureFn = int (*)();
  using PyGILStateReleaseFn = void (*)(int);
  using PySysGetObjectFn = void *(*)(const char *);
  using PyUnicodeFromStringFn = void *(*)(const char *);
  using PySequenceIndexFn = ssize_t (*)(void *, void *);
  using PySequenceDelItemFn = int (*)(void *, ssize_t);
  using PyErrClearFn = void (*)();
  using PyDecRefFn = void (*)(void *);
  auto *py_is_initialized = reinterpret_cast<PyIsInitializedFn>(dlsym(RTLD_DEFAULT, "Py_IsInitialized"));
  if ((py_is_initialized == nullptr) || (py_is_initialized() == 0)) {
    return;
  }
  auto *gil_ensure = reinterpret_cast<PyGILStateEnsureFn>(dlsym(RTLD_DEFAULT, "PyGILState_Ensure"));
  auto *gil_release = reinterpret_cast<PyGILStateReleaseFn>(dlsym(RTLD_DEFAULT, "PyGILState_Release"));
  auto *py_sys_get_object = reinterpret_cast<PySysGetObjectFn>(dlsym(RTLD_DEFAULT, "PySys_GetObject"));
  auto *py_unicode_from_string = reinterpret_cast<PyUnicodeFromStringFn>(dlsym(RTLD_DEFAULT, "PyUnicode_FromString"));
  auto *py_sequence_index = reinterpret_cast<PySequenceIndexFn>(dlsym(RTLD_DEFAULT, "PySequence_Index"));
  auto *py_sequence_del_item = reinterpret_cast<PySequenceDelItemFn>(dlsym(RTLD_DEFAULT, "PySequence_DelItem"));
  auto *py_err_clear = reinterpret_cast<PyErrClearFn>(dlsym(RTLD_DEFAULT, "PyErr_Clear"));
  auto *py_dec_ref = reinterpret_cast<PyDecRefFn>(dlsym(RTLD_DEFAULT, "Py_DecRef"));
  if ((gil_ensure == nullptr) || (gil_release == nullptr) || (py_sys_get_object == nullptr) ||
      (py_unicode_from_string == nullptr) || (py_sequence_index == nullptr) || (py_sequence_del_item == nullptr) ||
      (py_err_clear == nullptr) || (py_dec_ref == nullptr)) {
    return;
  }
  const auto state = gil_ensure();
  void *sys_path = py_sys_get_object("path");
  void *py_path = py_unicode_from_string(path.c_str());
  if ((sys_path != nullptr) && (py_path != nullptr)) {
    while (true) {
      const ssize_t index = py_sequence_index(sys_path, py_path);
      if (index < 0) {
        py_err_clear();
        break;
      }
      (void)py_sequence_del_item(sys_path, index);
    }
  }
  if (py_path != nullptr) {
    py_dec_ref(py_path);
  }
  gil_release(state);
}

void PrependPythonPathForCustomOpSt(const std::string &path) {
  using PyIsInitializedFn = int (*)();
  using PyGILStateEnsureFn = int (*)();
  using PyGILStateReleaseFn = void (*)(int);
  using PySysGetObjectFn = void *(*)(const char *);
  using PyUnicodeFromStringFn = void *(*)(const char *);
  using PyListInsertFn = int (*)(void *, ssize_t, void *);
  using PyDecRefFn = void (*)(void *);
  auto *py_is_initialized = reinterpret_cast<PyIsInitializedFn>(dlsym(RTLD_DEFAULT, "Py_IsInitialized"));
  if ((py_is_initialized == nullptr) || (py_is_initialized() == 0)) {
    return;
  }
  RemovePythonPathForCustomOpSt(path);
  auto *gil_ensure = reinterpret_cast<PyGILStateEnsureFn>(dlsym(RTLD_DEFAULT, "PyGILState_Ensure"));
  auto *gil_release = reinterpret_cast<PyGILStateReleaseFn>(dlsym(RTLD_DEFAULT, "PyGILState_Release"));
  auto *py_sys_get_object = reinterpret_cast<PySysGetObjectFn>(dlsym(RTLD_DEFAULT, "PySys_GetObject"));
  auto *py_unicode_from_string = reinterpret_cast<PyUnicodeFromStringFn>(dlsym(RTLD_DEFAULT, "PyUnicode_FromString"));
  auto *py_list_insert = reinterpret_cast<PyListInsertFn>(dlsym(RTLD_DEFAULT, "PyList_Insert"));
  auto *py_dec_ref = reinterpret_cast<PyDecRefFn>(dlsym(RTLD_DEFAULT, "Py_DecRef"));
  if ((gil_ensure == nullptr) || (gil_release == nullptr) || (py_sys_get_object == nullptr) ||
      (py_unicode_from_string == nullptr) || (py_list_insert == nullptr) || (py_dec_ref == nullptr)) {
    return;
  }
  const auto state = gil_ensure();
  void *sys_path = py_sys_get_object("path");
  void *py_path = py_unicode_from_string(path.c_str());
  if ((sys_path != nullptr) && (py_path != nullptr)) {
    (void)py_list_insert(sys_path, 0, py_path);
  }
  if (py_path != nullptr) {
    py_dec_ref(py_path);
  }
  gil_release(state);
}
}  // namespace

class CustomOpRefreshTest : public testing::Test {
 protected:
  void SetUp() {
    ModelManager::GetInstance().ClearAicpuSo();
    MockGenerateTask();
  }
  void TearDown() {
    OpsKernelBuilderRegistry::GetInstance().Unregister("AiCoreLib");
    OpsKernelBuilderRegistry::GetInstance().Unregister("RTSLib");
  }
};

class CustomOpFactoryStTest : public testing::Test {
 protected:
  void SetUp() override {
    PreparePythonPathForSt();
    (void)unsetenv(kEnvPythonCustomOpPath);
  }

  void TearDown() override {
    custom_op::UnloadPythonCustomOps();
    (void)unsetenv(kEnvPythonCustomOpPath);
    RestorePythonPathForSt();
  }

 private:
  void PreparePythonPathForSt() {
#ifdef ST_FUSION_PASS_PY_INSTALL_DIR
    const char *old_python_path = getenv(kEnvPythonPath);
    if (old_python_path != nullptr) {
      has_python_path_bak_ = true;
      python_path_bak_ = old_python_path;
      if (python_path_bak_.find(ST_FUSION_PASS_PY_INSTALL_DIR) == std::string::npos) {
        const std::string new_python_path = std::string(ST_FUSION_PASS_PY_INSTALL_DIR) + ":" + python_path_bak_;
        (void)setenv(kEnvPythonPath, new_python_path.c_str(), 1);
      }
      PrependPythonPathForCustomOpSt(ST_FUSION_PASS_PY_INSTALL_DIR);
      return;
    }
    (void)setenv(kEnvPythonPath, ST_FUSION_PASS_PY_INSTALL_DIR, 1);
    PrependPythonPathForCustomOpSt(ST_FUSION_PASS_PY_INSTALL_DIR);
#endif
  }

  void RestorePythonPathForSt() {
#ifdef ST_FUSION_PASS_PY_INSTALL_DIR
    RemovePythonPathForCustomOpSt(ST_FUSION_PASS_PY_INSTALL_DIR);
    if (has_python_path_bak_) {
      (void)setenv(kEnvPythonPath, python_path_bak_.c_str(), 1);
    } else {
      (void)unsetenv(kEnvPythonPath);
    }
    python_path_bak_.clear();
    has_python_path_bak_ = false;
#endif
  }

  std::string python_path_bak_;
  bool has_python_path_bak_{false};
};

class InferMetaCoverageCustomOpForSt final : public CustomOpInferMetaProvider {
 public:
  graphStatus InferMeta(gert::InferShapeContext *, CustomOpInferMetaResult *result) override {
    result->outputs.resize(1U);
    result->outputs[0U].shape = gert::StorageShape{{4, 5}, {4, 5}};
    result->outputs[0U].data_type = DT_FLOAT;
    return GRAPH_SUCCESS;
  }
};

class TestBaseCustomOp : public EagerExecuteOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    auto input_tensor0 = ctx->GetInputTensor(0);
    GE_ASSERT_NOTNULL(input_tensor0);
    auto input_shape0 = input_tensor0->GetShape().GetStorageShape();
    std::cout << "input shape dimnum " << input_shape0.GetDimNum() << std::endl;
    GE_ASSERT_TRUE(input_shape0.GetDimNum() == 3);
    GE_ASSERT_TRUE(input_shape0.GetDim(0) == 2);
    auto input_tensor1 = ctx->GetInputTensor(1);
    GE_ASSERT_NOTNULL(input_tensor1);
    auto input_shape1 = input_tensor1->GetShape().GetStorageShape();
    GE_ASSERT_TRUE(input_shape1.GetDimNum() == 3);
    GE_ASSERT_TRUE(input_shape1.GetDim(0) == 2);
    auto input_tensor2 = ctx->GetInputTensor(2);
    GE_ASSERT_NOTNULL(input_tensor2);
    auto input_shape2 = input_tensor2->GetShape().GetStorageShape();
    GE_ASSERT_TRUE(input_shape2.GetDimNum() == 3);
    GE_ASSERT_TRUE(input_shape2.GetDim(0) == 2);

    // allocator
    // 申请workspace有问题，taskinfo传入的是MemoryBlockManager但是在eager_op_execution_context里是按照GertAllocator来使用的
    auto workspaces = ctx->MallocWorkSpace(1024);
    GE_ASSERT_NOTNULL(workspaces);

    auto output_tensor = ctx->MallocOutputTensor(0, gert::StorageShape({2, 2, 2}, {2, 2, 2}),
                                                 gert::StorageFormat(FORMAT_ND, FORMAT_ND, ExpandDimsType()), DT_FLOAT);
    GE_ASSERT_NOTNULL(output_tensor);
    auto output_shape = output_tensor->GetShape().GetStorageShape();
    GE_ASSERT_TRUE(output_shape.GetDimNum() == 3);
    GE_ASSERT_TRUE(output_shape.GetDim(0) == 2);
    output_addr = output_tensor->GetAddr();
    GE_ASSERT_NOTNULL(output_addr);

    rtSetTaskTag("custom_op");
    void *input_0 = const_cast<void *>(ctx->GetInputTensor(0)->GetAddr());
    void *input_1 = const_cast<void *>(ctx->GetInputTensor(1)->GetAddr());
    void *input_2 = const_cast<void *>(ctx->GetInputTensor(2)->GetAddr());
    void *output_0 = const_cast<void *>(ctx->GetOutputTensor(0)->GetAddr());
    args_table[0] = static_cast<void *>(input_0);
    args_table[1] = static_cast<void *>(input_1);
    args_table[2] = static_cast<void *>(input_2);
    args_table[3] = static_cast<void *>(output_0);

    aclrtLaunchKernelWithHostArgs(nullptr, 0, nullptr, nullptr, &args_table[0], 32, nullptr, 0);
    return SUCCESS;
  }
};

class TestCompileOutputCustomOp : public EagerExecuteOp, public CompilableOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    return SUCCESS;
  }

  graphStatus Compile(gert::OpCompileContext *ctx) override {
    GE_ASSERT_NOTNULL(ctx);
    const auto required_output = ctx->GetRequiredOutputTensor(0U);
    GE_ASSERT_NOTNULL(required_output);
    GE_ASSERT_TRUE(required_output->GetShape().GetStorageShape() == gert::Shape({8, 16}));
    GE_ASSERT_TRUE(required_output->GetDataType() == DT_FLOAT16);
    GE_ASSERT_TRUE(required_output->GetStorageFormat() == FORMAT_ND);

    const auto dynamic_output0 = ctx->GetDynamicOutputTensor(1U, 0U);
    GE_ASSERT_NOTNULL(dynamic_output0);
    GE_ASSERT_TRUE(dynamic_output0->GetShape().GetStorageShape() == gert::Shape({16, 16}));
    GE_ASSERT_TRUE(dynamic_output0->GetDataType() == DT_FLOAT);

    const auto dynamic_output1 = ctx->GetDynamicOutputTensor(1U, 1U);
    GE_ASSERT_NOTNULL(dynamic_output1);
    GE_ASSERT_TRUE(dynamic_output1->GetShape().GetStorageShape() == gert::Shape({32, 16}));
    GE_ASSERT_TRUE(dynamic_output1->GetDataType() == DT_INT32);
    return SUCCESS;
  }
};

/**
 * 用例描述：fm外部设置，fm地址段不支持刷新，单次执行模型，args table正确
 *
 * 预置条件：
 * 1.构造计算图1，自定义算子直联Data和输出
 *  data0  data1  data2
 *     \    |      /
 *     \    |     /
 *       customop
 *          |
 *          |
 *       netoutput
 *
 * 测试步骤
 * 1.构造单个计算图1，设置fm地址段
 * 2.编译后执行计算图1
 * 3.判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.argstable的一致性和正确性均为成功，custom_op的args table为Execute流程分配，不走model args table的统一更新
 * 2.从dump图看产生了MemcpyAsyncTaskInfo
 */
TEST_F(CustomOpRefreshTest, model_execute_ok_with_customop_link_to_data) {
  MockForGenerateTask("DNN_VM_CUSTOM_OP_STORE", GenerateTaskForCustomOp);
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t *const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;
  args_table = new void *[4];

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto compute_graph = ShareGraph::BuildOnlyCustomOpKnowShapeGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  CustomOpFactory::RegisterCustomOpCreator(
      "CustomOp", []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestBaseCustomOp>(); });

  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto input_shape = context->GetInputShape(0U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  const auto infer_data_type_func = [](gert::InferDataTypeContext *context) -> graphStatus {
    const auto date_type = context->GetInputDataType(0U);
    EXPECT_EQ(context->SetOutputDataType(0, date_type), SUCCESS);
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
  };

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("CustomOp");

  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->infer_datatype = infer_data_type_func;
  op_impl_func->infer_shape_range = infer_shape_range_func;
  op_impl_func->output_shape_depend_compute = 1UL;

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));

  std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
  EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
  EXPECT_EQ(io_indexes.size(), 0U);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructCustomInputOutputTensor(3, 1, inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1, 2}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());
  EXPECT_EQ(SUCCESS, args_checker->CheckNodesArgsNotUpdated({"custom_op"}));

  delete[] args_table;
  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

/**
 * 用例描述：fm外部设置，fm地址段不支持刷新，单次执行模型，args table正确
 *
 * 预置条件：
 * 1.构造计算图1，自定义算子不在模型边界
 *
 *  data0  data1    data2  data3    data4  data5
 *     \    |         \     /          /    /
 *     \    |         \   /          /     /
 *         add0       add1          add2
 *              \       |         /
 *                \     |       /
 *                  customop           data6
 *                    |             /
 *                     |          /
 *                       add3
 *                        |
 *                      netoutput
 *
 *
 * 测试步骤
 * 1.构造单个计算图1，设置fm地址段
 * 2.编译后执行计算图1
 * 3.判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.argstable的一致性和正确性均为成功，custom_op的args table为Execute流程分配，不走model args table的统一更新
 * 2.从dump图看产生了未插入MemcpyAsyncTaskInfo
 */
TEST_F(CustomOpRefreshTest, model_execute_ok_with_customop_link_to_add) {
  MockForGenerateTask("DNN_VM_CUSTOM_OP_STORE", GenerateTaskForCustomOp);
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t *const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;
  args_table = new void *[4];

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto compute_graph = ShareGraph::BuildCustomOpWithAddKnowShapeGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  CustomOpFactory::RegisterCustomOpCreator(
      "CustomOp", []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestBaseCustomOp>(); });

  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto input_shape = context->GetInputShape(0U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  const auto infer_data_type_func = [](gert::InferDataTypeContext *context) -> graphStatus {
    const auto date_type = context->GetInputDataType(0U);
    EXPECT_EQ(context->SetOutputDataType(0, date_type), SUCCESS);
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
  };

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("CustomOp");

  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->infer_datatype = infer_data_type_func;
  op_impl_func->infer_shape_range = infer_shape_range_func;
  op_impl_func->output_shape_depend_compute = 1UL;

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));

  std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
  EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
  EXPECT_EQ(io_indexes.size(), 0U);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructCustomInputOutputTensor(7, 1, inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1, 2, 3, 4, 5, 6}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());
  EXPECT_EQ(SUCCESS, args_checker->CheckNodesArgsNotUpdated({"custom_op"}));

  delete[] args_table;
  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

/**
 * 用例描述：fm外部设置，fm地址段支持刷新，单次执行模型，args table正确
 *
 * 预置条件：
 * 1.构造计算图1，自定义算子不在模型边界
 *
 *  data0  data1    data2  data3    data4  data5
 *     \    |         \     /          /    /
 *     \    |         \   /          /     /
 *         add0       add1          add2
 *              \       |         /
 *                \     |       /
 *                  customop           data6
 *                    |             /
 *                     |          /
 *                       add3
 *                        |
 *                      netoutput
 *
 *
 * 测试步骤
 * 1.构造单个计算图1，设置fm地址段
 * 2.编译后执行计算图1
 * 3.判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.argstable的一致性和正确性均为成功，custom_op的args table为Execute流程分配，不走model args table的统一更新
 * 2.从dump图看插入MemcpyAsyncTaskInfo
 */
TEST_F(CustomOpRefreshTest, model_execute_ok_with_customop_link_to_add_and_fm_refresh) {
  MockForGenerateTask("DNN_VM_CUSTOM_OP_STORE", GenerateTaskForCustomOp);
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t *const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;
  args_table = new void *[4];

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto compute_graph = ShareGraph::BuildCustomOpWithAddKnowShapeGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  CustomOpFactory::RegisterCustomOpCreator(
      "CustomOp", []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestBaseCustomOp>(); });

  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto input_shape = context->GetInputShape(0U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  const auto infer_data_type_func = [](gert::InferDataTypeContext *context) -> graphStatus {
    const auto date_type = context->GetInputDataType(0U);
    EXPECT_EQ(context->SetOutputDataType(0, date_type), SUCCESS);
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
  };

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("CustomOp");

  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->infer_datatype = infer_data_type_func;
  op_impl_func->infer_shape_range = infer_shape_range_func;
  op_impl_func->output_shape_depend_compute = 1UL;

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));

  std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
  EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
  EXPECT_EQ(io_indexes.size(), 0U);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructCustomInputOutputTensor(7, 1, inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1, 2, 3, 4, 5, 6}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());
  EXPECT_EQ(SUCCESS, args_checker->CheckNodesArgsNotUpdated({"custom_op"}));

  delete[] args_table;
  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

/**
 * 用例描述：验证自定义编译算子编译上下文可构造输出Tensor
 * 预置条件：
 * 1. 注册一个包含1个required输出和2个dynamic输出实例的CompilableOp
 * 2. 构造仅包含该自定义算子的计算图
 * 测试步骤：
 * 1. 调用CustomGraphOptimizer执行自定义算子编译
 * 2. 在自定义算子的Compile函数中按IR输出读取输出Tensor
 * 预期结果：
 * 1. 自定义算子编译成功
 * 2. Compile函数中可以获取到3个输出Tensor，shape、format、datatype符合OpDesc描述
 */
TEST_F(CustomOpRefreshTest, custom_op_compile_context_construct_outputs_success) {
  const char *const op_type = "StCompileOutputCustomOp";
  auto graph = std::make_shared<ComputeGraph>("compile_output_custom_op_graph");
  auto op_desc = std::make_shared<OpDesc>("custom_op", op_type);
  op_desc->AppendIrOutput("y", kIrOutputRequired);
  op_desc->AppendIrOutput("dy", kIrOutputDynamic);

  GeTensorDesc required_output_desc(GeShape({8, 16}), FORMAT_ND, DT_FLOAT16);
  required_output_desc.SetOriginFormat(FORMAT_NCHW);
  ASSERT_EQ(op_desc->AddOutputDesc("y", required_output_desc), GRAPH_SUCCESS);
  GeTensorDesc dynamic_output_desc0(GeShape({16, 16}), FORMAT_ND, DT_FLOAT);
  dynamic_output_desc0.SetOriginFormat(FORMAT_ND);
  ASSERT_EQ(op_desc->AddOutputDesc("dy0", dynamic_output_desc0), GRAPH_SUCCESS);
  GeTensorDesc dynamic_output_desc1(GeShape({32, 16}), FORMAT_FRACTAL_NZ, DT_INT32);
  dynamic_output_desc1.SetOriginFormat(FORMAT_ND);
  ASSERT_EQ(op_desc->AddOutputDesc("dy1", dynamic_output_desc1), GRAPH_SUCCESS);

  ASSERT_NE(graph->AddNode(op_desc), nullptr);
  ASSERT_EQ(
      CustomOpFactory::RegisterCustomOpCreator(
          op_type, []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestCompileOutputCustomOp>(); }),
      GRAPH_SUCCESS);

  CustomGraphOptimizer optimizer;
  ASSERT_EQ(optimizer.OptimizeSubgraphPostProc(*graph), GRAPH_SUCCESS);
}

class TestArgsUpdaterCustomOp : public ArgsUpdater, public EagerExecuteOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    auto input_tensor0 = ctx->GetInputTensor(0);
    GE_ASSERT_NOTNULL(input_tensor0);
    auto input_shape0 = input_tensor0->GetShape().GetStorageShape();
    GE_ASSERT_TRUE(input_shape0.GetDimNum() == 3);
    GE_ASSERT_TRUE(input_shape0.GetDim(0) == 2);
    auto input_tensor1 = ctx->GetInputTensor(1);
    GE_ASSERT_NOTNULL(input_tensor1);
    auto input_shape1 = input_tensor1->GetShape().GetStorageShape();
    GE_ASSERT_TRUE(input_shape1.GetDimNum() == 3);
    GE_ASSERT_TRUE(input_shape1.GetDim(0) == 2);
    auto input_tensor2 = ctx->GetInputTensor(2);
    GE_ASSERT_NOTNULL(input_tensor2);
    auto input_shape2 = input_tensor2->GetShape().GetStorageShape();
    GE_ASSERT_TRUE(input_shape2.GetDimNum() == 3);
    GE_ASSERT_TRUE(input_shape2.GetDim(0) == 2);

    auto workspaces = ctx->MallocWorkSpace(1024);
    GE_ASSERT_NOTNULL(workspaces);

    auto output_tensor = ctx->MallocOutputTensor(0, gert::StorageShape({2, 2, 2}, {2, 2, 2}),
                                                 gert::StorageFormat(FORMAT_ND, FORMAT_ND, ExpandDimsType()), DT_FLOAT);
    GE_ASSERT_NOTNULL(output_tensor);
    auto output_shape = output_tensor->GetShape().GetStorageShape();
    GE_ASSERT_TRUE(output_shape.GetDimNum() == 3);
    GE_ASSERT_TRUE(output_shape.GetDim(0) == 2);
    output_addr = output_tensor->GetAddr();
    GE_ASSERT_NOTNULL(output_addr);

    rtSetTaskTag("custom_op");
    void *input_0 = const_cast<void *>(ctx->GetInputTensor(0)->GetAddr());
    void *input_1 = const_cast<void *>(ctx->GetInputTensor(1)->GetAddr());
    void *input_2 = const_cast<void *>(ctx->GetInputTensor(2)->GetAddr());
    void *output_0 = const_cast<void *>(ctx->GetOutputTensor(0)->GetAddr());
    args_table[0] = static_cast<void *>(input_0);
    args_table[1] = static_cast<void *>(input_1);
    args_table[2] = static_cast<void *>(input_2);
    args_table[3] = static_cast<void *>(output_0);

    aclrtLaunchKernelWithHostArgs(nullptr, 0, nullptr, nullptr, &args_table[0], 32, nullptr, 0);
    return SUCCESS;
  }

  graphStatus UpdateHostArgs(gert::UpdateArgsContext *ctx) override {
    return GRAPH_SUCCESS;
  }
};

/**
 * 用例描述：fm地址段支持刷新，ArgsUpdater算子端到端执行，args table被刷新
 *
 * 预置条件：
 * 1.构造计算图1，自定义ArgsUpdater算子直联Data和输出
 *  data0  data1  data2
 *     \    |      /
 *     \    |     /
 *       ArgsUpdaterOp
 *          |
 *          |
 *       netoutput
 *
 * 测试步骤
 * 1.构造单个计算图1，设置fm地址段且支持刷新
 * 2.注册ArgsUpdater类型算子（继承ArgsUpdater+EagerExecuteOp）
 * 3.编译后执行计算图1
 * 4.判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.argstable的一致性和正确性均为成功
 * 2.ArgsUpdater算子的args table通过预留段分配，走model args table的统一更新(H2D memcpy)
 * 3.CheckNodesArgsUpdated验证custom_op的args被刷新
 */
TEST_F(CustomOpRefreshTest, args_updater_end_to_end_with_fm_refresh) {
  MockForGenerateTask("DNN_VM_CUSTOM_OP_STORE", GenerateTaskForCustomOp);
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t *const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;
  args_table = new void *[4];

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto compute_graph = ShareGraph::BuildOnlyCustomOpKnowShapeGraph();
  auto custom_op_node = compute_graph->FindNode("custom_op");
  custom_op_node->GetOpDesc()->SetType("ArgsUpdaterOp");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  CustomOpFactory::RegisterCustomOpCreator(
      "ArgsUpdaterOp", []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestArgsUpdaterCustomOp>(); });

  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto input_shape = context->GetInputShape(0U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  const auto infer_data_type_func = [](gert::InferDataTypeContext *context) -> graphStatus {
    const auto date_type = context->GetInputDataType(0U);
    EXPECT_EQ(context->SetOutputDataType(0, date_type), SUCCESS);
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
  };

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("ArgsUpdaterOp");

  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->infer_datatype = infer_data_type_func;
  op_impl_func->infer_shape_range = infer_shape_range_func;
  op_impl_func->output_shape_depend_compute = 1UL;

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));

  std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
  EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
  EXPECT_EQ(io_indexes.size(), 0U);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructCustomInputOutputTensor(3, 1, inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1, 2}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());

  EXPECT_TRUE(CustomOpFactory::IsAddressRefreshable(AscendString("ArgsUpdaterOp")));

  delete[] args_table;
  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

class TestArgsUpdaterWithMallocCustomOp : public ArgsUpdater, public EagerExecuteOp {
 public:
  static int update_host_args_count_;

  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    auto input_tensor0 = ctx->GetInputTensor(0);
    GE_ASSERT_NOTNULL(input_tensor0);
    auto input_tensor1 = ctx->GetInputTensor(1);
    GE_ASSERT_NOTNULL(input_tensor1);
    auto input_tensor2 = ctx->GetInputTensor(2);
    GE_ASSERT_NOTNULL(input_tensor2);

    auto output_tensor = ctx->MallocOutputTensor(0, gert::StorageShape({2, 2, 2}, {2, 2, 2}),
                                                 gert::StorageFormat(FORMAT_ND, FORMAT_ND, ExpandDimsType()), DT_FLOAT);
    GE_ASSERT_NOTNULL(output_tensor);

    void *input_0 = const_cast<void *>(input_tensor0->GetAddr());
    void *input_1 = const_cast<void *>(input_tensor1->GetAddr());
    void *input_2 = const_cast<void *>(input_tensor2->GetAddr());
    void *output_0 = const_cast<void *>(ctx->GetOutputTensor(0)->GetAddr());

    uint64_t host_args[4] = {reinterpret_cast<uint64_t>(input_0), reinterpret_cast<uint64_t>(input_1),
                             reinterpret_cast<uint64_t>(input_2), reinterpret_cast<uint64_t>(output_0)};

    auto *dev_args = ctx->MallocReadOnlyDevArgs(host_args, sizeof(host_args));
    GE_ASSERT_NOTNULL(dev_args);

    rtSetTaskTag("custom_op");
    args_table[0] = input_0;
    args_table[1] = input_1;
    args_table[2] = input_2;
    args_table[3] = output_0;
    aclrtLaunchKernelWithHostArgs(nullptr, 0, nullptr, nullptr, &args_table[0], 32, nullptr, 0);
    return SUCCESS;
  }

  graphStatus UpdateHostArgs(gert::UpdateArgsContext *ctx) override {
    update_host_args_count_++;

    auto *input_tensor = ctx->GetInputTensor(0);
    auto *output_tensor = ctx->GetOutputTensor(0);
    auto *host_args = ctx->GetKernelArgs(gert::Placement::kPlacementHost, 0);

    if (host_args != nullptr && host_args->args_size >= sizeof(uint64_t) * 4 && input_tensor != nullptr &&
        output_tensor != nullptr) {
      auto *args = static_cast<uint64_t *>(host_args->args_data);
      args[0] = reinterpret_cast<uint64_t>(input_tensor->GetData<void>());
      args[3] = reinterpret_cast<uint64_t>(output_tensor->GetData<void>());
    }

    return GRAPH_SUCCESS;
  }
};
int TestArgsUpdaterWithMallocCustomOp::update_host_args_count_ = 0;

class TestArgsUpdaterMultiAllocCustomOp : public ArgsUpdater, public EagerExecuteOp {
 public:
  static int update_host_args_count_;

  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    auto output_tensor = ctx->MallocOutputTensor(0, gert::StorageShape({2, 2, 2}, {2, 2, 2}),
                                                 gert::StorageFormat(FORMAT_ND, FORMAT_ND, ExpandDimsType()), DT_FLOAT);
    GE_ASSERT_NOTNULL(output_tensor);

    for (int i = 0; i < 5; i++) {
      uint64_t host_args[8] = {static_cast<uint64_t>(i), 0, 0, 0, 0, 0, 0, 0};
      auto *dev_args = ctx->MallocReadOnlyDevArgs(host_args, sizeof(host_args));
      GE_ASSERT_NOTNULL(dev_args);
    }

    rtSetTaskTag("custom_op");
    void *input_0 = const_cast<void *>(ctx->GetInputTensor(0)->GetAddr());
    void *input_1 = const_cast<void *>(ctx->GetInputTensor(1)->GetAddr());
    void *input_2 = const_cast<void *>(ctx->GetInputTensor(2)->GetAddr());
    void *output_0 = const_cast<void *>(ctx->GetOutputTensor(0)->GetAddr());
    args_table[0] = input_0;
    args_table[1] = input_1;
    args_table[2] = input_2;
    args_table[3] = output_0;
    aclrtLaunchKernelWithHostArgs(nullptr, 0, nullptr, nullptr, &args_table[0], 32, nullptr, 0);
    return SUCCESS;
  }

  graphStatus UpdateHostArgs(gert::UpdateArgsContext *ctx) override {
    update_host_args_count_++;
    auto *host_args = ctx->GetKernelArgs(gert::Placement::kPlacementHost, 0);
    (void)host_args;
    return GRAPH_SUCCESS;
  }
};
int TestArgsUpdaterMultiAllocCustomOp::update_host_args_count_ = 0;

class TestEagerOnlyWithMallocCustomOp : public EagerExecuteOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    auto output_tensor = ctx->MallocOutputTensor(0, gert::StorageShape({2, 2, 2}, {2, 2, 2}),
                                                 gert::StorageFormat(FORMAT_ND, FORMAT_ND, ExpandDimsType()), DT_FLOAT);
    GE_ASSERT_NOTNULL(output_tensor);

    uint64_t host_args[4] = {0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD};
    auto *dev_args = ctx->MallocReadOnlyDevArgs(host_args, sizeof(host_args));
    GE_ASSERT_NOTNULL(dev_args);

    rtSetTaskTag("custom_op");
    void *input_0 = const_cast<void *>(ctx->GetInputTensor(0)->GetAddr());
    void *input_1 = const_cast<void *>(ctx->GetInputTensor(1)->GetAddr());
    void *input_2 = const_cast<void *>(ctx->GetInputTensor(2)->GetAddr());
    void *output_0 = const_cast<void *>(ctx->GetOutputTensor(0)->GetAddr());
    args_table[0] = input_0;
    args_table[1] = input_1;
    args_table[2] = input_2;
    args_table[3] = output_0;
    aclrtLaunchKernelWithHostArgs(nullptr, 0, nullptr, nullptr, &args_table[0], 32, nullptr, 0);
    return SUCCESS;
  }
};

/**
 * 用例描述：ArgsUpdater算子完整生命周期：MallocReadOnlyDevArgs分配kernel args + 两轮执行触发UpdateHostArgs
 *
 * 预置条件：
 * 1. 构造计算图，ArgsUpdater算子直联Data和输出，FM支持刷新
 *  data0  data1  data2
 *     \    |      /
 *     \    |     /
 *    ArgsUpdaterLifecycleOp
 *          |
 *       netoutput
 *
 * 测试步骤：
 * 1. 注册ArgsUpdater算子，Execute中调用MallocReadOnlyDevArgs分配kernel args
 * 2. 编译并第一轮执行（触发Distribute + IntegrateCustomOpArgs + 预留段分配）
 * 3. 使用不同输入地址进行第二轮执行（触发UpdateForExecute + UpdateHostArgs回调）
 * 预期结果：
 * 1. 两轮执行均成功
 * 2. 第二轮执行触发UpdateHostArgs回调（计数 > 0）
 * 3. ArgsUpdater算子走预留段分配路径，args通过统一H2D刷新
 */
TEST_F(CustomOpRefreshTest, args_updater_lifecycle_with_malloc_and_two_rounds) {
  MockForGenerateTask("DNN_VM_CUSTOM_OP_STORE", GenerateTaskForCustomOp);
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t *const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  args_table = new void *[4];
  TestArgsUpdaterWithMallocCustomOp::update_host_args_count_ = 0;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto compute_graph = ShareGraph::BuildOnlyCustomOpKnowShapeGraph();
  auto custom_op_node = compute_graph->FindNode("custom_op");
  custom_op_node->GetOpDesc()->SetType("ArgsUpdaterLifecycleOp");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  CustomOpFactory::RegisterCustomOpCreator("ArgsUpdaterLifecycleOp", []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<TestArgsUpdaterWithMallocCustomOp>();
  });

  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto input_shape = context->GetInputShape(0U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  const auto infer_data_type_func = [](gert::InferDataTypeContext *context) -> graphStatus {
    const auto date_type = context->GetInputDataType(0U);
    EXPECT_EQ(context->SetOutputDataType(0, date_type), SUCCESS);
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
  };

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("ArgsUpdaterLifecycleOp");
  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->infer_datatype = infer_data_type_func;
  op_impl_func->infer_shape_range = infer_shape_range_func;
  op_impl_func->output_shape_depend_compute = 1UL;

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  EXPECT_EQ(session.CompileGraph(graph_id), SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs1;
  std::vector<ge::Tensor> outputs1;
  ConstructCustomInputOutputTensor(3, 1, inputs1, outputs1);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs1, outputs1));

  std::vector<ge::Tensor> inputs2;
  std::vector<ge::Tensor> outputs2;
  ConstructCustomInputOutputTensor(3, 1, inputs2, outputs2);
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs2, outputs2));

  EXPECT_GT(TestArgsUpdaterWithMallocCustomOp::update_host_args_count_, 0);

  delete[] args_table;
  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

/**
 * 用例描述：ArgsUpdater算子多次MallocReadOnlyDevArgs，预留段耗尽后回退到Extra Pool
 *
 * 预置条件：
 * 1. 构造计算图，ArgsUpdater算子直联Data和输出，FM支持刷新
 * 2. 预留段大小 = (3 input + 1 output + 16 reserved) * 8 = 160 bytes
 * 3. 算子Execute中调用5次MallocReadOnlyDevArgs，每次64 bytes (总计320 > 160)
 *    调用1-3: 预留段分配 (192 > 160, 第3次溢出)
 *    调用3: 回退到新建Extra Pool (Tier3)
 *    调用4-5: 从已有Extra Pool分配 (Tier2)
 *
 * 测试步骤：
 * 1. 注册ArgsUpdater算子，Execute中5次调用MallocReadOnlyDevArgs
 * 2. 编译并执行（覆盖AllocateFromReservedSegment/AllocateFromNewPool/AllocateFromExistingPool）
 * 3. 第二轮执行（覆盖IntegrateExtraH2DCopyDatas/IntegrateExtraUpdateDatas的UpdateForExecute路径）
 * 预期结果：
 * 1. 两轮执行均成功
 * 2. 三级分配策略全部覆盖：reserved segment → new pool → existing pool
 * 3. Extra pool的H2D刷新和UpdateHostArgs回调正常工作
 */
TEST_F(CustomOpRefreshTest, args_updater_reserved_exhausted_fallback_to_extra_pool) {
  MockForGenerateTask("DNN_VM_CUSTOM_OP_STORE", GenerateTaskForCustomOp);
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t *const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  args_table = new void *[4];
  TestArgsUpdaterMultiAllocCustomOp::update_host_args_count_ = 0;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto compute_graph = ShareGraph::BuildOnlyCustomOpKnowShapeGraph();
  auto custom_op_node = compute_graph->FindNode("custom_op");
  custom_op_node->GetOpDesc()->SetType("MultiAllocOp");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  CustomOpFactory::RegisterCustomOpCreator("MultiAllocOp", []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<TestArgsUpdaterMultiAllocCustomOp>();
  });

  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto input_shape = context->GetInputShape(0U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  const auto infer_data_type_func = [](gert::InferDataTypeContext *context) -> graphStatus {
    const auto date_type = context->GetInputDataType(0U);
    EXPECT_EQ(context->SetOutputDataType(0, date_type), SUCCESS);
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
  };

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("MultiAllocOp");
  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->infer_datatype = infer_data_type_func;
  op_impl_func->infer_shape_range = infer_shape_range_func;
  op_impl_func->output_shape_depend_compute = 1UL;

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  EXPECT_EQ(session.CompileGraph(graph_id), SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs1;
  std::vector<ge::Tensor> outputs1;
  ConstructCustomInputOutputTensor(3, 1, inputs1, outputs1);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs1, outputs1));

  std::vector<ge::Tensor> inputs2;
  std::vector<ge::Tensor> outputs2;
  ConstructCustomInputOutputTensor(3, 1, inputs2, outputs2);
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs2, outputs2));

  EXPECT_GT(TestArgsUpdaterMultiAllocCustomOp::update_host_args_count_, 0);

  delete[] args_table;
  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

/**
 * 用例描述：EagerOnly算子（非ArgsUpdater）调用MallocReadOnlyDevArgs，走动态内存分配路径
 *
 * 预置条件：
 * 1. 构造计算图，EagerOnly算子直联Data和输出，FM不支持刷新
 *  data0  data1  data2
 *     \    |      /
 *     \    |     /
 *    EagerOnlyMallocOp
 *          |
 *       netoutput
 *
 * 测试步骤：
 * 1. 注册仅继承EagerExecuteOp的算子（不继承ArgsUpdater），Execute中调用MallocReadOnlyDevArgs
 * 2. 编译并执行
 * 预期结果：
 * 1. 执行成功
 * 2. MallocReadOnlyDevArgs走MallocDynamicMemory + H2D拷贝路径（非预留段）
 * 3. 算子不参与统一地址刷新（NeedReserveArgsTable=false）
 */
TEST_F(CustomOpRefreshTest, eager_only_op_with_malloc_read_only_dev_args) {
  MockForGenerateTask("DNN_VM_CUSTOM_OP_STORE", GenerateTaskForCustomOp);
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t *const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  args_table = new void *[4];

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto compute_graph = ShareGraph::BuildOnlyCustomOpKnowShapeGraph();
  auto custom_op_node = compute_graph->FindNode("custom_op");
  custom_op_node->GetOpDesc()->SetType("EagerOnlyMallocOp");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  CustomOpFactory::RegisterCustomOpCreator("EagerOnlyMallocOp", []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<TestEagerOnlyWithMallocCustomOp>();
  });

  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto input_shape = context->GetInputShape(0U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  const auto infer_data_type_func = [](gert::InferDataTypeContext *context) -> graphStatus {
    const auto date_type = context->GetInputDataType(0U);
    EXPECT_EQ(context->SetOutputDataType(0, date_type), SUCCESS);
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
  };

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("EagerOnlyMallocOp");
  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->infer_datatype = infer_data_type_func;
  op_impl_func->infer_shape_range = infer_shape_range_func;
  op_impl_func->output_shape_depend_compute = 1UL;

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  EXPECT_EQ(session.CompileGraph(graph_id), SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructCustomInputOutputTensor(3, 1, inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  EXPECT_FALSE(CustomOpFactory::IsAddressRefreshable(AscendString("EagerOnlyMallocOp")));

  delete[] args_table;
  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

/**
 * 用例描述：测试Python自定义算子在注册阶段拒绝与IR不匹配的回调签名。
 * 预置条件：
 * 1. 构造一个合法legacy实现和一个属性名与REG_OP定义不一致的Python实现。
 * 测试步骤：
 * 1. 通过LoadPythonCustomOps加载Python实现。
 * 2. 调用UnloadPythonCustomOps清理失败注册产生的部分状态。
 * 3. 查询CustomOpFactory中是否存在该Python自定义算子creator。
 * 预期结果：
 * 1. 注册阶段签名校验失败，LoadPythonCustomOps返回FAILED。
 * 2. 卸载后CustomOpFactory中不存在合法或非法Python自定义算子creator。
 */
TEST_F(CustomOpFactoryStTest, PythonCustomOpLoaderRejectsInvalidSignatureDuringRegistration) {
  EnsureInvalidSignaturePybindCustomOpFileForSt();
  ScopedEnvVarForCustomOpSt scoped_custom_opp_path(kEnvPythonCustomOpPath,
                                                   GetInvalidSignaturePybindCustomOpFilePathForSt());

  ASSERT_EQ(GePythonRuntimeManager::Instance().EnsureReady(), SUCCESS);
  EXPECT_EQ(custom_op::LoadPythonCustomOps(), FAILED);
  custom_op::UnloadPythonCustomOps();
  EXPECT_EQ(CustomOpFactory::CreateOrGetCustomOp(AscendString(kPythonCustomOpTypeForSt)), nullptr);
  EXPECT_EQ(CustomOpFactory::CreateOrGetCustomOp(AscendString(kPythonAnnotatedArgsBadAttrOpTypeForSt)), nullptr);
}

/**
 * 用例描述：测试Python自定义算子原型和实现通过loader注册、执行后，可以按类型移除注册信息。
 * 预置条件：
 * 1. 构造Python自定义算子原型和实现文件，并配置到ASCEND_CUSTOM_OPP_PATH。
 * 测试步骤：
 * 1. 通过LoadPythonCustomOps将原型和实现分别注册到OperatorFactory和CustomOpFactory。
 * 2. 校验原型creator生效，并通过CustomOpFactory创建Python自定义算子实例执行。
 * 3. 校验Python execute写入的标记文件。
 * 4. 调用UnloadPythonCustomOps移除该算子并清理runtime descriptor。
 * 预期结果：
 * 1. Python自定义算子execute被成功调用。
 * 2. 原型和实现creator均被移除，后续查询不存在，再次创建返回空指针。
 */
TEST_F(CustomOpFactoryStTest, register_and_remove_python_custom_op_proto_and_impl) {
  EnsureSharedPybindCustomOpFileForSt();
  const auto &marker_file = GetSharedPybindCustomOpMarkerFilePathForSt();
  (void)remove(marker_file.c_str());
  ScopedEnvVarForCustomOpSt scoped_custom_opp_path(kEnvPythonCustomOpPath, GetSharedPybindCustomOpFilePathForSt());

  ASSERT_EQ(GePythonRuntimeManager::Instance().EnsureReady(), SUCCESS);
  ASSERT_EQ(custom_op::LoadPythonCustomOps(), SUCCESS);
  ScopedLoadedPythonCustomOpsForSt loaded_python_custom_ops;

  ASSERT_TRUE(OperatorFactory::IsExistOp(kPythonCustomOpTypeForSt));
  EXPECT_FALSE(OperatorFactory::CreateOperator("st_python_proto", kPythonCustomOpTypeForSt).IsEmpty());

  const AscendString op_type(kPythonCustomOpTypeForSt);
  ASSERT_TRUE(CustomOpFactory::IsExistOp(op_type));
  auto *op = CustomOpFactory::CreateOrGetCustomOp(op_type);
  ASSERT_NE(op, nullptr);
  auto *eager_op = dynamic_cast<EagerExecuteOp *>(op);
  ASSERT_NE(eager_op, nullptr);

  gert::Tensor input;
  gert::Tensor output;
  // Eager context 在 1 个输入/输出之外还携带 4 个附加输入和 1 个附加输出。
  auto context_holder = gert::KernelRunContextFaker()
                            .NodeIoNum(1, 1)
                            .IrInputNum(1)
                            .NodeInputTd(0, DT_FLOAT, FORMAT_ND, FORMAT_ND)
                            .NodeOutputTd(0, DT_FLOAT, FORMAT_ND, FORMAT_ND)
                            .NodeAttrs({{"axis", AnyValue::CreateFrom<int64_t>(0)}})
                            .Inputs({&input, nullptr, nullptr, nullptr, nullptr})
                            .Outputs({&output, nullptr})
                            .Build();
  auto *ctx = context_holder.GetContext<gert::EagerOpExecutionContext>();
  ASSERT_NE(ctx, nullptr);
  EXPECT_EQ(eager_op->Execute(ctx), SUCCESS);
  EXPECT_EQ(ReadTextFileForCustomOpSt(marker_file), "executed");

  custom_op::UnloadPythonCustomOps();
  loaded_python_custom_ops.Dismiss();

  EXPECT_FALSE(OperatorFactory::IsExistOp(kPythonCustomOpTypeForSt));
  EXPECT_FALSE(CustomOpFactory::IsExistOp(op_type));
  EXPECT_EQ(CustomOpFactory::CreateOrGetCustomOp(op_type), nullptr);
}

TEST_F(CustomOpFactoryStTest, PythonCompilableCustomOpRealCallbackCompiles) {
  EnsureSharedPybindCustomOpFileForSt();
  const auto &marker_file = GetSharedPybindCustomOpCompileMarkerFilePathForSt();
  (void)remove(marker_file.c_str());
  ScopedEnvVarForCustomOpSt scoped_custom_opp_path(kEnvPythonCustomOpPath, GetSharedPybindCustomOpFilePathForSt());
  ScopedGraphOptionsForCustomOpSt scoped_graph_options(
      std::map<std::string, std::string>{{"st.python.compile.option", "enabled"}});

  ASSERT_EQ(GePythonRuntimeManager::Instance().EnsureReady(), SUCCESS);
  ASSERT_EQ(custom_op::LoadPythonCustomOps(), SUCCESS);
  ScopedLoadedPythonCustomOpsForSt loaded_python_custom_ops;

  const AscendString op_type(kPythonCompilableOpTypeForSt);
  ASSERT_TRUE(CustomOpFactory::IsExistOp(op_type));
  auto *const base_op = CustomOpFactory::CreateOrGetCustomOp(op_type);
  ASSERT_NE(base_op, nullptr);
  EXPECT_NE(CustomOpCast<CompilableOp>(base_op), nullptr);

  auto graph = std::make_shared<ComputeGraph>("st_python_compilable_graph");
  ASSERT_NE(graph, nullptr);
  auto op_desc = std::make_shared<OpDesc>("st_python_compilable_node", kPythonCompilableOpTypeForSt);
  ASSERT_NE(op_desc, nullptr);
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  op_desc->AppendIrAttrName("bias");
  ASSERT_TRUE(AttrUtils::SetInt(op_desc, "bias", 4));
  GeTensorDesc input_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  input_desc.SetOriginShape(GeShape({1, 16}));
  GeTensorDesc output_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  output_desc.SetOriginShape(GeShape({1, 16}));
  ASSERT_EQ(op_desc->AddInputDesc("x", input_desc), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->AddOutputDesc("z", output_desc), GRAPH_SUCCESS);
  ASSERT_NE(graph->AddNode(op_desc), nullptr);

  CustomGraphOptimizer optimizer;
  ASSERT_EQ(optimizer.OptimizeWholeGraph(*graph), SUCCESS);
  ASSERT_EQ(optimizer.OptimizeWholeGraph(*graph), SUCCESS);
  EXPECT_EQ(ReadTextFileForCustomOpSt(marker_file), "compiled");
}

/**
 * 验证 Python infer_meta 通过真实 RT2 InferShape kernel 使用运行时 shape，
 * 并从 native RuntimeAttrs 读取设计支持的全部 12 类属性。
 */
TEST_F(CustomOpFactoryStTest, PythonCustomOpInferMetaRunsThroughRt2WithNativeAttrs) {
  EnsureRt2InferMetaCustomOpFileForSt();
  const auto &marker_file = GetRt2InferMetaMarkerFilePathForSt();
  (void)remove(marker_file.c_str());
  ScopedEnvVarForCustomOpSt scoped_custom_opp_path(kEnvPythonCustomOpPath, GetRt2InferMetaCustomOpFilePathForSt());

  ASSERT_EQ(GePythonRuntimeManager::Instance().EnsureReady(), SUCCESS);
  ASSERT_EQ(custom_op::LoadPythonCustomOps(), SUCCESS);
  ScopedLoadedPythonCustomOpsForSt loaded_python_custom_ops;

  auto *base_op = CustomOpFactory::CreateOrGetCustomOp(AscendString(kPythonRt2InferMetaOpTypeForSt));
  ASSERT_NE(base_op, nullptr);

  gert::StorageShape input_shape({7, 13}, {7, 13});
  gert::Tensor output;
  auto attr_tensor = FakeGeTensorHolder()
                         .DataType(DT_FLOAT16)
                         .OriginFormat(FORMAT_ND)
                         .StorageFormat(FORMAT_ND)
                         .OriginShape({1})
                         .StorageShape({1})
                         .Build();
  auto infer_shape_func = kernel::InferCustomOpShapeFromInput;
  auto run_context =
      gert::KernelRunContextFaker()
          .KernelIONum(3, 1)
          .NodeIoNum(1, 1)
          .IrInputNum(1)
          .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
          .NodeOutputTd(0, DT_INT32, FORMAT_ND, FORMAT_ND)
          .NodeAttrs({{"attr_int", AnyValue::CreateFrom<int64_t>(7)},
                      {"attr_float", AnyValue::CreateFrom<float>(1.5F)},
                      {"attr_bool", AnyValue::CreateFrom<bool>(true)},
                      {"attr_str", AnyValue::CreateFrom<std::string>("native")},
                      {"attr_dtype", AnyValue::CreateFrom<DataType>(DT_INT32)},
                      {"attr_tensor", AnyValue::CreateFrom<GeTensor>(*attr_tensor)},
                      {"attr_list_int", AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
                      {"attr_list_float", AnyValue::CreateFrom<std::vector<float>>({2.5F, 3.5F})},
                      {"attr_list_bool", AnyValue::CreateFrom<std::vector<bool>>({true, false})},
                      {"attr_list_str", AnyValue::CreateFrom<std::vector<std::string>>({"a", "b"})},
                      {"attr_list_dtype", AnyValue::CreateFrom<std::vector<DataType>>({DT_FLOAT, DT_INT32})},
                      {"attr_list_list_int", AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>({{3, 4}, {5}})}})
          .Inputs({&input_shape, base_op, reinterpret_cast<void *>(infer_shape_func)})
          .Outputs({&output})
          .Build();

  const auto funcs = gert::KernelRegistry::GetInstance().FindKernelFuncs("InferShape");
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->run_func(run_context), GRAPH_SUCCESS);
  EXPECT_EQ(output.GetOriginShape(), gert::Shape({7, 7}));
  EXPECT_EQ(ReadTextFileForCustomOpSt(marker_file), "[7, 13]");
}

TEST_F(CustomOpFactoryStTest, CustomOpInferMetaCompilePath) {
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(
                AscendString(kInferMetaCoverageOpTypeForSt),
                []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<InferMetaCoverageCustomOpForSt>(); }),
            GRAPH_SUCCESS);

  auto op_desc = std::make_shared<OpDesc>("st_infer_meta", kInferMetaCoverageOpTypeForSt);
  ASSERT_EQ(op_desc->AddInputDesc(GeTensorDesc(GeShape({2, 3}), FORMAT_ND, DT_FLOAT)), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->AddOutputDesc(GeTensorDesc(GeShape({1}), FORMAT_ND, DT_UNDEFINED)), GRAPH_SUCCESS);
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("y", kIrOutputRequired);
  op_desc->AddInferFunc([](Operator &) { return GRAPH_SUCCESS; });
  auto op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  EXPECT_EQ(op.InferShapeAndType(), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetOutputDesc(0U).GetShape(), GeShape({4, 5}));
  EXPECT_EQ(op_desc->GetOutputDesc(0U).GetDataType(), DT_FLOAT);

  CustomOpFactory::RemoveCustomOps({AscendString(kInferMetaCoverageOpTypeForSt)});
}

TEST_F(CustomOpFactoryStTest, PythonAnnotatedArgsCustomOpLoaderGeneratesTaskDef) {
  EnsureSharedPybindCustomOpFileForSt();
  ScopedEnvVarForCustomOpSt scoped_custom_opp_path(kEnvPythonCustomOpPath, GetSharedPybindCustomOpFilePathForSt());

  ASSERT_EQ(GePythonRuntimeManager::Instance().EnsureReady(), SUCCESS);
  ASSERT_EQ(custom_op::LoadPythonCustomOps(), SUCCESS);
  ScopedLoadedPythonCustomOpsForSt loaded_python_custom_ops;

  const AscendString op_type(kPythonAnnotatedArgsOpTypeForSt);
  ASSERT_TRUE(CustomOpFactory::IsExistOp(op_type));
  auto *const base_op = CustomOpFactory::CreateOrGetCustomOp(op_type);
  ASSERT_NE(base_op, nullptr);
  EXPECT_NE(CustomOpCast<AnnotatedArgsOp>(base_op), nullptr);
  EXPECT_EQ(CustomOpCast<EagerExecuteOp>(base_op), nullptr);

  std::vector<domi::TaskDef> tasks;
  ASSERT_EQ(GeneratePythonAnnotatedArgsTaskForSt(kPythonAnnotatedArgsOpTypeForSt, 0, "Ascend910B", tasks), SUCCESS);

  ASSERT_EQ(tasks.size(), 1U);
  const auto &task = tasks[0];
  EXPECT_EQ(task.stream_id(), 3U);
  const auto &kernel = task.kernel();
  EXPECT_EQ(kernel.kernel_name(), "st_python_dla");
  EXPECT_EQ(kernel.stub_func(), "st_python_dla");
  EXPECT_EQ(kernel.block_dim(), 1U);
  EXPECT_EQ(kernel.args_size(), 24U);
  std::vector<ArgDesc> arg_descs;
  ASSERT_EQ(ArgsFormatDescUtils::Parse(kernel.context().args_format(), arg_descs), GRAPH_SUCCESS);
  ASSERT_EQ(arg_descs.size(), 3U);
  EXPECT_EQ(arg_descs[0].addr_type, AddrType::INPUT_INSTANCE);
  EXPECT_EQ(arg_descs[0].ir_idx, 0);
  EXPECT_EQ(arg_descs[1].addr_type, AddrType::OUTPUT_INSTANCE);
  EXPECT_EQ(arg_descs[1].ir_idx, 0);
  EXPECT_EQ(arg_descs[2].addr_type, AddrType::CUSTOM_VALUE);

  custom_op::UnloadPythonCustomOps();
  loaded_python_custom_ops.Dismiss();
  EXPECT_FALSE(CustomOpFactory::IsExistOp(op_type));
}

TEST_F(CustomOpFactoryStTest, PythonAnnotatedArgsUnknownShapeGeneratesBasicTaskDef) {
  EnsureSharedPybindCustomOpFileForSt();
  ScopedEnvVarForCustomOpSt scoped_custom_opp_path(kEnvPythonCustomOpPath, GetSharedPybindCustomOpFilePathForSt());

  ASSERT_EQ(GePythonRuntimeManager::Instance().EnsureReady(), SUCCESS);
  ASSERT_EQ(custom_op::LoadPythonCustomOps(), SUCCESS);
  ScopedLoadedPythonCustomOpsForSt loaded_python_custom_ops;

  std::vector<domi::TaskDef> tasks;
  ASSERT_EQ(GeneratePythonAnnotatedArgsTaskForSt(kPythonAnnotatedArgsOpTypeForSt, 0, "Ascend910B", tasks, true),
            SUCCESS);

  ASSERT_EQ(tasks.size(), 1U);
  const auto &task = tasks[0];
  EXPECT_EQ(task.stream_id(), 3U);
  EXPECT_EQ(task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  EXPECT_EQ(task.sqe_num(), 5U);
  EXPECT_EQ(task.kernel().context().op_index(), 0U);
  EXPECT_TRUE(task.kernel().kernel_name().empty());
  EXPECT_TRUE(task.kernel().context().args_format().empty());
}

TEST_F(CustomOpFactoryStTest, PythonAnnotatedArgsRealCallbackRejectsInvalidNativeUsage) {
  EnsureSharedPybindCustomOpFileForSt();
  ScopedEnvVarForCustomOpSt scoped_custom_opp_path(kEnvPythonCustomOpPath, GetSharedPybindCustomOpFilePathForSt());

  ASSERT_EQ(GePythonRuntimeManager::Instance().EnsureReady(), SUCCESS);
  ASSERT_EQ(custom_op::LoadPythonCustomOps(), SUCCESS);
  ScopedLoadedPythonCustomOpsForSt loaded_python_custom_ops;

  for (const int64_t alpha : {1, 2, 3, 4, 5}) {
    std::vector<domi::TaskDef> tasks;
    EXPECT_NE(GeneratePythonAnnotatedArgsTaskForSt(kPythonAnnotatedArgsOpTypeForSt, alpha, "Ascend910B", tasks),
              SUCCESS)
        << "alpha=" << alpha;
    EXPECT_TRUE(tasks.empty()) << "alpha=" << alpha;
  }

  std::vector<domi::TaskDef> index_message_tasks;
  EXPECT_EQ(GeneratePythonAnnotatedArgsTaskForSt(kPythonAnnotatedArgsOpTypeForSt, 9, "Ascend910B", index_message_tasks),
            SUCCESS);
  EXPECT_EQ(index_message_tasks.size(), 1U);
}

TEST_F(CustomOpFactoryStTest, PythonAnnotatedArgsRealCallbackEnforcesBorrowedLifetime) {
  EnsureSharedPybindCustomOpFileForSt();
  ScopedEnvVarForCustomOpSt scoped_custom_opp_path(kEnvPythonCustomOpPath, GetSharedPybindCustomOpFilePathForSt());

  ASSERT_EQ(GePythonRuntimeManager::Instance().EnsureReady(), SUCCESS);
  ASSERT_EQ(custom_op::LoadPythonCustomOps(), SUCCESS);
  ScopedLoadedPythonCustomOpsForSt loaded_python_custom_ops;

  std::vector<domi::TaskDef> capture_tasks;
  ASSERT_EQ(GeneratePythonAnnotatedArgsTaskForSt(kPythonAnnotatedArgsOpTypeForSt, 6, "Ascend910B", capture_tasks),
            SUCCESS);
  ASSERT_EQ(capture_tasks.size(), 1U);
  EXPECT_EQ(capture_tasks[0].kernel().args_size(), 32U);

  std::vector<domi::TaskDef> verify_tasks;
  ASSERT_EQ(GeneratePythonAnnotatedArgsTaskForSt(kPythonAnnotatedArgsOpTypeForSt, 7, "Ascend910B", verify_tasks),
            SUCCESS);
  ASSERT_EQ(verify_tasks.size(), 1U);
  EXPECT_EQ(verify_tasks[0].kernel().args_size(), 24U);
}

TEST_F(CustomOpFactoryStTest, PythonAnnotatedArgsMobileOmcAcceptsSingleLaunch) {
  EnsureSharedPybindCustomOpFileForSt();
  ScopedEnvVarForCustomOpSt scoped_custom_opp_path(kEnvPythonCustomOpPath, GetSharedPybindCustomOpFilePathForSt());

  ASSERT_EQ(GePythonRuntimeManager::Instance().EnsureReady(), SUCCESS);
  ASSERT_EQ(custom_op::LoadPythonCustomOps(), SUCCESS);
  ScopedLoadedPythonCustomOpsForSt loaded_python_custom_ops;

  std::vector<domi::TaskDef> tasks;
  ASSERT_EQ(GeneratePythonAnnotatedArgsTaskForSt(kPythonAnnotatedArgsOpTypeForSt, 8, "KirinX90", tasks), SUCCESS);
  ASSERT_EQ(tasks.size(), 1U);
  EXPECT_EQ(tasks[0].type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  EXPECT_EQ(tasks[0].kernel().kernel_name(), "st_python_dla");
  EXPECT_EQ(tasks[0].kernel().args_size(), 24U);
  const auto op_index = tasks[0].kernel().context().op_index();

  DavinciModel model(0, nullptr);
  std::vector<uint8_t> feature_mem(4096U, 0U);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(feature_mem.data());
  model.runtime_param_.mem_size = feature_mem.size();
  OpDescPtr op_desc;
  ASSERT_EQ(AddPythonAnnotatedArgsOpToModelForSt(model, op_index, op_desc), SUCCESS);
  ASSERT_EQ(model.GetOpByIndex(op_index), op_desc);

  auto ge_model = MakeShared<GeModel>();
  ASSERT_NE(ge_model, nullptr);
  std::vector<char> kernel_bin = {0x01, 0x02};
  ge_model->GetTBEKernelStore().AddTBEKernel(MakeShared<OpKernelBin>("st_python_dla", std::move(kernel_bin)));
  model.ge_model_ = ge_model;

  const uint64_t input_addr = model.runtime_param_.mem_base + 1024U;
  const uint64_t output_addr = model.runtime_param_.mem_base + 2048U;
  model.logical_mem_allocations_.push_back({0U, input_addr, 32U, MemAllocation::INPUT, 0U, 0U, 0U, 32U});
  model.logical_mem_allocations_.push_back({1U, output_addr, 32U, MemAllocation::OUTPUT, 0U, 0U, 0U, 0U});
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  std::vector<rtStream_t> streams(4U, nullptr);
  for (auto &stream : streams) {
    ASSERT_EQ(model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0U, 0, 0U), SUCCESS);
  }
  model.stream_list_ = streams;

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  ASSERT_EQ(task_info.ParseTaskRunParam(tasks[0], &model, task_run_param), SUCCESS);
  std::vector<uint8_t> loaded_args(tasks[0].kernel().args().cbegin(), tasks[0].kernel().args().cend());
  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = reinterpret_cast<uint64_t>(loaded_args.data());
  IowAddrs iow_addrs;
  iow_addrs.input_logic_addrs = {{input_addr, static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)}};
  iow_addrs.output_logic_addrs = {{output_addr, static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)}};
  ASSERT_EQ(task_info.Init(tasks[0], &model, args, {}, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
}

TEST_F(CustomOpFactoryStTest, load_python_custom_ops_if_needed_fails_for_missing_python_file) {
  ScopedTempDirForCustomOpSt temp_dir;
  const auto missing_python_file = temp_dir.FilePath("missing_custom_op.py");
  ScopedEnvVarForCustomOpSt scoped_custom_opp_path(kEnvPythonCustomOpPath, missing_python_file);

  EXPECT_EQ(custom_op::LoadPythonCustomOpsIfNeeded(), FAILED);
}

TEST_F(CustomOpFactoryStTest, check_need_load_python_custom_ops_skips_missing_non_python_path) {
  ScopedTempDirForCustomOpSt temp_dir;
  const auto missing_so_file = temp_dir.FilePath("missing_custom_op.so");
  ScopedEnvVarForCustomOpSt scoped_custom_opp_path(kEnvPythonCustomOpPath, missing_so_file);

  bool need_load = true;
  EXPECT_EQ(custom_op::CheckNeedLoadPythonCustomOps(need_load), SUCCESS);
  EXPECT_FALSE(need_load);
}

}  // namespace ge
