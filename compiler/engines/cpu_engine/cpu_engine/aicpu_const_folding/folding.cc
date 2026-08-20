/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "folding.h"

#include <vector>
#include <set>
#include <string>
#include <memory>
#include <new>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <dirent.h>
#include <dlfcn.h>
#include <cstring>
#include "cpu_kernel_register.h"
#include "cpu_kernel.h"
#include "cpu_context.h"
#include "proto/aicpu/cpu_attr.pb.h"
#include "proto/aicpu/cpu_node_def.pb.h"
#include "proto/aicpu/cpu_tensor.pb.h"
#include "proto/aicpu/cpu_tensor_shape.pb.h"
#include "util/aicpu_log.h"
#include "graph/types.h"
#include "mmpa/mmpa_api.h"

namespace {
const char *const kVtString = "VT_STRING";
const char *const kVtListString = "VT_LIST_STRING";
const char *const kVtFloat = "VT_FLOAT";
const char *const kVtListFloat = "VT_LIST_FLOAT";
const char *const kVtInt = "VT_INT";
const char *const kVtListInt = "VT_LIST_INT";
const char *const kVtListListInt = "VT_LIST_LIST_INT";
const char *const kVtBool = "VT_BOOL";
const char *const kVtListBool = "VT_LIST_BOOL";
const char *const kVtDataType = "VT_DATA_TYPE";
const char *const kVtListDataType = "VT_LIST_DATA_TYPE";
const char *const kVtTensor = "VT_TENSOR";
const char *const kVtListTensor = "VT_LIST_TENSOR";
const char kPathSeparator = '/';
const char *const kConstantFoldingSoPrefix = "libopconstant_folding_";
const char *const kConstantFoldingSoSuffix = ".so";
const char *const kExcludedConstantFoldingSo = "libconstant_folding_ops.so";
const char *const kSymGetAllRegisteredOpTypesV2 = "GetAllRegisteredOpTypesV2";
const char *const kSymIsRegisteredV2 = "IsRegisteredV2";
const char *const kSymRunCpuKernelV2 = "RunCpuKernelV2";
constexpr uint32_t kFusedHostCpuShapeChanged = 1U;
constexpr uint32_t kFusedHostCpuDataChanged = 2U;

using AttrValueMap = google::protobuf::Map<string, aicpuops::AttrValue>;

using GetAllRegisteredOpTypesV2Fn = std::vector<std::string> (*)();
using IsRegisteredV2Fn = bool (*)(const std::string &);
using RunCpuKernelV2Fn = uint32_t (*)(aicpu::CpuKernelContext &);

struct V2ModuleBinding {
  void *handle = nullptr;
  GetAllRegisteredOpTypesV2Fn get_all_op_types = nullptr;
  IsRegisteredV2Fn is_registered = nullptr;
  RunCpuKernelV2Fn run_cpu_kernel = nullptr;
  std::string so_name;
};

struct FusedTensorBindingState {
  ge::DataType data_type = ge::DT_UNDEFINED;
  ge::Format format = ge::FORMAT_RESERVED;
  const void *data = nullptr;
  size_t data_size = 0U;
  std::vector<int64_t> dims;
  bool initialized = false;
};

struct FusedCpuKernelPlan {
  std::unique_ptr<aicpuops::NodeDef> node_def;
  std::unique_ptr<aicpu::CpuKernelContext> context;
  const V2ModuleBinding *v2_binding = nullptr;
  std::shared_ptr<aicpu::CpuKernel> v1_kernel;
  std::vector<aicpu::Tensor *> input_tensors;
  std::vector<aicpu::Tensor *> output_tensors;
  std::vector<FusedTensorBindingState> input_states;
  std::vector<FusedTensorBindingState> output_states;
};

struct FusedCpuKernelChainNodeDesc {
  const ge::Operator *op;
  const ge::Tensor *const *inputs;
  size_t input_num;
  ge::Tensor *const *outputs;
  size_t output_num;
  const int32_t *input_binding_indices;
  const int32_t *output_binding_indices;
};

struct FusedHostCpuTensorBinding {
  const int64_t *dims;
  uint8_t *data;
  size_t dim_num;
  size_t data_size;
  uint32_t flags;
};

struct FusedCpuKernelBinding {
  const ge::Tensor *source;
  aicpu::Tensor *target;
  FusedTensorBindingState *state;
  size_t binding_index;
};

struct FusedCpuKernelChainNode {
  FusedCpuKernelPlan plan;
};

struct FusedCpuKernelChainPlan {
  std::vector<FusedCpuKernelChainNode> nodes;
  std::vector<FusedCpuKernelBinding> bindings;
};

std::vector<V2ModuleBinding> g_v2_bindings;
std::unordered_set<std::string> g_v1_op_types;
// op_type->binding反向索引, Init阶段一次性构建, 运行期只读。
std::unordered_map<std::string, const V2ModuleBinding *> g_v2_op_index;

void ConvertGeToAicpuTensor(const ge::GeTensorDesc &tensor_desc, const std::string &tensor_name,
                            const ge::Tensor &ge_tensor, aicpuops::Tensor *aicpu_tensor) {
  aicpu_tensor->set_name(tensor_name);
  aicpu_tensor->set_tensor_type(tensor_desc.GetDataType());
  aicpu_tensor->set_data_ptr(static_cast<uint64_t>(reinterpret_cast<intptr_t>(ge_tensor.GetData())));
  aicpu_tensor->set_data_size(static_cast<uint64_t>(ge_tensor.GetSize()));
  auto shape = aicpu_tensor->mutable_tensor_shape();
  if (shape != nullptr) {
    shape->clear_dim();
    std::vector<int64_t> dims = tensor_desc.GetShape().GetDims();
    for (size_t i = 0; i < dims.size(); i++) {
      aicpuops::TensorShape_Dim *aicpu_dims = shape->add_dim();
      if (aicpu_dims != nullptr) {
        aicpu_dims->set_size(dims[i]);
      }
    }
    shape->set_data_format(static_cast<ge::Format>(tensor_desc.GetFormat()));
  }
  AICPUE_LOGI("Op set tensor[%s], tensor info[type:%d, data:%p, size:%llu].", tensor_name.c_str(),
              static_cast<int>(tensor_desc.GetDataType()), ge_tensor.GetData(), ge_tensor.GetSize());
}

void ConvertFusedGeToAicpuTensor(const std::string &tensor_name, const ge::Tensor &ge_tensor,
                                 aicpuops::Tensor *aicpu_tensor) {
  aicpu_tensor->set_name(tensor_name);
  aicpu_tensor->set_tensor_type(ge_tensor.GetDataType());
  aicpu_tensor->set_data_ptr(static_cast<uint64_t>(reinterpret_cast<intptr_t>(ge_tensor.GetData())));
  aicpu_tensor->set_data_size(static_cast<uint64_t>(ge_tensor.GetSize()));
  auto shape = aicpu_tensor->mutable_tensor_shape();
  if (shape != nullptr) {
    shape->clear_dim();
    for (size_t i = 0U; i < ge_tensor.GetShapeDimNum(); ++i) {
      aicpuops::TensorShape_Dim *aicpu_dim = shape->add_dim();
      if (aicpu_dim != nullptr) {
        aicpu_dim->set_size(ge_tensor.GetShapeDim(i));
      }
    }
    shape->set_data_format(ge_tensor.GetFormat());
  }
  AICPUE_LOGI("Op set fused tensor[%s], tensor info[type:%d, data:%p, size:%llu].", tensor_name.c_str(),
              static_cast<int>(ge_tensor.GetDataType()), ge_tensor.GetData(), ge_tensor.GetSize());
}

int32_t AddStringAttrToNodeDef(const ge::Operator &op, const char *name, [[maybe_unused]] aicpuops::NodeDef node_def,
                               aicpuops::AttrValue &attr_value) {
  std::string s;
  ge::graphStatus ret = op.GetAttr(name, s);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  attr_value.set_s(s);

  AICPUE_LOGD("Finish add string attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddListStringAttrToNodeDef(const ge::Operator &op, const char *name,
                                   [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  std::vector<std::string> list_s;
  ge::graphStatus ret = op.GetAttr(name, list_s);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto array = attr_value.mutable_array();
  if (array == nullptr) {
    return -1;
  }

  for (std::string value : list_s) {
    array->add_s(value);
  }

  AICPUE_LOGD("Finish add list string attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddFloatAttrToNodeDef(const ge::Operator &op, const char *name, [[maybe_unused]] aicpuops::NodeDef node_def,
                              aicpuops::AttrValue &attr_value) {
  float f = 0;
  ge::graphStatus ret = op.GetAttr(name, f);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  attr_value.set_f(f);

  AICPUE_LOGD("Finish add float attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddListFloatAttrToNodeDef(const ge::Operator &op, const char *name, [[maybe_unused]] aicpuops::NodeDef node_def,
                                  aicpuops::AttrValue &attr_value) {
  std::vector<float> list_f;
  ge::graphStatus ret = op.GetAttr(name, list_f);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto array = attr_value.mutable_array();
  if (array == nullptr) {
    return -1;
  }

  for (float value : list_f) {
    array->add_f(value);
  }

  AICPUE_LOGD("Finish add list float attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddBoolAttrToNodeDef(const ge::Operator &op, const char *name, [[maybe_unused]] aicpuops::NodeDef node_def,
                             aicpuops::AttrValue &attr_value) {
  bool b = false;
  ge::graphStatus ret = op.GetAttr(name, b);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  attr_value.set_b(b);

  AICPUE_LOGD("Finish add bool attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddListBoolAttrToNodeDef(const ge::Operator &op, const char *name, [[maybe_unused]] aicpuops::NodeDef node_def,
                                 aicpuops::AttrValue &attr_value) {
  std::vector<bool> list_b;
  ge::graphStatus ret = op.GetAttr(name, list_b);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto array = attr_value.mutable_array();
  if (array == nullptr) {
    return -1;
  }

  for (bool value : list_b) {
    array->add_b(value);
  }

  AICPUE_LOGD("Finish add list bool attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddIntAttrToNodeDef(const ge::Operator &op, const char *name, [[maybe_unused]] aicpuops::NodeDef node_def,
                            aicpuops::AttrValue &attr_value) {
  int64_t i = 0;
  ge::graphStatus ret = op.GetAttr(name, i);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  attr_value.set_i(i);

  AICPUE_LOGD("Finish add int attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddListIntAttrToNodeDef(const ge::Operator &op, const char *name, [[maybe_unused]] aicpuops::NodeDef node_def,
                                aicpuops::AttrValue &attr_value) {
  std::vector<int64_t> list_i;
  ge::graphStatus ret = op.GetAttr(name, list_i);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto array = attr_value.mutable_array();
  if (array == nullptr) {
    return -1;
  }

  for (int64_t value : list_i) {
    array->add_i(value);
  }

  AICPUE_LOGD("Finish add list int attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddListListIntAttrToNodeDef(const ge::Operator &op, const char *name,
                                    [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  std::vector<std::vector<int64_t>> list_i;
  ge::graphStatus ret = op.GetAttr(name, list_i);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto array = attr_value.mutable_list_list_int();
  if (array == nullptr) {
    return -1;
  }

  array->clear_list_list_i();
  for (const std::vector<int64_t> &i : list_i) {
    const auto list_i = array->add_list_list_i();
    for (const int64_t val : i) {
      list_i->add_list_i(val);
    }
  }

  AICPUE_LOGD("Finish add list int int attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddDataTypeAttrToNodeDef(const ge::Operator &op, const char *name, [[maybe_unused]] aicpuops::NodeDef node_def,
                                 aicpuops::AttrValue &attr_value) {
  ge::DataType data_type = ge::DT_UNDEFINED;
  ge::graphStatus ret = op.GetAttr(name, data_type);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  attr_value.set_type(data_type);

  AICPUE_LOGD("Finish add datatype attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddListDataTypeAttrToNodeDef(const ge::Operator &op, const char *name,
                                     [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  std::vector<ge::DataType> list_type;
  ge::graphStatus ret = op.GetAttr(name, list_type);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto array = attr_value.mutable_array();
  if (array == nullptr) {
    return -1;
  }

  for (ge::DataType value : list_type) {
    array->add_type(value);
  }

  AICPUE_LOGD("Finish add list datatype attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddTensorAttrToNodeDef(const ge::Operator &op, const char *name, [[maybe_unused]] aicpuops::NodeDef node_def,
                               aicpuops::AttrValue &attr_value) {
  ge::Tensor ge_tensor;
  ge::graphStatus ret = op.GetAttr(name, ge_tensor);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto aicpu_tensor = attr_value.mutable_tensor();
  if (aicpu_tensor == nullptr) {
    return -1;
  }

  ge::TensorDesc ge_tensor_desc = ge_tensor.GetTensorDesc();
  aicpu_tensor->set_tensor_type(ge_tensor_desc.GetDataType());
  aicpu_tensor->set_data_ptr(static_cast<uint64_t>(reinterpret_cast<intptr_t>(ge_tensor.GetData())));
  aicpu_tensor->set_data_size(static_cast<uint64_t>(ge_tensor.GetSize()));
  auto shape = aicpu_tensor->mutable_tensor_shape();
  if (shape == nullptr) {
    return -1;
  }

  shape->clear_dim();
  std::vector<int64_t> dims = ge_tensor_desc.GetShape().GetDims();
  for (size_t i = 0; i < dims.size(); i++) {
    aicpuops::TensorShape_Dim *aicpu_dims = shape->add_dim();
    if (aicpu_dims == nullptr) {
      return -1;
    }
    aicpu_dims->set_size(dims[i]);
  }

  AICPUE_LOGD("Finish add tensor attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddListTensorAttrToNodeDef(const ge::Operator &op, const char *name,
                                   [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  std::vector<ge::Tensor> ge_list_tensor;
  ge::graphStatus ret = op.GetAttr(name, ge_list_tensor);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto array = attr_value.mutable_array();
  if (array == nullptr) {
    return -1;
  }

  for (const ge::Tensor &ge_tensor : ge_list_tensor) {
    auto aicpu_tensor = array->add_tensor();
    if (aicpu_tensor == nullptr) {
      return -1;
    }

    ge::TensorDesc ge_tensor_desc = ge_tensor.GetTensorDesc();
    aicpu_tensor->set_tensor_type(ge_tensor_desc.GetDataType());
    aicpu_tensor->set_data_ptr(static_cast<uint64_t>(reinterpret_cast<intptr_t>(ge_tensor.GetData())));
    aicpu_tensor->set_data_size(static_cast<uint64_t>(ge_tensor.GetSize()));
    auto shape = aicpu_tensor->mutable_tensor_shape();
    if (shape == nullptr) {
      return -1;
    }

    shape->clear_dim();
    std::vector<int64_t> dims = ge_tensor_desc.GetShape().GetDims();
    for (size_t i = 0; i < dims.size(); i++) {
      aicpuops::TensorShape_Dim *aicpu_dims = shape->add_dim();
      if (aicpu_dims == nullptr) {
        return -1;
      }
      aicpu_dims->set_size(dims[i]);
    }
  }

  AICPUE_LOGD("Finish add list tensor attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddListAttrToNodeDef(const ge::Operator &op, const char *name, const std::string &type,
                             aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  int32_t ret = 0;
  if (type == kVtListString) {
    ret = AddListStringAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtListFloat) {
    ret = AddListFloatAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtListInt) {
    ret = AddListIntAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtListListInt) {
    ret = AddListListIntAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtListBool) {
    ret = AddListBoolAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtListDataType) {
    ret = AddListDataTypeAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtListTensor) {
    ret = AddListTensorAttrToNodeDef(op, name, node_def, attr_value);
  } else {
    AICPUE_LOGW("Attr type is unsupported, name: [%s], type: [%s].", name, type.c_str());
  }
  return ret;
}

int32_t AddAttrToNodeDef(const ge::Operator &op, const char *name, const std::string type, aicpuops::NodeDef node_def,
                         aicpuops::AttrValue &attr_value) {
  int32_t ret = 0;
  if (type.empty() || type[0] == '_') {
    return ret;
  }
  if (type == kVtString) {
    ret = AddStringAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtFloat) {
    ret = AddFloatAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtInt) {
    ret = AddIntAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtBool) {
    ret = AddBoolAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtDataType) {
    ret = AddDataTypeAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtTensor) {
    ret = AddTensorAttrToNodeDef(op, name, node_def, attr_value);
  } else {
    ret = AddListAttrToNodeDef(op, name, type, node_def, attr_value);
  }
  return ret;
}

std::string GetRealPath(const std::string &path) {
  char resoved_path[PATH_MAX] = {0};
  if (realpath(path.c_str(), resoved_path) != nullptr) {
    return std::string(resoved_path);
  }
  AICPUE_LOGW("path %s does not exist", path.c_str());
  return "";
}

std::string EnsureTrailingSlash(const std::string &path) {
  if (path.empty() || path.back() == kPathSeparator) {
    return path;
  }
  return path + kPathSeparator;
}

bool IsConstantFoldingSo(const std::string &file_name) {
  if (file_name.find(kConstantFoldingSoPrefix) != 0U) {
    return false;
  }
  size_t suffix_len = strlen(kConstantFoldingSoSuffix);
  if (file_name.length() < suffix_len) {
    return false;
  }
  if (file_name.compare(file_name.length() - suffix_len, suffix_len, kConstantFoldingSoSuffix) != 0) {
    return false;
  }
  return file_name != kExcludedConstantFoldingSo;
}

void TryBindV2Symbols(void *handle, const std::string &so_name) {
  V2ModuleBinding binding;
  binding.handle = handle;
  binding.so_name = so_name;

  binding.get_all_op_types =
      reinterpret_cast<GetAllRegisteredOpTypesV2Fn>(dlsym(handle, kSymGetAllRegisteredOpTypesV2));
  binding.is_registered = reinterpret_cast<IsRegisteredV2Fn>(dlsym(handle, kSymIsRegisteredV2));
  binding.run_cpu_kernel = reinterpret_cast<RunCpuKernelV2Fn>(dlsym(handle, kSymRunCpuKernelV2));

  if ((binding.get_all_op_types == nullptr) && (binding.run_cpu_kernel == nullptr)) {
    AICPUE_LOGW("V2 C ABI symbols not found in so[%s], skip V2 binding.", so_name.c_str());
    return;
  }
  g_v2_bindings.emplace_back(binding);
}

void LoadConstantFoldingSo(const std::string &base_path) {
  std::string dir_path = base_path + "opp/built-in/op_impl/host_cpu/";
  DIR *dir = opendir(dir_path.c_str());
  if (dir == nullptr) {
    AICPUE_LOGW("Failed to open directory: %s", dir_path.c_str());
    return;
  }
  struct dirent *entry = nullptr;
  while ((entry = readdir(dir)) != nullptr) {
    if (!IsConstantFoldingSo(entry->d_name)) {
      continue;
    }
    std::string lib_path = GetRealPath(dir_path + entry->d_name);
    if (lib_path.empty()) {
      continue;
    }
    AICPUE_LOGI("Found constant folding so: %s", lib_path.c_str());
    void *handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
      AICPUE_LOGW("dlopen failed: %s, reason: %s", lib_path.c_str(), dlerror());
      continue;
    }
    AICPUE_LOGI("Successfully loaded: %s", lib_path.c_str());
    TryBindV2Symbols(handle, entry->d_name);
  }
  closedir(dir);
}
}  // namespace

void RegisterHostCpuOp(std::vector<std::string> ops, ge::HostCpuOp *(*create_fn)()) {
  std::set<std::string> black_list = {"Assign", "NoOp", "TruncatedNormal"};
  for (const std::string &op_type : ops) {
    if (black_list.find(op_type) != black_list.end()) {
      continue;
    }
    AICPUE_LOGI("Register op[%s].", op_type.c_str());
    ::ge::HostCpuOpRegistrar registrar __attribute__((unused)) = ::ge::HostCpuOpRegistrar(op_type.c_str(), create_fn);
  }
}

extern "C" {
__attribute__((visibility("default"))) int32_t InitCpuConstantFoldingNew(ge::HostCpuOp *(*create_fn)()) {
  AICPUE_LOGI("Init cpu constant folding begin.");

  const char *path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, path_env);
  if (path_env == nullptr) {
    AICPUE_LOGE("ASCEND_HOME_PATH environment variable not found");
    return -1;
  }

  std::string base_path = EnsureTrailingSlash(path_env);
  AICPUE_LOGI("ASCEND_HOME_PATH is %s", base_path.c_str());

  LoadConstantFoldingSo(base_path);

  std::vector<std::string> ops = aicpu::CpuKernelRegister::Instance().GetAllRegisteredOpTypes();
  AICPUE_LOGI("Registered V1 ops: %llu", static_cast<uint64_t>(ops.size()));
  g_v1_op_types.clear();
  g_v1_op_types.insert(ops.cbegin(), ops.cend());
  RegisterHostCpuOp(ops, create_fn);

  // 枚举每个ops so的V2算子, 同时构建op_type->binding反向索引。
  g_v2_op_index.clear();
  for (const auto &binding : g_v2_bindings) {
    if (binding.get_all_op_types == nullptr) {
      continue;
    }
    std::vector<std::string> ops_v2 = binding.get_all_op_types();
    for (const std::string &op_type : ops_v2) {
      // 多so重复注册同一op_type时保留先加载者, 避免后加载者静默覆盖。
      auto insert_ret = g_v2_op_index.emplace(op_type, &binding);
      if (!insert_ret.second) {
        AICPUE_LOGW("op type [%s] V2 already indexed by so[%s], so[%s] skipped.", op_type.c_str(),
                    insert_ret.first->second->so_name.c_str(), binding.so_name.c_str());
      }
    }
    AICPUE_LOGI("Registered V2 ops from so[%s]: %llu", binding.so_name.c_str(), static_cast<uint64_t>(ops_v2.size()));
    RegisterHostCpuOp(ops_v2, create_fn);
  }

  return 0;
}
int32_t BuildInputTensors(const ge::OpDescPtr &op_desc, const std::map<std::string, const ge::Tensor> &inputs,
                          const char *op_type, aicpuops::NodeDef &node_def) {
  uint32_t count = static_cast<uint32_t>(op_desc->GetAllInputsSize());
  for (uint32_t i = 0; i < count; ++i) {
    ge::GeTensorDescPtr desc = op_desc->MutableInputDesc(i);
    if (desc == nullptr) {
      continue;
    }
    std::string name = op_desc->GetInputNameByIndex(i);
    auto iter = inputs.find(name);
    if (iter == inputs.end()) {
      AICPUE_LOGW("Op[%s] input[%s] not found.", op_type, name.c_str());
      return -1;
    }
    aicpuops::Tensor *tensor = node_def.add_inputs();
    if (tensor == nullptr) {
      return -1;
    }
    ConvertGeToAicpuTensor(*desc, name, iter->second, tensor);
  }
  return 0;
}

int32_t BuildOutputTensors(const ge::OpDescPtr &op_desc, const std::map<std::string, ge::Tensor> &outputs,
                           const char *op_type, aicpuops::NodeDef &node_def) {
  uint32_t count = static_cast<uint32_t>(op_desc->GetOutputsSize());
  for (uint32_t i = 0; i < count; ++i) {
    ge::GeTensorDesc desc = op_desc->GetOutputDesc(i);
    std::string name = op_desc->GetOutputNameByIndex(i);
    auto iter = outputs.find(name);
    if (iter == outputs.end()) {
      AICPUE_LOGW("Op[%s] output[%s] not found.", op_type, name.c_str());
      return -1;
    }
    aicpuops::Tensor *tensor = node_def.add_outputs();
    if (tensor == nullptr) {
      return -1;
    }
    ConvertGeToAicpuTensor(desc, name, iter->second, tensor);
  }
  return 0;
}

int32_t BuildFusedInputTensorArray(const ge::OpDescPtr &op_desc, const ge::Tensor *const *inputs,
                                   const size_t input_num, aicpuops::NodeDef &node_def) {
  const size_t count = op_desc->GetAllInputsSize();
  if ((count != input_num) || ((count != 0U) && (inputs == nullptr))) {
    AICPUE_LOGE("Invalid fused input tensor array: op[%s], expected_num[%zu], actual_num[%zu], inputs_null[%d].",
                AICPUE_ERROR_CODE, op_desc->GetTypePtr(), count, input_num, static_cast<int32_t>(inputs == nullptr));
    return -1;
  }
  for (size_t i = 0U; i < count; ++i) {
    if (inputs[i] == nullptr) {
      AICPUE_LOGE("Fused input tensor is null: op[%s], input_index[%zu].", AICPUE_ERROR_CODE, op_desc->GetTypePtr(), i);
      return -1;
    }
    aicpuops::Tensor *tensor = node_def.add_inputs();
    if (tensor == nullptr) {
      AICPUE_LOGE("Failed to add fused input tensor to NodeDef: op[%s], input_index[%zu].", AICPUE_ERROR_CODE,
                  op_desc->GetTypePtr(), i);
      return -1;
    }
    ConvertFusedGeToAicpuTensor(op_desc->GetInputNameByIndex(static_cast<uint32_t>(i)), *inputs[i], tensor);
  }
  return 0;
}

int32_t BuildFusedOutputTensorArray(const ge::OpDescPtr &op_desc, ge::Tensor *const *outputs, const size_t output_num,
                                    aicpuops::NodeDef &node_def) {
  const size_t count = op_desc->GetOutputsSize();
  if ((count != output_num) || ((count != 0U) && (outputs == nullptr))) {
    AICPUE_LOGE("Invalid fused output tensor array: op[%s], expected_num[%zu], actual_num[%zu], outputs_null[%d].",
                AICPUE_ERROR_CODE, op_desc->GetTypePtr(), count, output_num, static_cast<int32_t>(outputs == nullptr));
    return -1;
  }
  for (size_t i = 0U; i < count; ++i) {
    if (outputs[i] == nullptr) {
      AICPUE_LOGE("Fused output tensor is null: op[%s], output_index[%zu].", AICPUE_ERROR_CODE, op_desc->GetTypePtr(),
                  i);
      return -1;
    }
    aicpuops::Tensor *tensor = node_def.add_outputs();
    if (tensor == nullptr) {
      AICPUE_LOGE("Failed to add fused output tensor to NodeDef: op[%s], output_index[%zu].", AICPUE_ERROR_CODE,
                  op_desc->GetTypePtr(), i);
      return -1;
    }
    ConvertFusedGeToAicpuTensor(op_desc->GetOutputNameByIndex(static_cast<uint32_t>(i)), *outputs[i], tensor);
  }
  return 0;
}

int32_t BuildNodeDefAttrs(const ge::Operator &op, aicpuops::NodeDef &node_def) {
  std::map<ge::AscendString, ge::AscendString> attrs;
  if (op.GetAllAttrNamesAndTypes(attrs) != ge::GRAPH_SUCCESS) {
    return -1;
  }
  for (const auto &attr : attrs) {
    const char *name = attr.first.GetString();
    std::string type = std::string(attr.second.GetString());
    aicpuops::AttrValue attr_value;
    int32_t ret = AddAttrToNodeDef(op, name, type, node_def, attr_value);
    if (ret != 0) {
      return ret;
    }
    auto *node_def_attrs = node_def.mutable_attrs();
    if (node_def_attrs == nullptr) {
      return -1;
    }
    auto pair = node_def_attrs->insert(AttrValueMap::value_type(std::string(name), attr_value));
    if (!pair.second) {
      return -1;
    }
  }
  return 0;
}

int32_t BuildNodeDef(const ge::Operator &op, const std::string &op_type_str,
                     const std::map<std::string, const ge::Tensor> &inputs, std::map<std::string, ge::Tensor> &outputs,
                     aicpuops::NodeDef &node_def) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    AICPUE_LOGW("Op[%s] get op desc failed.", op_type_str.c_str());
    return -1;
  }
  node_def.set_op(op_type_str);
  int32_t ret = BuildInputTensors(op_desc, inputs, op_type_str.c_str(), node_def);
  if (ret != 0) {
    return ret;
  }
  ret = BuildOutputTensors(op_desc, outputs, op_type_str.c_str(), node_def);
  if (ret != 0) {
    return ret;
  }
  return BuildNodeDefAttrs(op, node_def);
}

int32_t BuildFusedNodeDefFromTensorArray(const ge::Operator &op, const std::string &op_type_str,
                                         const ge::Tensor *const *inputs, const size_t input_num,
                                         ge::Tensor *const *outputs, const size_t output_num,
                                         aicpuops::NodeDef &node_def) {
  const ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    AICPUE_LOGW("Op[%s] get op desc failed.", op_type_str.c_str());
    return -1;
  }
  node_def.set_op(op_type_str);
  int32_t ret = BuildFusedInputTensorArray(op_desc, inputs, input_num, node_def);
  if (ret != 0) {
    return ret;
  }
  ret = BuildFusedOutputTensorArray(op_desc, outputs, output_num, node_def);
  if (ret != 0) {
    return ret;
  }
  return BuildNodeDefAttrs(op, node_def);
}

bool HasSameShape(const ge::Tensor &source, const FusedTensorBindingState &state) {
  const size_t dim_num = source.GetShapeDimNum();
  if (state.dims.size() != dim_num) {
    return false;
  }
  for (size_t i = 0U; i < dim_num; ++i) {
    if (state.dims[i] != source.GetShapeDim(i)) {
      return false;
    }
  }
  return true;
}

int32_t InitializeFusedTensor(const ge::Tensor &source, aicpu::Tensor *target, FusedTensorBindingState &state) {
  const void *data = static_cast<const void *>(source.GetData());
  const size_t data_size = source.GetSize();
  target->SetData(const_cast<void *>(data));
  target->SetDataSize(static_cast<uint64_t>(data_size));
  target->SetDataType(static_cast<aicpu::DataType>(source.GetDataType()));

  const std::shared_ptr<aicpu::TensorShape> tensor_shape = target->GetTensorShape();
  if (tensor_shape == nullptr) {
    AICPUE_LOGE("Failed to get target TensorShape while initializing fused Tensor.", AICPUE_ERROR_CODE);
    return -1;
  }
  state.dims.resize(source.GetShapeDimNum());
  for (size_t i = 0U; i < state.dims.size(); ++i) {
    state.dims[i] = source.GetShapeDim(i);
  }
  tensor_shape->SetDimSizes(state.dims);
  const ge::Format format = source.GetFormat();
  tensor_shape->SetFormat(static_cast<aicpu::Format>(format));

  state.data = data;
  state.data_size = data_size;
  state.data_type = source.GetDataType();
  state.format = format;
  state.initialized = true;
  return 0;
}

int32_t RebindFusedTensor(const ge::Tensor &source, aicpu::Tensor *target, FusedTensorBindingState &state) {
  if ((target == nullptr) || ((source.GetSize() != 0U) && (source.GetData() == nullptr))) {
    AICPUE_LOGE("Failed to rebind fused Tensor: target_null[%d], data_null[%d], data_size[%zu].", AICPUE_ERROR_CODE,
                static_cast<int32_t>(target == nullptr),
                static_cast<int32_t>((source.GetSize() != 0U) && (source.GetData() == nullptr)), source.GetSize());
    return -1;
  }
  if (!state.initialized) {
    return InitializeFusedTensor(source, target, state);
  }
  const void *data = static_cast<const void *>(source.GetData());
  const size_t data_size = source.GetSize();
  if (state.data != data) {
    target->SetData(const_cast<void *>(data));
    state.data = data;
  }
  if (state.data_size != data_size) {
    target->SetDataSize(static_cast<uint64_t>(data_size));
    state.data_size = data_size;
  }

  const ge::DataType data_type = source.GetDataType();
  if (state.data_type != data_type) {
    target->SetDataType(static_cast<aicpu::DataType>(data_type));
    state.data_type = data_type;
  }

  const ge::Format format = source.GetFormat();
  const bool shape_changed = !HasSameShape(source, state);
  if (shape_changed || (state.format != format)) {
    const std::shared_ptr<aicpu::TensorShape> tensor_shape = target->GetTensorShape();
    if (tensor_shape == nullptr) {
      AICPUE_LOGE("Failed to get target TensorShape while rebinding fused Tensor.", AICPUE_ERROR_CODE);
      return -1;
    }
    if (shape_changed) {
      state.dims.resize(source.GetShapeDimNum());
      for (size_t i = 0U; i < state.dims.size(); ++i) {
        state.dims[i] = source.GetShapeDim(i);
      }
      tensor_shape->SetDimSizes(state.dims);
    }
    tensor_shape->SetFormat(static_cast<aicpu::Format>(format));
    state.format = format;
  }
  state.initialized = true;
  return 0;
}

int32_t RebindFusedTensorDataByFlags(const ge::Tensor &source, aicpu::Tensor *target, FusedTensorBindingState &state,
                                     const uint32_t binding_flags) {
  if ((binding_flags & kFusedHostCpuDataChanged) == 0U) {
    return 0;
  }
  const void *data = static_cast<const void *>(source.GetData());
  const size_t data_size = source.GetSize();
  if ((data_size != 0U) && (data == nullptr)) {
    AICPUE_LOGE("Failed to rebind fused Tensor data by flags: data is null, data_size[%zu], binding_flags[%u].",
                AICPUE_ERROR_CODE, data_size, binding_flags);
    return -1;
  }
  if (state.data != data) {
    target->SetData(const_cast<void *>(data));
    state.data = data;
  }
  if (state.data_size != data_size) {
    target->SetDataSize(static_cast<uint64_t>(data_size));
    state.data_size = data_size;
  }
  return 0;
}

int32_t RebindFusedTensorShapeByFlags(const ge::Tensor &source, aicpu::Tensor *target, FusedTensorBindingState &state,
                                      const uint32_t binding_flags) {
  if ((binding_flags & kFusedHostCpuShapeChanged) == 0U) {
    return 0;
  }
  const ge::DataType data_type = source.GetDataType();
  if (state.data_type != data_type) {
    target->SetDataType(static_cast<aicpu::DataType>(data_type));
    state.data_type = data_type;
  }

  const ge::Format format = source.GetFormat();
  const bool shape_changed = !HasSameShape(source, state);
  if (shape_changed || (state.format != format)) {
    const std::shared_ptr<aicpu::TensorShape> tensor_shape = target->GetTensorShape();
    if (tensor_shape == nullptr) {
      AICPUE_LOGE("Failed to get target TensorShape while rebinding fused Tensor by flags: binding_flags[%u].",
                  AICPUE_ERROR_CODE, binding_flags);
      return -1;
    }
    if (shape_changed) {
      state.dims.resize(source.GetShapeDimNum());
      for (size_t i = 0U; i < state.dims.size(); ++i) {
        state.dims[i] = source.GetShapeDim(i);
      }
      tensor_shape->SetDimSizes(state.dims);
    }
    tensor_shape->SetFormat(static_cast<aicpu::Format>(format));
    state.format = format;
  }
  return 0;
}

int32_t RebindFusedTensorByFlags(const ge::Tensor &source, aicpu::Tensor *target, FusedTensorBindingState &state,
                                 uint32_t binding_flags) {
  if (target == nullptr) {
    AICPUE_LOGE("Failed to rebind fused Tensor by flags: target is null, binding_flags[%u].", AICPUE_ERROR_CODE,
                binding_flags);
    return -1;
  }
  if (!state.initialized) {
    return RebindFusedTensor(source, target, state);
  }
  if ((RebindFusedTensorDataByFlags(source, target, state, binding_flags) != 0) ||
      (RebindFusedTensorShapeByFlags(source, target, state, binding_flags) != 0)) {
    return -1;
  }
  state.initialized = true;
  return 0;
}

int32_t RebindFusedTensorByBinding(const FusedHostCpuTensorBinding &binding, aicpu::Tensor *target,
                                   FusedTensorBindingState &state) {
  if ((target == nullptr) || ((binding.dim_num != 0U) && (binding.dims == nullptr)) ||
      ((binding.data_size != 0U) && (binding.data == nullptr))) {
    AICPUE_LOGE(
        "Invalid fused Tensor binding: target_null[%d], dim_num[%zu], dims_null[%d], data_size[%zu], "
        "data_null[%d], flags[%u].",
        AICPUE_ERROR_CODE, static_cast<int32_t>(target == nullptr), binding.dim_num,
        static_cast<int32_t>((binding.dim_num != 0U) && (binding.dims == nullptr)), binding.data_size,
        static_cast<int32_t>((binding.data_size != 0U) && (binding.data == nullptr)), binding.flags);
    return -1;
  }
  uint32_t binding_flags = binding.flags;
  if (!state.initialized) {
    binding_flags |= kFusedHostCpuShapeChanged | kFusedHostCpuDataChanged;
  }
  if ((binding_flags & kFusedHostCpuDataChanged) != 0U) {
    target->SetData(binding.data);
    target->SetDataSize(static_cast<uint64_t>(binding.data_size));
    state.data = binding.data;
    state.data_size = binding.data_size;
  }
  if ((binding_flags & kFusedHostCpuShapeChanged) != 0U) {
    const std::shared_ptr<aicpu::TensorShape> tensor_shape = target->GetTensorShape();
    if (tensor_shape == nullptr) {
      AICPUE_LOGE("Failed to get target TensorShape from fused binding: dim_num[%zu], flags[%u].", AICPUE_ERROR_CODE,
                  binding.dim_num, binding.flags);
      return -1;
    }
    state.dims.resize(binding.dim_num);
    for (size_t i = 0U; i < binding.dim_num; ++i) {
      state.dims[i] = binding.dims[i];
    }
    tensor_shape->SetDimSizes(state.dims);
  }
  state.initialized = true;
  return 0;
}

int32_t RebindFusedPlan(FusedCpuKernelPlan &plan, const ge::Tensor *const *inputs, const size_t input_num,
                        ge::Tensor *const *outputs, const size_t output_num) {
  if ((plan.context == nullptr) || (input_num != plan.input_tensors.size()) ||
      (output_num != plan.output_tensors.size()) || ((input_num != 0U) && (inputs == nullptr)) ||
      ((output_num != 0U) && (outputs == nullptr))) {
    AICPUE_LOGE(
        "Invalid fused CPU plan binding: context_null[%d], input_num[%zu], expected_inputs[%zu], "
        "output_num[%zu], expected_outputs[%zu], inputs_null[%d], outputs_null[%d].",
        AICPUE_ERROR_CODE, static_cast<int32_t>(plan.context == nullptr), input_num, plan.input_tensors.size(),
        output_num, plan.output_tensors.size(), static_cast<int32_t>(inputs == nullptr),
        static_cast<int32_t>(outputs == nullptr));
    return -1;
  }
  for (size_t i = 0U; i < input_num; ++i) {
    if ((inputs[i] == nullptr) || (RebindFusedTensor(*inputs[i], plan.input_tensors[i], plan.input_states[i]) != 0)) {
      AICPUE_LOGE("Failed to rebind fused CPU plan input: input_index[%zu], input_null[%d].", AICPUE_ERROR_CODE, i,
                  static_cast<int32_t>(inputs[i] == nullptr));
      return -1;
    }
  }
  for (size_t i = 0U; i < output_num; ++i) {
    if ((outputs[i] == nullptr) ||
        (RebindFusedTensor(*outputs[i], plan.output_tensors[i], plan.output_states[i]) != 0)) {
      AICPUE_LOGE("Failed to rebind fused CPU plan output: output_index[%zu], output_null[%d].", AICPUE_ERROR_CODE, i,
                  static_cast<int32_t>(outputs[i] == nullptr));
      return -1;
    }
  }
  return 0;
}

int32_t RunFusedCpuKernelPlan(FusedCpuKernelPlan &plan) {
  uint32_t ret = 0U;
  if (plan.v2_binding != nullptr) {
    ret = plan.v2_binding->run_cpu_kernel(*plan.context);
  } else if (plan.v1_kernel != nullptr) {
    ret = plan.v1_kernel->Compute(*plan.context);
  } else {
    AICPUE_LOGE("Fused CPU kernel plan has neither V1 kernel nor V2 binding.", AICPUE_ERROR_CODE);
    return -1;
  }
  if (ret != 0U) {
    AICPUE_LOGE("Fused CPU kernel execution failed: ret[%u].", AICPUE_ERROR_CODE, ret);
  }
  return (ret == 0U) ? 0 : -1;
}

// 查找op_type对应的V2 binding。未命中返回nullptr表示走V1路径。
const V2ModuleBinding *LookupV2Binding(const std::string &op_type) {
  auto iter = g_v2_op_index.find(op_type);
  if (iter == g_v2_op_index.end()) {
    return nullptr;
  }
  const V2ModuleBinding *binding = iter->second;
  if ((binding == nullptr) || (binding->run_cpu_kernel == nullptr)) {
    return nullptr;
  }
  return binding;
}

__attribute__((visibility("default"))) int32_t IsCpuConstantFoldingFusedOpSupported(const char *op_type) {
  if ((op_type == nullptr) || (op_type[0] == '\0')) {
    return 0;
  }
  const std::string op_type_str(op_type);
  return ((LookupV2Binding(op_type_str) != nullptr) || (g_v1_op_types.count(op_type_str) > 0U)) ? 1 : 0;
}

__attribute__((visibility("default"))) int32_t
CpuConstantFoldingComputeNew(const ge::Operator &op, const std::map<std::string, const ge::Tensor> &inputs,
                             std::map<std::string, ge::Tensor> outputs) {
  ge::AscendString op_type;
  if (op.GetOpType(op_type) != ge::GRAPH_SUCCESS) {
    return -1;
  }
  AICPUE_LOGI("Enter cpu op[%s].", op_type.GetString());
  std::string op_type_str(op_type.GetString());

  const V2ModuleBinding *hit_binding = LookupV2Binding(op_type_str);
  if (hit_binding == nullptr) {
    if (aicpu::CpuKernelRegister::Instance().GetCpuKernel(op_type_str) == nullptr) {
      AICPUE_LOGW("op type [%s] is not registered in v1 nor v2.", op_type.GetString());
      return -1;
    }
    AICPUE_LOGI("op type [%s] use v1 kernel from local register.", op_type.GetString());
  } else {
    AICPUE_LOGI("op type [%s] hit v2 in so[%s].", op_type.GetString(), hit_binding->so_name.c_str());
  }

  aicpuops::NodeDef node_def;
  int32_t ret = BuildNodeDef(op, op_type_str, inputs, outputs, node_def);
  if (ret != 0) {
    return ret;
  }

  aicpu::CpuKernelContext ctx(aicpu::HOST);
  ret = ctx.Init(&node_def);
  if (ret != 0) {
    return -1;
  }

  if (hit_binding != nullptr) {
    AICPUE_LOGI("op type [%s] run cpu kernel v2 in so[%s].", op_type.GetString(), hit_binding->so_name.c_str());
    ret = static_cast<int32_t>(hit_binding->run_cpu_kernel(ctx));
  } else {
    AICPUE_LOGI("op type [%s] run cpu kernel v1.", op_type.GetString());
    ret = static_cast<int32_t>(aicpu::CpuKernelRegister::Instance().RunCpuKernel(ctx));
  }
  if (ret != 0) {
    return -1;
  }

  AICPUE_LOGI("Finish cpu op[%s].", op_type.GetString());
  return 0;
}

int32_t InitializeFusedCpuKernel(const ge::Operator &op, FusedCpuKernelPlan &plan, std::string &op_type_str,
                                 ge::AscendString &op_type) {
  if (op.GetOpType(op_type) != ge::GRAPH_SUCCESS) {
    return -1;
  }
  op_type_str = op_type.GetString();
  plan.v2_binding = LookupV2Binding(op_type_str);
  if (plan.v2_binding == nullptr) {
    plan.v1_kernel = aicpu::CpuKernelRegister::Instance().GetCpuKernel(op_type_str);
    if (plan.v1_kernel == nullptr) {
      AICPUE_LOGW("op type [%s] is not registered in v1 nor v2.", op_type.GetString());
      return -1;
    }
  }
  return 0;
}

int32_t AllocateFusedCpuKernelPlan(FusedCpuKernelPlan &plan, const ge::AscendString &op_type) {
  plan.node_def.reset(new (std::nothrow) aicpuops::NodeDef());
  plan.context.reset(new (std::nothrow) aicpu::CpuKernelContext(aicpu::HOST));
  if ((plan.node_def == nullptr) || (plan.context == nullptr)) {
    AICPUE_LOGE("Failed to allocate fused CPU kernel plan objects: op[%s], node_def_null[%d], context_null[%d].",
                AICPUE_ERROR_CODE, op_type.GetString(), static_cast<int32_t>(plan.node_def == nullptr),
                static_cast<int32_t>(plan.context == nullptr));
    return -1;
  }
  return 0;
}

int32_t InitializeFusedCpuKernelTensors(FusedCpuKernelPlan &plan, const size_t input_num, const size_t output_num,
                                        const ge::AscendString &op_type) {
  plan.input_tensors.resize(input_num);
  plan.output_tensors.resize(output_num);
  plan.input_states.resize(input_num);
  plan.output_states.resize(output_num);
  for (size_t i = 0U; i < input_num; ++i) {
    plan.input_tensors[i] = plan.context->Input(static_cast<uint32_t>(i));
    if (plan.input_tensors[i] == nullptr) {
      AICPUE_LOGE("Fused CPU kernel context input is null: op[%s], input_index[%zu], input_num[%zu].",
                  AICPUE_ERROR_CODE, op_type.GetString(), i, input_num);
      return -1;
    }
  }
  for (size_t i = 0U; i < output_num; ++i) {
    plan.output_tensors[i] = plan.context->Output(static_cast<uint32_t>(i));
    if (plan.output_tensors[i] == nullptr) {
      AICPUE_LOGE("Fused CPU kernel context output is null: op[%s], output_index[%zu], output_num[%zu].",
                  AICPUE_ERROR_CODE, op_type.GetString(), i, output_num);
      return -1;
    }
  }
  return 0;
}

int32_t InitializeFusedCpuKernelPlan(const ge::Operator &op, const ge::Tensor *const *inputs, const size_t input_num,
                                     ge::Tensor *const *outputs, const size_t output_num, FusedCpuKernelPlan &plan) {
  ge::AscendString op_type;
  std::string op_type_str;
  if (InitializeFusedCpuKernel(op, plan, op_type_str, op_type) != 0) {
    return -1;
  }
  if (AllocateFusedCpuKernelPlan(plan, op_type) != 0) {
    return -1;
  }
  if (BuildFusedNodeDefFromTensorArray(op, op_type_str, inputs, input_num, outputs, output_num, *plan.node_def) != 0) {
    AICPUE_LOGE("Failed to build fused CPU NodeDef: op[%s], inputs[%zu], outputs[%zu].", AICPUE_ERROR_CODE,
                op_type.GetString(), input_num, output_num);
    return -1;
  }
  const int32_t context_ret = plan.context->Init(plan.node_def.get());
  if (context_ret != 0) {
    AICPUE_LOGE("Failed to initialize fused CPU kernel context: op[%s], ret[%d].", AICPUE_ERROR_CODE,
                op_type.GetString(), context_ret);
    return -1;
  }
  if (InitializeFusedCpuKernelTensors(plan, input_num, output_num, op_type) != 0) {
    return -1;
  }
  if (RebindFusedPlan(plan, inputs, input_num, outputs, output_num) != 0) {
    AICPUE_LOGE("Failed to bind fused CPU kernel plan tensors: op[%s], inputs[%zu], outputs[%zu].", AICPUE_ERROR_CODE,
                op_type.GetString(), input_num, output_num);
    return -1;
  }
  AICPUE_LOGD("Created fused cpu execution plan for op[%s], inputs[%zu], outputs[%zu].", op_type.GetString(), input_num,
              output_num);
  return 0;
}

int32_t CalculateFusedChainBindingCapacity(const FusedCpuKernelChainNodeDesc *descs, const size_t node_num,
                                           size_t &binding_capacity) {
  binding_capacity = 0U;
  for (size_t i = 0U; i < node_num; ++i) {
    if (descs[i].input_num > (std::numeric_limits<size_t>::max() - descs[i].output_num)) {
      AICPUE_LOGE("Fused CPU chain node binding count overflows size_t: node_index[%zu], inputs[%zu], outputs[%zu].",
                  AICPUE_ERROR_CODE, i, descs[i].input_num, descs[i].output_num);
      return -1;
    }
    const size_t node_binding_count = descs[i].input_num + descs[i].output_num;
    if (binding_capacity > (std::numeric_limits<size_t>::max() - node_binding_count)) {
      AICPUE_LOGE(
          "Fused CPU chain binding capacity overflows size_t: node_index[%zu], current_capacity[%zu], "
          "node_binding_count[%zu].",
          AICPUE_ERROR_CODE, i, binding_capacity, node_binding_count);
      return -1;
    }
    binding_capacity += node_binding_count;
  }
  return 0;
}

int32_t AddFusedChainInputBindings(const FusedCpuKernelChainNodeDesc &desc, const size_t node_index,
                                   const size_t external_binding_num, FusedCpuKernelChainNode &node,
                                   FusedCpuKernelChainPlan &chain) {
  for (size_t j = 0U; j < desc.input_num; ++j) {
    const int32_t binding_index =
        (desc.input_binding_indices == nullptr) ? static_cast<int32_t>(j) : desc.input_binding_indices[j];
    if ((binding_index >= 0) && (static_cast<size_t>(binding_index) < external_binding_num) &&
        (desc.inputs[j] != nullptr)) {
      chain.bindings.push_back(
          {desc.inputs[j], node.plan.input_tensors[j], &node.plan.input_states[j], static_cast<size_t>(binding_index)});
    } else if ((binding_index >= 0) &&
               ((desc.inputs[j] == nullptr) || (static_cast<size_t>(binding_index) >= external_binding_num))) {
      AICPUE_LOGE(
          "Invalid fused input binding: node_index[%zu], input_index[%zu], binding_index[%d], "
          "external_binding_num[%zu], input_null[%d].",
          AICPUE_ERROR_CODE, node_index, j, binding_index, external_binding_num,
          static_cast<int32_t>(desc.inputs[j] == nullptr));
      return -1;
    }
  }
  return 0;
}

int32_t AddFusedChainOutputBindings(const FusedCpuKernelChainNodeDesc &desc, const size_t node_index,
                                    const size_t external_binding_num, FusedCpuKernelChainNode &node,
                                    FusedCpuKernelChainPlan &chain) {
  for (size_t j = 0U; j < desc.output_num; ++j) {
    const int32_t binding_index =
        (desc.output_binding_indices == nullptr) ? static_cast<int32_t>(j) : desc.output_binding_indices[j];
    if ((binding_index >= 0) && (static_cast<size_t>(binding_index) < external_binding_num) &&
        (desc.outputs[j] != nullptr)) {
      chain.bindings.push_back({desc.outputs[j], node.plan.output_tensors[j], &node.plan.output_states[j],
                                static_cast<size_t>(binding_index)});
    } else if ((binding_index >= 0) &&
               ((desc.outputs[j] == nullptr) || (static_cast<size_t>(binding_index) >= external_binding_num))) {
      AICPUE_LOGE(
          "Invalid fused output binding: node_index[%zu], output_index[%zu], binding_index[%d], "
          "external_binding_num[%zu], output_null[%d].",
          AICPUE_ERROR_CODE, node_index, j, binding_index, external_binding_num,
          static_cast<int32_t>(desc.outputs[j] == nullptr));
      return -1;
    }
  }
  return 0;
}

int32_t InitializeFusedChainNode(const FusedCpuKernelChainNodeDesc &desc, const size_t node_index,
                                 const size_t external_binding_num, FusedCpuKernelChainPlan &chain) {
  if ((desc.op == nullptr) || ((desc.input_num != 0U) && (desc.inputs == nullptr)) ||
      ((desc.output_num != 0U) && (desc.outputs == nullptr))) {
    AICPUE_LOGE(
        "Invalid fused CPU chain node descriptor: node_index[%zu], op_null[%d], input_num[%zu], "
        "inputs_null[%d], output_num[%zu], outputs_null[%d].",
        AICPUE_ERROR_CODE, node_index, static_cast<int32_t>(desc.op == nullptr), desc.input_num,
        static_cast<int32_t>(desc.inputs == nullptr), desc.output_num, static_cast<int32_t>(desc.outputs == nullptr));
    return -1;
  }
  chain.nodes.emplace_back();
  FusedCpuKernelChainNode &node = chain.nodes.back();
  const int32_t init_ret =
      InitializeFusedCpuKernelPlan(*desc.op, desc.inputs, desc.input_num, desc.outputs, desc.output_num, node.plan);
  if (init_ret != 0) {
    AICPUE_LOGE("Initialize fused CPU kernel plan failed: node_index[%zu], input_num[%zu], output_num[%zu], ret[%d].",
                AICPUE_ERROR_CODE, node_index, desc.input_num, desc.output_num, init_ret);
    return -1;
  }
  if ((node.plan.input_tensors.size() != desc.input_num) || (node.plan.input_states.size() != desc.input_num) ||
      (node.plan.output_tensors.size() != desc.output_num) || (node.plan.output_states.size() != desc.output_num)) {
    AICPUE_LOGE(
        "Fused CPU kernel plan size mismatch: node_index[%zu], expected inputs[%zu], input_states[%zu], "
        "outputs[%zu], output_states[%zu], actual inputs[%zu], input_states[%zu], outputs[%zu], "
        "output_states[%zu].",
        AICPUE_ERROR_CODE, node_index, desc.input_num, desc.input_num, desc.output_num, desc.output_num,
        node.plan.input_tensors.size(), node.plan.input_states.size(), node.plan.output_tensors.size(),
        node.plan.output_states.size());
    return -1;
  }
  return (AddFusedChainInputBindings(desc, node_index, external_binding_num, node, chain) == 0) &&
                 (AddFusedChainOutputBindings(desc, node_index, external_binding_num, node, chain) == 0)
             ? 0
             : -1;
}

__attribute__((visibility("default"))) void *CreateCpuConstantFoldingFusedChainPlan(const void *node_descs,
                                                                                    const size_t node_num,
                                                                                    const size_t external_input_num,
                                                                                    const size_t external_output_num) {
  if ((node_descs == nullptr) || (node_num == 0U)) {
    AICPUE_LOGE(
        "Invalid fused CPU chain plan arguments: node_descs_null[%d], node_num[%zu], external_inputs[%zu], "
        "external_outputs[%zu].",
        AICPUE_ERROR_CODE, static_cast<int32_t>(node_descs == nullptr), node_num, external_input_num,
        external_output_num);
    return nullptr;
  }
  const auto *descs = static_cast<const FusedCpuKernelChainNodeDesc *>(node_descs);
  std::unique_ptr<FusedCpuKernelChainPlan> chain = std::make_unique<FusedCpuKernelChainPlan>();

  // 防止计算外部 binding 数量时发生 size_t 整数溢出
  if (external_input_num > (std::numeric_limits<size_t>::max() - external_output_num)) {
    AICPUE_LOGE("Fused CPU chain external binding count overflows size_t: external_inputs[%zu], external_outputs[%zu].",
                AICPUE_ERROR_CODE, external_input_num, external_output_num);
    return nullptr;
  }
  const size_t external_binding_num = external_input_num + external_output_num;
  chain->nodes.reserve(node_num);
  size_t binding_capacity = 0U;
  if (CalculateFusedChainBindingCapacity(descs, node_num, binding_capacity) != 0) {
    return nullptr;
  }
  chain->bindings.reserve(binding_capacity);
  for (size_t i = 0U; i < node_num; ++i) {
    if (InitializeFusedChainNode(descs[i], i, external_binding_num, *chain) != 0) {
      return nullptr;
    }
  }
  AICPUE_LOGD("Created fused cpu chain execution plan, nodes[%zu], dynamic bindings[%zu].", node_num,
              chain->bindings.size());
  return chain.release();
}

__attribute__((visibility("default"))) int32_t RunCpuConstantFoldingFusedChainPlan(void *plan,
                                                                                   const uint32_t binding_flags) {
  FusedCpuKernelChainPlan *chain = static_cast<FusedCpuKernelChainPlan *>(plan);
  if (chain == nullptr) {
    AICPUE_LOGE("Run fused CPU chain plan received null plan: binding_flags[%u].", AICPUE_ERROR_CODE, binding_flags);
    return -1;
  }
  if (binding_flags != 0U) {
    for (FusedCpuKernelBinding &binding : chain->bindings) {
      if (RebindFusedTensorByFlags(*binding.source, binding.target, *binding.state, binding_flags) != 0) {
        AICPUE_LOGE("Failed to rebind fused CPU chain binding: binding_index[%zu], flags[%u].", AICPUE_ERROR_CODE,
                    binding.binding_index, binding_flags);
        return -1;
      }
    }
  }
  for (size_t node_index = 0U; node_index < chain->nodes.size(); ++node_index) {
    if (RunFusedCpuKernelPlan(chain->nodes[node_index].plan) != 0) {
      AICPUE_LOGE("Failed to run fused CPU chain node: node_index[%zu], node_count[%zu].", AICPUE_ERROR_CODE,
                  node_index, chain->nodes.size());
      return -1;
    }
  }
  return 0;
}

__attribute__((visibility("default"))) int32_t
RunCpuConstantFoldingFusedChainPlanBindings(void *plan, const void *binding_data, const uint32_t binding_flags) {
  FusedCpuKernelChainPlan *chain = static_cast<FusedCpuKernelChainPlan *>(plan);
  if (chain == nullptr) {
    AICPUE_LOGE("Run fused CPU chain bindings received null plan: binding_flags[%u].", AICPUE_ERROR_CODE,
                binding_flags);
    return -1;
  }
  if (binding_flags != 0U) {
    if (binding_data == nullptr) {
      AICPUE_LOGE("Run fused CPU chain bindings received null binding data: binding_flags[%u], binding_count[%zu].",
                  AICPUE_ERROR_CODE, binding_flags, chain->bindings.size());
      return -1;
    }
    const auto *bindings = static_cast<const FusedHostCpuTensorBinding *>(binding_data);
    for (FusedCpuKernelBinding &binding : chain->bindings) {
      const FusedHostCpuTensorBinding *runtime_binding = &bindings[binding.binding_index];
      if ((runtime_binding->flags != 0U) &&
          (RebindFusedTensorByBinding(*runtime_binding, binding.target, *binding.state) != 0)) {
        AICPUE_LOGE(
            "Failed to rebind fused CPU runtime binding: binding_index[%zu], runtime_flags[%u], "
            "global_flags[%u].",
            AICPUE_ERROR_CODE, binding.binding_index, runtime_binding->flags, binding_flags);
        return -1;
      }
    }
  }
  for (size_t node_index = 0U; node_index < chain->nodes.size(); ++node_index) {
    if (RunFusedCpuKernelPlan(chain->nodes[node_index].plan) != 0) {
      AICPUE_LOGE("Failed to run fused CPU chain node with runtime bindings: node_index[%zu], node_count[%zu].",
                  AICPUE_ERROR_CODE, node_index, chain->nodes.size());
      return -1;
    }
  }
  return 0;
}

__attribute__((visibility("default"))) void DestroyCpuConstantFoldingFusedChainPlan(void *plan) {
  delete static_cast<FusedCpuKernelChainPlan *>(plan);
}
}
