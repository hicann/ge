/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_common.h"

#include <cstdio>
#include <cstring>
#include <queue>
#include <sstream>
#include <vector>
#include "common/log_inner.h"
#include "model_desc_internal.h"
#include "framework/common/framework_types_internal.h"
#include "securec.h"
#include "rt_external_base.h"

namespace {
constexpr size_t DYNAMIC_BATCH_SIZE = 1U;
constexpr size_t DYNAMIC_HW_SIZE = 2U;
constexpr size_t MIN_OUTPUT_SHAPE_INFO_SIZE = 2U;
constexpr size_t MAX_OUTPUT_SHAPE_INFO_SIZE = MIN_OUTPUT_SHAPE_INFO_SIZE + static_cast<size_t>(ACL_MAX_DIM_CNT);
}  // namespace

namespace acl {
// Parse batch info helper (inline implementation)
static aclError ParseBatchInfoInline(aclmdlDesc *const modelDesc, const int32_t dynamicType,
                                     const std::vector<std::vector<int64_t>> &batchInfo) {
  const uint32_t modelId = modelDesc->modelId;
  if (dynamicType == static_cast<int32_t>(ge::DYNAMIC_DIMS)) {
    // dynamic dims, size can be [1, 4]
    const size_t dimCount = batchInfo[0U].size();
    for (size_t i = 0U; i < batchInfo.size(); ++i) {
      if (batchInfo[i].size() != dimCount) {
        ACL_LOG_INNER_ERROR(
            "[Check][Size]Get dynamic model info invalid, model id[%u], one dim count is %zu "
            "while another is %zu",
            modelId, dimCount, batchInfo[i].size());
        modelDesc->dynamicDims.clear();
        return ACL_ERROR_GE_FAILURE;
      }
      std::vector<uint64_t> oneDims;
      for (size_t j = 0U; j < dimCount; ++j) {
        oneDims.push_back(static_cast<uint64_t>(batchInfo[i][j]));
      }
      modelDesc->dynamicDims.push_back(oneDims);
    }
  } else if (batchInfo[0U].size() == DYNAMIC_BATCH_SIZE) {
    // dynamic batch, size is 1
    for (size_t i = 0U; i < batchInfo.size(); ++i) {
      if (batchInfo[i].size() != DYNAMIC_BATCH_SIZE) {
        ACL_LOG_INNER_ERROR("[Check][Size]get dynamic model info invalid, model id[%u]", modelId);
        modelDesc->dynamicBatch.clear();
        return ACL_ERROR_GE_FAILURE;
      }
      modelDesc->dynamicBatch.push_back(static_cast<uint64_t>(batchInfo[i][0U]));
    }
  } else if (batchInfo[0U].size() == DYNAMIC_HW_SIZE) {
    // dynamic hw, size is 2
    for (size_t i = 0U; i < batchInfo.size(); ++i) {
      if (batchInfo[i].size() != DYNAMIC_HW_SIZE) {
        ACL_LOG_INNER_ERROR("[Check][Size]get dynamic model info invalid, model id[%u]", modelId);
        modelDesc->dynamicHW.clear();
        return ACL_ERROR_GE_FAILURE;
      }
      modelDesc->dynamicHW.push_back(
          {static_cast<uint64_t>(batchInfo[i][0U]), static_cast<uint64_t>(batchInfo[i][1U])});
    }
  } else {
    ACL_LOG_INNER_ERROR("[Get][DynamicModel]get dynamic model info invalid, model id[%u]", modelId);
    return ACL_ERROR_GE_FAILURE;
  }
  return ACL_SUCCESS;
}

static bool CheckMdlLoadConfigFromFile(const aclmdlConfigHandle *const handle) {
  if (handle->attrState.find(ACL_MDL_PATH_PTR) == handle->attrState.end()) {
    ACL_LOG_ERROR(
        "[Check][Type]model load type[%zu]: model path is not set in aclmdlConfigHandle "
        "when load type is from file",
        handle->mdlLoadType);
    const std::string errMsg = "model path is not set in aclmdlConfigHandle when load type is from file";
    acl::AclErrorLogManager::ReportInputError(
        acl::INVALID_PARAM_MSG, std::vector<const char *>({"param", "value", "reason"}),
        std::vector<const char *>({"handle", "inner model path", errMsg.c_str()}));
    return false;
  }
  if (handle->attrState.find(ACL_MDL_WEIGHT_PATH_PTR) != handle->attrState.end()) {
    ACL_LOG_ERROR("[Check][Type]model load type[%zu]: should not set ACL_MDL_WEIGHT_PATH_PTR", handle->mdlLoadType);
    const std::string errMsg = "should not set ACL_MDL_WEIGHT_PATH_PTR";
    acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                                              std::vector<const char *>({"param", "value", "reason"}),
                                              std::vector<const char *>({"handle", "weight path", errMsg.c_str()}));
    return false;
  }
  return true;
}

static bool CheckMdlLoadConfigFromMem(const aclmdlConfigHandle *const handle) {
  if (handle->attrState.find(ACL_MDL_MEM_ADDR_PTR) == handle->attrState.end()) {
    ACL_LOG_ERROR("[Check][Type]model load type[%zu]: model memory ptr is not set in aclmdlConfigHandle",
                  handle->mdlLoadType);
    const std::string errMsg = "model memory ptr is not set in aclmdlConfigHandle when load type is from mem";
    acl::AclErrorLogManager::ReportInputError(
        acl::INVALID_PARAM_MSG, std::vector<const char *>({"param", "value", "reason"}),
        std::vector<const char *>({"handle", "inner model memory", errMsg.c_str()}));
    return false;
  }

  if (handle->attrState.find(ACL_MDL_MEM_SIZET) == handle->attrState.end()) {
    ACL_LOG_ERROR("[Check][Type]model load type[%zu]: model memory size is not set in aclmdlConfigHandle",
                  handle->mdlLoadType);
    const std::string errMsg = "model memory size is not set in aclmdlConfigHandle when load type is from mem";
    acl::AclErrorLogManager::ReportInputError(
        acl::INVALID_PARAM_MSG, std::vector<const char *>({"param", "value", "reason"}),
        std::vector<const char *>({"handle", "inner model memory size", errMsg.c_str()}));
    return false;
  }
  return true;
}

static bool CheckMdlLoadConfigWithQ(const aclmdlConfigHandle *const handle) {
  if (handle->attrState.find(ACL_MDL_INPUTQ_ADDR_PTR) == handle->attrState.end()) {
    ACL_LOG_ERROR("[Check][Type]model load type[%zu]: inputQ ptr is not set in aclmdlConfigHandle",
                  handle->mdlLoadType);
    const std::string errMsg =
        "ACL_MDL_INPUTQ_ADDR_PTR is not set in aclmdlConfigHandle "
        "when load type is with queue";
    acl::AclErrorLogManager::ReportInputError(
        acl::INVALID_PARAM_MSG, std::vector<const char *>({"param", "value", "reason"}),
        std::vector<const char *>({"handle", "inner inputq addr", errMsg.c_str()}));
    return false;
  }

  if (handle->attrState.find(ACL_MDL_INPUTQ_NUM_SIZET) == handle->attrState.end()) {
    ACL_LOG_ERROR("[Check][Type]model load type[%zu]: inputQ num is not set in aclmdlConfigHandle",
                  handle->mdlLoadType);
    const std::string errMsg =
        "ACL_MDL_INPUTQ_NUM_SIZET is not set in aclmdlConfigHandle "
        "when load type is with queue";
    acl::AclErrorLogManager::ReportInputError(
        acl::INVALID_PARAM_MSG, std::vector<const char *>({"param", "value", "reason"}),
        std::vector<const char *>({"handle", "inner inputq num", errMsg.c_str()}));
    return false;
  }

  if (handle->attrState.find(ACL_MDL_OUTPUTQ_ADDR_PTR) == handle->attrState.end()) {
    ACL_LOG_ERROR("[Check][Type]model load type[%zu]: outputQ ptr is not set in aclmdlConfigHandle",
                  handle->mdlLoadType);
    const std::string errMsg =
        "ACL_MDL_OUTPUTQ_ADDR_PTR is not set in aclmdlConfigHandle "
        "when load type is with queue";
    acl::AclErrorLogManager::ReportInputError(
        acl::INVALID_PARAM_MSG, std::vector<const char *>({"param", "value", "reason"}),
        std::vector<const char *>({"handle", "inner outputq addr", errMsg.c_str()}));
    return false;
  }

  if (handle->attrState.find(ACL_MDL_OUTPUTQ_NUM_SIZET) == handle->attrState.end()) {
    ACL_LOG_ERROR("[Check][Type]model load type[%zu]: outputQ num is not set in aclmdlConfigHandle",
                  handle->mdlLoadType);
    const std::string errMsg =
        "ACL_MDL_OUTPUTQ_NUM_SIZET is not set in aclmdlConfigHandle "
        "when load type is with queue";
    acl::AclErrorLogManager::ReportInputError(
        acl::INVALID_PARAM_MSG, std::vector<const char *>({"param", "value", "reason"}),
        std::vector<const char *>({"handle", "inner outputq num", errMsg.c_str()}));
    return false;
  }

  return true;
}

ACL_FUNC_VISIBILITY bool CheckMdlConfigHandle(const aclmdlConfigHandle *const handle) {
  if (handle->attrState.find(ACL_MDL_LOAD_TYPE_SIZET) == handle->attrState.end()) {
    ACL_LOG_ERROR("[Find][Type]model load type is not set in aclmdlConfigHandle");
    const std::string errMsg = "ACL_MDL_LOAD_TYPE_SIZET is not set in aclmdlConfigHandle";
    acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                                              std::vector<const char *>({"param", "value", "reason"}),
                                              std::vector<const char *>({"handle", "inner load type", errMsg.c_str()}));
    return false;
  }

  if ((handle->mdlLoadType == static_cast<size_t>(ACL_MDL_LOAD_FROM_FILE)) ||
      (handle->mdlLoadType == static_cast<size_t>(ACL_MDL_LOAD_FROM_FILE_WITH_MEM))) {
    if (!CheckMdlLoadConfigFromFile(handle)) {
      return false;
    }
  }

  if ((handle->mdlLoadType == static_cast<size_t>(ACL_MDL_LOAD_FROM_MEM)) ||
      (handle->mdlLoadType == static_cast<size_t>(ACL_MDL_LOAD_FROM_MEM_WITH_MEM))) {
    if ((!CheckMdlLoadConfigFromMem(handle))) {
      return false;
    }
  }

  if (handle->mdlLoadType == static_cast<size_t>(ACL_MDL_LOAD_FROM_FILE_WITH_Q)) {
    if ((!CheckMdlLoadConfigFromFile(handle)) || (!CheckMdlLoadConfigWithQ(handle))) {
      return false;
    }
  }

  if (handle->mdlLoadType == static_cast<size_t>(ACL_MDL_LOAD_FROM_MEM_WITH_Q)) {
    if ((!CheckMdlLoadConfigFromMem(handle)) || (!CheckMdlLoadConfigWithQ(handle))) {
      return false;
    }
  }
  return true;
}

ACL_FUNC_VISIBILITY aclError GetDynamicTensorInfoHelp(aclmdlDesc *const modelDesc, const int32_t dynamicType,
                                                      const std::vector<std::vector<int64_t>> &batchInfo) {
  if (batchInfo.empty()) {
    ACL_LOG_INFO("model is not dynamic, batchInfo is empty, modelId[%u]", modelDesc->modelId);
    return ACL_SUCCESS;
  }

  ACL_LOG_INFO("model is dynamic, modelId[%u]", modelDesc->modelId);
  const aclError retVal = ParseBatchInfoInline(modelDesc, dynamicType, batchInfo);
  if (retVal != ACL_SUCCESS) {
    ACL_LOG_INNER_ERROR("[Parse][BatchInfo]get model dynamic info failed, result[%d], model id[%u]", retVal,
                        modelDesc->modelId);
    return retVal;
  }

  return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError GetCurGearIndex(const aclmdlDesc *const modelDesc, const std::vector<uint64_t> &shapeInfo,
                                             const int32_t dynamicType, size_t &curGearIndex) {
  if (dynamicType == static_cast<int32_t>(ge::DYNAMIC_DIMS)) {
    ACL_LOG_DEBUG("Get dynamic dims gear index, dynamicType[%d], modelId[%u]", dynamicType, modelDesc->modelId);
    for (size_t i = 0U; i < modelDesc->dynamicDims.size(); ++i) {
      if (shapeInfo == modelDesc->dynamicDims[i]) {
        curGearIndex = i;
        return ACL_SUCCESS;
      }
    }
  } else {
    const size_t shapeSize = shapeInfo.size();
    if (shapeSize == DYNAMIC_BATCH_SIZE) {
      ACL_LOG_DEBUG("Get dynamic batch gear index, dynamicType[%d], modelId[%u]", dynamicType, modelDesc->modelId);
      for (size_t i = 0U; i < modelDesc->dynamicBatch.size(); ++i) {
        if (shapeInfo[0U] == modelDesc->dynamicBatch[i]) {
          curGearIndex = i;
          return ACL_SUCCESS;
        }
      }
    } else if (shapeSize == DYNAMIC_HW_SIZE) {
      ACL_LOG_DEBUG("Get dynamic hw gear index, dynamicType[%d], modelId[%u]", dynamicType, modelDesc->modelId);
      for (size_t i = 0U; i < modelDesc->dynamicHW.size(); ++i) {
        if (shapeInfo == modelDesc->dynamicHW[i]) {
          curGearIndex = i;
          return ACL_SUCCESS;
        }
      }
    } else {
      ACL_LOG_INNER_ERROR("[Check][dynamicType]dynamicType[%d] is invalid", dynamicType);
    }
  }
  return ACL_ERROR_FAILURE;
}

ACL_FUNC_VISIBILITY aclError GetCurOuputShapeInfo(const aclmdlDesc *const modelDesc, const size_t index,
                                                  const size_t curGearIndex, aclmdlIODims *const dims) {
  ACL_LOG_DEBUG("curGearIndex is %zu, dynamicOutputShapeInfoSize is %zu , modelId is %u", curGearIndex,
                modelDesc->dynamicOutputShape.size(), modelDesc->modelId);
  for (auto &it : modelDesc->dynamicOutputShape) {
    if ((it.size() < MIN_OUTPUT_SHAPE_INFO_SIZE) || (it.size() > MAX_OUTPUT_SHAPE_INFO_SIZE)) {
      ACL_LOG_INNER_ERROR(
          "[Check][dynamicOutputShape]output shape info size[%zu] is invalid, range is "
          "[%zu, %zu]",
          it.size(), MIN_OUTPUT_SHAPE_INFO_SIZE, MAX_OUTPUT_SHAPE_INFO_SIZE);
      return ACL_ERROR_FAILURE;
    }
    if (((static_cast<int64_t>(curGearIndex) == it[0U]) || (it[0U] == -1)) && (static_cast<int64_t>(index) == it[1U])) {
      int32_t idx = 0;
      for (size_t i = 2U; i < it.size(); ++i) {
        dims->dims[idx] = it[i];
        idx++;
      }
      dims->dimCount = it.size() - 2U;
      const aclmdlTensorDesc &tensorDesc = modelDesc->outputDesc[index];
      const auto ret = GetTensorDescNameToDims(modelDesc, tensorDesc.name, TensorType::OUTPUT_TENSOR_TYPE, index, dims);
      if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Get][TensorDescName]get tensor desc name to dims failed, errorCode = %d", ret);
        return ret;
      }
      return ACL_SUCCESS;
    }
  }
  return ACL_ERROR_FAILURE;
}

ACL_FUNC_VISIBILITY aclError GetModelOutputShapeInfoHelp(aclmdlDesc *const modelDesc,
                                                         std::vector<std::string> &geDynamicOutputShape) {
  if (geDynamicOutputShape.empty()) {
    ACL_LOG_INFO("model is not dynamic, geDynamicOutputShape is empty, modelId[%u]", modelDesc->modelId);
    return ACL_SUCCESS;
  }

  std::vector<std::vector<int64_t>> &dynamicOutputShape = modelDesc->dynamicOutputShape;
  for (auto &it : geDynamicOutputShape) {
    int64_t val = 0;
    int64_t negativeFlag = 1;
    std::vector<int64_t> outputShape;
    for (auto &strIt : it) {
      if ((strIt >= '0') && (strIt <= '9')) {
        val = (val * 10) + static_cast<int64_t>(strIt - '0');
      } else if (strIt == '-') {
        negativeFlag = -1;
        ACL_LOG_DEBUG("dynamic model include static output");
      } else {
        val *= negativeFlag;
        outputShape.emplace_back(val);
        val = 0;
        negativeFlag = 1;
      }
    }
    val *= negativeFlag;
    outputShape.emplace_back(val);
    dynamicOutputShape.emplace_back(outputShape);
  }
  return ACL_SUCCESS;
}

// get real tensor name from modelDesc, it will return nullptr if tensorName isn't in modelDesc
ACL_FUNC_VISIBILITY const char_t *GetRealTensorName(const aclmdlDesc *const modelDesc, const std::string &tensorName) {
  for (size_t idx = 0U; idx < modelDesc->inputDesc.size(); ++idx) {
    if (modelDesc->inputDesc[idx].name == tensorName) {
      return modelDesc->inputDesc[idx].name.c_str();
    }
  }

  for (size_t idx = 0U; idx < modelDesc->outputDesc.size(); ++idx) {
    if (modelDesc->outputDesc[idx].name == tensorName) {
      return modelDesc->outputDesc[idx].name.c_str();
    }
  }
  return nullptr;
}

// Check if conversion tensor name is legal
bool IsConvertTensorNameLegal(const aclmdlDesc *const modelDesc, const std::string &tensorName) {
  return (GetRealTensorName(modelDesc, tensorName) == nullptr);
}

// current conversion tensor name illegal needs to be transformed
ACL_FUNC_VISIBILITY bool TransConvertTensorNameToLegal(const aclmdlDesc *const modelDesc, std::string &tensorName) {
  size_t depth = 0U;
  tensorName = tensorName + "_";
  std::queue<std::string> q;
  q.push(tensorName);
  constexpr size_t maxDepth = 3U;
  while (!q.empty()) {
    if (depth == maxDepth) {
      ACL_LOG_INFO("reach max depth[%zu], cannot generate legal convert tensor name", maxDepth);
      tensorName = tensorName.substr(0U, tensorName.size() - 1U);
      return false;
    }
    const size_t len = q.size();
    size_t idx = 0U;
    while (idx < len) {
      std::string curTensorName = q.front();
      q.pop();
      ++idx;
      for (char_t c = 'a'; c <= 'z'; ++c) {
        curTensorName += c;
        if (IsConvertTensorNameLegal(modelDesc, curTensorName)) {
          tensorName = curTensorName;
          return true;
        }
        q.push(curTensorName);
        curTensorName = curTensorName.substr(0U, curTensorName.size() - 1U);
      }
    }
    depth++;
  }
  return false;
}

// Get conversion tensor name from params
ACL_FUNC_VISIBILITY void GetConvertTensorName(const aclmdlDesc *const modelDesc, const size_t idx,
                                              const TensorType tensorType, std::string &convertName) {
  convertName =
      std::string(TENSOR_NAME_PREFIX) + "_" + std::string(MODEL_ID_STR) + "_" + std::to_string(modelDesc->modelId);
  if (tensorType == TensorType::INPUT_TENSOR_TYPE) {
    convertName += ("_" + std::string(TENSOR_INPUT_STR));
  } else {
    convertName += ("_" + std::string(TENSOR_OUTPUT_STR));
  }
  convertName += ("_" + std::to_string(idx));
  ACL_LOG_INFO("convert realname of tensor success, conversion name = %s", convertName.c_str());
}

// get tensor name to dims with or without realname
ACL_FUNC_VISIBILITY aclError GetTensorDescNameToDims(const aclmdlDesc *const modelDesc, const std::string &realName,
                                                     const TensorType tensorType, const size_t idx,
                                                     aclmdlIODims *const dims) {
  const size_t dimsNameLen = sizeof(dims->name);
  std::string tensorName;
  if ((realName.size() + 1U) > dimsNameLen) {
    // use conversion name because realname is too long
    ACL_LOG_INFO("use conversion name because real tensor name is longer than %zu characters", dimsNameLen);
    GetConvertTensorName(modelDesc, idx, tensorType, tensorName);
    if (!IsConvertTensorNameLegal(modelDesc, tensorName)) {
      if (!TransConvertTensorNameToLegal(modelDesc, tensorName)) {
        ACL_LOG_WARN("cannot generate legal tensor name, use conversion name %s may have conflict risk",
                     tensorName.c_str());
      }
    }
  } else {
    tensorName = realName;
  }

  const auto ret = strncpy_s(dims->name, dimsNameLen, tensorName.c_str(), tensorName.size());
  if (ret != EOK) {
    ACL_LOG_INNER_ERROR("[Copy][Str]call strncpy_s failed, result = %d", ret);
    return ACL_ERROR_FAILURE;
  }
  return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError GetDims(const aclmdlDesc *const modelDesc, const TensorType tensorType,
                                     const DimsType dimsType, const size_t idx, aclmdlIODims *const dims) {
  ACL_REQUIRES_NOT_NULL(dims);
  std::vector<aclmdlTensorDesc> desc;
  if (tensorType == TensorType::INPUT_TENSOR_TYPE) {
    desc = modelDesc->inputDesc;
  } else {
    desc = modelDesc->outputDesc;
  }

  const size_t descSize = desc.size();
  if (idx >= descSize) {
    ACL_LOG_INNER_ERROR(
        "[Check][Params]GetDims failed, index[%zu] cannot greater than or equal to tensor "
        "size[%zu]",
        idx, descSize);
    return ACL_ERROR_INVALID_PARAM;
  }

  const aclmdlTensorDesc &tensorDesc = desc[idx];
  const auto ret = GetTensorDescNameToDims(modelDesc, tensorDesc.name, tensorType, idx, dims);
  if (ret != ACL_SUCCESS) {
    ACL_LOG_INNER_ERROR("[Get][TensorDescName]get tensor desc name to dims failed, errorCode = %d", ret);
    return ret;
  }
  std::vector<int64_t> tensorDims;
  if (dimsType == DimsType::DIMS_TYPE_V1) {
    tensorDims = tensorDesc.dims;
  } else if (dimsType == DimsType::DIMS_TYPE_V2) {
    tensorDims = tensorDesc.dimsV2;
  } else {
    ACL_LOG_INNER_ERROR("[Check][dimsType]dims type[%d] is invalid", static_cast<int32_t>(dimsType));
    return ACL_ERROR_FAILURE;
  }

  const size_t dimSize = tensorDims.size();
  if (dimSize > static_cast<size_t>(ACL_MAX_DIM_CNT)) {
    ACL_LOG_INNER_ERROR("[Check][dimSize]get dims failed, dims count[%zu] cannot larger than max[%d]", dims->dimCount,
                        ACL_MAX_DIM_CNT);
    return ACL_ERROR_STORAGE_OVER_LIMIT;
  }
  dims->dimCount = dimSize;

  for (size_t i = 0U; i < dimSize; ++i) {
    dims->dims[i] = tensorDims[i];
  }

  return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY void SetAippInfo(aclAippInfo *const aippInfo, const ge::AippConfigInfo &aippParams) {
  ACL_LOG_DEBUG("start to execute SetAippInfo");
  if (aippInfo == nullptr) {
    ACL_LOG_INNER_ERROR("[Check][AippInfo]param aippInfo must not be null");
    return;
  }
  aippInfo->inputFormat = static_cast<aclAippInputFormat>(aippParams.input_format);
  aippInfo->srcImageSizeW = aippParams.src_image_size_w;
  aippInfo->srcImageSizeH = aippParams.src_image_size_h;

  aippInfo->cropSwitch = aippParams.crop;
  aippInfo->loadStartPosW = aippParams.load_start_pos_w;
  aippInfo->loadStartPosH = aippParams.load_start_pos_h;
  aippInfo->cropSizeW = aippParams.crop_size_w;
  aippInfo->cropSizeH = aippParams.crop_size_h;

  aippInfo->resizeSwitch = aippParams.resize;
  aippInfo->resizeOutputW = aippParams.resize_output_w;
  aippInfo->resizeOutputH = aippParams.resize_output_h;

  aippInfo->paddingSwitch = aippParams.padding;
  aippInfo->leftPaddingSize = aippParams.left_padding_size;
  aippInfo->rightPaddingSize = aippParams.right_padding_size;
  aippInfo->topPaddingSize = aippParams.top_padding_size;
  aippInfo->bottomPaddingSize = aippParams.bottom_padding_size;

  aippInfo->cscSwitch = aippParams.csc_switch;
  aippInfo->rbuvSwapSwitch = aippParams.rbuv_swap_switch;
  aippInfo->axSwapSwitch = aippParams.ax_swap_switch;
  aippInfo->singleLineMode = aippParams.single_line_mode;

  aippInfo->matrixR0C0 = aippParams.matrix_r0c0;
  aippInfo->matrixR0C1 = aippParams.matrix_r0c1;
  aippInfo->matrixR0C2 = aippParams.matrix_r0c2;
  aippInfo->matrixR1C0 = aippParams.matrix_r1c0;
  aippInfo->matrixR1C1 = aippParams.matrix_r1c1;
  aippInfo->matrixR1C2 = aippParams.matrix_r1c2;
  aippInfo->matrixR2C0 = aippParams.matrix_r2c0;
  aippInfo->matrixR2C1 = aippParams.matrix_r2c1;
  aippInfo->matrixR2C2 = aippParams.matrix_r2c2;

  aippInfo->outputBias0 = aippParams.output_bias_0;
  aippInfo->outputBias1 = aippParams.output_bias_1;
  aippInfo->outputBias2 = aippParams.output_bias_2;
  aippInfo->inputBias0 = aippParams.input_bias_0;
  aippInfo->inputBias1 = aippParams.input_bias_1;
  aippInfo->inputBias2 = aippParams.input_bias_2;

  aippInfo->meanChn0 = aippParams.mean_chn_0;
  aippInfo->meanChn1 = aippParams.mean_chn_1;
  aippInfo->meanChn2 = aippParams.mean_chn_2;
  aippInfo->meanChn3 = aippParams.mean_chn_3;
  aippInfo->minChn0 = aippParams.min_chn_0;
  aippInfo->minChn1 = aippParams.min_chn_1;
  aippInfo->minChn2 = aippParams.min_chn_2;
  aippInfo->minChn3 = aippParams.min_chn_3;

  aippInfo->varReciChn0 = aippParams.var_reci_chn_0;
  aippInfo->varReciChn1 = aippParams.var_reci_chn_1;
  aippInfo->varReciChn2 = aippParams.var_reci_chn_2;
  aippInfo->varReciChn3 = aippParams.var_reci_chn_3;
  ACL_LOG_DEBUG("end to execute SetAippInfo");
}

ACL_FUNC_VISIBILITY std::string GetNpuArch() {
  char npu_arch[MAX_NPU_ARCH_LEN] = {0};
  const auto ret = rtGetSocSpec("version", "NpuArch", npu_arch, sizeof(npu_arch));
  if (ret != RT_ERROR_NONE) {
    return "";
  }
  return std::string(npu_arch);
}

ACL_FUNC_VISIBILITY aclError SetIODims(const ge::InputOutputDims &oriDims, aclmdlIODims &dstDims) {
  ACL_LOG_DEBUG("start to execute SetIODims");
  dstDims.dimCount = oriDims.dim_num;
  if (oriDims.dims.size() > static_cast<size_t>(ACL_MAX_DIM_CNT)) {
    ACL_LOG_INNER_ERROR("[Check][Params]size of dims[%zu] must be smaller than ACL_MAX_DIM_CNT(128)",
                        oriDims.dims.size());
    return ACL_ERROR_GE_FAILURE;
  }
  for (size_t i = 0U; i < oriDims.dims.size(); ++i) {
    dstDims.dims[i] = oriDims.dims[i];
  }
  if (oriDims.name.empty()) {
    ACL_LOG_DEBUG("the name of oriDims is empty");
    return ACL_SUCCESS;
  }
  const auto ret = strncpy_s(dstDims.name, sizeof(dstDims.name), oriDims.name.c_str(), oriDims.name.size());
  if (ret != EOK) {
    ACL_LOG_INNER_ERROR("[Copy][Str]call strncpy_s failed");
    return ACL_ERROR_FAILURE;
  }
  return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY std::string AippInfoDebugString(const aclAippInfo *aippInfo) {
  if (aippInfo == nullptr) {
    ACL_LOG_INNER_ERROR("[Check][aippInfo]param aippInfo must not be null");
    return "";
  }
  std::stringstream ss;
  ss << "aclAippInfo[";
  ss << " inputFormat:" << static_cast<int32_t>(aippInfo->inputFormat);
  ss << " srcImageSizeW:" << aippInfo->srcImageSizeW;
  ss << " srcImageSizeH:" << aippInfo->srcImageSizeH;

  ss << " cropSwitch:" << static_cast<int32_t>(aippInfo->cropSwitch);
  ss << " loadStartPosW:" << aippInfo->loadStartPosW;
  ss << " loadStartPosH:" << aippInfo->loadStartPosH;
  ss << " cropSizeW:" << aippInfo->cropSizeW;
  ss << " cropSizeH:" << aippInfo->cropSizeH;

  ss << " resizeSwitch:" << static_cast<int32_t>(aippInfo->resizeSwitch);
  ss << " resizeOutputW:" << aippInfo->resizeOutputW;
  ss << " resizeOutputH:" << aippInfo->resizeOutputH;

  ss << " paddingSwitch:" << static_cast<int32_t>(aippInfo->paddingSwitch);
  ss << " leftPaddingSize:" << aippInfo->leftPaddingSize;
  ss << " rightPaddingSize:" << aippInfo->rightPaddingSize;
  ss << " topPaddingSize:" << aippInfo->topPaddingSize;
  ss << " bottomPaddingSize:" << aippInfo->bottomPaddingSize;

  ss << " cscSwitch:" << static_cast<int32_t>(aippInfo->cscSwitch);
  ss << " rbuvSwapSwitch:" << static_cast<int32_t>(aippInfo->rbuvSwapSwitch);
  ss << " axSwapSwitch:" << static_cast<int32_t>(aippInfo->axSwapSwitch);
  ss << " singleLineMode:" << static_cast<int32_t>(aippInfo->singleLineMode);

  ss << " matrixR0C0:" << aippInfo->matrixR0C0;
  ss << " matrixR0C1:" << aippInfo->matrixR0C1;
  ss << " matrixR0C2:" << aippInfo->matrixR0C2;
  ss << " matrixR1C0:" << aippInfo->matrixR1C0;
  ss << " matrixR1C1:" << aippInfo->matrixR1C1;
  ss << " matrixR1C2:" << aippInfo->matrixR1C2;
  ss << " matrixR2C0:" << aippInfo->matrixR2C0;
  ss << " matrixR2C1:" << aippInfo->matrixR2C1;
  ss << " matrixR2C2:" << aippInfo->matrixR2C2;

  ss << " outputBias0:" << aippInfo->outputBias0;
  ss << " outputBias1:" << aippInfo->outputBias1;
  ss << " outputBias2:" << aippInfo->outputBias2;
  ss << " inputBias0:" << aippInfo->inputBias0;
  ss << " inputBias1:" << aippInfo->inputBias1;
  ss << " inputBias2:" << aippInfo->inputBias2;

  ss << " meanChn0:" << aippInfo->meanChn0;
  ss << " meanChn1:" << aippInfo->meanChn1;
  ss << " meanChn2:" << aippInfo->meanChn2;
  ss << " meanChn3:" << aippInfo->meanChn3;
  ss << " minChn0:" << aippInfo->minChn0;
  ss << " minChn1:" << aippInfo->minChn1;
  ss << " minChn2:" << aippInfo->minChn2;
  ss << " minChn3:" << aippInfo->minChn3;
  ss << " varReciChn0:" << aippInfo->varReciChn0;
  ss << " varReciChn1:" << aippInfo->varReciChn1;
  ss << " varReciChn2:" << aippInfo->varReciChn2;
  ss << " varReciChn3:" << aippInfo->varReciChn3;

  ss << " shapeCount:" << aippInfo->shapeCount;
  ss << " srcFormat:" << aippInfo->srcFormat;
  ss << " srcDatatype:" << aippInfo->srcDatatype;
  ss << " srcDimNum:" << aippInfo->srcDimNum;
  ss << " ]";
  return ss.str();
}

ACL_FUNC_VISIBILITY std::string DimsDebugString(const aclmdlIODims &ioDims) {
  std::stringstream ss;
  ss << "[" << " tensorName:" << ioDims.name;
  ss << " dimcount:" << static_cast<int32_t>(ioDims.dimCount);
  ss << " dims:";
  for (size_t i = 0U; i < ioDims.dimCount; ++i) {
    ss << " " << ioDims.dims[i];
  }
  ss << "]; ";
  return ss.str();
}

ACL_FUNC_VISIBILITY std::string AippDimsDebugString(const aclAippDims *aippDims, size_t shapeCount) {
  if (aippDims == nullptr) {
    ACL_LOG_INNER_ERROR("[Check][aippDims]param aippDims must not be null");
    return "";
  }
  std::stringstream ssDims;
  for (size_t i = 0U; i < shapeCount; ++i) {
    ssDims << " aclAippDims[" << i << "]: ";
    ssDims << DimsDebugString(aippDims[i].srcDims);
    ssDims << " srcSize:" << aippDims[i].srcSize;
    ssDims << DimsDebugString(aippDims[i].aippOutdims);
    ssDims << " aippOutSize:" << aippDims[i].aippOutSize;
  }
  return ssDims.str();
}

// FP16 debug helpers, internal use only
union TypeUnion {
  float32_t fVal;
  uint32_t uVal;
};

#define FP16_EXTRAC_SIGN(x) (((x) >> 15U) & 1U)
#define FP16_EXTRAC_EXP(x) (((x) >> 10U) & acl::FP16_MAX_EXP)
#define FP16_EXTRAC_MAN(x) ((((x) >> 0U) & 0x3FFU) | ((((((x) >> 10U) & 0x1FU) > 0U) ? 1U : 0U) * 0x400U))
#define FP32_CONSTRUCTOR(s, e, m) \
  (((s) << acl::FP32_SIGN_INDEX) | ((e) << acl::FP32_MAN_LEN) | ((m) & acl::FP32_MAX_MAN))

static void ExtractFP16(const uint16_t val, uint16_t *const s, int16_t *const e, uint16_t *const m) {
  *s = FP16_EXTRAC_SIGN(val);
  *e = static_cast<int16_t>(FP16_EXTRAC_EXP(val));
  *m = FP16_EXTRAC_MAN(val);
  if ((*e) == 0) {
    *e = 1;
  }
}

static float32_t Fp16ToFloat(const uint16_t val) {
  uint16_t hf_sign;
  uint16_t hf_man;
  int16_t hf_exp;
  ExtractFP16(val, &hf_sign, &hf_exp, &hf_man);

  while ((hf_man != 0U) && ((hf_man & acl::FP16_MAN_HIDE_BIT) == 0U)) {
    hf_man <<= 1U;
    hf_exp--;
  }

  uint32_t exp_ret;
  uint32_t man_ret;
  if (hf_man == 0U) {
    exp_ret = 0U;
    man_ret = 0U;
  } else {
    exp_ret = static_cast<uint32_t>(hf_exp + static_cast<int16_t>(acl::FP32_EXP_BIAS - acl::FP16_EXP_BIAS));
    man_ret = static_cast<uint32_t>(hf_man & acl::FP16_MAN_MASK);
    man_ret = man_ret << (acl::FP32_MAN_LEN - acl::FP16_MAN_LEN);
  }

  const uint32_t sign_ret = hf_sign;
  TypeUnion type_union;
  type_union.uVal = FP32_CONSTRUCTOR(sign_ret, exp_ret, man_ret);
  return type_union.fVal;
}

ACL_FUNC_VISIBILITY std::string AippParmsDebugString(const kAippDynamicPara &aippParms) {
  std::stringstream ss;
  ss << "kAippDynamicPara[";
  ss << " inputFormat:" << static_cast<uint32_t>(aippParms.inputFormat);
  ss << " cscSwitch:" << static_cast<int32_t>(aippParms.cscSwitch);
  ss << " rbuvSwapSwitch:" << static_cast<int32_t>(aippParms.rbuvSwapSwitch);
  ss << " axSwapSwitch:" << static_cast<int32_t>(aippParms.axSwapSwitch);
  ss << " batchNum:" << static_cast<int32_t>(aippParms.batchNum);
  ss << " srcImageSizeW:" << aippParms.srcImageSizeW;
  ss << " srcImageSizeH:" << aippParms.srcImageSizeH;
  ss << " cscMatrixR0C0:" << static_cast<int32_t>(aippParms.cscMatrixR0C0);
  ss << " cscMatrixR0C1:" << static_cast<int32_t>(aippParms.cscMatrixR0C1);
  ss << " cscMatrixR0C2:" << static_cast<int32_t>(aippParms.cscMatrixR0C2);
  ss << " cscMatrixR1C0:" << static_cast<int32_t>(aippParms.cscMatrixR1C0);
  ss << " cscMatrixR1C1:" << static_cast<int32_t>(aippParms.cscMatrixR1C1);
  ss << " cscMatrixR1C2:" << static_cast<int32_t>(aippParms.cscMatrixR1C2);
  ss << " cscMatrixR2C0:" << static_cast<int32_t>(aippParms.cscMatrixR2C0);
  ss << " cscMatrixR2C1:" << static_cast<int32_t>(aippParms.cscMatrixR2C1);
  ss << " cscMatrixR2C2:" << static_cast<int32_t>(aippParms.cscMatrixR2C2);
  ss << " cscOutputBiasR0:" << static_cast<uint32_t>(aippParms.cscOutputBiasR0);
  ss << " cscOutputBiasR1:" << static_cast<uint32_t>(aippParms.cscOutputBiasR1);
  ss << " cscOutputBiasR2:" << static_cast<uint32_t>(aippParms.cscOutputBiasR2);
  ss << " cscInputBiasR0:" << static_cast<uint32_t>(aippParms.cscInputBiasR0);
  ss << " cscInputBiasR1:" << static_cast<uint32_t>(aippParms.cscInputBiasR1);
  ss << " cscInputBiasR2:" << static_cast<uint32_t>(aippParms.cscInputBiasR2);
  ss << " ]";
  return ss.str();
}

ACL_FUNC_VISIBILITY std::string AippBatchParaDebugString(const kAippDynamicBatchPara &aippBatchPara) {
  std::stringstream ss;
  ss << "kAippDynamicBatchPara[";
  ss << " cropSwitch:" << static_cast<int32_t>(aippBatchPara.cropSwitch);
  ss << " cropStartPosW:" << aippBatchPara.cropStartPosW;
  ss << " cropStartPosH:" << aippBatchPara.cropStartPosH;
  ss << " cropSizeW:" << aippBatchPara.cropSizeW;
  ss << " cropSizeH:" << aippBatchPara.cropSizeH;
  ss << " scfSwitch:" << static_cast<int32_t>(aippBatchPara.scfSwitch);
  ss << " scfInputSizeW:" << aippBatchPara.scfInputSizeW;
  ss << " scfInputSizeH:" << aippBatchPara.scfInputSizeH;
  ss << " scfOutputSizeW:" << aippBatchPara.scfOutputSizeW;
  ss << " scfOutputSizeH:" << aippBatchPara.scfOutputSizeH;
  ss << " paddingSwitch:" << static_cast<int32_t>(aippBatchPara.paddingSwitch);
  ss << " paddingSizeTop:" << aippBatchPara.paddingSizeTop;
  ss << " paddingSizeBottom:" << aippBatchPara.paddingSizeBottom;
  ss << " paddingSizeLeft:" << aippBatchPara.paddingSizeLeft;
  ss << " paddingSizeRight:" << aippBatchPara.paddingSizeRight;
  ss << " rotateSwitch:" << static_cast<int32_t>(aippBatchPara.rotateSwitch);
  ss << " dtcPixelMeanChn0:" << static_cast<int32_t>(aippBatchPara.dtcPixelMeanChn0);
  ss << " dtcPixelMeanChn1:" << static_cast<int32_t>(aippBatchPara.dtcPixelMeanChn1);
  ss << " dtcPixelMeanChn2:" << static_cast<int32_t>(aippBatchPara.dtcPixelMeanChn2);
  ss << " dtcPixelMeanChn3:" << static_cast<int32_t>(aippBatchPara.dtcPixelMeanChn3);
  ss << " dtcPixelMinChn0:" << static_cast<uint32_t>(aippBatchPara.dtcPixelMinChn0);
  ss << " dtcPixelMinChn1:" << static_cast<uint32_t>(aippBatchPara.dtcPixelMinChn1);
  ss << " dtcPixelMinChn2:" << static_cast<uint32_t>(aippBatchPara.dtcPixelMinChn2);
  ss << " dtcPixelMinChn3:" << static_cast<uint32_t>(aippBatchPara.dtcPixelMinChn3);
  ss << " dtcPixelVarReciChn0:" << Fp16ToFloat(aippBatchPara.dtcPixelVarReciChn0);
  ss << " dtcPixelVarReciChn1:" << Fp16ToFloat(aippBatchPara.dtcPixelVarReciChn1);
  ss << " dtcPixelVarReciChn2:" << Fp16ToFloat(aippBatchPara.dtcPixelVarReciChn2);
  ss << " dtcPixelVarReciChn3:" << Fp16ToFloat(aippBatchPara.dtcPixelVarReciChn3);
  ss << " ]";
  return ss.str();
}

ACL_FUNC_VISIBILITY size_t GetMaxShapeIndex(const std::vector<ge::InputOutputDims> &inputDims) {
  size_t maxShapeIndex = 0U;
  uint32_t shapeSize = 0U;
  for (size_t i = 0U; i < inputDims.size(); ++i) {
    if (inputDims[i].size > shapeSize) {
      shapeSize = inputDims[i].size;
      maxShapeIndex = i;
    }
  }
  ACL_LOG_INFO("GetMaxShapeIndex success, maxShapeIndex[%zu]", maxShapeIndex);
  return maxShapeIndex;
}

}  // namespace acl
