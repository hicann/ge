/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <sstream>
#include "acl/acl_mdl.h"
#include "common/prof_api_reg.h"
#include "acl/acl_rt.h"
#include "common/dynamic_aipp.h"
#include "model_desc_internal.h"
#include "common/log_inner.h"
#include "utils/math_utils.h"
#include "model/acl_model_impl_om2.h"
#include "model_common.h"
#include "acl_resource_manager_om2.h"
#include "framework/runtime/om2_model_executor.h"
#include "error_codes_inner.h"
#include "securec.h"
#include "aipp_param_check.h"

namespace acl {
struct Fp16Type {
 public:
  Fp16Type(void) : val_(0x0U) {}

  explicit Fp16Type(const float32_t fVal) : val_(0x0U) {
    this->operator=(fVal);
  }

  ~Fp16Type() = default;

  Fp16Type &operator=(const float32_t fVal) {
    uint16_t sRet;
    uint16_t mRet;
    int16_t eRet;
    uint32_t eF;
    uint32_t mF;

    const void *const pV = static_cast<const void *>(&fVal);
    const uint32_t ui32V = *(static_cast<const uint32_t *>(pV));
    uint32_t mLenDelta;

    sRet = static_cast<uint16_t>((ui32V & acl::FP32_SIGN_MASK) >> acl::FP32_SIGN_INDEX);
    eF = (ui32V & acl::FP32_EXP_MASK) >> acl::FP32_MAN_LEN;
    mF = (ui32V & acl::FP32_MAN_MASK);
    mLenDelta = acl::FP32_MAN_LEN - acl::FP16_MAN_LEN;

    bool needRound = false;
    if (eF > 0x8FU) {
      eRet = acl::FP16_MAX_EXP - 1;
      mRet = static_cast<uint16_t>(acl::FP16_MAX_MAN);
    }

    if (eF <= 0x70U) {
      eRet = 0;
      if (eF >= 0x67U) {
        mF = (mF | acl::FP32_MAN_HIDE_BIT);
        constexpr uint16_t shiftOut = static_cast<uint16_t>(acl::FP32_MAN_LEN);
        const uint64_t m_tmp = (static_cast<uint64_t>(mF)) << (eF - 0x67U);
        needRound = acl::IsRoundOne(m_tmp, shiftOut);
        mRet = static_cast<uint16_t>(m_tmp >> static_cast<uint64_t>(shiftOut));
        if (needRound) {
          mRet++;
        }
      } else if ((eF == 0x66U) && (mF > 0U)) {
        mRet = 1U;
      } else {
        mRet = 0U;
      }
    }

    if ((eF <= 0x8FU) && (eF > 0x70U)) {
      eRet = static_cast<int16_t>(eF - 0x70U);
      needRound = acl::IsRoundOne(static_cast<uint64_t>(mF), static_cast<uint16_t>(mLenDelta));
      mRet = static_cast<uint16_t>(mF >> mLenDelta);
      if (needRound) {
        mRet++;
      }
      if ((mRet & static_cast<uint16_t>(acl::FP16_MAN_HIDE_BIT)) != 0) {
        eRet++;
      }
    }
    acl::Fp16Normalize(eRet, mRet);
    val_ = static_cast<uint16_t>(((sRet) << static_cast<uint16_t>(acl::FP16_SIGN_INDEX)) |
                                 ((static_cast<uint16_t>(eRet)) << acl::FP16_MAN_LEN) |
                                 ((mRet) & static_cast<uint16_t>(acl::FP16_MAX_MAN)));
    return *this;
  }

  uint16_t GetVal() const {
    return val_;
  }

 private:
  uint16_t val_;
};
}  // namespace acl

namespace {
aclError VerifyIndexOm2(const std::shared_ptr<gert::Om2ModelExecutor> &executor, size_t index) {
  const std::vector<ge::Om2TensorDesc> *input_desc = nullptr;
  const std::vector<ge::Om2TensorDesc> *output_desc = nullptr;
  const auto ret = executor->GetModelDescInfo(input_desc, output_desc);
  if (ret != ge::SUCCESS || input_desc == nullptr) {
    ACL_LOG_INNER_ERROR("[Get][ModelDesc]GetModelDescInfo failed, ret[%u]", ret);
    return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
  }
  if (index >= input_desc->size()) {
    ACL_LOG_ERROR("[Check][Index]index[%zu] is invalid, input tensor count is %zu", index, input_desc->size());
    return ACL_ERROR_INVALID_PARAM;
  }
  return ACL_SUCCESS;
}

static aclError CheckAippDataIndexOm2(const std::shared_ptr<gert::Om2ModelExecutor> &executor, const uint32_t modelId,
                                      const size_t idx) {
  ACL_LOG_INFO("call ge interface executor.GetAippType, modelId[%u]", modelId);
  ge::InputAippType type;
  size_t aippIndex = 0U;
  const ge::Status ret = executor->GetAippType(static_cast<uint32_t>(idx), type, aippIndex);
  if (ret != ge::SUCCESS) {
    ACL_LOG_CALL_ERROR("[Get][AippType]Get aipp type failed, ge result[%u]", ret);
    return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
  }
  if (type == ge::DYNAMIC_AIPP_NODE) {
    ACL_LOG_INFO("Index [%zu] entered by the user is dynamic aipp data", idx);
    return ACL_SUCCESS;
  } else if (type == ge::DATA_WITHOUT_AIPP) {
    // maybe this is old om when getaipptype interface is unsupported, ensure compatibility
    const std::vector<ge::Om2TensorDesc> *input_desc = nullptr;
    const std::vector<ge::Om2TensorDesc> *output_desc = nullptr;
    const auto desc_ret = executor->GetModelDescInfo(input_desc, output_desc);
    if (desc_ret != ge::SUCCESS || input_desc == nullptr) {
      ACL_LOG_INNER_ERROR("[Get][ModelDesc]GetModelDescInfo failed, ret[%u]", desc_ret);
      return ACL_ERROR_INVALID_PARAM;
    }
    size_t index_in_model = 0U;
    bool found = false;
    for (size_t i = 0U; i < input_desc->size(); ++i) {
      if ((*input_desc)[i].GetName() == "ascend_dynamic_aipp_data") {
        index_in_model = i;
        found = true;
        break;
      }
    }
    if (!found) {
      ACL_LOG_INNER_ERROR("[Get][InputIndex]the model is not a dynamic aipp model, there is no dynamic aipp node");
      return ACL_ERROR_INVALID_PARAM;
    }
    if (index_in_model != idx) {
      ACL_LOG_INNER_ERROR("[Check][indexInModel]index[%zu] entered by the user is not dynamic aipp index[%zu]", idx,
                          index_in_model);
      return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
  } else {
    ACL_LOG_INNER_ERROR("[Check][Index]index[%zu] entered by the user is not dynamic aipp data index.", idx);
    return ACL_ERROR_INVALID_PARAM;
  }
}

static aclError GetModelOriDimsOm2(const std::shared_ptr<gert::Om2ModelExecutor> &executor, const uint32_t model_id,
                                   const uint32_t related_input_rank, bool &is_get_dim, int64_t &mdl_ori_h,
                                   int64_t &mdl_ori_w, int64_t &mdl_ori_n) {
  // get model origin input info
  ge::OriginInputInfo input_info;
  auto ret = executor->GetOrigInputInfo(related_input_rank, input_info);
  if (ret != ge::SUCCESS) {
    ACL_LOG_CALL_ERROR("[Get][OrigInputInfo]GetOrigInputInfo failed, modelId[%u], index[%u], ret[%u]", model_id,
                       related_input_rank, ret);
    return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
  }
  const aclFormat src_format = static_cast<aclFormat>(input_info.format);

  // get model origin input dims
  std::vector<ge::InputOutputDims> input_dims;
  std::vector<ge::InputOutputDims> output_dims;
  ret = executor->GetAllAippInputOutputDims(related_input_rank, input_dims, output_dims);
  if (ret != ge::SUCCESS) {
    ACL_LOG_CALL_ERROR("[Get][AllAippInputOutputDims]GetAllAippInputOutputDims failed, modelId[%u], index[%u], ret[%u]",
                       model_id, related_input_rank, ret);
    return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
  }

  // Parse NHW from input Dims when inputDims is not empty
  if (input_dims.empty()) {
    ACL_LOG_INNER_ERROR("[Check][InputDims]get model origin input dims fail, origin input dims is empty");
    return ACL_ERROR_GE_FAILURE;
  }
  // Get the index of the maximum gear
  const size_t max_shape_index = acl::GetMaxShapeIndex(input_dims);
  aclmdlIODims src_dims;
  const aclError io_ret = acl::SetIODims(input_dims[max_shape_index], src_dims);
  if (io_ret != ACL_SUCCESS) {
    ACL_LOG_INNER_ERROR("[Set][IODims]srcDims SetIODims failed, modelId[%u], result[%d]", model_id, io_ret);
    return io_ret;
  }
  switch (src_format) {
    case ACL_FORMAT_NCHW:
      if (src_dims.dimCount == 4U) {
        mdl_ori_h = src_dims.dims[2];
        mdl_ori_w = src_dims.dims[3];
        mdl_ori_n = src_dims.dims[0];
      }
      is_get_dim = true;
      break;
    case ACL_FORMAT_NHWC:
      if (src_dims.dimCount == 4U) {
        mdl_ori_h = src_dims.dims[1];
        mdl_ori_w = src_dims.dims[2];
        mdl_ori_n = src_dims.dims[0];
      }
      is_get_dim = true;
      break;
    default:
      ACL_LOG_INFO("the model origin format[%d] is invalid, only support ACL_FORMAT_NHWC or ACL_FORMAT_NCHW",
                   static_cast<int32_t>(src_format));
      is_get_dim = false;
      break;
  }
  return ACL_SUCCESS;
}

static aclError GetAndCheckAippOutputShapeOm2(const std::shared_ptr<gert::Om2ModelExecutor> &executor,
                                              const uint32_t model_id, const uint32_t idx,
                                              const aclmdlAIPP *const aipp_parms_set) {
  const int64_t batch_size = static_cast<int64_t>(aipp_parms_set->batchSize);

  const std::vector<ge::Om2TensorDesc> *input_desc = nullptr;
  const std::vector<ge::Om2TensorDesc> *output_desc = nullptr;
  const auto desc_ret = executor->GetModelDescInfo(input_desc, output_desc);
  if (desc_ret != ge::SUCCESS || input_desc == nullptr) {
    ACL_LOG_INNER_ERROR("[Get][ModelDesc]GetModelDescInfo failed, ret[%u]", desc_ret);
    return ACL_ERROR_INVALID_PARAM;
  }
  if (idx >= input_desc->size()) {
    ACL_LOG_INNER_ERROR("[Check][Params]index[%u] cannot greater than or equal to tensor size[%zu]", idx,
                        input_desc->size());
    return ACL_ERROR_INVALID_PARAM;
  }
  const auto &shape_ranges = (*input_desc)[idx].GetShapeRange();
  if (!shape_ranges.empty()) {
    ACL_LOG_INFO("check aipp parameters of dynamic shape model[%u]", model_id);
    return ACL_SUCCESS;
  }

  ACL_LOG_INFO("check aipp parameters of static shape model[%u]", model_id);
  int32_t aipp_output_w = 0;
  int32_t aipp_output_h = 0;
  const aclError hw_ret = acl::GetAippOutputHW(aipp_parms_set, 0U, acl::GetNpuArch(), aipp_output_w, aipp_output_h);
  if (hw_ret != ACL_SUCCESS) {
    return hw_ret;
  }

  bool is_get_dim = false;
  int64_t mdl_ori_h = 0;
  int64_t mdl_ori_w = 0;
  int64_t mdl_ori_n = 0;
  const aclError mdl_ret = GetModelOriDimsOm2(executor, model_id, idx, is_get_dim, mdl_ori_h, mdl_ori_w, mdl_ori_n);
  if (mdl_ret != ACL_SUCCESS) {
    ACL_LOG_INNER_ERROR("[Get][ModelOriDims]get model original dims fail");
    return mdl_ret;
  }

  if (is_get_dim) {
    ACL_LOG_INFO("relatedInputRank[%u], mdlOriH[%ld], mdlOriW[%ld], mdlOriN[%ld]", idx, mdl_ori_h, mdl_ori_w,
                 mdl_ori_n);
    // check batchSize
    if ((batch_size != mdl_ori_n) || (aipp_output_w != mdl_ori_w) || (aipp_output_h != mdl_ori_h)) {
      ACL_LOG_ERROR(
          "[Check][Params]aipp output shape set by ACL must be equal to aipp output shape in the model! "
          "AclAippBatchSize = %ld, AclAippOutputW = %d, AclAippOutputH = %d, "
          "ModelAippBatchSize = %ld, ModelAippOutputW = %ld, ModelAippOutputH = %ld.",
          batch_size, aipp_output_w, aipp_output_h, mdl_ori_n, mdl_ori_w, mdl_ori_h);
      const std::string errMsg = acl::AclErrorLogManager::FormatStr(
          "aipp output shape set by ACL "
          "must be equal to aipp output shape in the model! AclAippBatchSize = %ld, AclAippOutputW = %d, "
          "AclAippOutputH = %d, ModelAippBatchSize = %ld, ModelAippOutputW = %ld, ModelAippOutputH = %ld.",
          batch_size, aipp_output_w, aipp_output_h, mdl_ori_n, mdl_ori_w, mdl_ori_h);
      acl::AclErrorLogManager::ReportInputError(acl::INVALID_AIPP_MSG, std::vector<const char *>({"param", "reason"}),
                                                std::vector<const char *>({"dynamic aipp shape", errMsg.c_str()}));
      return ACL_ERROR_INVALID_PARAM;
    }
  } else {
    ACL_LOG_INFO("cant not get model H W N, current used model is old");
  }
  return ACL_SUCCESS;
}

static aclError GetAndCheckAippParamsOm2(const std::shared_ptr<gert::Om2ModelExecutor> &executor,
                                         const uint32_t modelId, const size_t idx,
                                         const aclmdlAIPP *const aippParmsSet) {
  // check dynamic aipp parameters
  ge::AippConfigInfo aippParams;
  const auto aipp_ret = executor->GetAippInfo(static_cast<uint32_t>(idx), aippParams);
  if (aipp_ret == ge::SUCCESS) {
    const uint32_t relatedInputRank = aippParams.related_input_rank;
    const uint64_t maxSrcImageSize = static_cast<uint64_t>(aippParams.max_src_image_size);
    // check max_src_image_size
    const uint64_t size = acl::GetSrcImageSize(aippParmsSet);
    ACL_LOG_INFO("Input SrcImageSize = %lu", size);
    if (size > maxSrcImageSize) {
      ACL_LOG_ERROR("[Check][Size]the dynamic aipp size[%lu] is bigger than max_src_image_size[%lu]", size,
                    maxSrcImageSize);
      const std::string errMsg =
          acl::AclErrorLogManager::FormatStr("bigger than max_src_image_size[%lu]", maxSrcImageSize);
      acl::AclErrorLogManager::ReportInputError(acl::INVALID_AIPP_MSG, std::vector<const char *>({"param", "reason"}),
                                                std::vector<const char *>({"dynamic aipp size", errMsg.c_str()}));
      return ACL_ERROR_INVALID_PARAM;
    }
    const aclError ret =
        GetAndCheckAippOutputShapeOm2(executor, modelId, static_cast<uint32_t>(relatedInputRank), aippParmsSet);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  } else {
    ACL_LOG_INFO("current used model is old");
  }
  return acl::AippParamsCheck(aippParmsSet, acl::GetNpuArch());
}
}  // namespace

aclmdlAIPP *aclmdlCreateAIPPImplOm2(uint64_t batchSize) {
  aclmdlAIPP *aippParmsSet = nullptr;
  try {
    ACL_LOG_INFO("start to execute aclmdlCreateAIPP, batchSize[%lu]", batchSize);
    if (batchSize == 0U) {
      ACL_LOG_INNER_ERROR("[Check][BatchSize]the batchSize can't be zero");
      return nullptr;
    }
    aippParmsSet = new (std::nothrow) aclmdlAIPP();
    if (aippParmsSet == nullptr) {
      ACL_LOG_INNER_ERROR("[Check][ParmsSet]new aclmdlAIPP fail");
      return nullptr;
    }

    aippParmsSet->batchSize = batchSize;
    aippParmsSet->aippParms.batchNum = static_cast<int8_t>(batchSize);
    aippParmsSet->aippBatchPara.resize(batchSize);
    ACL_LOG_INFO("the size of aippBatchPara is [%zu]", aippParmsSet->aippBatchPara.size());
    for (uint64_t i = 0U; i < batchSize; i++) {
      aippParmsSet->aippBatchPara[i].dtcPixelVarReciChn0 = acl::Fp16Type(1.0F).GetVal();
      aippParmsSet->aippBatchPara[i].dtcPixelVarReciChn1 = acl::Fp16Type(1.0F).GetVal();
      aippParmsSet->aippBatchPara[i].dtcPixelVarReciChn2 = acl::Fp16Type(1.0F).GetVal();
      aippParmsSet->aippBatchPara[i].dtcPixelVarReciChn3 = acl::Fp16Type(1.0F).GetVal();
    }
    return aippParmsSet;
  } catch (std::bad_alloc &) {
    ACL_LOG_INNER_ERROR("[Create][AIPP]aclmdlCreateAIPP fail, bad memory allocation, batchSize[%lu]", batchSize);
  } catch (std::length_error &) {
    ACL_LOG_INNER_ERROR("[Create][AIPP]aclmdlCreateAIPP fail with length error, batchSize[%lu]", batchSize);
  }

  ACL_DELETE(aippParmsSet);
  return nullptr;
}

aclError aclmdlDestroyAIPPImplOm2(const aclmdlAIPP *aippParmsSet) {
  ACL_LOG_INFO("start to execute aclmdlDestroyAIPP.");
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippParmsSet);
  ACL_DELETE_AND_SET_NULL(aippParmsSet);
  return ACL_SUCCESS;
}

aclError aclmdlGetAippDataSizeImplOm2(uint64_t batchSize, size_t *size) {
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(size);
  ACL_REQUIRES_POSITIVE_WITH_INPUT_REPORT(batchSize);
  ACL_LOG_INFO("start to execute aclmdlGetAippDataSize, batchSize is %lu", batchSize);
  constexpr size_t realAippParamSize = sizeof(kAippDynamicPara) - sizeof(kAippDynamicBatchPara);
  size_t retTmp = 0U;
  ACL_CHECK_ASSIGN_SIZET_MULTI(batchSize, sizeof(kAippDynamicBatchPara), retTmp);
  ACL_CHECK_ASSIGN_SIZET_ADD(retTmp, realAippParamSize, *size);
  ACL_LOG_INFO("end to execute aclmdlGetAippDataSize, batchSize is %lu, size is %zu", batchSize, *size);
  return ACL_SUCCESS;
}

aclError aclmdlSetAIPPInputFormatImplOm2(aclmdlAIPP *aippParmsSet, aclAippInputFormat inputFormat) {
  ACL_LOG_INFO("start to execute aclmdlSetAIPPInputFormat");
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippParmsSet);
  const std::map<aclAippInputFormat, acl::CceAippInputFormat> inputFormatMap = {
      {ACL_YUV420SP_U8, acl::CCE_YUV420SP_U8},
      {ACL_XRGB8888_U8, acl::CCE_XRGB8888_U8},
      {ACL_RGB888_U8, acl::CCE_RGB888_U8},
      {ACL_YUV400_U8, acl::CCE_YUV400_U8},
      {ACL_NC1HWC0DI_FP16, acl::CCE_NC1HWC0DI_FP16},
      {ACL_NC1HWC0DI_S8, acl::CCE_NC1HWC0DI_S8},
      {ACL_ARGB8888_U8, acl::CCE_ARGB8888_U8},
      {ACL_YUYV_U8, acl::CCE_YUYV_U8},
      {ACL_YUV422SP_U8, acl::CCE_YUV422SP_U8},
      {ACL_AYUV444_U8, acl::CCE_AYUV444_U8},
      {ACL_RAW10, acl::CCE_RAW10},
      {ACL_RAW12, acl::CCE_RAW12},
      {ACL_RAW16, acl::CCE_RAW16},
      {ACL_RAW24, acl::CCE_RAW24}};

  auto it = inputFormatMap.find(inputFormat);
  if (it == inputFormatMap.end()) {
    ACL_LOG_INNER_ERROR("[Unsupported][Format]unsupported inputFormat[%d]", static_cast<int32_t>(inputFormat));
    return ACL_ERROR_INVALID_PARAM;
  }

  aippParmsSet->aippParms.inputFormat = static_cast<uint8_t>(it->second);
  return ACL_SUCCESS;
}

aclError aclmdlSetAIPPCscParamsImplOm2(aclmdlAIPP *aippParmsSet, int8_t cscSwitch, int16_t cscMatrixR0C0,
                                       int16_t cscMatrixR0C1, int16_t cscMatrixR0C2, int16_t cscMatrixR1C0,
                                       int16_t cscMatrixR1C1, int16_t cscMatrixR1C2, int16_t cscMatrixR2C0,
                                       int16_t cscMatrixR2C1, int16_t cscMatrixR2C2, uint8_t cscOutputBiasR0,
                                       uint8_t cscOutputBiasR1, uint8_t cscOutputBiasR2, uint8_t cscInputBiasR0,
                                       uint8_t cscInputBiasR1, uint8_t cscInputBiasR2) {
  ACL_LOG_INFO("start to execute aclmdlSetAIPPCscParams");
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippParmsSet);
  ACL_CHECK_RANGE_INT(cscSwitch, acl::AIPP_SWITCH_OFF, acl::AIPP_SWITCH_ON);
  aippParmsSet->aippParms.cscSwitch = cscSwitch;
  if (cscSwitch == 0) {
    ACL_LOG_INFO("cscSwitch[%d] is off", static_cast<int32_t>(cscSwitch));
    return ACL_SUCCESS;
  }
  ACL_CHECK_RANGE_INT(cscMatrixR0C0, acl::CSC_MATRIX_MIN, acl::CSC_MATRIX_MAX);
  ACL_CHECK_RANGE_INT(cscMatrixR0C1, acl::CSC_MATRIX_MIN, acl::CSC_MATRIX_MAX);
  ACL_CHECK_RANGE_INT(cscMatrixR0C2, acl::CSC_MATRIX_MIN, acl::CSC_MATRIX_MAX);

  ACL_CHECK_RANGE_INT(cscMatrixR1C0, acl::CSC_MATRIX_MIN, acl::CSC_MATRIX_MAX);
  ACL_CHECK_RANGE_INT(cscMatrixR1C1, acl::CSC_MATRIX_MIN, acl::CSC_MATRIX_MAX);
  ACL_CHECK_RANGE_INT(cscMatrixR1C2, acl::CSC_MATRIX_MIN, acl::CSC_MATRIX_MAX);

  ACL_CHECK_RANGE_INT(cscMatrixR2C0, acl::CSC_MATRIX_MIN, acl::CSC_MATRIX_MAX);
  ACL_CHECK_RANGE_INT(cscMatrixR2C1, acl::CSC_MATRIX_MIN, acl::CSC_MATRIX_MAX);
  ACL_CHECK_RANGE_INT(cscMatrixR2C2, acl::CSC_MATRIX_MIN, acl::CSC_MATRIX_MAX);

  aippParmsSet->aippParms.cscMatrixR0C0 = cscMatrixR0C0;
  aippParmsSet->aippParms.cscMatrixR0C1 = cscMatrixR0C1;
  aippParmsSet->aippParms.cscMatrixR0C2 = cscMatrixR0C2;
  aippParmsSet->aippParms.cscMatrixR1C0 = cscMatrixR1C0;
  aippParmsSet->aippParms.cscMatrixR1C1 = cscMatrixR1C1;
  aippParmsSet->aippParms.cscMatrixR1C2 = cscMatrixR1C2;
  aippParmsSet->aippParms.cscMatrixR2C0 = cscMatrixR2C0;
  aippParmsSet->aippParms.cscMatrixR2C1 = cscMatrixR2C1;
  aippParmsSet->aippParms.cscMatrixR2C2 = cscMatrixR2C2;
  aippParmsSet->aippParms.cscOutputBiasR0 = cscOutputBiasR0;
  aippParmsSet->aippParms.cscOutputBiasR1 = cscOutputBiasR1;
  aippParmsSet->aippParms.cscOutputBiasR2 = cscOutputBiasR2;
  aippParmsSet->aippParms.cscInputBiasR0 = cscInputBiasR0;
  aippParmsSet->aippParms.cscInputBiasR1 = cscInputBiasR1;
  aippParmsSet->aippParms.cscInputBiasR2 = cscInputBiasR2;
  return ACL_SUCCESS;
}

aclError aclmdlSetAIPPRbuvSwapSwitchImplOm2(aclmdlAIPP *aippParmsSet, int8_t rbuvSwapSwitch) {
  ACL_LOG_INFO("start to execute aclmdlSetAIPPRbuvSwapSwitch");
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippParmsSet);
  ACL_CHECK_RANGE_INT(rbuvSwapSwitch, acl::AIPP_SWITCH_OFF, acl::AIPP_SWITCH_ON);
  aippParmsSet->aippParms.rbuvSwapSwitch = rbuvSwapSwitch;
  ACL_LOG_INFO("successfully execute aclmdlSetAIPPRbuvSwapSwitch");
  return ACL_SUCCESS;
}

aclError aclmdlSetAIPPAxSwapSwitchImplOm2(aclmdlAIPP *aippParmsSet, int8_t axSwapSwitch) {
  ACL_LOG_INFO("start to execute aclmdlSetAIPPAxSwapSwitch");
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippParmsSet);
  ACL_CHECK_RANGE_INT(axSwapSwitch, acl::AIPP_SWITCH_OFF, acl::AIPP_SWITCH_ON);
  aippParmsSet->aippParms.axSwapSwitch = axSwapSwitch;
  return ACL_SUCCESS;
}

aclError aclmdlSetAIPPSrcImageSizeImplOm2(aclmdlAIPP *aippParmsSet, int32_t srcImageSizeW, int32_t srcImageSizeH) {
  ACL_LOG_INFO("start to execute aclmdlSetAIPPSrcImageSize");
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippParmsSet);
  ACL_CHECK_RANGE_INT(srcImageSizeW, acl::IMAGE_SIZE_MIN + 1, acl::IMAGE_SIZE_MAX);
  ACL_CHECK_RANGE_INT(srcImageSizeH, acl::IMAGE_SIZE_MIN, acl::IMAGE_SIZE_MAX);
  aippParmsSet->aippParms.srcImageSizeW = srcImageSizeW;
  aippParmsSet->aippParms.srcImageSizeH = srcImageSizeH;
  ACL_LOG_INFO("successfully execute aclmdlSetAIPPSrcImageSize.");
  return ACL_SUCCESS;
}

aclError aclmdlSetAIPPScfParamsImplOm2(aclmdlAIPP *aippParmsSet, int8_t scfSwitch, int32_t scfInputSizeW,
                                       int32_t scfInputSizeH, int32_t scfOutputSizeW, int32_t scfOutputSizeH,
                                       uint64_t batchIndex) {
  ACL_LOG_INFO("start to execute aclmdlSetAIPPScfParams");
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippParmsSet);
  const uint64_t aippBatchParaSize = static_cast<uint64_t>(aippParmsSet->aippBatchPara.size());
  if (batchIndex >= aippBatchParaSize) {
    ACL_LOG_ERROR(
        "[Check][Param]Set batch parameter Failed, batch_index (%lu) is greater than or "
        "equal to batch_number (%lu)",
        batchIndex, aippBatchParaSize);
    const std::string errMsg = acl::AclErrorLogManager::FormatStr(
        "batch_index (%lu) is greater than or "
        "equal to batch_number (%lu)",
        batchIndex, aippBatchParaSize);
    acl::AclErrorLogManager::ReportInputError(acl::INVALID_AIPP_MSG, std::vector<const char *>({"param", "reason"}),
                                              std::vector<const char *>({"batch_index", errMsg.c_str()}));
    return ACL_ERROR_INVALID_PARAM;
  }
  ACL_CHECK_RANGE_INT(scfSwitch, acl::AIPP_SWITCH_OFF, acl::AIPP_SWITCH_ON);
  aippParmsSet->aippBatchPara[batchIndex].scfSwitch = scfSwitch;
  if (scfSwitch == 0) {
    ACL_LOG_INFO("scfSwitch[%d] is off", static_cast<int32_t>(scfSwitch));
    return ACL_SUCCESS;
  }
  ACL_CHECK_RANGE_INT(scfInputSizeW, acl::SCF_SIZE_MIN, acl::IMAGE_SIZE_MAX);
  ACL_CHECK_RANGE_INT(scfInputSizeH, acl::SCF_SIZE_MIN, acl::IMAGE_SIZE_MAX);
  ACL_CHECK_RANGE_INT(scfOutputSizeW, acl::SCF_SIZE_MIN, acl::SCF_SIZEW_MAX);
  ACL_CHECK_RANGE_INT(scfOutputSizeH, acl::SCF_SIZE_MIN, acl::IMAGE_SIZE_MAX);
  aippParmsSet->aippBatchPara[batchIndex].scfInputSizeW = scfInputSizeW;
  aippParmsSet->aippBatchPara[batchIndex].scfInputSizeH = scfInputSizeH;
  aippParmsSet->aippBatchPara[batchIndex].scfOutputSizeW = scfOutputSizeW;
  aippParmsSet->aippBatchPara[batchIndex].scfOutputSizeH = scfOutputSizeH;

  ACL_LOG_INFO("successfully execute aclmdlSetAIPPScfParams");
  return ACL_SUCCESS;
}

aclError aclmdlSetAIPPCropParamsImplOm2(aclmdlAIPP *aippParmsSet, int8_t cropSwitch, int32_t cropStartPosW,
                                        int32_t cropStartPosH, int32_t cropSizeW, int32_t cropSizeH,
                                        uint64_t batchIndex) {
  ACL_LOG_INFO("start to execute aclmdlSetAIPPCropParams");
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippParmsSet);
  const uint64_t aippBatchParaSize = static_cast<uint64_t>(aippParmsSet->aippBatchPara.size());
  if (batchIndex >= aippBatchParaSize) {
    ACL_LOG_ERROR(
        "[Check][Param]Set batch parameter Failed, batch_index (%lu) is greater than or "
        "equal to batch_number (%lu)",
        batchIndex, aippBatchParaSize);
    const std::string errMsg = acl::AclErrorLogManager::FormatStr(
        "batch_index (%lu) is greater than or "
        "equal to batch_number (%lu)",
        batchIndex, aippBatchParaSize);
    acl::AclErrorLogManager::ReportInputError(acl::INVALID_AIPP_MSG, std::vector<const char *>({"param", "reason"}),
                                              std::vector<const char *>({"batch_index", errMsg.c_str()}));
    return ACL_ERROR_INVALID_PARAM;
  }
  ACL_CHECK_RANGE_INT(cropSwitch, acl::AIPP_SWITCH_OFF, acl::AIPP_SWITCH_ON);
  aippParmsSet->aippBatchPara[batchIndex].cropSwitch = cropSwitch;
  if (cropSwitch == 0) {
    ACL_LOG_INFO("cropSwitch[%d] is off", static_cast<int32_t>(cropSwitch));
    return ACL_SUCCESS;
  }
  ACL_CHECK_RANGE_INT(cropStartPosW, acl::IMAGE_SIZE_MIN - 1, acl::IMAGE_SIZE_MAX - 1);
  ACL_CHECK_RANGE_INT(cropStartPosH, acl::IMAGE_SIZE_MIN - 1, acl::IMAGE_SIZE_MAX - 1);
  ACL_CHECK_RANGE_INT(cropSizeW, acl::IMAGE_SIZE_MIN, acl::IMAGE_SIZE_MAX);
  ACL_CHECK_RANGE_INT(cropSizeH, acl::IMAGE_SIZE_MIN, acl::IMAGE_SIZE_MAX);
  aippParmsSet->aippBatchPara[batchIndex].cropStartPosW = cropStartPosW;
  aippParmsSet->aippBatchPara[batchIndex].cropStartPosH = cropStartPosH;
  aippParmsSet->aippBatchPara[batchIndex].cropSizeW = cropSizeW;
  aippParmsSet->aippBatchPara[batchIndex].cropSizeH = cropSizeH;

  ACL_LOG_INFO("successfully execute aclmdlSetAIPPCropParams");
  return ACL_SUCCESS;
}

aclError aclmdlSetAIPPPaddingParamsImplOm2(aclmdlAIPP *aippParmsSet, int8_t paddingSwitch, int32_t paddingSizeTop,
                                           int32_t paddingSizeBottom, int32_t paddingSizeLeft, int32_t paddingSizeRight,
                                           uint64_t batchIndex) {
  ACL_LOG_INFO("start to execute aclmdlSetAIPPPaddingParams");
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippParmsSet);
  const uint64_t aippBatchParaSize = static_cast<uint64_t>(aippParmsSet->aippBatchPara.size());
  if (batchIndex >= aippBatchParaSize) {
    ACL_LOG_ERROR(
        "[Check][Param]Set batch parameter Failed, batch_index (%lu) is greater "
        "than or equal to batch_number (%lu)",
        batchIndex, aippBatchParaSize);
    const std::string errMsg = acl::AclErrorLogManager::FormatStr(
        "batch_index (%lu) is greater "
        "than or equal to batch_number (%lu)",
        batchIndex, aippBatchParaSize);
    acl::AclErrorLogManager::ReportInputError(acl::INVALID_AIPP_MSG, std::vector<const char *>({"param", "reason"}),
                                              std::vector<const char *>({"batch_index", errMsg.c_str()}));
    return ACL_ERROR_INVALID_PARAM;
  }
  ACL_CHECK_RANGE_INT(paddingSwitch, acl::AIPP_SWITCH_OFF, acl::AIPP_SWITCH_ON);
  aippParmsSet->aippBatchPara[batchIndex].paddingSwitch = paddingSwitch;
  if (paddingSwitch == 0) {
    ACL_LOG_INFO("paddingSwitch[%d] is off", static_cast<int32_t>(paddingSwitch));
    return ACL_SUCCESS;
  }
  ACL_CHECK_RANGE_INT(paddingSizeTop, acl::PADDING_MIN, acl::PADDING_MAX);
  ACL_CHECK_RANGE_INT(paddingSizeBottom, acl::PADDING_MIN, acl::PADDING_MAX);
  ACL_CHECK_RANGE_INT(paddingSizeLeft, acl::PADDING_MIN, acl::PADDING_MAX);
  ACL_CHECK_RANGE_INT(paddingSizeRight, acl::PADDING_MIN, acl::PADDING_MAX);
  aippParmsSet->aippBatchPara[batchIndex].paddingSizeTop = paddingSizeTop;
  aippParmsSet->aippBatchPara[batchIndex].paddingSizeBottom = paddingSizeBottom;
  aippParmsSet->aippBatchPara[batchIndex].paddingSizeLeft = paddingSizeLeft;
  aippParmsSet->aippBatchPara[batchIndex].paddingSizeRight = paddingSizeRight;

  ACL_LOG_INFO("successfully execute aclmdlSetAIPPPaddingParams");
  return ACL_SUCCESS;
}

aclError aclmdlSetAIPPDtcPixelMeanImplOm2(aclmdlAIPP *aippParmsSet, int16_t dtcPixelMeanChn0, int16_t dtcPixelMeanChn1,
                                          int16_t dtcPixelMeanChn2, int16_t dtcPixelMeanChn3, uint64_t batchIndex) {
  ACL_LOG_INFO("start to execute aclmdlSetAIPPDtcPixelMean");
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippParmsSet);
  const uint64_t aippBatchParaSize = static_cast<uint64_t>(aippParmsSet->aippBatchPara.size());
  if (batchIndex >= aippBatchParaSize) {
    ACL_LOG_ERROR(
        "[Check][Param]Set batch parameter Failed, batch_index (%lu) is greater than or "
        "equal to batch_number (%lu)",
        batchIndex, aippBatchParaSize);
    acl::AclErrorLogManager::ReportInputError(
        acl::INVALID_AIPP_MSG, std::vector<const char *>({"param", "reason"}),
        std::vector<const char *>({"InputAIPP", "parameters verification failed"}));
    return ACL_ERROR_INVALID_PARAM;
  }
  ACL_CHECK_RANGE_INT(dtcPixelMeanChn0, acl::MEAN_CHN_MIN, acl::MEAN_CHN_MAX);
  ACL_CHECK_RANGE_INT(dtcPixelMeanChn1, acl::MEAN_CHN_MIN, acl::MEAN_CHN_MAX);
  ACL_CHECK_RANGE_INT(dtcPixelMeanChn2, acl::MEAN_CHN_MIN, acl::MEAN_CHN_MAX);
  ACL_CHECK_RANGE_INT(dtcPixelMeanChn3, acl::MEAN_CHN_MIN, acl::MEAN_CHN_MAX);
  aippParmsSet->aippBatchPara[batchIndex].dtcPixelMeanChn0 = dtcPixelMeanChn0;
  aippParmsSet->aippBatchPara[batchIndex].dtcPixelMeanChn1 = dtcPixelMeanChn1;
  aippParmsSet->aippBatchPara[batchIndex].dtcPixelMeanChn2 = dtcPixelMeanChn2;
  aippParmsSet->aippBatchPara[batchIndex].dtcPixelMeanChn3 = dtcPixelMeanChn3;

  ACL_LOG_INFO("successfully execute aclmdlSetAIPPDtcPixelMean");
  return ACL_SUCCESS;
}

aclError aclmdlSetAIPPDtcPixelMinImplOm2(aclmdlAIPP *aippParmsSet, float dtcPixelMinChn0, float dtcPixelMinChn1,
                                         float dtcPixelMinChn2, float dtcPixelMinChn3, uint64_t batchIndex) {
  ACL_LOG_INFO("start to execute aclmdlSetAIPPDtcPixelMin");
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippParmsSet);
  const uint64_t aippBatchParaSize = static_cast<uint64_t>(aippParmsSet->aippBatchPara.size());
  if (batchIndex >= aippBatchParaSize) {
    ACL_LOG_ERROR(
        "[Check][Param]Set batch parameter Failed, batch_index (%lu) is greater than or "
        "equal to batch_number (%lu)",
        batchIndex, aippBatchParaSize);
    const std::string errMsg = acl::AclErrorLogManager::FormatStr(
        "batch_index (%lu) is greater than or "
        "equal to batch_number (%lu)",
        batchIndex, aippBatchParaSize);
    acl::AclErrorLogManager::ReportInputError(acl::INVALID_AIPP_MSG, std::vector<const char *>({"param", "reason"}),
                                              std::vector<const char *>({"batch_index", errMsg.c_str()}));
    return ACL_ERROR_INVALID_PARAM;
  }
  ACL_CHECK_RANGE_FLOAT(dtcPixelMinChn0, acl::MIN_CHN_MIN, acl::MIN_CHN_MAX);
  ACL_CHECK_RANGE_FLOAT(dtcPixelMinChn1, acl::MIN_CHN_MIN, acl::MIN_CHN_MAX);
  ACL_CHECK_RANGE_FLOAT(dtcPixelMinChn2, acl::MIN_CHN_MIN, acl::MIN_CHN_MAX);
  ACL_CHECK_RANGE_FLOAT(dtcPixelMinChn3, acl::MIN_CHN_MIN, acl::MIN_CHN_MAX);
  aippParmsSet->aippBatchPara[batchIndex].dtcPixelMinChn0 = acl::Fp16Type(dtcPixelMinChn0).GetVal();
  aippParmsSet->aippBatchPara[batchIndex].dtcPixelMinChn1 = acl::Fp16Type(dtcPixelMinChn1).GetVal();
  aippParmsSet->aippBatchPara[batchIndex].dtcPixelMinChn2 = acl::Fp16Type(dtcPixelMinChn2).GetVal();
  aippParmsSet->aippBatchPara[batchIndex].dtcPixelMinChn3 = acl::Fp16Type(dtcPixelMinChn3).GetVal();
  ACL_LOG_INFO(
      "successfully execute aclmdlSetAIPPDtcPixelMin, input dtcPixelMinChn0:%f,"
      "the dtcPixelMinChn0:%u of batchIndex:%lu",
      static_cast<float64_t>(dtcPixelMinChn0),
      static_cast<uint32_t>(aippParmsSet->aippBatchPara[batchIndex].dtcPixelMinChn0), batchIndex);
  return ACL_SUCCESS;
}

aclError aclmdlSetAIPPPixelVarReciImplOm2(aclmdlAIPP *aippParmsSet, float dtcPixelVarReciChn0,
                                          float dtcPixelVarReciChn1, float dtcPixelVarReciChn2,
                                          float dtcPixelVarReciChn3, uint64_t batchIndex) {
  ACL_LOG_INFO("start to execute aclmdlSetAIPPPixelVarReci");
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippParmsSet);
  const uint64_t aippBatchParaSize = static_cast<uint64_t>(aippParmsSet->aippBatchPara.size());
  if (batchIndex >= aippBatchParaSize) {
    ACL_LOG_ERROR(
        "[Check][Param]Set batch parameter Failed, batch_index (%lu) is greater than or "
        "equal to batch_number (%lu)",
        batchIndex, aippBatchParaSize);
    const std::string errMsg = acl::AclErrorLogManager::FormatStr(
        "batch_index (%lu) is greater than or "
        "equal to batch_number (%lu)",
        batchIndex, aippBatchParaSize);
    acl::AclErrorLogManager::ReportInputError(acl::INVALID_AIPP_MSG, std::vector<const char *>({"param", "reason"}),
                                              std::vector<const char *>({"batch_index", errMsg.c_str()}));
    return ACL_ERROR_INVALID_PARAM;
  }
  ACL_CHECK_RANGE_FLOAT(dtcPixelVarReciChn0, acl::VR_CHN_MIN, acl::VR_CHN_MAX);
  ACL_CHECK_RANGE_FLOAT(dtcPixelVarReciChn1, acl::VR_CHN_MIN, acl::VR_CHN_MAX);
  ACL_CHECK_RANGE_FLOAT(dtcPixelVarReciChn2, acl::VR_CHN_MIN, acl::VR_CHN_MAX);
  ACL_CHECK_RANGE_FLOAT(dtcPixelVarReciChn3, acl::VR_CHN_MIN, acl::VR_CHN_MAX);
  aippParmsSet->aippBatchPara[batchIndex].dtcPixelVarReciChn0 = acl::Fp16Type(dtcPixelVarReciChn0).GetVal();
  aippParmsSet->aippBatchPara[batchIndex].dtcPixelVarReciChn1 = acl::Fp16Type(dtcPixelVarReciChn1).GetVal();
  aippParmsSet->aippBatchPara[batchIndex].dtcPixelVarReciChn2 = acl::Fp16Type(dtcPixelVarReciChn2).GetVal();
  aippParmsSet->aippBatchPara[batchIndex].dtcPixelVarReciChn3 = acl::Fp16Type(dtcPixelVarReciChn3).GetVal();
  ACL_LOG_INFO("input dtcPixelVarReciChn0:%f, the dtcPixelVarReciChn0:%u of batchIndex:%lu",
               static_cast<float64_t>(dtcPixelVarReciChn0),
               static_cast<int32_t>(aippParmsSet->aippBatchPara[batchIndex].dtcPixelVarReciChn0), batchIndex);
  return ACL_SUCCESS;
}

aclError aclmdlGetAippTypeImplOm2(uint32_t modelId, size_t index, aclmdlInputAippType *type,
                                  size_t *dynamicAttachedDataIndex) {
  ACL_LOG_INFO("start to execute aclmdlGetAippType(OM2), modelId[%u], index[%zu]", modelId, index);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dynamicAttachedDataIndex);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(type);
  auto executor = acl::AclResourceManagerOm2::GetInstance().GetOm2Executor(modelId);
  if (executor == nullptr) {
    ACL_LOG_INNER_ERROR("[Check][Executor]Om2 executor not found, modelId[%u]", modelId);
    return ACL_ERROR_INVALID_PARAM;
  }
  ACL_REQUIRES_OK(VerifyIndexOm2(executor, index));
  *dynamicAttachedDataIndex = ACL_INVALID_NODE_INDEX;
  ge::InputAippType aipp_type = ge::DATA_WITHOUT_AIPP;
  const auto ret = executor->GetAippType(static_cast<uint32_t>(index), aipp_type, *dynamicAttachedDataIndex);
  if (ret != ge::SUCCESS) {
    ACL_LOG_CALL_ERROR("[Get][AippType]GetAippType failed, modelId[%u], index[%zu], ret[%u]", modelId, index, ret);
    return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
  }
  *type = static_cast<aclmdlInputAippType>(aipp_type);
  ACL_LOG_INFO("successfully execute aclmdlGetAippType(OM2), modelId[%u], index[%zu], type[%d], dataIdx[%zu]", modelId,
               index, static_cast<int32_t>(*type), *dynamicAttachedDataIndex);
  return ACL_SUCCESS;
}

aclError aclmdlGetFirstAippInfoImplOm2(uint32_t modelId, size_t index, aclAippInfo *aippInfo) {
  ACL_LOG_DEBUG("start to execute aclmdlGetFirstAippInfo(OM2), modelId[%u], index[%zu]", modelId, index);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippInfo);
  auto executor = acl::AclResourceManagerOm2::GetInstance().GetOm2Executor(modelId);
  if (executor == nullptr) {
    ACL_LOG_INNER_ERROR("[Check][Executor]Om2 executor not found, modelId[%u]", modelId);
    return ACL_ERROR_INVALID_PARAM;
  }
  ACL_REQUIRES_OK(VerifyIndexOm2(executor, index));
  ge::AippConfigInfo aippParams;
  auto ret = executor->GetAippInfo(static_cast<uint32_t>(index), aippParams);
  if (ret == ACL_ERROR_GE_AIPP_NOT_EXIST) {
    ACL_LOG_INFO("the tensor index[%zu] is not configured with aipp, modelId[%u], ret[%u]", index, modelId, ret);
    return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
  }
  if (ret != ge::SUCCESS) {
    ACL_LOG_CALL_ERROR("[Get][AippInfo]GetAippInfo failed, modelId[%u], index[%zu], ret[%u]", modelId, index, ret);
    return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
  }
  acl::SetAippInfo(aippInfo, aippParams);

  // Get batch info size for shapeCount, matching OM1 behavior
  size_t shapeCount = 0U;
  ret = executor->GetBatchInfoSize(shapeCount);
  if (ret != ge::SUCCESS) {
    ACL_LOG_CALL_ERROR("[Get][BatchInfo]GetBatchInfoSize failed, modelId[%u], index[%zu], ret[%u]", modelId, index,
                       ret);
    return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
  }

  ACL_LOG_DEBUG("get shapeCount[%zu]", shapeCount);
  aippInfo->shapeCount = shapeCount;

  // Get origin input info, fail on error matching OM1
  ge::OriginInputInfo inputInfo;
  ret = executor->GetOrigInputInfo(static_cast<uint32_t>(index), inputInfo);
  if (ret != ge::SUCCESS) {
    ACL_LOG_CALL_ERROR("[Get][OrigInputInfo]GetOrigInputInfo failed, modelId[%u], index[%zu], ret[%u]", modelId, index,
                       ret);
    return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
  }
  aippInfo->srcFormat = static_cast<aclFormat>(inputInfo.format);
  aippInfo->srcDatatype = static_cast<aclDataType>(inputInfo.data_type);
  aippInfo->srcDimNum = inputInfo.dim_num;
  ACL_LOG_DEBUG("aclAippInfo: %s .", acl::AippInfoDebugString(aippInfo).c_str());

  // Get AllAippInputOutputDims, fail on error matching OM1
  std::vector<ge::InputOutputDims> input_dims;
  std::vector<ge::InputOutputDims> output_dims;
  ret = executor->GetAllAippInputOutputDims(static_cast<uint32_t>(index), input_dims, output_dims);
  if (ret != ge::SUCCESS) {
    ACL_LOG_CALL_ERROR(
        "[Get][AllAippInputOutputDims]GetAllAippInputOutputDims failed, modelId[%u], index[%zu], ret[%u]", modelId,
        index, ret);
    return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
  }
  ACL_LOG_DEBUG("GetAllAippInputOutputDims success");
  if ((shapeCount > static_cast<size_t>(ACL_MAX_SHAPE_COUNT)) || (shapeCount != input_dims.size()) ||
      (shapeCount != output_dims.size())) {
    ACL_LOG_INNER_ERROR(
        "[Check][Params]shapeCount[%zu] should be smaller than ACL_MAX_SHAPE_COUNT(128) and it "
        "should be equal to size of inputDims[%zu], size of outputDims[%zu]",
        shapeCount, input_dims.size(), output_dims.size());
    return ACL_ERROR_GE_FAILURE;
  }

  // Copy dims using SetIODims including tensor names matching OM1
  for (size_t i = 0U; i < shapeCount; ++i) {
    aclError io_ret = acl::SetIODims(input_dims[i], aippInfo->outDims[i].srcDims);
    if (io_ret != ACL_SUCCESS) {
      ACL_LOG_INNER_ERROR("[Set][IODims]srcDims SetIODims failed, modelId[%u], index[%zu], ret[%d]", modelId, index,
                          io_ret);
      return io_ret;
    }
    aippInfo->outDims[i].srcSize = input_dims[i].size;
    io_ret = acl::SetIODims(output_dims[i], aippInfo->outDims[i].aippOutdims);
    if (io_ret != ACL_SUCCESS) {
      ACL_LOG_INNER_ERROR("[Set][IODims]aippOutdims SetIODims failed, modelId[%u], index[%zu], ret[%d]", modelId, index,
                          io_ret);
      return io_ret;
    }
    aippInfo->outDims[i].aippOutSize = output_dims[i].size;
  }
  ACL_LOG_DEBUG("successfully execute aclmdlGetFirstAippInfo(om2), aclAippDims: %s",
                acl::AippDimsDebugString(aippInfo->outDims, aippInfo->shapeCount).c_str());
  return ACL_SUCCESS;
}

aclError aclmdlSetInputAIPPImplOm2(uint32_t modelId, aclmdlDataset *dataset, size_t index,
                                   const aclmdlAIPP *aippParmsSet) {
  ACL_PROFILING_REG(acl::AclProfType::AclmdlSetInputAIPP);
  ACL_LOG_DEBUG("start to execute aclmdlSetInputAIPP(OM2), modelId[%u], index[%zu]", modelId, index);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dataset);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippParmsSet);
  auto executor = acl::AclResourceManagerOm2::GetInstance().GetOm2Executor(modelId);
  if (executor == nullptr) {
    ACL_LOG_INNER_ERROR("[Check][Executor]Om2 executor not found, modelId[%u]", modelId);
    return ACL_ERROR_INVALID_PARAM;
  }
  // check dynamic aipp index
  ACL_REQUIRES_OK(VerifyIndexOm2(executor, index));

  auto mdlRet = CheckAippDataIndexOm2(executor, modelId, index);
  if (mdlRet != ACL_SUCCESS) {
    ACL_LOG_ERROR("[Check][AippData]Dynamic AIPP data index %zu is invalid, parameters verification failed", index);
    const std::string errMsg = acl::AclErrorLogManager::FormatStr("index %zu is invalid", index);
    acl::AclErrorLogManager::ReportInputError(acl::INVALID_AIPP_MSG, std::vector<const char *>({"param", "reason"}),
                                              std::vector<const char *>({"Dynamic AIPP data index", errMsg.c_str()}));
    return mdlRet;
  }

  mdlRet = GetAndCheckAippParamsOm2(executor, modelId, index, aippParmsSet);
  if (mdlRet != ACL_SUCCESS) {
    ACL_LOG_ERROR("[Check][AippParams]Dynamic AIPP parameters is invalid, parameters verification failed");
    acl::AclErrorLogManager::ReportInputError(
        acl::INVALID_AIPP_MSG, std::vector<const char *>({"param", "reason"}),
        std::vector<const char *>({"parameters", "parameters verification failed"}));
    return mdlRet;
  }

  const aclDataBuffer *const buff = aclmdlGetDatasetBufferImplOm2(dataset, index);
  if (buff == nullptr) {
    ACL_LOG_INNER_ERROR("[Check][Buff]failed to get data buffer by index[%zu]", index);
    return ACL_ERROR_INVALID_PARAM;
  }
  void *const dev_ptr = aclGetDataBufferAddr(buff);
  if (dev_ptr == nullptr) {
    ACL_LOG_INNER_ERROR("[Check][DevPtr]failed to get addr by index[%zu]", index);
    return ACL_ERROR_INVALID_PARAM;
  }
  const uint64_t mem_size = aclGetDataBufferSizeV2(buff);
  ACL_LOG_DEBUG("aippParmsSet->aippParms: %s .", acl::AippParmsDebugString(aippParmsSet->aippParms).c_str());
  for (size_t i = 0U; i < aippParmsSet->aippBatchPara.size(); ++i) {
    ACL_LOG_DEBUG("batchIndex[%lu] aippParmsSet->aippBatchPara: %s .", i,
                  acl::AippBatchParaDebugString(aippParmsSet->aippBatchPara[i]).c_str());
  }
  // send dynamic aipp to device
  ACL_LOG_INFO("call om2 executor.SetDynamicAippData, modelId[%u]", modelId);
  const auto ret =
      executor->SetDynamicAippData(dev_ptr, mem_size, aippParmsSet->aippBatchPara, aippParmsSet->aippParms);
  if (ret != ge::SUCCESS) {
    ACL_LOG_CALL_ERROR("[Set][DynamicAippData]SetDynamicAippData failed, ge result[%u]", ret);
    return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
  }
  ACL_LOG_INFO("successfully execute aclmdlSetInputAIPP(OM2), modelId[%u], index[%zu]", modelId, index);
  return ACL_SUCCESS;
}

aclError aclmdlSetAIPPByInputIndexImplOm2(uint32_t modelId, aclmdlDataset *dataset, size_t index,
                                          const aclmdlAIPP *aippParmsSet) {
  ACL_LOG_INFO("start to execute aclmdlSetAIPPByInputIndex(OM2), modelId[%u], index[%zu]", modelId, index);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippParmsSet);
  if ((dataset == nullptr) || (index >= dataset->blobs.size())) {
    ACL_LOG_ERROR("[Check][Dataset]input param is invalid, dataset[%p], index[%zu]", static_cast<void *>(dataset),
                  index);
    const std::string errMsg =
        acl::AclErrorLogManager::FormatStr("dataset[%p], index[%zu]", static_cast<void *>(dataset), index);
    acl::AclErrorLogManager::ReportInputError(acl::INVALID_AIPP_MSG, std::vector<const char *>({"param", "reason"}),
                                              std::vector<const char *>({"params", errMsg.c_str()}));
    return ACL_ERROR_INVALID_PARAM;
  }
  aclmdlInputAippType aipp_type = ACL_DATA_WITHOUT_AIPP;
  size_t dynamicAttachedDataIndex = 0U;
  const auto ret = aclmdlGetAippTypeImplOm2(modelId, index, &aipp_type, &dynamicAttachedDataIndex);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  if (aipp_type != ACL_DATA_WITH_DYNAMIC_AIPP) {
    ACL_LOG_INNER_ERROR("[Check][Type]input[%zu] has no dynamic aipp linked, modelId[%u]", index, modelId);
    return ACL_ERROR_FAILURE;
  }
  ACL_LOG_INFO("successfully execute aclmdlSetAIPPByInputIndex(OM2), modelId[%u], index[%zu]", modelId, index);
  return aclmdlSetInputAIPPImplOm2(modelId, dataset, dynamicAttachedDataIndex, aippParmsSet);
}

aclError aclmdlSetAttributeImplOm2(uint32_t modelId, aclmdlAttr attr, aclmdlAttrValue_t *attrValue) {
  (void)modelId;
  (void)attr;
  (void)attrValue;
  return ACL_ERROR_API_NOT_SUPPORT;
}

aclError aclmdlGetAttributeImplOm2(uint32_t modelId, aclmdlAttr attr, aclmdlAttrValue_t *attrValue) {
  (void)modelId;
  (void)attr;
  (void)attrValue;
  return ACL_ERROR_API_NOT_SUPPORT;
}
