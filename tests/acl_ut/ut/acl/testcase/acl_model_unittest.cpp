/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "acl/acl.h"
#define private public
#include "model/acl_resource_manager.h"
#undef private
#include "runtime/gert_api.h"
#include "runtime/stream.h"
#include "runtime/rt_model.h"
#include "model/aipp_param_check.h"
#include "framework/executor/ge_executor.h"
#include "common/ge_types.h"
#include "acl_stub.h"
#include "graph/ge_context.h"


using namespace testing;
using namespace std;
using namespace acl;

class UTEST_ACL_Model : public testing::Test {
public:
    UTEST_ACL_Model()
    {
    }

protected:
    void SetUp() override
    {
        MockFunctionTest::aclStubInstance().ResetToDefaultMock();
    }
    void TearDown() override
    {
        Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));
    }
};

ge::graphStatus ExecuteSync_Invoke(gert::Tensor **inputs, size_t input_num, gert::Tensor **outputs, size_t output_num) {
    (void) inputs;
    (void) input_num;
    (void) output_num;
    static uint64_t tmp{1};
    outputs[0]->MutableTensorData().SetAddr(&tmp, nullptr);
    outputs[0]->MutableTensorData().SetSize(sizeof(uint64_t));
    return ge::GRAPH_SUCCESS;
}

ge::Status GetModelDescInfo_Invoke(uint32_t modelId, std::vector<ge::TensorDesc>& inputDesc,
                                        std::vector<ge::TensorDesc>& outputDesc, bool new_model_desc)
{
    (void) modelId;
    (void) new_model_desc;
    ge::TensorDesc desc1;
    inputDesc.push_back(desc1);
    outputDesc.push_back(desc1);
    return ge::SUCCESS;
}

ge::Status GetModelDescInfoFromMem_Invoke(const ModelData &model_data, ModelInOutInfo &info)
{
    (void) model_data;
    ge::ModelInOutTensorDesc test;
    info.input_desc.emplace_back(test);
    info.output_desc.emplace_back(test);
    return ge::SUCCESS;
}

ge::Status GetModelDescInfo_Invoke2(uint32_t modelId, std::vector<ge::TensorDesc>& inputDesc,
                                   std::vector<ge::TensorDesc>& outputDesc, bool new_model_desc)
{
    (void) modelId;
    (void) new_model_desc;
    ge::TensorDesc desc1;
    ge::TensorDesc desc2;
    inputDesc.push_back(desc1);
    inputDesc.push_back(desc2);
    outputDesc.push_back(desc1);
    return ge::SUCCESS;
}

ge::Status GetModelDescInfo_Invoke3(uint32_t modelId, std::vector<ge::TensorDesc>& inputDesc,
                                   std::vector<ge::TensorDesc>& outputDesc, bool new_model_desc)
{
    if (modelId == 2U && new_model_desc == false) {
        ge::TensorDesc desc1;
        ge::TensorDesc desc2;
        inputDesc.push_back(desc1);
        outputDesc.push_back(desc1);
    }
    if (modelId == 1U) {
        ge::TensorDesc desc1;
        ge::TensorDesc desc2;
        inputDesc.push_back(desc1);
        outputDesc.push_back(desc1);
    }

    return ge::SUCCESS;
}

ge::Status GetDynamicBatchInfo_Invoke(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                      int32_t &dynamic_type)
{
    (void) model_id;
    dynamic_type = 2;
    batch_info.push_back({224, 224});
    batch_info.push_back({600, 600});
    return ge::SUCCESS;
}

ge::Status GetDynamicBatchInfo_Invoke3(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                       int32_t &dynamic_type)
{
    (void) model_id;
    dynamic_type = 3;
    batch_info.push_back({224, 224});
    batch_info.push_back({600, 600});
    return ge::SUCCESS;
}

ge::Status ExecModelInvoke(uint32_t model_id, void *stream, const ge::RunModelData &run_input_data,
                            const std::vector<ge::GeTensorDesc> &input_desc, ge::RunModelData &run_output_data,
                            std::vector<ge::GeTensorDesc> &output_desc, bool async_mode)
{
    (void) model_id;
    (void) stream;
    (void) run_input_data;
    (void) input_desc;
    (void) run_output_data;
    (void) async_mode;
    ge::GeTensorDesc geDescTmp;
    output_desc.push_back(geDescTmp);
    output_desc.push_back(geDescTmp);
    return SUCCESS;
}

ge::Status ExecModelInvokeOneOut(uint32_t model_id, void *stream, const ge::RunModelData &run_input_data,
                           const std::vector<ge::GeTensorDesc> &input_desc, ge::RunModelData &run_output_data,
                           std::vector<ge::GeTensorDesc> &output_desc, bool async_mode)
{
  (void) model_id;
  (void) stream;
  (void) run_input_data;
  (void) input_desc;
  (void) run_output_data;
  (void) async_mode;
  ge::GeTensorDesc geDescTmp;
  output_desc.push_back(geDescTmp);
  return SUCCESS;
}

ge::Status GetDynamicBatchInfo_Invoke4(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                       int32_t &dynamic_type)
{
    (void) model_id;
    dynamic_type = 3;
    batch_info.push_back({224, 224});
    batch_info.push_back({600, 600, 600});
    return ge::SUCCESS;
}

ge::Status GetDynamicBatchInfo_Invoke5(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
    int32_t &dynamic_type)
{
  (void) model_id;
  (void) batch_info;
  (void) dynamic_type;
  return ge::SUCCESS;
}

ge::Status GetUserDesignateShapeOrderInvoke(uint32_t model_id, vector<string> &user_designate_shape_order) {
    (void) model_id;
    user_designate_shape_order.emplace_back("resnet50");
    user_designate_shape_order.emplace_back("resnet50");
    return ge::SUCCESS;
}

ge::Status GetAippTypeFailInvoke(uint32_t model_id, uint32_t index,
    ge::InputAippType &type, size_t &aippindex) {
    (void) model_id;
    (void) index;
    type = ge::DATA_WITHOUT_AIPP;
    aippindex = 0xFFFFFFFF;
    return ge::FAILED;
}

ge::Status GetAippTypeSuccessInvoke(uint32_t model_id, uint32_t index,
    ge::InputAippType &type, size_t &aippindex) {
    (void) model_id;
    (void) index;
    (void) aippindex;
    type = ge::DYNAMIC_AIPP_NODE;
    return ge::SUCCESS;
}

ge::Status GetAippTypeNoAippInvoke(uint32_t model_id, uint32_t index,
    ge::InputAippType &type, size_t &aippindex) {
    (void) model_id;
    (void) index;
    type = ge::DATA_WITHOUT_AIPP;
    aippindex = 0xFFFFFFFF;
    return ge::SUCCESS;
}

ge::Status GetAippTypeStaticAippInvoke(uint32_t model_id, uint32_t index, ge::InputAippType &type, size_t &aippindex) {
    (void) model_id;
    (void) index;
    type = ge::DATA_WITH_STATIC_AIPP;
    aippindex = 0xFFFFFFFF;
    return ge::SUCCESS;
}

ge::Status GetOpAttr_Invoke(uint32_t model_id, const std::string &op_name, const std::string &attr_name, std::string &attr_value)
{
    (void) model_id;
    (void) op_name;
    (void) attr_name;
    (void) attr_value;
    return FAILED;
}

ge::Status GetOpAttr_Invoke_1(uint32_t model_id, const std::string &op_name, const std::string &attr_name, std::string &attr_value)
{
    (void) model_id;
    (void) op_name;
    (void) attr_name;
    attr_value = "";
    return SUCCESS;
}

ge::Status GetOpAttr_Invoke_2(uint32_t model_id, const std::string &op_name, const std::string &attr_name, std::string &attr_value)
{
    (void) model_id;
    (void) op_name;
    (void) attr_name;
    attr_value = "attr_finded";
    return SUCCESS;
}

ge::graphStatus GetShapeRange_Invoke(std::vector<std::pair<int64_t,int64_t>> &range)
{
    range.push_back(std::make_pair(1, 16));
    range.push_back(std::make_pair(1, 16));
    range.push_back(std::make_pair(1, 16));
    range.push_back(std::make_pair(1, 16));
    return GRAPH_SUCCESS;
}

std::unique_ptr<gert::ModelV2Executor>
LoadExecutorFromModelDataReturnError(const ge::ModelData &model_data,  const gert::LoadExecutorArgs &args,
                                     ge::graphStatus &error_code)
{
    (void) model_data;
    (void) args;
    error_code = ge::GRAPH_FAILED;
    return nullptr;
}

std::unique_ptr<gert::ModelV2Executor>
LoadExecutorFromModelDataSuccess(const ge::ModelData &model_data, const gert::LoadExecutorArgs &args,
                                 ge::graphStatus &error_code)
{
    (void) model_data;
    (void) args;
    auto executor = std::unique_ptr<gert::ModelV2Executor>(new (std::nothrow) gert::ModelV2Executor);
    error_code = ge::GRAPH_SUCCESS;
    return executor;
}

std::unique_ptr<gert::ModelV2Executor>
LoadExecutorFromModelDataCheckFileConstantMemSuccess(const ge::ModelData &model_data,
                                                     const gert::LoadExecutorArgs &args,
                                                     ge::graphStatus &error_code)
{
    (void) model_data;
    if (args.file_constant_mems.size() != 2U) {
        return nullptr;
    }
    if (args.file_constant_mems[0].file_name != "fileconstant1.bin" ||
        args.file_constant_mems[1].file_name != "fileconstant2.bin") {
        return nullptr;
    }
    auto executor = std::unique_ptr<gert::ModelV2Executor>(new (std::nothrow) gert::ModelV2Executor);
    error_code = ge::GRAPH_SUCCESS;
    return executor;
}

std::unique_ptr<gert::ModelV2Executor> LoadExecutorFromFileReturnUnSupported(const char *file_path, ge::graphStatus &error_code)
{
    (void) file_path;
    error_code = ge::GE_GRAPH_UNSUPPORTED;
    return std::unique_ptr<gert::ModelV2Executor>(new(std::nothrow) gert::ModelV2Executor);
}

ge::graphStatus IsDynamicModelReturnTrue(const char *file_path, bool &is_dynamic_model) {
  (void) file_path;
  is_dynamic_model = true;
  return GRAPH_SUCCESS;
}

ge::graphStatus IsDynamicModelReturnFailed(const char *file_path, bool &is_dynamic_model) {
  (void) file_path;
  is_dynamic_model = false;
  return ACL_ERROR_GE_PARAM_INVALID;
}

TEST_F(UTEST_ACL_Model, aclmdlGetOpAttr)
{
    aclmdlDesc *mdlDesc = aclmdlCreateDesc();
    aclmdlDesc *mdlDescNullptr = nullptr;
    const char *opName = "anyOp";
    const char *opNameNullptr = nullptr;
    const char *attr = "_datadump_original_op_names";
    const char *attrNullptr = nullptr;

    const char *attrNotSupported = "anyAttr";
    const char *result = aclmdlGetOpAttr(mdlDesc, opName, attrNotSupported);
    EXPECT_EQ(result, nullptr);

    const char *resultNullptr_1 = aclmdlGetOpAttr(mdlDescNullptr, opName, attr);
    EXPECT_EQ(resultNullptr_1, nullptr);

    const char *resultNullptr_2 = aclmdlGetOpAttr(mdlDesc, opNameNullptr, attr);
    EXPECT_EQ(resultNullptr_2, nullptr);

    const char *resultNullptr_3 = aclmdlGetOpAttr(mdlDesc, opName, attrNullptr);
    EXPECT_EQ(resultNullptr_3, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetOpAttr(_,_,_,_))
        .WillOnce(Invoke(GetOpAttr_Invoke));
    const char *resultGeFailed  = aclmdlGetOpAttr(mdlDesc, opName, attr);
    EXPECT_EQ(resultGeFailed, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetOpAttr(_,_,_,_))
        .WillOnce(Invoke(GetOpAttr_Invoke_1));
    const char *resultEmptyStr = aclmdlGetOpAttr(mdlDesc, opName, attr);
    EXPECT_EQ(string(resultEmptyStr), "");

    const char *opName3 = "anyOp3";
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetOpAttr(_,_,_,_))
        .WillOnce(Invoke(GetOpAttr_Invoke_2));
    const char *resultValue = aclmdlGetOpAttr(mdlDesc, opName3, attr);
    EXPECT_EQ(string(resultValue), "attr_finded");

    const char *resultGeSuccess = aclmdlGetOpAttr(mdlDesc, opName3, attr);
    EXPECT_EQ(string(resultGeSuccess), "attr_finded");

    aclmdlDestroyDesc(mdlDesc);
}

TEST_F(UTEST_ACL_Model, desc)
{
    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    size_t size = aclmdlGetNumInputs(nullptr);
    EXPECT_EQ(size, 0);
    size = aclmdlGetNumInputs(desc);
    EXPECT_EQ(size, 0);

    size = aclmdlGetNumOutputs(nullptr);
    EXPECT_EQ(size, 0);
    size = aclmdlGetNumOutputs(desc);
    EXPECT_EQ(size, 0);

    aclError ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlDestroyDesc(nullptr);
    EXPECT_NE(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetDesc)
{
    aclError ret = aclmdlGetDesc(nullptr, 1);
    EXPECT_NE(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_,_,_,_))
        .WillRepeatedly(Invoke(GetModelDescInfo_Invoke));
    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_,_,_))
         .WillOnce(Invoke(GetDynamicBatchInfo_Invoke3));
    ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);
    Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillOnce(Invoke(GetDynamicBatchInfo_Invoke4));
    ret = aclmdlGetDesc(desc, 1);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetDesc_fail_1)
{
    aclError ret = aclmdlGetDesc(nullptr, 1);
    EXPECT_NE(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_,_,_,_))
        .WillRepeatedly(Invoke(GetModelDescInfo_Invoke));
    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_,_,_))
         .WillRepeatedly(Return(ACL_ERROR_INVALID_PARAM));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetUserDesignateShapeOrder(_,_))
        .WillRepeatedly(Return(ACL_ERROR_INVALID_PARAM));
    ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetDesc_2)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_,_,_,_))
        .WillRepeatedly(Invoke(GetModelDescInfo_Invoke3));

    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    aclError ret = aclmdlGetDesc(desc, 2);

    EXPECT_NE(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_,_,_))
         .WillOnce(Invoke(GetDynamicBatchInfo_Invoke3));
    ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(aclmdlDestroyDesc(desc), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetDesc_Fail)
{
    aclError ret = aclmdlGetDesc(nullptr, 1);
    EXPECT_NE(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillOnce(Return((PARAM_INVALID)));

    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    ret = aclmdlGetDesc(desc, 1);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetDescFromFile)
{
  aclError ret = aclmdlGetDescFromFile(nullptr, "./fake.om");
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

  aclmdlDesc* desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);

  ret = aclmdlGetDescFromFile(desc, nullptr);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

  EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadDataFromFile(_,_))
          .WillOnce(Return((PARAM_INVALID)))
          .WillRepeatedly(Return(SUCCESS));

  ret = aclmdlGetDescFromFile(desc, "./fake.om");
  EXPECT_NE(ret, ACL_SUCCESS);

  EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfoFromMem(_, _))
          .WillOnce(Return((PARAM_INVALID)))
          .WillRepeatedly(Invoke(GetModelDescInfoFromMem_Invoke));

  ret = aclmdlGetDescFromFile(desc, "./fake.om");
  EXPECT_NE(ret, ACL_SUCCESS);

  ret = aclmdlGetDescFromFile(desc, "./fake.om");
  EXPECT_EQ(ret, ACL_SUCCESS);

  ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetDescFromMem)
{
  uint32_t fakeModel = 0;
  aclError ret = aclmdlGetDescFromMem(nullptr, &fakeModel, sizeof(fakeModel));
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

  aclmdlDesc* desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);

  ret = aclmdlGetDescFromMem(desc, nullptr, sizeof(fakeModel));
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

  ret = aclmdlGetDescFromMem(desc, &fakeModel, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

  EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfoFromMem(_, _))
          .WillOnce(Return((PARAM_INVALID)))
          .WillRepeatedly(Return(SUCCESS));

  ret = aclmdlGetDescFromMem(desc, &fakeModel, sizeof(fakeModel));
  EXPECT_NE(ret, ACL_SUCCESS);

  ret = aclmdlGetDescFromMem(desc, &fakeModel, sizeof(fakeModel));
  EXPECT_EQ(ret, ACL_SUCCESS);

  ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetInputSizeByIndex)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));
    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    size_t size = aclmdlGetInputSizeByIndex(nullptr, 0);
    EXPECT_EQ(size, 0);

    size = aclmdlGetInputSizeByIndex(desc, 0);
    EXPECT_EQ(size, 1);

    size = aclmdlGetOutputSizeByIndex(nullptr, 0);
    EXPECT_EQ(size, 0);

    size = aclmdlGetOutputSizeByIndex(desc, 0);
    EXPECT_EQ(size, 1);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, Dataset)
{
    aclmdlDataset *dataset = aclmdlCreateDataset();
    EXPECT_NE(dataset, nullptr);

    aclError ret = aclmdlAddDatasetBuffer(dataset, nullptr);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlAddDatasetBuffer(dataset, (aclDataBuffer *)0x01);
    EXPECT_EQ(ret, ACL_SUCCESS);

    size_t size = aclmdlGetDatasetNumBuffers(nullptr);
    EXPECT_EQ(size, 0);

    size = aclmdlGetDatasetNumBuffers(dataset);
    EXPECT_EQ(size, 1);

    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(dataset, 1);
    EXPECT_EQ(dataBuffer, nullptr);

    dataBuffer = aclmdlGetDatasetBuffer(dataset, 0);
    EXPECT_NE(dataBuffer, nullptr);

    ret = aclmdlAddDatasetBuffer(nullptr, nullptr);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlDestroyDataset(nullptr);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlDestroyDataset(dataset);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlLoadFromFile)
{
    aclError ret = aclmdlLoadFromFile(nullptr, nullptr);
    EXPECT_NE(ret, ACL_SUCCESS);

    uint32_t modelId = 1;
    const char *modelPath = "/";

    ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadModelFromDataWithArgs(_,_,_))
        .WillOnce(Return((ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED)));
    ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_NE(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadModelFromDataWithArgs(_,_,_))
        .WillOnce(Return((PARAM_INVALID)));
    ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_NE(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlLoadFromFileRTV2)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), IsDynamicModel(_, _))
            .WillRepeatedly(Invoke((IsDynamicModelReturnTrue)));
    uint32_t modelId = 1;
    const char *modelPath = "/";
    // case 1
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
            .WillOnce(Invoke(LoadExecutorFromModelDataSuccess));
    auto ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    // case 2
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
            .WillOnce(Invoke(LoadExecutorFromModelDataReturnError));
    ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_NE(ret, ACL_SUCCESS);

    // case 3
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadDataFromFileV2(_,_))
            .WillOnce(Return((GRAPH_PARAM_INVALID)));
    ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_NE(ret, ACL_SUCCESS);

    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, aclmdlLoadFromFileWithMemRTV2)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), IsDynamicModel(_, _))
            .WillRepeatedly(Invoke((IsDynamicModelReturnTrue)));
    uint32_t modelId = 1;
    const char *modelPath = "/";
    void *weightPtr = reinterpret_cast<void *>(0x02);
    size_t weightSize = 10U;
    // case 1
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
            .WillOnce(Invoke(LoadExecutorFromModelDataSuccess));
    auto ret = aclmdlLoadFromFileWithMem(modelPath, &modelId, nullptr, 0U, weightPtr, weightSize);
    EXPECT_EQ(ret, ACL_SUCCESS);

    // case 2
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
            .WillOnce(Invoke(LoadExecutorFromModelDataReturnError));
    ret = aclmdlLoadFromFileWithMem(modelPath, &modelId, nullptr, 0U, weightPtr, weightSize);
    EXPECT_NE(ret, ACL_SUCCESS);

    // case 3
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadDataFromFileV2(_,_))
            .WillOnce(Return((GRAPH_PARAM_INVALID)));
    ret = aclmdlLoadFromFileWithMem(modelPath, &modelId, nullptr, 0U, weightPtr, weightSize);
    EXPECT_NE(ret, ACL_SUCCESS);

    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, aclmdlLoadFromMemRTV2)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;

    aclError ret = aclmdlLoadFromMem(nullptr, 0, nullptr);
    EXPECT_NE(ret, ACL_SUCCESS);

    ge::ModelFileHeader head;
    head.version = ge::MODEL_VERSION + 1U;
    head.model_num = 2U;
    void *model = (void *)&head;
    uint32_t modelId = 1;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
            .WillOnce(Invoke(LoadExecutorFromModelDataSuccess));
    ret = aclmdlLoadFromMem(model, 0, &modelId);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlLoadFromMem(model, sizeof(head), &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
        .WillOnce(Invoke(LoadExecutorFromModelDataReturnError));
    ret = aclmdlLoadFromMem(model, sizeof(head), &modelId);
    EXPECT_NE(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, aclmdlLoadFromMemWithMemRTV2)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    void *weightPtr = reinterpret_cast<void *>(0x02);
    size_t weightSize = 10U;
    // case 1
    aclError ret = aclmdlLoadFromMemWithMem(nullptr, 0, nullptr, nullptr, 0U, weightPtr, weightSize);
    EXPECT_NE(ret, ACL_SUCCESS);

    // case 2
    ge::ModelFileHeader head;
    head.version = ge::MODEL_VERSION + 1U;
    head.model_num = 2U;
    void *model = (void *)&head;
    uint32_t modelId = 1;
    ret = aclmdlLoadFromMemWithMem(model, 0, &modelId, nullptr, 0U, weightPtr, weightSize);
    EXPECT_NE(ret, ACL_SUCCESS);

    // case 3
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
            .WillOnce(Invoke(LoadExecutorFromModelDataSuccess));
    ret = aclmdlLoadFromMemWithMem(model, sizeof(head), &modelId, nullptr, 0U, weightPtr, weightSize);
    EXPECT_EQ(ret, ACL_SUCCESS);

    // case 4
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
            .WillOnce(Invoke(LoadExecutorFromModelDataReturnError));
    ret = aclmdlLoadFromMemWithMem(model, sizeof(head), &modelId, nullptr, 0U, weightPtr, weightSize);
    EXPECT_NE(ret, ACL_SUCCESS);

    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, aclmdlLoadFromFileWithMem)
{
    aclError ret = aclmdlLoadFromFileWithMem(nullptr, nullptr, nullptr, 0, nullptr, 0);
    EXPECT_NE(ret, ACL_SUCCESS);

    uint32_t modelId = 1;
    const char *modelPath = "/";

    ret = aclmdlLoadFromFileWithMem(modelPath, &modelId, nullptr, 0, nullptr, 0);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadModelFromDataWithArgs(_,_,_))
        .WillOnce(Return((PARAM_INVALID)));
    ret = aclmdlLoadFromFileWithMem(modelPath, &modelId, nullptr, 0, nullptr, 0);
    EXPECT_NE(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadDataFromFile(_,_))
        .WillOnce(Return((PARAM_INVALID)));
    ret = aclmdlLoadFromFileWithMem(modelPath, &modelId, nullptr, 0, nullptr, 0);
    EXPECT_NE(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlLoadFromMem)
{
    aclError ret = aclmdlLoadFromMem(nullptr, 0, nullptr);
    EXPECT_NE(ret, ACL_SUCCESS);

    void *model = (void *)0x01;
    uint32_t modelId = 1;
    ret = aclmdlLoadFromMem(model, 0, &modelId);
    EXPECT_NE(ret, ACL_SUCCESS);

    ge::ModelFileHeader head;
    head.version = 0U;
    head.model_num = 1U;
    model = (void *)&head;
    ret = aclmdlLoadFromMem(model, sizeof(head), &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadModelFromDataWithArgs(_,_,_))
        .WillOnce(Return((PARAM_INVALID)));
    ret = aclmdlLoadFromMem(model, sizeof(head), &modelId);
    EXPECT_NE(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlLoadFromMemWithMem)
{
    aclError ret = aclmdlLoadFromMemWithMem(nullptr, 0, nullptr, nullptr, 0, nullptr, 0);
    EXPECT_NE(ret, ACL_SUCCESS);

    void *model = (void *)0x01;
    uint32_t modelId = 1;
    ret = aclmdlLoadFromMemWithMem(model, 0, &modelId, nullptr, 0, nullptr, 0);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlLoadFromMemWithMem(model, 1, &modelId, nullptr, 0, nullptr, 0);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadModelFromDataWithArgs(_,_,_))
        .WillOnce(Return((PARAM_INVALID)));
    ret = aclmdlLoadFromMemWithMem(model, 1, &modelId, nullptr, 0, nullptr, 0);
    EXPECT_NE(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlLoadFromFileWithQ)
{
    aclError ret = aclmdlLoadFromFileWithQ(nullptr, nullptr, nullptr, 0, nullptr, 0);
    EXPECT_NE(ret, ACL_SUCCESS);

    const char *modelPath = "/";
    uint32_t modelId = 1;

    uint32_t *input = new(std::nothrow) uint32_t[1];
    uint32_t *output = new(std::nothrow) uint32_t[1];
    ret = aclmdlLoadFromFileWithQ(modelPath, &modelId, input, 1, output, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadModelWithQ(_,_,_))
        .WillOnce(Return((PARAM_INVALID)));
    ret = aclmdlLoadFromFileWithQ(modelPath, &modelId, input, 1, output, 1);
    EXPECT_NE(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadDataFromFile(_,_))
        .WillOnce(Return((PARAM_INVALID)));
    ret = aclmdlLoadFromFileWithQ(modelPath, &modelId, (uint32_t*)input, 1, (uint32_t*)output, 1);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlLoadFromFileWithQ(modelPath, &modelId, (uint32_t*)input, 0, (uint32_t*)output, 1);
    EXPECT_NE(ret, ACL_SUCCESS);
    delete []input;
    delete []output;
}

TEST_F(UTEST_ACL_Model, aclmdlLoadFromMemWithQ)
{
    aclError ret = aclmdlLoadFromMemWithQ(nullptr, 0, nullptr, nullptr, 0, nullptr, 0);
    EXPECT_NE(ret, ACL_SUCCESS);

    uint32_t *input = new(std::nothrow) uint32_t[1];
    uint32_t *output = new(std::nothrow) uint32_t[1];
    const char *modelPath = "/";
    uint32_t modelId = 1;
    ret = aclmdlLoadFromMemWithQ(modelPath, 1, &modelId, input, 1, output, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadModelWithQ(_,_,_))
        .WillRepeatedly(Return(PARAM_INVALID));
    ret = aclmdlLoadFromMemWithQ(modelPath, 1, &modelId, input, 1, output, 1);
    EXPECT_NE(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadModelWithQ(_,_,_))
        .WillRepeatedly(Return(PARAM_INVALID));
    ret = aclmdlLoadFromMemWithQ(modelPath, 0, &modelId, input, 1, output, 1);
    EXPECT_NE(ret, ACL_SUCCESS);

    delete []input;
    delete []output;
}

TEST_F(UTEST_ACL_Model, aclmdlUnload)
{
    aclError ret = aclmdlUnload(0);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), UnloadModel(_))
        .WillOnce(Return((PARAM_INVALID)));
    ret = aclmdlUnload(0);
    EXPECT_NE(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlQuerySize)
{
    const char *fileName = "/";
    size_t memSize;
    size_t weightSize;

    aclError ret = aclmdlQuerySize(nullptr, nullptr, nullptr);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlQuerySize(fileName, &memSize, &weightSize);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetMemAndWeightSize(_,_,_))
        .WillOnce(Return((PARAM_INVALID)));
    ret = aclmdlQuerySize(fileName, &memSize, &weightSize);
    EXPECT_NE(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlQuerySizeFromMem)
{
    void *model = (void *)0x01;
    size_t memSize;
    size_t weightSize;

    aclError ret = aclmdlQuerySizeFromMem(nullptr, 1,  nullptr, nullptr);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlQuerySizeFromMem(model, 1, &memSize, &weightSize);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetMemAndWeightSize(_,_,_,_))
        .WillOnce(Return((PARAM_INVALID)));
    ret = aclmdlQuerySizeFromMem(model, 1, &memSize, &weightSize);
    EXPECT_NE(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlExecute)
{
    aclmdlDataset *dataset = aclmdlCreateDataset();
    EXPECT_NE(dataset, nullptr);

    aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
    aclError ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecModel(_, _, _, _, _, _, _))
        .WillOnce(Invoke(ExecModelInvokeOneOut));
    ret = aclmdlExecute(1, dataset, dataset);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlExecute(1, nullptr, nullptr);
    EXPECT_NE(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecModel(_,_,_,_,_,_,_))
        .WillOnce(Return((PARAM_INVALID)));
    ret = aclmdlExecute(1, dataset, dataset);
    EXPECT_NE(ret, ACL_SUCCESS);

    free(dataBuffer);
    ret = aclmdlDestroyDataset(dataset);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlExecuteAsync)
{
    aclmdlDataset *dataset = aclmdlCreateDataset();
    EXPECT_NE(dataset, nullptr);

    aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
    aclError ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecModel(_, _, _, _, _, _, _))
        .WillOnce(Invoke(ExecModelInvokeOneOut));

    ret = aclmdlExecuteAsync(1, dataset, dataset, nullptr);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlExecuteAsync(1, nullptr, nullptr, nullptr);
    EXPECT_NE(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecModel(_,_,_,_,_,_,_))
        .WillOnce(Return((PARAM_INVALID)));
    ret = aclmdlExecuteAsync(1, dataset, dataset, nullptr);
    EXPECT_NE(ret, ACL_SUCCESS);

    free(dataBuffer);
    ret = aclmdlDestroyDataset(dataset);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlSetDynamicBatchSize)
{
    aclError ret = aclmdlSetDynamicBatchSize(1, (aclmdlDataset*)0x1, 0, 0);
    EXPECT_NE(ret, ACL_SUCCESS);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlSetDynamicBatchSize(1, dataset, 0, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclDataBuffer *buffer2 = aclCreateDataBuffer(nullptr, 0);
    ret = aclmdlAddDatasetBuffer(dataset, buffer2);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlSetDynamicBatchSize(1, dataset, 1, 1);
    EXPECT_NE(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), SetDynamicBatchSize(_,_,_,_))
        .WillOnce(Return((PARAM_INVALID)));
    ret = aclmdlSetDynamicBatchSize(1, dataset, 0, 1);
    EXPECT_NE(ret, ACL_SUCCESS);

    aclDestroyDataBuffer(buffer);
    aclDestroyDataBuffer(buffer2);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclGetDataBufferSizeV2)
{
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    EXPECT_EQ(aclGetDataBufferSizeV2(nullptr), 0);
    EXPECT_EQ(aclGetDataBufferSizeV2(buffer), 1);
    aclDestroyDataBuffer(buffer);
}

TEST_F(UTEST_ACL_Model, aclmdlSetDynamicHWSize)
{
    aclError ret = aclmdlSetDynamicHWSize(1, (aclmdlDataset*)0x1, 0, 0, 0);
    EXPECT_NE(ret, ACL_SUCCESS);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlSetDynamicHWSize(1, dataset, 0, 1, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclDataBuffer *buffer2 = aclCreateDataBuffer(nullptr, 0);
    ret = aclmdlAddDatasetBuffer(dataset, buffer2);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlSetDynamicHWSize(1, dataset, 1, 1, 1);
    EXPECT_NE(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), SetDynamicImageSize(_,_,_,_,_))
        .WillOnce(Return((PARAM_INVALID)));
    ret = aclmdlSetDynamicHWSize(1, dataset, 0, 1, 1);
    EXPECT_NE(ret, ACL_SUCCESS);

    aclDestroyDataBuffer(buffer);
    aclDestroyDataBuffer(buffer2);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetInputDynamicDims01)
{
    aclError ret = aclmdlSetInputDynamicDims(1, (aclmdlDataset*)0x1, 0, nullptr);
    EXPECT_NE(ret, ACL_SUCCESS);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), SetDynamicDims(_,_,_,_))
        .Times(2)
        .WillOnce(Return((SUCCESS)));
    aclmdlIODims dims[1];
    dims[0].dimCount = 1;
    dims[0].dims[0] = 1;
    ret = aclmdlSetInputDynamicDims(1, dataset, 0, dims);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclDataBuffer *buffer2 = aclCreateDataBuffer(nullptr, 0);
    ret = aclmdlAddDatasetBuffer(dataset, buffer2);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlSetInputDynamicDims(1, dataset, 0, dims);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), SetDynamicDims(_,_,_,_))
        .WillOnce(Return((PARAM_INVALID)));
    ret = aclmdlSetInputDynamicDims(1, dataset, 0, dims);
    EXPECT_NE(ret, ACL_SUCCESS);

    aclDestroyDataBuffer(buffer);
    aclDestroyDataBuffer(buffer2);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetInputDynamicDims02)
{
    aclError ret = aclmdlSetInputDynamicDims(1, (aclmdlDataset*)0x1, 0, nullptr);
    EXPECT_NE(ret, ACL_SUCCESS);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclmdlIODims dims[1];
    dims[0].dimCount = 1;
    dims[0].dims[0] = 1;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetCurDynamicDims(_,_,_))
        .WillOnce(Return((FAILED)));
    ret = aclmdlSetInputDynamicDims(1, dataset, 0, dims);
    EXPECT_NE(ret, ACL_SUCCESS);
    aclDestroyDataBuffer(buffer);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlGetInputNameByIndex)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));

    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    const char *res = aclmdlGetInputNameByIndex(nullptr, 0);
    EXPECT_STREQ(res, "");

    res = aclmdlGetInputNameByIndex(desc, 3);
    EXPECT_STREQ(res, "");

    res = aclmdlGetInputNameByIndex(desc, 0);
    EXPECT_STRNE(res, "");

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetOutputNameByIndex)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));

    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    const char *res = aclmdlGetOutputNameByIndex(nullptr, 0);
    EXPECT_STREQ(res, "");

    res = aclmdlGetOutputNameByIndex(desc, 3);
    EXPECT_STREQ(res, "");

    res = aclmdlGetOutputNameByIndex(desc, 0);
    EXPECT_STRNE(res, "");

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetInputFormat)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));

    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclFormat formatVal = aclmdlGetInputFormat(nullptr, 0);
    EXPECT_EQ(formatVal, ACL_FORMAT_UNDEFINED);

    formatVal = aclmdlGetInputFormat(desc, 3);
    EXPECT_EQ(formatVal, ACL_FORMAT_UNDEFINED);

    formatVal = aclmdlGetInputFormat(desc, 0);
    EXPECT_EQ(formatVal, ACL_FORMAT_NCHW);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetOutputFormat)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));

    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclFormat formatVal = aclmdlGetOutputFormat(nullptr, 0);
    EXPECT_EQ(formatVal, ACL_FORMAT_UNDEFINED);

    formatVal = aclmdlGetOutputFormat(desc, 3);
    EXPECT_EQ(formatVal, ACL_FORMAT_UNDEFINED);

    formatVal = aclmdlGetOutputFormat(desc, 0);
    EXPECT_EQ(formatVal, ACL_FORMAT_NCHW);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetInputDataType)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));

    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclDataType typeVal = aclmdlGetInputDataType(nullptr, 0);
    EXPECT_EQ(typeVal, ACL_DT_UNDEFINED);

    typeVal = aclmdlGetInputDataType(desc, 3);
    EXPECT_EQ(typeVal, ACL_DT_UNDEFINED);

    typeVal = aclmdlGetInputDataType(desc, 0);
    EXPECT_EQ(typeVal, ACL_FLOAT);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetOutputDataType)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));

    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclDataType typeVal = aclmdlGetOutputDataType(nullptr, 0);
    EXPECT_EQ(typeVal, ACL_DT_UNDEFINED);

    typeVal = aclmdlGetOutputDataType(desc, 3);
    EXPECT_EQ(typeVal, ACL_DT_UNDEFINED);

    typeVal = aclmdlGetOutputDataType(desc, 0);
    EXPECT_EQ(typeVal, ACL_FLOAT);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}


TEST_F(UTEST_ACL_Model, aclmdlGetInputIndexByName)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));

    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    size_t idx = 0;
    ret = aclmdlGetInputIndexByName(desc, "resnet50", &idx);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlGetInputIndexByName(desc, "resnet18", &idx);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetOutputIndexByName)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));

    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    size_t idx = 0;
    ret = aclmdlGetOutputIndexByName(desc, "resnet50", &idx);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlGetOutputIndexByName(desc, "resnet18", &idx);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetAippType)
{
    uint32_t modelId = 0;
    size_t index = 0;
    aclmdlInputAippType type;
    size_t aippIndex;
    EXPECT_EQ(aclmdlGetAippType(modelId, index, &type, nullptr), ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(aclmdlGetAippType(modelId, index, nullptr, &aippIndex), ACL_ERROR_INVALID_PARAM);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
            .WillRepeatedly(Invoke((GetModelDescInfo_Invoke2)));
    aclError ret = aclmdlGetAippType(modelId, index, &type, &aippIndex);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetAippType1)
{
    uint32_t modelId = 0;
    size_t index = 0;
    aclmdlInputAippType type;
    size_t aippIndex;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
            .WillRepeatedly(Invoke((GetModelDescInfo_Invoke2)));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAippType(_, _,_,_))
        .WillRepeatedly(Invoke((GetAippTypeFailInvoke)));
    aclError ret = aclmdlGetAippType(modelId, index, &type, &aippIndex);
    EXPECT_EQ(ret, ACL_ERROR_GE_FAILURE);
}

TEST_F(UTEST_ACL_Model, aclmdlGetDynamicBatch)
{
    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclmdlBatch batch;
    ret = aclmdlGetDynamicBatch(desc, &batch);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetDynamicHW)
{
    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    aclmdlHW hw;
    aclError ret = aclmdlGetDynamicHW(nullptr, -1, &hw);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlGetDynamicHW(desc, -1, &hw);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetInputDynamicGearCount)
{
    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    size_t dimsCount;
    ret = aclmdlGetInputDynamicGearCount(desc, -1, &dimsCount);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke5)));
    ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlGetInputDynamicGearCount(desc, -1, &dimsCount);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlGetInputDynamicGearCount(desc, 1, &dimsCount);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetInputDynamicDims)
{
    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke3)));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetUserDesignateShapeOrder(_,_))
        .WillRepeatedly(Invoke((GetUserDesignateShapeOrderInvoke)));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke2)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    size_t dimsCount = 2;
    aclmdlIODims dims[2];
    ret = aclmdlGetInputDynamicDims(desc, 2, dims, dimsCount);
    EXPECT_NE(ret, ACL_SUCCESS);
    ret = aclmdlGetInputDynamicDims(desc, -1, dims, dimsCount);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlGetInputDynamicDims(desc, 1, dims, dimsCount);
    EXPECT_NE(ret, ACL_SUCCESS);
    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetInputDynamicDims_)
{
  aclmdlDesc desc;
  for (size_t i = 0; i < (ACL_MAX_DIM_CNT + 1); ++i) {
    std::string test{"test"};
    desc.dataNameOrder.emplace_back(test);
    aclmdlTensorDesc tensorDesc;
    tensorDesc.name = test;
    tensorDesc.dims.emplace_back(1);
    desc.inputDesc.emplace_back(tensorDesc);
  }
  desc.dynamicDims.emplace_back(1);
  desc.dynamicDims.emplace_back(1);
  size_t dimsCount = 2;
  aclmdlIODims dims[2];
  auto ret = aclmdlGetInputDynamicDims(&desc, -1, dims, dimsCount);
  EXPECT_EQ(ret, ACL_ERROR_FAILURE);
}

TEST_F(UTEST_ACL_Model, aclmdlGetInputDims)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));

    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclmdlIODims dims;
    ret = aclmdlGetInputDims(desc, 0, &dims);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlGetInputDims(desc, 1, &dims);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetInputDimsV2)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));

    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclmdlIODims dims;
    ret = aclmdlGetInputDimsV2(desc, 0, &dims);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlGetInputDimsV2(desc, 1, &dims);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetOutputDims01)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));

    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclmdlIODims dims;
    ret = aclmdlGetOutputDims(desc, 0, &dims);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlGetOutputDims(desc, 1, &dims);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetOutputDims02)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));
    aclmdlDesc* desc = aclmdlCreateDesc();

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclmdlGetDesc(desc, 1);
    aclmdlDestroyDesc(desc);
}

TEST_F(UTEST_ACL_Model, aclmdlGetInputDimsRange_DynamicGearScenario)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));

    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
    aclError ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclmdlIODimsRange dimsRange;
    ret = aclmdlGetInputDimsRange(desc, 0, &dimsRange);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(dimsRange.rangeCount, 0);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetInputDimsRange_DynamicScenario)
{
    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    aclmdlIODimsRange dimsRange;

    aclError ret = aclmdlGetInputDimsRange(desc, 0, &dimsRange);
    EXPECT_NE(ret, ACL_SUCCESS);

    aclmdlTensorDesc tensorDesc;
    tensorDesc.dims.push_back(-1);
    tensorDesc.shapeRanges.push_back(std::make_pair(1, 3));
    desc->inputDesc.push_back(tensorDesc);
    ret = aclmdlGetInputDimsRange(desc, 0, &dimsRange);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(dimsRange.rangeCount, 1);
    EXPECT_EQ(dimsRange.range[0][0], 1);
    EXPECT_EQ(dimsRange.range[0][1], 3);

    aclmdlTensorDesc tensorDesc1;
    tensorDesc1.dims.resize(static_cast<size_t>(ACL_MAX_DIM_CNT) + 1);
    tensorDesc1.shapeRanges.resize(static_cast<size_t>(ACL_MAX_DIM_CNT) + 1);
    desc->inputDesc.push_back(tensorDesc1);
    ret = aclmdlGetInputDimsRange(desc, 1, &dimsRange);
    EXPECT_NE(ret, ACL_SUCCESS);

    aclmdlTensorDesc tensorDesc2;
    tensorDesc2.shapeRanges.resize(4);
    tensorDesc2.dims.resize(10);
    desc->inputDesc.push_back(tensorDesc2);
    ret = aclmdlGetInputDimsRange(desc, 2, &dimsRange);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlGetInputDimsRange_StaticScenario)
{
    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    aclmdlIODimsRange dimsRange;

    aclError ret = aclmdlGetInputDimsRange(desc, 0, &dimsRange);
    EXPECT_NE(ret, ACL_SUCCESS);

    aclmdlTensorDesc tensorDesc;
    tensorDesc.dims.push_back(1);
    desc->inputDesc.push_back(tensorDesc);
    ret = aclmdlGetInputDimsRange(desc, 0, &dimsRange);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(dimsRange.rangeCount, 1);
    EXPECT_EQ(dimsRange.range[0][0], 1);
    EXPECT_EQ(dimsRange.range[0][1], 1);

    aclmdlTensorDesc tensorDesc1;
    tensorDesc1.dims.resize(static_cast<size_t>(ACL_MAX_DIM_CNT) + 1);
    desc->inputDesc.push_back(tensorDesc1);
    ret = aclmdlGetInputDimsRange(desc, 1, &dimsRange);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

ge::Status GetDynamicBatchInfo_Invoke1(uint32_t model_id,
                                      std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type)
{
    (void) model_id;
    dynamic_type = 2;
    batch_info.push_back({224, 224});
    batch_info.push_back({600, 600});
    return ge::SUCCESS;
}

ge::Status GetDynamicBatchInfo_Invoke2(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type)
{
    (void) model_id;
    dynamic_type = 1;
    batch_info.push_back({224});
    batch_info.push_back({600});
    return ge::SUCCESS;
}

ge::Status GetCurShape_Invoke(const uint32_t model_id, std::vector<int64_t> &batch_info,
                              int32_t &dynamic_type)
{
    (void) model_id;
    dynamic_type = 1;
    batch_info.push_back(224);
    return ge::SUCCESS;
}

ge::Status GetCurShape_Invoke1(const uint32_t model_id, std::vector<int64_t> &batch_info,
                               int32_t &dynamic_type)
{
    (void) model_id;
    (void) batch_info;
    (void) dynamic_type;
    return ge::SUCCESS;
}

ge::Status GetCurShape_Invoke2(const uint32_t model_id, std::vector<int64_t> &batch_info,
                               int32_t &dynamic_type)
{
    (void) model_id;
    (void) dynamic_type;
    batch_info.push_back(224);
    batch_info.push_back(224);
    batch_info.push_back(224);
    return ge::SUCCESS;
}

ge::Status GetCurShape_Invoke3(const uint32_t model_id, std::vector<int64_t> &batch_info,
                               int32_t &dynamic_type)
{
    (void) model_id;
    (void) batch_info;
    (void) dynamic_type;
    return ge::FAILED;
}

ge::Status GetCurShape_Invoke4(const uint32_t model_id, std::vector<int64_t> &batch_info,
                               int32_t &dynamic_type)
{
    (void) model_id;
    (void) dynamic_type;
    batch_info.push_back(224);
    batch_info.push_back(224);
    return ge::SUCCESS;
}

ge::Status GetModelAttr_Invoke(uint32_t model_id,
                               std::vector<std::string> &dynamic_output_shape_info)
{
    (void) model_id;
    dynamic_output_shape_info.push_back({"1:0:1,3,224,224"});
    return ge::SUCCESS;
}

ge::Status GetModelAttr_Invoke1(uint32_t model_id,
                               std::vector<std::string> &dynamic_output_shape_info)
{
    (void) model_id;
    dynamic_output_shape_info.push_back({"-1:0:1,3,224,224"});
    return ge::SUCCESS;
}


TEST_F(UTEST_ACL_Model, aclmdlGetCurOutputDims)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));

    aclmdlDesc* desc = aclmdlCreateDesc();
     EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke)));
     aclError ret = aclmdlGetDesc(desc, 1);
     EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetCurShape(_, _,_))
        .WillRepeatedly(Invoke((GetCurShape_Invoke4)));

    aclmdlIODims dims;
    ret = aclmdlGetCurOutputDims(desc, 0, &dims);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlGetCurOutputDims(desc, 1, &dims);
    EXPECT_NE(ret, ACL_SUCCESS);
    Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke2)));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetCurShape(_, _,_))
        .WillRepeatedly(Invoke((GetCurShape_Invoke)));

    ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlGetCurOutputDims(desc, 0, &dims);
    EXPECT_EQ(ret, ACL_SUCCESS);
    Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetCurShape(_, _,_))
        .WillRepeatedly(Invoke((GetCurShape_Invoke1)));

    ret = aclmdlGetCurOutputDims(desc, 0, &dims);
    EXPECT_EQ(ret, ACL_SUCCESS);
    Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetCurShape(_, _,_))
        .WillRepeatedly(Invoke((GetCurShape_Invoke2)));
    ret = aclmdlGetCurOutputDims(desc, 0, &dims);
    EXPECT_NE(ret, ACL_SUCCESS);
    Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetCurShape(_, _,_))
        .WillRepeatedly(Invoke((GetCurShape_Invoke3)));
    ret = aclmdlGetCurOutputDims(desc, 0, &dims);
    EXPECT_NE(ret, ACL_SUCCESS);
    Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke(GetModelDescInfo_Invoke));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke(GetDynamicBatchInfo_Invoke1));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetCurShape(_, _,_))
        .WillRepeatedly(Invoke(GetCurShape_Invoke4));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelAttr(_, _))
        .WillRepeatedly(Invoke(GetModelAttr_Invoke));
    ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlGetCurOutputDims(desc, 0, &dims);
    EXPECT_EQ(ret, ACL_SUCCESS); //modify
    Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelAttr(_, _))
        .WillRepeatedly(Invoke((GetModelAttr_Invoke1)));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_, _,_))
        .WillRepeatedly(Invoke((GetDynamicBatchInfo_Invoke2)));
    ret = aclmdlGetDesc(desc, 1);
    EXPECT_EQ(ret, ACL_SUCCESS);
    // check api not support to rt2 dynamic shape
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    auto executor = std::unique_ptr<gert::ModelV2Executor>(new(std::nothrow) gert::ModelV2Executor);
    auto rtSession = acl::AclResourceManager::GetInstance().CreateRtSession();
    acl::AclResourceManager::GetInstance().AddExecutor(desc->modelId, std::move(executor), rtSession);
    ret = aclmdlGetCurOutputDims(desc, 0, nullptr);
    EXPECT_EQ(ret, ACL_ERROR_API_NOT_SUPPORT);
    // check input invalid
    ret = aclmdlGetCurOutputDims(nullptr, 0, nullptr);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, aclmdlGetAippDataSize)
{
    size_t size = 0;
    EXPECT_EQ(ACL_ERROR_INVALID_PARAM, aclmdlGetAippDataSize(0, &size));
    EXPECT_EQ(ACL_ERROR_INVALID_PARAM, aclmdlGetAippDataSize(1, nullptr));
    EXPECT_EQ(ACL_SUCCESS, aclmdlGetAippDataSize(1, &size));
    EXPECT_EQ(size, 160);
}

TEST_F(UTEST_ACL_Model, aclmdlSetInputAIPP)
{
    aclmdlAIPP *aippmdlAipp = aclmdlCreateAIPP(0);
    aclError ret = aclmdlSetAIPPInputFormat(aippmdlAipp, ACL_YUV420SP_U8);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);
    ret = aclmdlSetAIPPInputFormat(aippDynamicSet, ACL_YUV420SP_U8);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPCscParams(aippDynamicSet, 1, 256, 443, 0, 256, -86, -178, 256, 0, 350, 0, 0, 0, 0, 128, 128);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPCscParams(aippDynamicSet, 0, 256, 443, 0, 256, -86, -178, 256, 0, 350, 0, 0, 0, 0, 128, 128);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPRbuvSwapSwitch(aippDynamicSet, 0);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPDtcPixelMean(aippDynamicSet, 0, 0, 0, 0, 0);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPDtcPixelMin(aippDynamicSet, 0, 0, 0, 0, 0);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPPixelVarReci(aippDynamicSet, 1, 1, 1, 0, 0);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPScfParams(aippDynamicSet, 0, 1, 1, 1, 1, 0);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPScfParams(aippDynamicSet, 1, 224, 224, 16, 224, 0);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPAxSwapSwitch(aippDynamicSet, 0);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPSrcImageSize(aippDynamicSet, 224, 224);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPCropParams(aippDynamicSet, 0, 0, 0, 1, 1, 0);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPCropParams(aippDynamicSet, 1, 0, 0, 1, 1, 0);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPPaddingParams(aippDynamicSet, 0, 0, 0, 0, 0, 0);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPPaddingParams(aippDynamicSet, 1, 0, 0, 0, 0, 0);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlSetAIPPPaddingParams(aippDynamicSet, 1, 0, 0, 0, 0, 10);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_NE(ret, ACL_SUCCESS);
    ret = aclmdlDestroyAIPP(aippDynamicSet);
    EXPECT_EQ(ret, ACL_SUCCESS);
    aclDestroyDataBuffer(buffer);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetInputAIPP_SetDynamicAippData_fail)
{
    aclmdlAIPP *aippmdlAipp = aclmdlCreateAIPP(0);
    aclError ret = aclmdlSetAIPPInputFormat(aippmdlAipp, ACL_YUV420SP_U8);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);
    ret = aclmdlSetAIPPInputFormat(aippDynamicSet, ACL_YUV420SP_U8);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPCscParams(aippDynamicSet, 1, 256, 443, 0, 256, -86, -178, 256, 0, 350, 0, 0, 0, 0, 128, 128);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetAIPPSrcImageSize(aippDynamicSet, 224, 224);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAippType(_, _,_,_))
        .WillRepeatedly(Invoke(GetAippTypeSuccessInvoke));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), SetDynamicAippData(_, _,_,_, _))
        .WillRepeatedly(Return(FAILED));

    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_NE(ret, ACL_SUCCESS);
    ret = aclmdlDestroyAIPP(aippDynamicSet);
    EXPECT_EQ(ret, ACL_SUCCESS);
    aclDestroyDataBuffer(buffer);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetAIPPByInputIndex_fail1)
{
    aclmdlAIPP *aippmdlAipp = aclmdlCreateAIPP(0);
    aclError ret = aclmdlSetAIPPInputFormat(aippmdlAipp, ACL_YUV420SP_U8);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
            .WillRepeatedly(Invoke((GetModelDescInfo_Invoke2)));
    ret = aclmdlSetAIPPByInputIndex(1, dataset, 6, aippDynamicSet);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlDestroyAIPP(aippDynamicSet);
    EXPECT_EQ(ret, ACL_SUCCESS);
    aclDestroyDataBuffer(buffer);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetAIPPByInputIndex_fail2)
{
    aclmdlAIPP *aippmdlAipp = aclmdlCreateAIPP(0);
    aclError ret = aclmdlSetAIPPInputFormat(aippmdlAipp, ACL_YUV420SP_U8);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
            .WillRepeatedly(Invoke((GetModelDescInfo_Invoke2)));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAippType(_, _,_,_))
        .WillRepeatedly(Return(ACL_ERROR_FAILURE));
    ret = aclmdlSetAIPPByInputIndex(1, dataset, 0, aippDynamicSet);
    EXPECT_EQ(ret, ACL_ERROR_FAILURE);
    ret = aclmdlDestroyAIPP(aippDynamicSet);
    EXPECT_EQ(ret, ACL_SUCCESS);
    aclDestroyDataBuffer(buffer);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetAIPPByInputIndex_fail3)
{
    aclmdlAIPP *aippmdlAipp = aclmdlCreateAIPP(0);
    aclError ret = aclmdlSetAIPPInputFormat(aippmdlAipp, ACL_YUV420SP_U8);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAippType(_, _,_,_))
        .WillRepeatedly(Invoke((GetAippTypeFailInvoke)));

    ret = aclmdlSetAIPPByInputIndex(1, dataset, 0, aippDynamicSet);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
            .WillRepeatedly(Invoke((GetModelDescInfo_Invoke2)));
    ret = aclmdlSetAIPPByInputIndex(1, dataset, 0, aippDynamicSet);
    EXPECT_EQ(ret, ACL_ERROR_GE_FAILURE);
    ret = aclmdlDestroyAIPP(aippDynamicSet);
    EXPECT_EQ(ret, ACL_SUCCESS);
    aclDestroyDataBuffer(buffer);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetAIPPByInputIndex_SUCCESS)
{
    aclmdlAIPP *aippmdlAipp = aclmdlCreateAIPP(0);
    EXPECT_EQ(aippmdlAipp, nullptr);
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    auto ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke(GetModelDescInfo_Invoke2));
    ret = aclmdlSetAIPPByInputIndex(1, dataset, 0, aippDynamicSet);
    EXPECT_EQ(ret, ACL_ERROR_FAILURE);
    ret = aclmdlDestroyAIPP(aippDynamicSet);
    EXPECT_EQ(ret, ACL_SUCCESS);
    aclDestroyDataBuffer(buffer);
    aclmdlDestroyDataset(dataset);
}

aclError aclmdlGetInputIndexByName_Invoke(const aclmdlDesc *modelDesc, const char *name, size_t *index)
{
    (void) modelDesc;
    (void) name;
    *index = 0;
    return ACL_SUCCESS;
}

TEST_F(UTEST_ACL_Model, aclmdlSetInputAIPP_Fail)
{
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);
    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    aclError ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_,_,_,_))
        .WillOnce(Return(ACL_ERROR_INVALID_PARAM))
        .WillRepeatedly(Return(ACL_SUCCESS));
    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_EQ(ret, ACL_ERROR_GE_FAILURE);
    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    ret = aclmdlDestroyAIPP(aippDynamicSet);
    EXPECT_EQ(ret, ACL_SUCCESS);
    aclDestroyDataBuffer(buffer);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetInputAIPP_Fail1)
{
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    auto ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAippType(_, _,_,_))
        .WillRepeatedly(Return(FAILED));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
            .WillRepeatedly(Invoke((GetModelDescInfo_Invoke2)));
    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_NE(ret, ACL_SUCCESS);
    ret = aclmdlDestroyAIPP(aippDynamicSet);
    EXPECT_EQ(ret, ACL_SUCCESS);
    aclDestroyDataBuffer(buffer);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetInputAIPP_Fail2)
{
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    auto ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
            .WillRepeatedly(Invoke((GetModelDescInfo_Invoke2)));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAippType(_, _,_,_))
        .WillOnce(Invoke(GetAippTypeFailInvoke))
        .WillRepeatedly(Invoke(GetAippTypeStaticAippInvoke));

    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_NE(ret, ACL_SUCCESS);
    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_NE(ret, ACL_SUCCESS);
    ret = aclmdlDestroyAIPP(aippDynamicSet);
    EXPECT_EQ(ret, ACL_SUCCESS);
    aclDestroyDataBuffer(buffer);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetInputAIPP_Fail3)
{
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    auto ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAippType(_, _,_,_))
        .WillRepeatedly(Invoke(GetAippTypeNoAippInvoke));
    ret = aclmdlSetInputAIPP(1, dataset, 6, aippDynamicSet);
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = aclmdlDestroyAIPP(aippDynamicSet);
    EXPECT_EQ(ret, ACL_SUCCESS);
    aclDestroyDataBuffer(buffer);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlGetFirstAippInfo)
{
    aclAippInfo aippInfo;
    aclError ret = aclmdlGetFirstAippInfo(1, 0, &aippInfo);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
            .WillRepeatedly(Invoke((GetModelDescInfo_Invoke2)));
    ret = aclmdlGetFirstAippInfo(1, 0, &aippInfo);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAIPPInfo(_, _, _))
    .WillOnce(Return(ACL_ERROR_GE_AIPP_NOT_EXIST));
    ret = aclmdlGetFirstAippInfo(1, 0, &aippInfo);
    EXPECT_EQ(ret, ACL_ERROR_GE_AIPP_NOT_EXIST);
}

TEST_F(UTEST_ACL_Model, AippParamsCheck)
{
    aclError ret;
    uint32_t batchNumber = 2;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    //InputFormat not setted
    (void)GetSrcImageSize(aippDynamicSet);

    //aipp not support Ascend910
    std::string socVersion = "Ascend910";
    (void)aclmdlSetAIPPInputFormat(aippDynamicSet, ACL_ARGB8888_U8);
    (void)GetSrcImageSize(aippDynamicSet);
    ret = AippParamsCheck(aippDynamicSet, socVersion);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    //ES,YUV420SP_U8, src_image_w must be multiples of 16
    (void)aclmdlSetAIPPSrcImageSize(aippDynamicSet, 18, 224);
    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    //enable scf, disable crop,scfInputSizeW==srcImageSizeW,scfInputSizeH==srcImageSizeH
    (void)aclmdlSetAIPPInputFormat(aippDynamicSet, ACL_YUV420SP_U8);
    (void)aclmdlSetAIPPSrcImageSize(aippDynamicSet, 224, 224);
    ret = aclmdlSetAIPPScfParams(aippDynamicSet, 1, 210, 210, 1, 1, 0);
    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    ret = aclmdlDestroyAIPP(aippDynamicSet);
    EXPECT_EQ(ret, ACL_SUCCESS);
    aclDestroyDataBuffer(buffer);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetInputAIPP_Check)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelAttr(_, _))
        .WillRepeatedly(Invoke(GetModelAttr_Invoke));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_,_,_,_))
        .WillRepeatedly(Invoke(GetModelDescInfo_Invoke));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetDynamicBatchInfo(_,_,_))
         .WillRepeatedly(Invoke(GetDynamicBatchInfo_Invoke5));
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);
    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    aclError ret = aclmdlAddDatasetBuffer(dataset, buffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAippType(_, _,_,_))
        .WillRepeatedly(Invoke(GetAippTypeNoAippInvoke));

    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlDestroyAIPP(aippDynamicSet);
    EXPECT_EQ(ret, ACL_SUCCESS);
    aclDestroyDataBuffer(buffer);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlCreateAndGetOpDesc)
{
    char opName[256];
    memset(opName, '\0', 256);
    aclTensorDesc *inputDesc = nullptr;
    aclTensorDesc *outputDesc = nullptr;
    size_t inputCnt = 0;
    size_t outputCnt = 0;
    aclError ret = aclmdlCreateAndGetOpDesc(0, 0, 0, opName, 256,  &inputDesc, &inputCnt, &outputDesc, &outputCnt);
    EXPECT_EQ(ret, ACL_SUCCESS);
    for (size_t i = 0; i < inputCnt; ++i) {
        (void)aclGetTensorDescByIndex(inputDesc, i);
    }
    for (size_t i = 0; i < outputCnt; ++i) {
        (void)aclGetTensorDescByIndex(outputDesc, i);
    }
    ret = aclmdlCreateAndGetOpDesc(0, 0, 0, opName, -1,  &inputDesc, &inputCnt, &outputDesc, &outputCnt);
    EXPECT_EQ(ret, ACL_ERROR_FAILURE);
    aclDestroyTensorDesc(inputDesc);
    aclDestroyTensorDesc(outputDesc);
}

TEST_F(UTEST_ACL_Model, aclGetTensorDescAddress)
{
    auto ret = aclGetTensorDescAddress(nullptr);
    EXPECT_EQ(ret, nullptr);
}

TEST_F(UTEST_ACL_Model, aclmdlSetExternalWeightAddress) {
    aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
    ASSERT_NE(handle, nullptr);
    std::string file_name = "fileconstant1.bin";
    size_t mem_size = 1024;
    uint32_t user_mem[1024];
    auto ret = aclmdlSetExternalWeightAddress(handle, file_name.c_str(), (void *)user_mem, mem_size);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetExternalWeightAddress(handle, file_name.c_str(), nullptr, mem_size);
    EXPECT_NE(ret, ACL_SUCCESS);
    ret = aclmdlSetExternalWeightAddress(handle, nullptr, (void *)user_mem, mem_size);
    EXPECT_NE(ret, ACL_SUCCESS);
    ret = aclmdlSetExternalWeightAddress(nullptr, file_name.c_str(), (void *)user_mem, mem_size);
    EXPECT_NE(ret, ACL_SUCCESS);
    ret = aclmdlSetExternalWeightAddress(handle, file_name.c_str(), (void *)user_mem, 0);
    EXPECT_NE(ret, ACL_SUCCESS);
    aclmdlDestroyConfigHandle(handle);
}

TEST_F(UTEST_ACL_Model, aclmdlLoadWithConfig_ExternalAddress)
{
    aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
    ASSERT_NE(handle, nullptr);
    std::string file_name1 = "fileconstant1.bin";
    size_t mem_size1 = 1024;
    uint32_t user_mem1[1024];
    auto ret = aclmdlSetExternalWeightAddress(handle, file_name1.c_str(), (void *)user_mem1, mem_size1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    std::string file_name2 = "fileconstant2.bin";
    size_t mem_size2 = 1024;
    uint32_t user_mem2[1024];
    ret = aclmdlSetExternalWeightAddress(handle, file_name2.c_str(), (void *)user_mem2, mem_size2);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ge::ModelFileHeader head;
    head.version = ge::MODEL_VERSION + 1U;
    head.model_num = 2U;
    void *p = (void *)&head;
    uint32_t modelId;
    size_t type = 99;

    ret = aclmdlSetConfigOpt(handle, ACL_MDL_MEM_ADDR_PTR, &p, sizeof(p));
    EXPECT_EQ(ret, ACL_SUCCESS);

    size_t modelSize1 = 1;
    size_t modelSize = sizeof(head);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_MEM_SIZET, &modelSize, sizeof(modelSize));
    EXPECT_EQ(ret, ACL_SUCCESS);
    const char *path = "/home";
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_PATH_PTR, &path, sizeof(path));
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WORKSPACE_ADDR_PTR, &p, sizeof(p));
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WORKSPACE_SIZET, &modelSize1, sizeof(modelSize1));
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WEIGHT_ADDR_PTR, &p, sizeof(p));
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WEIGHT_SIZET, &modelSize1, sizeof(modelSize1));
    EXPECT_EQ(ret, ACL_SUCCESS);
    size_t reuseMemory = 0;
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WORKSPACE_MEM_OPTIMIZE, &reuseMemory, sizeof(reuseMemory));
    EXPECT_EQ(ret, ACL_SUCCESS);
    reuseMemory = 0xFFFFFFFFFFFFFFFF;

    vector<uint32_t> inputQ(100);
    vector<uint32_t> outputQ(100);
    uint32_t *inputQPtr = inputQ.data();
    uint32_t *outputQPtr = outputQ.data();

    type = ACL_MDL_LOAD_FROM_FILE_WITH_Q;

    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_INPUTQ_ADDR_PTR, &inputQPtr, sizeof(inputQ.data()));
    EXPECT_EQ(ret, ACL_SUCCESS);
    size_t num = 100;
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_INPUTQ_NUM_SIZET, &num, sizeof(size_t));
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_OUTPUTQ_ADDR_PTR, &outputQPtr, sizeof(outputQ.data()));
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlSetConfigOpt(handle, ACL_MDL_OUTPUTQ_NUM_SIZET, &num, sizeof(size_t));
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    int32_t priority = 1;
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_PRIORITY_INT32, &priority, sizeof(priority));
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    type = ACL_MDL_LOAD_FROM_MEM;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    // rt2.0
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
        .WillRepeatedly(Invoke(LoadExecutorFromModelDataCheckFileConstantMemSuccess));
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;

    type = ACL_MDL_LOAD_FROM_MEM_WITH_MEM;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    // static
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    // rt2.0
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
    // static
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    type = ACL_MDL_LOAD_FROM_FILE_WITH_MEM;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    // rt2.0
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;

    type = ACL_MDL_LOAD_FROM_FILE;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    // rt2.0
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;

    type = ACL_MDL_LOAD_FROM_FILE_WITH_Q;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    type = ACL_MDL_LOAD_FROM_MEM_WITH_Q;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    type = ACL_MDL_LOAD_FROM_MEM;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    const char_t *weight_path = "weight_path";
    aclmdlSetConfigOpt(handle, ACL_MDL_WEIGHT_PATH_PTR, &weight_path, sizeof(weight_path));
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    type = ACL_MDL_LOAD_FROM_FILE;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    const char *om_path = "/home";
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_PATH_PTR, &om_path, sizeof(om_path));
    EXPECT_EQ(ret, ACL_SUCCESS);

    type = ACL_MDL_LOAD_FROM_MEM_WITH_MEM;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_NE(ret, ACL_SUCCESS);

    type = ACL_MDL_LOAD_FROM_MEM;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    // rt2.0
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
        .WillRepeatedly(Invoke(LoadExecutorFromModelDataSuccess));
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
    aclmdlDestroyConfigHandle(handle);
}

TEST_F(UTEST_ACL_Model, aclmdlLoadWithConfig)
{
    aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
    ge::ModelFileHeader head;
    head.version = ge::MODEL_VERSION + 1U;
    head.model_num = 2U;
    void *p = (void *)&head;
    uint32_t modelId;
    aclError ret;
    size_t invalid_num = 99;
    size_t type = 99;
    ret = aclmdlSetConfigOpt(handle, *(aclmdlConfigAttr *)(&type), &type, sizeof(type));
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    ret = aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    ret = aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, invalid_num);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    type = ACL_MDL_LOAD_FROM_FILE;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));

    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    type = ACL_MDL_LOAD_FROM_MEM_WITH_MEM;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));

    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    ret = aclmdlSetConfigOpt(handle, ACL_MDL_MEM_ADDR_PTR, &p, sizeof(p));
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    size_t modelSize1 = 1;
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_MEM_SIZET, &modelSize1, 0);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    size_t modelSize = sizeof(head);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_MEM_SIZET, &modelSize, sizeof(modelSize));
    EXPECT_EQ(ret, ACL_SUCCESS);
    const char *path = "/home";
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_PATH_PTR, &path, 0);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_PATH_PTR, &path, sizeof(path));
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WORKSPACE_ADDR_PTR, &p, 0);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WORKSPACE_ADDR_PTR, &p, sizeof(p));
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WORKSPACE_SIZET, &modelSize1, 0);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WORKSPACE_SIZET, &modelSize1, sizeof(modelSize1));
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WEIGHT_ADDR_PTR, &p, 0);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WEIGHT_ADDR_PTR, &p, sizeof(p));
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WEIGHT_SIZET, &modelSize1, 0);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WEIGHT_SIZET, &modelSize1, sizeof(modelSize1));
    EXPECT_EQ(ret, ACL_SUCCESS);
    size_t reuseMemory = 0;
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WORKSPACE_MEM_OPTIMIZE, &reuseMemory, 0);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WORKSPACE_MEM_OPTIMIZE, &reuseMemory, sizeof(reuseMemory));
    EXPECT_EQ(ret, ACL_SUCCESS);
    reuseMemory = 0xFFFFFFFFFFFFFFFF;
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_WORKSPACE_MEM_OPTIMIZE, &reuseMemory, sizeof(reuseMemory));
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    vector<uint32_t> inputQ(100);
    vector<uint32_t> outputQ(100);
    uint32_t *inputQPtr = inputQ.data();
    uint32_t *outputQPtr = outputQ.data();

    type = ACL_MDL_LOAD_FROM_FILE_WITH_Q;

    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    ret = aclmdlSetConfigOpt(handle, ACL_MDL_INPUTQ_ADDR_PTR, &inputQPtr, 0);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_INPUTQ_ADDR_PTR, &inputQPtr, sizeof(inputQ.data()));
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    size_t num = 100;
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_INPUTQ_NUM_SIZET, &num, 0);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_INPUTQ_NUM_SIZET, &num, sizeof(size_t));
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_OUTPUTQ_ADDR_PTR, &outputQPtr, 0);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_OUTPUTQ_ADDR_PTR, &outputQPtr, sizeof(outputQ.data()));
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    ret = aclmdlSetConfigOpt(handle, ACL_MDL_OUTPUTQ_NUM_SIZET, &num, 0);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_OUTPUTQ_NUM_SIZET, &num, sizeof(size_t));
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    int32_t priority = 1;

    ret = aclmdlSetConfigOpt(handle, ACL_MDL_PRIORITY_INT32, &priority, sizeof(priority));
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_PRIORITY_INT32, &priority, 0);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    type = ACL_MDL_LOAD_FROM_MEM;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    // rt2.0
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
            .WillRepeatedly(Invoke(LoadExecutorFromModelDataSuccess));
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;

    type = ACL_MDL_LOAD_FROM_MEM_WITH_MEM;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    // static
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    // rt2.0
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
    // static
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    type = ACL_MDL_LOAD_FROM_FILE_WITH_MEM;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    // rt2.0
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;

    type = ACL_MDL_LOAD_FROM_FILE;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    // rt2.0
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;

    type = ACL_MDL_LOAD_FROM_FILE_WITH_Q;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    type = ACL_MDL_LOAD_FROM_MEM_WITH_Q;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    type = ACL_MDL_LOAD_FROM_MEM;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    const char_t *weight_path = "weight_path";
    aclmdlSetConfigOpt(handle, ACL_MDL_WEIGHT_PATH_PTR, &weight_path, sizeof(weight_path));
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    type = ACL_MDL_LOAD_FROM_FILE;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    const char *om_path = "/home";
    ret = aclmdlSetConfigOpt(handle, ACL_MDL_PATH_PTR, &om_path, sizeof(om_path));
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    type = ACL_MDL_LOAD_FROM_MEM_WITH_MEM;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_NE(ret, ACL_SUCCESS);

    type = ACL_MDL_LOAD_FROM_MEM;
    aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
    // rt2.0
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
            .WillRepeatedly(Invoke(LoadExecutorFromModelDataSuccess));
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    ret = aclmdlLoadWithConfig(handle, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
    aclmdlDestroyConfigHandle(handle);
}

TEST_F(UTEST_ACL_Model, aclmdlGetRealTensorName)
{
    aclmdlDesc *mdlDesc = aclmdlCreateDesc();
    aclmdlTensorDesc desc;
    desc.name = "dhsdhasiodhsaiodhsiashdisdhsiahdisahdisoahisahdihdisahdaoidhaihdsaihdsaihdsahdishaodhsiahihdoiahdsioadhisahdasidhsaidashdiaoiahdisohdosahdsahdiasoidashoidaoidhahdaoidahioadhiahdsahdiahdaiodaidahdhdahidahdaoda";
    // desc.dimsV2 = {1};
    // desc.size = 1;
    mdlDesc->inputDesc.push_back(desc);
    mdlDesc->outputDesc.push_back(desc);

    aclmdlTensorDesc desc1;
    desc1.name = "a6872_1bu_idc";
    mdlDesc->inputDesc.push_back(desc1);

    aclmdlIODims dims;
    aclmdlIODims dims1;
    auto ret = aclmdlGetInputDimsV2(mdlDesc, 0, &dims);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlGetInputDimsV2(mdlDesc, 1, &dims1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_STREQ(dims.name, "acl_modelId_0_input_0");
    EXPECT_STREQ(dims1.name, "a6872_1bu_idc");

    const char *str = aclmdlGetTensorRealName(mdlDesc, dims.name);
    EXPECT_STREQ(desc.name.c_str(), str);

    str = aclmdlGetTensorRealName(mdlDesc, dims1.name);
    EXPECT_STREQ(desc1.name.c_str(), str);

    str = aclmdlGetTensorRealName(mdlDesc, "xxxdwdsdasd");
    EXPECT_EQ(str, nullptr);

    str = aclmdlGetTensorRealName(mdlDesc, "acl_modelId_0_input_xxx");
    EXPECT_EQ(str, nullptr);

    str = aclmdlGetTensorRealName(mdlDesc, "modelId_0_input_xxx");
    EXPECT_EQ(str, nullptr);

    str = aclmdlGetTensorRealName(mdlDesc, "acl_modelId_xxx_input_0");
    EXPECT_EQ(str, nullptr);

    str = aclmdlGetTensorRealName(mdlDesc, "acl_modelId_0_input_0");
    EXPECT_EQ(str, desc.name.c_str());

    str = aclmdlGetTensorRealName(mdlDesc, "acl_modelId_0_output_0");
    EXPECT_EQ(str, desc.name.c_str());

    ret = aclmdlGetOutputDims(mdlDesc, 0, &dims);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_STREQ(dims.name, "acl_modelId_0_output_0");

    ret = aclmdlGetInputDims(mdlDesc, 0, &dims);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_STREQ(dims.name, "acl_modelId_0_input_0");

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetCurShape(_,_,_))
        .WillOnce(Return(SUCCESS));
    ret = aclmdlGetCurOutputDims(mdlDesc, 0, &dims);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_STREQ(dims.name, "acl_modelId_0_output_0");

    str = aclmdlGetTensorRealName(mdlDesc, "xxxx_modelId_0_output_0");
    EXPECT_EQ(str, nullptr);

    str = aclmdlGetTensorRealName(mdlDesc, "acl_modelId_100_input_0");
    EXPECT_EQ(str, nullptr);

    str = aclmdlGetTensorRealName(mdlDesc, "acl_modelId_0_input_100");
    EXPECT_EQ(str, nullptr);

    str = aclmdlGetTensorRealName(mdlDesc, "acl_modelId_0_output_100");
    EXPECT_EQ(str, nullptr);

    aclmdlDestroyDesc(mdlDesc);
}

TEST_F(UTEST_ACL_Model, aclmdlSetTensorDesc)
{
    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    aclError ret = aclmdlAddDatasetBuffer(dataset, buffer);
    aclDataBuffer *buffer1 = aclCreateDataBuffer((void*)0x2, 1);
    ret = aclmdlAddDatasetBuffer(dataset, buffer1);
    aclmdlDataset *datasetOut = aclmdlCreateDataset();
    aclDataBuffer *buffer2 = aclCreateDataBuffer((void*)0x3, 1);
    ret = aclmdlAddDatasetBuffer(datasetOut, buffer);

    int64_t shape[2] = {16, 32};
    aclTensorDesc *inputDesc = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    size_t index = 1;
    ret = aclmdlSetDatasetTensorDesc (dataset, inputDesc, index);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetDatasetTensorDesc (dataset, nullptr, 0);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecModel(_, _, _, _, _, _, _))
        .WillOnce(Invoke(ExecModelInvokeOneOut));

    ret = aclmdlExecute(1, dataset, datasetOut);
    EXPECT_EQ(ret, ACL_SUCCESS);

    size_t index1 = 2;
    ret = aclmdlSetDatasetTensorDesc (dataset, inputDesc, index1);
    EXPECT_NE(ret, ACL_SUCCESS);

    aclmdlDestroyDataset(dataset);    
    aclmdlDestroyDataset(datasetOut);
    aclDestroyDataBuffer(buffer);
    aclDestroyDataBuffer(buffer1);
    aclDestroyDataBuffer(buffer2);
    aclDestroyTensorDesc(inputDesc); 
}

/// TODO:
TEST_F(UTEST_ACL_Model, aclmdlExecuteSyncWithNullOutput)
{
    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void *)0x1, 1);
    aclError ret = aclmdlAddDatasetBuffer(dataset, buffer);
    aclDataBuffer *buffer1 = aclCreateDataBuffer((void *)0x2, 1);
    ret = aclmdlAddDatasetBuffer(dataset, buffer1);
    aclmdlDataset *datasetOut = aclmdlCreateDataset();
    aclDataBuffer *buffer2 = aclCreateDataBuffer(nullptr, 1);
    ret = aclmdlAddDatasetBuffer(datasetOut, buffer2);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _, _, _))
        .WillRepeatedly(Invoke(GetModelDescInfo_Invoke));
    int64_t shape[2] = {16, 32};
    aclTensorDesc *inputDesc = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    size_t index = 1;
    ret = aclmdlSetDatasetTensorDesc(dataset, inputDesc, index);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetDatasetTensorDesc(dataset, nullptr, 0);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecModel(_, _, _, _, _, _, _))
        .WillRepeatedly(Invoke(ExecModelInvokeOneOut));
    ret = aclmdlExecute(1, dataset, datasetOut);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_NE(aclGetDataBufferAddr(buffer2), nullptr);
    EXPECT_EQ(aclrtFree(aclGetDataBufferAddr(buffer2)), ACL_SUCCESS);

    size_t index1 = 2;
    ret = aclmdlSetDatasetTensorDesc(dataset, inputDesc, index1);
    EXPECT_NE(ret, ACL_SUCCESS);

    aclmdlDestroyDataset(dataset);
    aclmdlDestroyDataset(datasetOut);
    aclDestroyDataBuffer(buffer);
    aclDestroyDataBuffer(buffer1);
    aclDestroyDataBuffer(buffer2);
    aclDestroyTensorDesc(inputDesc);
}

TEST_F(UTEST_ACL_Model, aclmdlGetDatasetTensorDesc)
{
    aclmdlDataset *dataset = aclmdlCreateDataset();
    aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
    aclError ret = aclmdlAddDatasetBuffer(dataset, buffer);
    aclDataBuffer *buffer1 = aclCreateDataBuffer((void*)0x2, 1);
    ret = aclmdlAddDatasetBuffer(dataset, buffer1);
    aclmdlDataset *datasetOut = aclmdlCreateDataset();
    aclDataBuffer *buffer2 = aclCreateDataBuffer((void*)0x3, 1);
    ret = aclmdlAddDatasetBuffer(datasetOut, buffer);
    aclDataBuffer *buffer3 = aclCreateDataBuffer((void*)0x4, 1);
    ret = aclmdlAddDatasetBuffer(datasetOut, buffer);

    int64_t shape[2] = {16, 32};
    aclTensorDesc *inputDesc = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    size_t index = 1;
    ret = aclmdlSetDatasetTensorDesc (dataset, inputDesc, index);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlSetDatasetTensorDesc (dataset, nullptr, 0);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecModel(_,_,_,_,_,_,_))
        .WillOnce(Invoke(ExecModelInvoke));
    ret = aclmdlExecute(1, dataset, datasetOut);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclTensorDesc *outputDesc = aclmdlGetDatasetTensorDesc(nullptr, 0);
    EXPECT_EQ(outputDesc, nullptr);

    outputDesc = aclmdlGetDatasetTensorDesc(datasetOut, 2);
    EXPECT_EQ(outputDesc, nullptr);

    outputDesc = aclmdlGetDatasetTensorDesc(datasetOut, 1);
    EXPECT_NE(outputDesc, nullptr);

    aclmdlDestroyDataset(dataset);    
    aclmdlDestroyDataset(datasetOut);
    aclDestroyDataBuffer(buffer);
    aclDestroyDataBuffer(buffer1);
    aclDestroyDataBuffer(buffer2);
    aclDestroyDataBuffer(buffer3);
    aclDestroyTensorDesc(inputDesc);
}

TEST_F(UTEST_ACL_Model, aclmdlGetRealTensorName2)
{
    aclmdlDesc *mdlDesc = aclmdlCreateDesc();
    aclmdlTensorDesc desc;
    desc.name = "dhsdhasiodhsaiodhsiashdisdhsiahdisahdisoahisahdihdisahdaoidhaihdsaihdsaihdsahdishaodhsiahihdoiahdsioadhisahdasidhsaidashdiaoiahdisohdosahdsahdiasoidashoidaoidhahdaoidahioadhiahdsahdiahdaiodaidahdhdahidahdaoda";

    mdlDesc->inputDesc.push_back(desc);
    mdlDesc->outputDesc.push_back(desc);

    aclmdlTensorDesc desc1;
    desc1.name = "acl_modelId_0_input_0";
    mdlDesc->inputDesc.push_back(desc1);

    aclmdlIODims dims;
    auto ret = aclmdlGetInputDimsV2(mdlDesc, 0, &dims);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_STREQ(dims.name, "acl_modelId_0_input_0_a");

    const char *realName = aclmdlGetTensorRealName(mdlDesc, dims.name);
    EXPECT_STREQ(realName, desc.name.c_str());

    desc1.name = "acl_modelId_0_input_0_a";
    mdlDesc->inputDesc.push_back(desc1);
    ret = aclmdlGetInputDimsV2(mdlDesc, 0, &dims);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_STREQ(dims.name, "acl_modelId_0_input_0_b");

    realName = aclmdlGetTensorRealName(mdlDesc, dims.name);
    EXPECT_STREQ(realName, desc.name.c_str());

    aclmdlDestroyDesc(mdlDesc);
}

TEST_F(UTEST_ACL_Model, aclmdlGetInputSizeByIndex2)
{
    aclmdlDesc *mdlDesc = aclmdlCreateDesc();
    aclmdlTensorDesc desc;
    desc.dims.push_back(-1);
    desc.dims.push_back(2);
    desc.shapeRanges.push_back(std::make_pair(1, 3));
    desc.dataType = ACL_FLOAT16;
    mdlDesc->inputDesc.push_back(desc);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclDataTypeSize(_))
        .WillOnce(Return((sizeof(int16_t))));
    size_t size = aclmdlGetInputSizeByIndex(mdlDesc, 0);
    EXPECT_NE(size, 0);

    size = aclmdlGetInputSizeByIndex(mdlDesc, 2);
    EXPECT_EQ(size, 0);
    aclmdlTensorDesc desc1;
    desc1.dims.push_back(-1);
    desc1.dims.push_back(2);
    desc1.shapeRanges.push_back(std::make_pair(-1, -1));
    mdlDesc->inputDesc.push_back(desc1);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclDataTypeSize(_))
        .WillOnce(Return((sizeof(int16_t))));
    size = aclmdlGetInputSizeByIndex(mdlDesc, 1);
    EXPECT_EQ(size, 0);

    aclmdlDestroyDesc(mdlDesc);
}

TEST_F(UTEST_ACL_Model, aclGetDataBufferSize)
{
    aclDataBuffer *dataBuffer = nullptr;
    EXPECT_EQ(aclGetDataBufferSize(dataBuffer), 0);

    dataBuffer = aclCreateDataBuffer((void*)0x1, 1);
    EXPECT_NE(aclGetDataBufferSize(dataBuffer), 0);
    aclDestroyDataBuffer(dataBuffer);
}

TEST_F(UTEST_ACL_Model, aclmdlSetAIPPPixelVarReciTest)
{
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);
    aclError ret = aclmdlSetAIPPPixelVarReci(aippDynamicSet, 1, 1, 1, 0, 3);
    EXPECT_NE(ret, ACL_SUCCESS);
    aclmdlDestroyAIPP(aippDynamicSet);
}

TEST_F(UTEST_ACL_Model, aclmdlSetInputAIPPTest01)
{
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);
    aclmdlDataset *dataset = aclmdlCreateDataset();
    aippDynamicSet->aippParms.inputFormat = CCE_YUV400_U8;
    aippDynamicSet->aippParms.srcImageSizeW = 1;
    aippDynamicSet->aippParms.srcImageSizeH = 511373560;
    aippDynamicSet->batchSize = 1;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAippType(_, _,_,_))
        .WillRepeatedly(Invoke(GetAippTypeSuccessInvoke));
    aclError ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_NE(ret, ACL_SUCCESS);

    aippDynamicSet->aippParms.srcImageSizeH = 1;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetOrigInputInfo(_, _, _))
        .WillOnce(Return(FAILED));
    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_NE(ret, ACL_SUCCESS);
    aclmdlDestroyAIPP(aippDynamicSet);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetInputAIPPTest02)
{
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);
    aippDynamicSet->aippParms.inputFormat = CCE_YUV420SP_U8;
    aippDynamicSet->aippParms.srcImageSizeW = 1;
    aippDynamicSet->aippParms.srcImageSizeH = 2;
    aclmdlDataset *dataset = aclmdlCreateDataset();
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAippType(_, _,_,_))
        .WillRepeatedly(Invoke(GetAippTypeSuccessInvoke));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAllAippInputOutputDims(_, _, _, _))
        .WillOnce(Return(FAILED));
    aclError ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_NE(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAllAippInputOutputDims(_, _, _, _))
        .WillOnce(Return(SUCCESS));
    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_NE(ret, ACL_SUCCESS);
    aclmdlDestroyAIPP(aippDynamicSet);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetInputAIPPTest03)
{
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);
    aclmdlDataset *dataset = aclmdlCreateDataset();
    aippDynamicSet->aippParms.inputFormat = CCE_YUV400_U8;
    aippDynamicSet->aippParms.srcImageSizeW = 1;
    aippDynamicSet->aippParms.srcImageSizeH = 511373560;
    aippDynamicSet->batchSize = 1;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAippType(_, _,_,_))
        .WillRepeatedly(Invoke(GetAippTypeSuccessInvoke));
    aclError ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_NE(ret, ACL_SUCCESS);

    aippDynamicSet->aippParms.srcImageSizeH = 1;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetOrigInputInfo(_, _, _))
        .WillOnce(Return(FAILED));
    ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_NE(ret, ACL_SUCCESS);
    aclmdlDestroyAIPP(aippDynamicSet);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetInputAIPPTest04)
{
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);
    aclmdlDataset *dataset = aclmdlCreateDataset();
    aippDynamicSet->aippParms.inputFormat = CCE_YUV400_U8;
    aippDynamicSet->aippParms.srcImageSizeW = 1;
    aippDynamicSet->aippParms.srcImageSizeH = 1;
    aippDynamicSet->batchSize = 1;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAippType(_, _,_,_))
        .WillRepeatedly(Invoke(GetAippTypeSuccessInvoke));

    aclError ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_NE(ret, ACL_SUCCESS);

    aclmdlDestroyAIPP(aippDynamicSet);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetInputAIPPWithDynamicShapeTest)
{
    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);
    aippDynamicSet->aippParms.inputFormat = CCE_YUV420SP_U8;
    aippDynamicSet->aippParms.srcImageSizeW = 1;
    aippDynamicSet->aippParms.srcImageSizeH = 2;
    aclmdlDataset *dataset = aclmdlCreateDataset();

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillRepeatedly(Invoke((GetModelDescInfo_Invoke)));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetShapeRange(_))
        .WillRepeatedly(Invoke((GetShapeRange_Invoke)));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAippType(_, _,_,_))
        .WillRepeatedly(Invoke(GetAippTypeSuccessInvoke));

    aclError ret = aclmdlSetInputAIPP(1, dataset, 0, aippDynamicSet);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    aclmdlDestroyAIPP(aippDynamicSet);
    aclmdlDestroyDataset(dataset);
}

TEST_F(UTEST_ACL_Model, aclmdlSetInputAIPPWithDynamicShapeTestRT2)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    auto executor = std::unique_ptr<gert::ModelV2Executor>(new(std::nothrow) gert::ModelV2Executor);
    auto rtSession = acl::AclResourceManager::GetInstance().CreateRtSession();
    uint32_t modelId = 1;
    acl::AclResourceManager::GetInstance().AddExecutor(modelId, std::move(executor), rtSession);

    uint32_t batchNumber = 1;
    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(batchNumber);
    aippDynamicSet->aippParms.inputFormat = CCE_YUV420SP_U8;
    aippDynamicSet->aippParms.srcImageSizeW = 1;
    aippDynamicSet->aippParms.srcImageSizeH = 2;
    aclmdlDataset *dataset = aclmdlCreateDataset();

    aclError ret = aclmdlSetInputAIPP(modelId, dataset, 0, aippDynamicSet);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    aclmdlDestroyAIPP(aippDynamicSet);
    aclmdlDestroyDataset(dataset);
    acl::AclResourceManager::GetInstance().DeleteExecutor(modelId);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, aclmdlGetFirstAippInfoTest)
{
    uint32_t modelId = 0;
    size_t index = 0;
    aclAippInfo aippInfo;

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
            .WillRepeatedly(Invoke((GetModelDescInfo_Invoke2)));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAIPPInfo(_, _, _))
        .WillOnce(Return(FAILED));
    aclError ret = aclmdlGetFirstAippInfo(modelId, index, &aippInfo);
    EXPECT_NE(ret, ACL_SUCCESS);
    Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
            .WillRepeatedly(Invoke((GetModelDescInfo_Invoke2)));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAIPPInfo(_, _, _))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetBatchInfoSize(_, _))
        .WillOnce(Return(FAILED));
    ret = aclmdlGetFirstAippInfo(modelId, index, &aippInfo);
    EXPECT_NE(ret, ACL_SUCCESS);
    Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
            .WillRepeatedly(Invoke((GetModelDescInfo_Invoke2)));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAIPPInfo(_, _, _))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetBatchInfoSize(_, _))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetOrigInputInfo(_, _, _))
        .WillOnce(Return(FAILED));
    ret = aclmdlGetFirstAippInfo(modelId, index, &aippInfo);
    EXPECT_NE(ret, ACL_SUCCESS);
    Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
            .WillRepeatedly(Invoke((GetModelDescInfo_Invoke2)));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAIPPInfo(_, _, _))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetBatchInfoSize(_, _))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAllAippInputOutputDims(_, _, _, _))
        .WillOnce(Return(FAILED));
    ret = aclmdlGetFirstAippInfo(modelId, index, &aippInfo);
    EXPECT_NE(ret, ACL_SUCCESS);
    Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
            .WillRepeatedly(Invoke((GetModelDescInfo_Invoke2)));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAIPPInfo(_, _, _))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetBatchInfoSize(_, _))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAllAippInputOutputDims(_, _, _, _))
        .WillOnce(Return(SUCCESS));
    ret = aclmdlGetFirstAippInfo(modelId, index, &aippInfo);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

Status GetAllAippInputOutputDims_invoke(uint32_t index,
                                        std::vector<InputOutputDims> &input_dims,
                                        std::vector<InputOutputDims> &output_dims)
{
    (void) index;
    InputOutputDims fake_dim{};
    fake_dim.name = "hello";
    fake_dim.dim_num = 4;
    fake_dim.size = 4;
    fake_dim.dims = std::vector<int64_t>{1,1,1,1};
    input_dims.push_back(fake_dim);
    output_dims.push_back(fake_dim);
    return SUCCESS;
}

Status GetOriginAippInputInfo_invoke( uint32_t index, OriginInputInfo &origOutputInfo)
{
    (void) index;
    origOutputInfo.format = ge::FORMAT_NCHW;
    origOutputInfo.data_type = ge::DT_FLOAT;
    origOutputInfo.dim_num = 4;
    return SUCCESS;
}

TEST_F(UTEST_ACL_Model, aclmdlGetFirstAippInfoTest_rt2)
{
    uint32_t modelId = 999;
    size_t index = 0;
    aclAippInfo aippInfo{};
    bool flag = acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_;
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    auto executor = std::unique_ptr<gert::ModelV2Executor>(new(std::nothrow) gert::ModelV2Executor);
    auto rtSession = acl::AclResourceManager::GetInstance().CreateRtSession();
    acl::AclResourceManager::GetInstance().AddExecutor(modelId, std::move(executor), rtSession);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAippInfo(_, _))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetOriginAippInputInfo(_, _))
        .WillOnce(Invoke(GetOriginAippInputInfo_invoke));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAllAippInputOutputDims(_, _, _))
        .WillOnce(Invoke(GetAllAippInputOutputDims_invoke));
    auto ret = aclmdlGetFirstAippInfo(modelId, index, &aippInfo);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(aippInfo.srcFormat, ACL_FORMAT_NCHW);
    EXPECT_EQ(aippInfo.srcDatatype, ACL_FLOAT);
    EXPECT_EQ(aippInfo.srcDimNum, 4);
    EXPECT_EQ(aippInfo.shapeCount, 1);
    EXPECT_EQ(std::string(aippInfo.outDims[0].srcDims.name), "hello");
    EXPECT_EQ(aippInfo.outDims[0].srcDims.dimCount, 4);
    EXPECT_EQ(aippInfo.outDims[0].srcDims.dims[2], 1);
    EXPECT_EQ(std::string(aippInfo.outDims[0].aippOutdims.name), "hello");
    EXPECT_EQ(aippInfo.outDims[0].aippOutdims.dimCount, 4);
    EXPECT_EQ(aippInfo.outDims[0].aippOutdims.dims[2], 1);
    acl::AclResourceManager::GetInstance().DeleteExecutor(modelId);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = flag;
}

TEST_F(UTEST_ACL_Model, AippScfSizeCheckTest)
{
    uint32_t batchNumber = 10;
    aclmdlAIPP *aippParmsSet = aclmdlCreateAIPP(batchNumber);
    int32_t batchIndex = 0;
    aclError ret = AippScfSizeCheck(aippParmsSet, batchIndex);
    EXPECT_NE(ret, ACL_SUCCESS);
    aclmdlDestroyAIPP(aippParmsSet);
}

TEST_F(UTEST_ACL_Model, RuntimeV2UnloadModel)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), UnloadModel(_))
        .WillOnce(Return(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID))
        .WillRepeatedly(Return(SUCCESS));
    uint32_t modelId = 999;
    auto ret = aclmdlUnload(modelId);
    EXPECT_EQ(ret, static_cast<aclError>(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID));

    acl::AclResourceManager::GetInstance().DeleteExecutor(modelId);
    const char *modelPath = "/";

    ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlUnload(modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = acl::AclResourceManager::GetInstance().DeleteExecutor(2);
    EXPECT_EQ(ret, static_cast<aclError>(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID));
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, RuntimeV2ExecuteModel)
{ 
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    
    aclmdlDataset *dataset = aclmdlCreateDataset();
    EXPECT_NE(dataset, nullptr);

    aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
    aclError ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_,_,_,_))
        .WillRepeatedly(Invoke(GetModelDescInfo_Invoke));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecModel(_,_,_,_,_,_,_))
        .WillOnce(Return(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID))
        .WillRepeatedly(Return(SUCCESS));
    ret = aclmdlExecute(1, dataset, dataset);
    EXPECT_EQ(ret, static_cast<aclError>(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID));

    uint32_t modelId = 0;
    const char *modelPath = "/";
    ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecModel(_, _, _, _, _, _, _))
        .WillOnce(Invoke(ExecModelInvokeOneOut));
    ret = aclmdlExecute(modelId, dataset, dataset);
    EXPECT_EQ(ret, ACL_SUCCESS);

    free(dataBuffer);
    ret = aclmdlDestroyDataset(dataset);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, RuntimeV2ExecuteModelFailed)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    auto executor = std::unique_ptr<gert::ModelV2Executor>(new(std::nothrow) gert::ModelV2Executor);
    uint32_t modelId = 0x1;
    auto rtSession = acl::AclResourceManager::GetInstance().CreateRtSession();
    acl::AclResourceManager::GetInstance().AddExecutor(modelId, std::move(executor), rtSession);
    aclmdlDataset *dataset = aclmdlCreateDataset();
    EXPECT_NE(dataset, nullptr);

    auto ret = aclmdlExecute(modelId, dataset, dataset);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
    ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), IsOriginShapeInRange(_))
        .WillOnce(Return(false))
        .WillRepeatedly(Return(true));
    ret = aclmdlExecute(modelId, dataset, dataset);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    ret = aclmdlExecute(modelId, dataset, dataset);
    EXPECT_EQ(ret, ACL_SUCCESS);

    free(dataBuffer);
    ret = aclmdlDestroyDataset(dataset);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = acl::AclResourceManager::GetInstance().DeleteExecutor(modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, RuntimeV2GetDesc)
{ 
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    aclmdlDesc* desc = aclmdlCreateDesc();
    EXPECT_NE(desc, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _,_,_))
        .WillOnce(Return(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID))
        .WillRepeatedly(Return(SUCCESS));
    auto ret = aclmdlGetDesc(desc, 1);

    EXPECT_EQ(ret, static_cast<aclError>(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID));

    uint32_t modelId = 0;
    const char *modelPath = "/";
    ret = aclmdlLoadFromFile(modelPath, &modelId);

    auto executor = std::unique_ptr<gert::ModelV2Executor>(new(std::nothrow) gert::ModelV2Executor);
    auto rtSession = acl::AclResourceManager::GetInstance().CreateRtSession();
    acl::AclResourceManager::GetInstance().AddExecutor(modelId, std::move(executor), rtSession);
    ret = aclmdlGetDesc(desc, modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlDestroyDesc(desc);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, RuntimeV2EsecuteAsync)
{ 
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    aclmdlDataset *dataset = aclmdlCreateDataset();
    EXPECT_NE(dataset, nullptr);

    aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
    aclError ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    int64_t shape[2] = {16, 32};
    aclTensorDesc *inputDesc = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    size_t index = 0;
    ret = aclmdlSetDatasetTensorDesc (dataset, inputDesc, index);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecModel(_,_,_,_,_,_,_))
        .WillOnce(Return((ACL_ERROR_GE_EXEC_MODEL_ID_INVALID)))
        .WillRepeatedly(Return(SUCCESS));
    ret = aclmdlExecuteAsync(1, dataset, dataset, nullptr);
    EXPECT_EQ(ret, static_cast<aclError>(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), IsDynamicModel(_,_))
        .WillOnce(Invoke((IsDynamicModelReturnTrue)));
    uint32_t modelId = 0;
    const char *modelPath = "/";
    ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecModel(_, _, _, _, _, _, _))
        .WillOnce(Invoke(ExecModelInvokeOneOut));
    ret = aclmdlExecuteAsync(modelId, dataset, dataset, nullptr);
    EXPECT_EQ(ret, ACL_SUCCESS);

    free(dataBuffer);
    aclDestroyTensorDesc(inputDesc);
    ret = aclmdlDestroyDataset(dataset);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, RuntimeV2ExecuteAsync_Ok_ExecuteAsyncDynamic)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    aclmdlDataset *dataset = aclmdlCreateDataset();
    EXPECT_NE(dataset, nullptr);

    aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
    aclError ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    int64_t shape[2] = {16, 32};
    aclTensorDesc *inputDesc = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    size_t index = 0;
    ret = aclmdlSetDatasetTensorDesc (dataset, inputDesc, index);

    uint32_t modelId = 0;
    const char *modelPath = "/";
    ret = aclmdlLoadFromFile(modelPath, &modelId);

    auto executor = std::unique_ptr<gert::ModelV2Executor>(new(std::nothrow) gert::ModelV2Executor);
    auto rtSession = acl::AclResourceManager::GetInstance().CreateRtSession();
    acl::AclResourceManager::GetInstance().AddExecutor(modelId, std::move(executor), rtSession);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), IsOriginShapeInRange(_)).WillRepeatedly(Return(true));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Execute(_, _, _, _, _)).WillOnce(Return(ge::GRAPH_SUCCESS));
    ret = aclmdlExecuteAsync(modelId, dataset, dataset, nullptr);
    EXPECT_EQ(ret, ACL_SUCCESS);

    free(dataBuffer);
    aclDestroyTensorDesc(inputDesc);
    ret = aclmdlDestroyDataset(dataset);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = acl::AclResourceManager::GetInstance().DeleteExecutor(modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}


TEST_F(UTEST_ACL_Model, RuntimeV2ExecuteWithNullOutput)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;

    uint32_t modelId = 1;
    const char *modelPath = "/";
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
    .WillOnce(Invoke(LoadExecutorFromModelDataSuccess));
    auto ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), IsDynamicModel(_,_))
    .WillOnce(Invoke((IsDynamicModelReturnTrue)));
    ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    EXPECT_NE(dataset, nullptr);

    aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
    ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclmdlDataset *datasetOut = aclmdlCreateDataset();
    EXPECT_NE(dataset, nullptr);
    aclDataBuffer *buffer2 = aclCreateDataBuffer(nullptr, 1);
    EXPECT_EQ(aclGetDataBufferAddr(buffer2), nullptr);
    ret = aclmdlAddDatasetBuffer(datasetOut, buffer2);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _, _, _))
        .WillRepeatedly(Invoke(GetModelDescInfo_Invoke));
    int64_t shape[2] = {16, 32};
    aclTensorDesc *inputDesc = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    size_t index = 0;
    ret = aclmdlSetDatasetTensorDesc(dataset, inputDesc, index);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecuteSync(_, _, _, _))
        .WillRepeatedly(Invoke(ExecuteSync_Invoke));

    EXPECT_EQ(aclGetDataBufferAddr(buffer2), nullptr);
    EXPECT_EQ(aclGetDataBufferAddr(datasetOut->blobs[0].dataBuf), nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), IsOriginShapeInRange(_)).WillRepeatedly(Return(true));

    ret = aclmdlExecute(modelId, dataset, datasetOut);
    EXPECT_EQ(ret, ACL_SUCCESS);
    /// TODO: rtMalloc
    /// TODO: rtFree
    // EXPECT_NE(aclGetDataBufferAddr(datasetOut->blobs[0].dataBuf), nullptr);
    EXPECT_EQ(aclrtFree(aclGetDataBufferAddr(datasetOut->blobs[0].dataBuf)), ACL_SUCCESS);

    free(dataBuffer);
    aclDestroyTensorDesc(inputDesc);
    aclDestroyDataBuffer(buffer2);
    ret = aclmdlDestroyDataset(dataset);
    ret = aclmdlDestroyDataset(datasetOut);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, RuntimeV2LoadFromFile)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;

    uint32_t modelId = 1;
    const char *modelPath = "/";
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
            .WillOnce(Invoke(LoadExecutorFromModelDataSuccess));
    auto ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), IsDynamicModel(_,_))
        .WillOnce(Invoke((IsDynamicModelReturnTrue)));
    ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, RuntimeV2LoadFromFileFailed)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    uint32_t modelId = 1;
    const char *modelPath = "/";
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
        .WillRepeatedly(Invoke(LoadExecutorFromModelDataSuccess));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), IsDynamicModel(_,_))
        .WillRepeatedly(Invoke((IsDynamicModelReturnTrue)));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtCreateStreamImpl(_))
        .WillOnce(Return(ACL_ERROR_RT_PARAM_INVALID))
        .WillOnce(Return(RT_ERROR_NONE));
    auto ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_EQ(ret, ACL_ERROR_RT_PARAM_INVALID);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtSynchronizeStreamImpl(_))
        .WillOnce(Return((ACL_ERROR_RT_PARAM_INVALID)));
    ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_EQ(ret, ACL_ERROR_RT_PARAM_INVALID);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, RuntimeV2LoadFromMemFailed)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    uint32_t modelId = 1;
    ge::ModelFileHeader head;
    head.version = ge::MODEL_VERSION + 1U;
    head.model_num = 2U;
    void *model = (void *)&head;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
        .WillRepeatedly(Invoke(LoadExecutorFromModelDataSuccess));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtCreateStreamImpl(_))
        .WillOnce(Return(ACL_ERROR_RT_PARAM_INVALID))
        .WillOnce(Return(RT_ERROR_NONE));
    auto ret = aclmdlLoadFromMem(model, sizeof(head), &modelId);
    EXPECT_EQ(ret, ACL_ERROR_RT_PARAM_INVALID);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtSynchronizeStreamImpl(_))
        .WillOnce(Return((ACL_ERROR_RT_PARAM_INVALID)));
    ret = aclmdlLoadFromMem(model, sizeof(head), &modelId);
    EXPECT_EQ(ret, ACL_ERROR_RT_PARAM_INVALID);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, RuntimeV2LoadFromFileWithDynamicModelFailed)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;

    uint32_t modelId = 1;
    const char *modelPath = "/";

    auto ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), IsDynamicModel(_,_))
      .WillOnce(Invoke((IsDynamicModelReturnFailed)));
    ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, aclmdlSetExecConfigOpt_invalid_timeout_failed)
{
    size_t invalid_num = 99;
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
    EXPECT_NE(handle, nullptr);
    int32_t stream_sync_timeout = -2;
    auto ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_STREAM_SYNC_TIMEOUT, &stream_sync_timeout, sizeof(stream_sync_timeout));
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(handle->streamSyncTimeout, -1);

    ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_STREAM_SYNC_TIMEOUT, &stream_sync_timeout, invalid_num);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    const size_t invalid_stream_sync_timeout = 1234567899876;
    ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_STREAM_SYNC_TIMEOUT, &invalid_stream_sync_timeout, sizeof(invalid_stream_sync_timeout));
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    ret = aclmdlSetExecConfigOpt(handle, static_cast<aclmdlExecConfigAttr>(100U), &stream_sync_timeout, sizeof(stream_sync_timeout));
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    const int32_t event_sync_timeout = -2;
    ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_EVENT_SYNC_TIMEOUT, &event_sync_timeout, invalid_num);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_EVENT_SYNC_TIMEOUT, &event_sync_timeout, sizeof(event_sync_timeout));
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(handle->eventSyncTimeout, -1);

    ret = aclmdlSetExecConfigOpt(handle, static_cast<aclmdlExecConfigAttr>(100U), &event_sync_timeout, sizeof(event_sync_timeout));
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    const size_t invalid_event_sync_timeout = 1234567899876;
    ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_STREAM_SYNC_TIMEOUT, &invalid_event_sync_timeout, sizeof(invalid_event_sync_timeout));
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(aclmdlDestroyExecConfigHandle(handle), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlSetExecConfigOpt_valid_timeout_success)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
    EXPECT_NE(handle, nullptr);
    const int32_t stream_sync_timeout = 100;
    aclError ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_STREAM_SYNC_TIMEOUT, &stream_sync_timeout, sizeof(stream_sync_timeout));
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(handle->streamSyncTimeout, 100);

    const int32_t event_sync_timeout = 200;
    ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_EVENT_SYNC_TIMEOUT, &event_sync_timeout, sizeof(event_sync_timeout));
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(handle->eventSyncTimeout, 200);
    EXPECT_EQ(aclmdlDestroyExecConfigHandle(handle), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlExecuteV2_runtimev2_ModelExecute_success)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    auto executor = std::unique_ptr<gert::ModelV2Executor>(new(std::nothrow) gert::ModelV2Executor);
    uint32_t modelId = 0x1;
    auto rtSession = acl::AclResourceManager::GetInstance().CreateRtSession();
    acl::AclResourceManager::GetInstance().AddExecutor(modelId, std::move(executor), rtSession);

    aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
    EXPECT_NE(handle, nullptr);

    const int32_t stream_sync_timeout = 100;
    aclError ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_STREAM_SYNC_TIMEOUT, &stream_sync_timeout,
                                          sizeof(stream_sync_timeout));
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(handle->streamSyncTimeout, 100);

    const int32_t event_sync_timeout = 200;
    ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_EVENT_SYNC_TIMEOUT, &event_sync_timeout,
                                 sizeof(event_sync_timeout));
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(handle->eventSyncTimeout, 200);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    EXPECT_NE(dataset, nullptr);

    aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
    ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), IsOriginShapeInRange(_))
            .WillOnce(Return(false))
            .WillRepeatedly(Return(true));
    ret = aclmdlExecuteV2(modelId, dataset, dataset, nullptr, handle);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlExecuteV2(modelId, dataset, dataset, nullptr, handle);
    EXPECT_EQ(ret, ACL_SUCCESS);

    auto stream_timeout = ge::GEContext().StreamSyncTimeout();
    EXPECT_EQ(stream_timeout, stream_sync_timeout);

    auto event_timeout = ge::GEContext().EventSyncTimeout();
    EXPECT_EQ(event_timeout, event_sync_timeout);

    free(dataBuffer);
    ret = aclmdlDestroyDataset(dataset);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = acl::AclResourceManager::GetInstance().DeleteExecutor(modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlDestroyExecConfigHandle(handle);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

ge::Status GetAippTypeDynamicAippInvoke(uint32_t index, ge::InputAippType &type, size_t &aippindex) {
    (void) index;
    type = ge::DATA_WITH_DYNAMIC_AIPP;
    aippindex = 0xFFFFFFFF;
    return ge::SUCCESS;
}

TEST_F(UTEST_ACL_Model, aclmdlExecuteV2_runtimev2_ModelExecute_failed)
{
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    auto executor = std::unique_ptr<gert::ModelV2Executor>(new(std::nothrow) gert::ModelV2Executor);
    uint32_t modelId = 0x1;
    auto rtSession = acl::AclResourceManager::GetInstance().CreateRtSession();
    acl::AclResourceManager::GetInstance().AddExecutor(modelId, std::move(executor), rtSession);

    aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
    EXPECT_NE(handle, nullptr);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    EXPECT_NE(dataset, nullptr);

    aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
    aclError ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), IsOriginShapeInRange(_))
            .WillOnce(Return(false));
    ret = aclmdlExecuteV2(modelId, dataset, dataset, nullptr, handle);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAippType(_,_,_))
            .WillOnce(Invoke(GetAippTypeDynamicAippInvoke));
    ret = aclmdlExecuteV2(modelId, dataset, dataset, nullptr, handle);
    EXPECT_EQ(ret, ACL_SUCCESS);
    free(dataBuffer);
    ret = aclmdlDestroyDataset(dataset);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = acl::AclResourceManager::GetInstance().DeleteExecutor(modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = aclmdlDestroyExecConfigHandle(handle);
    EXPECT_EQ(ret, ACL_SUCCESS);
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = false;
}

TEST_F(UTEST_ACL_Model, aclmdlExecuteV2_ModelExecute_success)
{
    aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
    EXPECT_NE(handle, nullptr);

    const int32_t stream_sync_timeout = 100;
    aclError ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_STREAM_SYNC_TIMEOUT, &stream_sync_timeout, sizeof(stream_sync_timeout));
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(handle->streamSyncTimeout, stream_sync_timeout);

    const int32_t event_sync_timeout = 200;
    ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_EVENT_SYNC_TIMEOUT, &event_sync_timeout, sizeof(event_sync_timeout));
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(handle->eventSyncTimeout, 200);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    EXPECT_NE(dataset, nullptr);

    aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
    ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_,_,_,_))
            .WillRepeatedly(Invoke(GetModelDescInfo_Invoke));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecModel(_, _, _, _, _, _, _))
        .WillOnce(Invoke(ExecModelInvokeOneOut));
    ret = aclmdlExecuteV2(1, dataset, dataset, nullptr, handle);
    EXPECT_EQ(ret, ACL_SUCCESS);

    auto stream_timeout = ge::GEContext().StreamSyncTimeout();
    EXPECT_EQ(stream_timeout, stream_sync_timeout);

    auto event_timeout = ge::GEContext().EventSyncTimeout();
    EXPECT_EQ(event_timeout, event_sync_timeout);

    free(dataBuffer);
    ret = aclmdlDestroyDataset(dataset);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlDestroyExecConfigHandle(handle);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlExecuteV2_ModelExecute_failed)
{
    aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
    EXPECT_NE(handle, nullptr);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    EXPECT_NE(dataset, nullptr);

    aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
    aclError ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_,_,_,_))
            .WillRepeatedly(Invoke(GetModelDescInfo_Invoke));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecModel(_,_,_,_,_,_,_))
            .WillOnce(Return(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID));

    uint32_t modelId = 0;
    const char *modelPath = "/";
    ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlExecuteV2(1, dataset, dataset, nullptr, handle);
    EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

    free(dataBuffer);
    ret = aclmdlDestroyDataset(dataset);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlDestroyExecConfigHandle(handle);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(UTEST_ACL_Model, aclmdlExecuteV2_ModelExecute_StreamNotNull_failed)
{
    aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
    EXPECT_NE(handle, nullptr);

    aclmdlDataset *dataset = aclmdlCreateDataset();
    EXPECT_NE(dataset, nullptr);

    aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
    aclError ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_,_,_,_))
            .WillRepeatedly(Invoke(GetModelDescInfo_Invoke));

    uint32_t modelId = 0;
    const char *modelPath = "/";
    ret = aclmdlLoadFromFile(modelPath, &modelId);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtSynchronizeStreamWithTimeoutImpl(_, _))
            .WillOnce(Return(ACL_ERROR_RT_STREAM_SYNC_TIMEOUT));
    aclrtStream stream = (aclrtStream)0x11;
    ret = aclmdlExecuteV2(1, dataset, dataset, stream, handle);
    EXPECT_EQ(ret, ACL_ERROR_RT_STREAM_SYNC_TIMEOUT);

    free(dataBuffer);
    ret = aclmdlDestroyDataset(dataset);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = aclmdlDestroyExecConfigHandle(handle);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

std::shared_ptr<uint8_t> ConstructBundleOm(size_t model_num, size_t &size)
{
  size = sizeof(ge::ModelFileHeader) + sizeof(ModelPartitionTable) +
         (sizeof(ModelPartitionMemInfo) * model_num) + ((sizeof(ModelFileHeader) + 10) * model_num);
  std::shared_ptr<uint8_t> model_p(new (std::nothrow) uint8_t[size], std::default_delete<uint8_t[]>());
  uint8_t *model_data = model_p.get();
  size_t offset = 0;
  ModelFileHeader *header = (ModelFileHeader *)model_data;
  header->modeltype = 4;
  header->modeltype = 5;
  offset += sizeof(ModelFileHeader);
  ModelPartitionTable *p = (ModelPartitionTable *)(model_data + offset);
  p->num = 3;
  offset += sizeof(ModelPartitionTable);
  for (size_t i = 0; i < model_num; ++i) {
    ModelPartitionMemInfo *part = (ModelPartitionMemInfo *)(model_data + offset);
    part->mem_size = sizeof(ModelFileHeader) + 10;
    offset += sizeof(ModelPartitionMemInfo);
  }
  for (size_t i = 0; i < model_num; ++i) {
    ModelFileHeader *inner_header = (ModelFileHeader *)(model_data + offset);
    inner_header->version = ge::MODEL_VERSION + 1U;
    if ( i == 0) {
      inner_header->model_num = 2U;
      inner_header->is_unknow_model = 1;
    } else {
      inner_header->model_num = 1U;
      inner_header->is_unknow_model = 0;
    }
    offset += sizeof(ModelFileHeader);
    offset += 10;
  }
  return model_p;
}

TEST_F(UTEST_ACL_Model, TestaclmdlBundleLoadFromMem_verify_input_param)
{
    uint32_t bundle_id = 0;
    size_t size = 0;
    auto model_p = ConstructBundleOm(3, size);
    uint8_t * model_data = model_p.get();
    auto ret = aclmdlBundleLoadFromMem(nullptr, size, &bundle_id);
    EXPECT_EQ(ret , ACL_ERROR_INVALID_PARAM);
    ret = aclmdlBundleLoadFromMem(model_data, 0, &bundle_id);
    EXPECT_EQ(ret , ACL_ERROR_INVALID_PARAM);
    ret = aclmdlBundleLoadFromMem(model_data, size, nullptr);
    EXPECT_EQ(ret , ACL_ERROR_INVALID_PARAM);

    ModelFileHeader *header = (ModelFileHeader *)model_data;
    header->modeltype = 4;
    ret = aclmdlBundleLoadFromMem(model_data, size, &bundle_id);
    EXPECT_EQ(ret , ACL_ERROR_INVALID_PARAM);
    header->modeltype = 5;
    ret = aclmdlBundleLoadFromMem(model_data, sizeof(ModelFileHeader), &bundle_id);
    EXPECT_EQ(ret , ACL_ERROR_INVALID_PARAM);
    ret = aclmdlBundleLoadFromMem(model_data, (size - 1), &bundle_id);
    EXPECT_EQ(ret , ACL_ERROR_INVALID_PARAM);
}

TEST_F(UTEST_ACL_Model, TestaclmdlBundleLoadFromMem_success)
{
    uint32_t bundle_id = 0;
    size_t size = 0;
    auto model_p = ConstructBundleOm(3, size);
    uint8_t * model_data = model_p.get();
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
            .WillRepeatedly(Invoke(LoadExecutorFromModelDataSuccess));
    auto ret = aclmdlBundleLoadFromMem(model_data, size, &bundle_id);
    EXPECT_EQ(ret , ACL_SUCCESS);
    ret = aclmdlBundleUnload(bundle_id);
    EXPECT_EQ(ret , ACL_SUCCESS);
}

ge::graphStatus LoadDataFromFileV2Stub(const char *path, ge::ModelData &model_data)
{
    (void) path;
    size_t size = 0;
    auto model_p = ConstructBundleOm(3, size);
    // delete is outside
    auto p_new = new uint8_t[size];
    memcpy_s(p_new, size, model_p.get(), size);
    model_data.model_len = size;
    model_data.model_data = p_new;
    return GRAPH_SUCCESS;
}

TEST_F(UTEST_ACL_Model, TestaclmdlBundleLoadFromFile)
{
    const char *path = "/";
    uint32_t bundle_id = 0;
    auto ret = aclmdlBundleLoadFromFile(nullptr, &bundle_id);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclmdlBundleLoadFromFile(path, nullptr);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadDataFromFileV2(_,_))
            .WillOnce(Return(ge::GRAPH_FAILED))
            .WillRepeatedly(Invoke(LoadDataFromFileV2Stub));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
            .WillRepeatedly(Invoke(LoadExecutorFromModelDataSuccess));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadModelFromDataWithArgs(_,_,_))
            .WillOnce(Return(ge::GRAPH_FAILED))
            .WillRepeatedly(Return(ge::SUCCESS));
    ret = aclmdlBundleLoadFromFile(path, &bundle_id);
    EXPECT_NE(ret, ACL_SUCCESS);
    ret = aclmdlBundleLoadFromFile(path, &bundle_id);
    EXPECT_NE(ret, ACL_SUCCESS);
    ret = aclmdlBundleLoadFromFile(path, &bundle_id);
    EXPECT_EQ(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), UnLoad())
            .WillOnce(Return(ge::GRAPH_FAILED))
            .WillRepeatedly(Return(ge::GRAPH_SUCCESS));
    ret = aclmdlBundleUnload(bundle_id);
    EXPECT_NE(ret, ACL_SUCCESS);
    ret = aclmdlBundleUnload(bundle_id);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

Status LoadModelFromDataWithArgsStub(uint32_t &model_id, const ModelData &model_data, const ModelLoadArg &load_arg)
{
    (void) model_data;
    (void) load_arg;
    static uint32_t cnt = 0;
    ++cnt;
    model_id = cnt;
    return SUCCESS;
}

TEST_F(UTEST_ACL_Model, TestBundleInfo)
{
    AclResourceManager::GetInstance().bundleInfos_.clear();
    AclResourceManager::GetInstance().bundleInnerIds_.clear();
    AclResourceManager::GetInstance().executorMap_.clear();
    AclResourceManager::GetInstance().rtSessionMap_.clear();
    // load
    uint32_t bundle_id = 0;
    size_t size = 0;
    auto model_p = ConstructBundleOm(3, size);
    uint8_t * model_data = model_p.get();
    acl::AclResourceManager::GetInstance().enableRuntimeV2ForModel_ = true;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadExecutorFromModelData(_,_,_))
            .WillRepeatedly(Invoke(LoadExecutorFromModelDataSuccess));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadModelFromDataWithArgs(_,_,_))
            .WillRepeatedly(Invoke(LoadModelFromDataWithArgsStub));
    auto ret = aclmdlBundleLoadFromMem(model_data, size, &bundle_id);
    EXPECT_EQ(ret , ACL_SUCCESS);

    // check bundle info
    size_t modelNum = 0;
    ret = aclmdlBundleGetModelNum(bundle_id, nullptr);
    EXPECT_EQ(ret , ACL_ERROR_INVALID_PARAM);
    ret = aclmdlBundleGetModelNum((bundle_id + 1), &modelNum);
    EXPECT_EQ(ret , ACL_ERROR_INVALID_BUNDLE_MODEL_ID);
    ret = aclmdlBundleGetModelNum(bundle_id, &modelNum);
    EXPECT_EQ(ret , ACL_SUCCESS);
    EXPECT_EQ(modelNum , 3);

    size_t index = 0;
    uint32_t model_id = 0;
    std::vector<uint32_t> model_ids;
    ret = aclmdlBundleGetModelId(bundle_id, 999, nullptr);
    EXPECT_EQ(ret , ACL_ERROR_INVALID_PARAM);
    ret = aclmdlBundleGetModelId(bundle_id, 999, &model_id);
    EXPECT_EQ(ret , ACL_ERROR_INVALID_PARAM);
    ret = aclmdlBundleGetModelId(bundle_id + 1, index, &model_id);
    EXPECT_EQ(ret , ACL_ERROR_INVALID_BUNDLE_MODEL_ID);
    for (size_t i = 0; i < modelNum; ++i) {
        ret = aclmdlBundleGetModelId(bundle_id, i, &model_id);
        EXPECT_EQ(ret , ACL_SUCCESS);
        model_ids.emplace_back(model_id);
    }
    for (auto id : model_ids) {
        ret = aclmdlUnload(id);
        EXPECT_EQ(ret , ACL_ERROR_INVALID_PARAM);
    }

    EXPECT_EQ(AclResourceManager::GetInstance().bundleInfos_.size(), 1);
    EXPECT_EQ(AclResourceManager::GetInstance().bundleInnerIds_.size(), 3);
    EXPECT_EQ(AclResourceManager::GetInstance().executorMap_.size(), 2);
    EXPECT_EQ(AclResourceManager::GetInstance().rtSessionMap_.size(), 2);
    EXPECT_EQ(AclResourceManager::GetInstance().rtSessionMap_[bundle_id].get(),
              AclResourceManager::GetInstance().rtSessionMap_[model_ids[0]].get());

    ret = aclmdlBundleUnload(bundle_id);
    EXPECT_EQ(ret , ACL_SUCCESS);
    EXPECT_EQ(AclResourceManager::GetInstance().bundleInfos_.size(), 0);
    EXPECT_EQ(AclResourceManager::GetInstance().bundleInnerIds_.size(), 0);
    EXPECT_EQ(AclResourceManager::GetInstance().executorMap_.size(), 0);
    EXPECT_EQ(AclResourceManager::GetInstance().rtSessionMap_.size(), 0);
}
