/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/executor/ge_executor.h"
#include "framework/generator/ge_generator.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/runtime/mem_allocator.h"
#include "framework/runtime/stream_executor.h"
#include "framework/runtime/gert_api.h"
#include "framework/memory/allocator_desc.h"
#include "framework/runtime/gert_api.h"
#include "exe_graph/runtime/tensor_data.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/ge_tensor.h"
#include "graph/utils/type_utils_inner.h"
#include "graph/model.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "graph/operator.h"
#include "ge/ge_api.h"
#include "common/ge_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/opsproto_manager.h"
#include "graph/operator_factory.h"
#include "graph/ge_local_context.h"
#include "graph/tensor.h"
#include "common/helper/om_file_helper.h"
#include "platform/platform_info.h"

// #include "tdt/tdt_host_interface.h"
#include "runtime/dev.h"
#include "runtime/rts/rts_device.h"
#include "runtime/stream.h"
#include "runtime/rts/rts_stream.h"
#include "runtime/context.h"
#include "runtime/rts/rts_context.h"
#include "runtime/event.h"
#include "runtime/rts/rts_event.h"
#include "runtime/mem.h"
#include "runtime/rts/rts_mem.h"
#include "runtime/kernel.h"
#include "runtime/rts/rts_kernel.h"
#include "runtime/base.h"
#include "runtime/config.h"
#include "runtime/rt_mem_queue.h"
#include "runtime/rt_preload_task.h"
#include "runtime/rt_stars.h"
#include "runtime/rt_model.h"
#include "runtime/rts/rts_model.h"
#include "runtime/rt_stars_define.h"
#include "runtime/rts/rts_stars.h"
#include "runtime/rt_ras.h"

#include "adx_datadump_server.h"
#include "adump_api.h"
#include "mmpa/mmpa_api.h"

#include "acl/acl_rt.h"
#include "acl/acl_op.h"
#include "acl/acl_rt_allocator.h"

#include <gmock/gmock.h>

// using namespace tdt;
using namespace ge;

typedef aclError (*aclDumpSetCallbackFunc)(const char *configStr);
typedef aclError (*aclDumpUnsetCallbackFunc)();

class aclStub
{
public:
    // error manager
    virtual std::unique_ptr<const char_t[]> GetErrMgrErrorMessage();

    // ge function
    virtual ge::Status SetDump(const ge::DumpConfig &dumpConfig);
    virtual ge::Status GEInitialize(const std::map<AscendString, AscendString> &options);
    virtual ge::Status Finalize();
    virtual ge::Status Ge_Generator_Finalize();
    virtual ge::Status GEFinalize();
    virtual ge::Status BuildSingleOpModel(ge::OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                          const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                                          int32_t compile_flag, ModelBufferData &model_buff);
    virtual ge::Status BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                          const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                                          int32_t compile_flag, ModelBufferData &model_buff,
                                          GraphStage graph_stage, ComputeGraphPtr &compute_graph);
    virtual graphStatus SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range);
    virtual bool ReadBytesFromBinaryFile(char const *file_name, char **buffer, int &length);
    virtual ge::Status Initialize(const std::map<std::string, std::string> &options);
    virtual ge::Status Initialize(const std::map<std::string, std::string> &options, OmgContext &omgContext);
    virtual ge::Status LoadSingleOpV2(const std::string &modelName,
                                      const ModelData &modelData,
                                      void *stream,
                                      SingleOp **single_op,
                                      const uint64_t model_id);
    virtual ge::Status LoadDynamicSingleOpV2(const std::string &model_name,
                                             const ge::ModelData &modelData,
                                             void *stream,
                                             DynamicSingleOp **single_op,
                                             const uint64_t model_id);
    virtual ge::Status ExecuteAsync(DynamicSingleOp *executor,
                                    const std::vector<GeTensorDesc> &input_desc,
                                    const std::vector<DataBuffer> &inputs,
                                    std::vector<GeTensorDesc> &output_desc,
                                    std::vector<DataBuffer> &outputs);
    virtual ge::Status ExecuteAsync(SingleOp *executor,
                                    const std::vector<DataBuffer> &inputs,
                                    std::vector<DataBuffer> &outputs);
    virtual graphStatus GetName(AscendString &name);
    virtual bool GetBool(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name, bool &value);
    virtual bool GetInt(AttrUtils::ConstAttrHolderAdapter &&obj, const std::string &name, int32_t &value);
    virtual bool GetListNamedAttrs(AttrUtils::ConstAttrHolderAdapter &&obj, std::string const &name, vector<GeAttrValue::NAMED_ATTRS> &value);
    virtual std::map<string, AnyValue> GetAllAttrs();
    virtual std::string RealPath(const char *path);
    virtual graphStatus GetOpsTypeList(std::vector<ge::AscendString> &all_ops);
    virtual ge::Status GetModelDescInfo(uint32_t modelId, std::vector<TensorDesc> &inputDesc,
                                        std::vector<TensorDesc> &outputDesc, bool new_model_desc);

    virtual ge::Status GetModelDescInfoFromMem(const ModelData &model_data, ModelInOutInfo &info);
    virtual graphStatus GetShapeRange(std::vector<std::pair<int64_t, int64_t>> &range);
    virtual Format GetFormat();
    virtual ge::Status GetDynamicBatchInfo(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                           int32_t &dynamic_type);
    virtual ge::Status LoadModelFromData(uint32_t &model_id, const ModelData &modelData,
                                         void *dev_ptr, size_t memsize, void *weight_ptr, size_t weightsize);
    virtual ge::Status LoadModelFromDataWithArgs(uint32_t &model_id, const ModelData &model_data, const ModelLoadArg &load_arg);
    virtual ge::graphStatus LoadDataFromFile(std::string const &path, ModelData &modelData);
    virtual ge::Status LoadModelWithQ(uint32_t &model_id, const ge::ModelData &ge_model_data,
                                      const std::vector<uint32_t> &input_queue_ids, const std::vector<uint32_t> &output_queue_ids);
    virtual ge::Status LoadModelWithQ(uint32_t &model_id, const ge::ModelData &ge_model_data,
                                      const ge::ModelQueueArg &queue_arg);
    virtual ge::Status UnloadModel(uint32_t modelId);
    virtual ge::Status GetMemAndWeightSize(const std::string &path, size_t &mem_size, size_t &weight_size);
    virtual ge::Status GetMemAndWeightSize(const void *model_data, size_t model_size, size_t &mem_size, size_t &weight_size);
    virtual ge::Status ExecModel(uint32_t model_id, void *stream, const ge::RunModelData &run_input_data,
                                 const std::vector<ge::GeTensorDesc> &input_desc, ge::RunModelData &run_output_data,
                                 std::vector<ge::GeTensorDesc> &output_desc, bool async_mode);
    virtual ge::Status SetDynamicBatchSize(uint32_t model_id, void *dynamic_input_addr, uint64_t length, uint64_t batch_size);
    virtual ge::Status SetDynamicImageSize(uint32_t model_id, void *dynamic_input_addr, uint64_t length, uint64_t image_height, uint64_t image_width);
    virtual ge::Status SetDynamicDims(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                      const vector<uint64_t> &dynamic_dims);
    virtual ge::Status GetCurDynamicDims(uint32_t model_id, const vector<uint64_t> &dynamic_dims,
                                         vector<uint64_t> &cur_dynamic_dims);
    virtual ge::Status GetAippType(uint32_t model_id, uint32_t index, ge::InputAippType &type, size_t &aippindex);
    virtual ge::Status GetAippType( uint32_t index, ge::InputAippType &type, size_t &aippindex);
    virtual ge::Status GetUserDesignateShapeOrder(uint32_t model_id, vector<string> &user_designate_shape_order);
    virtual ge::Status GetCurShape(const uint32_t model_id, std::vector<int64_t> &batch_info, int32_t &dynamic_type);
    virtual ge::Status GetModelAttr(uint32_t model_id, std::vector<std::string> &dynamic_output_shape_info);
    virtual ge::Status GetOpAttr(uint32_t model_id, const std::string &op_name, const std::string &attr_name, std::string &attr_value);
    virtual ge::Status GetAIPPInfo(uint32_t model_id, uint32_t index, AippConfigInfo &aipp_params);
    virtual ge::Status GetAippInfo(const uint32_t index, ge::AippConfigInfo &aipp_info);
    virtual ge::Status GetBatchInfoSize(uint32_t model_id, size_t &shape_count);
    virtual ge::Status GetOrigInputInfo(uint32_t model_id, uint32_t index, OriginInputInfo &origOutputInfo);
    virtual ge::Status GetOriginAippInputInfo(uint32_t index, OriginInputInfo &origOutputInfo);
    virtual ge::Status GetAllAippInputOutputDims(uint32_t model_id, uint32_t index,
                                                 std::vector<InputOutputDims> &input_dims,
                                                 std::vector<InputOutputDims> &output_dims);
    virtual ge::Status GetAllAippInputOutputDims(uint32_t index,
                                                 std::vector<InputOutputDims> &input_dims,
                                                 std::vector<InputOutputDims> &output_dims);
    virtual ge::Status SetDynamicAippData(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                          const std::vector<kAippDynamicBatchPara> &aippBatchPara,
                                          const kAippDynamicPara &aippParms);
    virtual int Init();
    virtual bool OpsProtoManager_Initialize(const std::map<std::string, std::string> &options);
    virtual ge::Status TransShape(const TensorDesc &src_desc,
                                  Format dst_format,
                                  std::vector<int64_t> &dst_shape);
    virtual ge::Status Init(uint8_t *model_data, const uint32_t model_data_size);
    virtual ge::Status GetModelPartition(ModelPartitionType type, ModelPartition &partition);
    virtual graphStatus Load(const uint8_t *data, size_t len, Model &model);
    virtual bool HasAttr(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name);
    virtual bool GetListTensor(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name, vector<ConstGeTensorPtr> &value);
    virtual bool IsOriginShapeInRange(const gert::Shape &shape);
    virtual ge::Status SetAllocator(void *const stream, ge::Allocator *const external_allocator);

    // RT2.0
    virtual gert::ModelV2Executor *GetOrCreateLoaded(rtStream_t stream, const gert::ModelExecuteArg &arg);
    virtual gert::ModelV2Executor *CreateAndLoad(rtStream_t stream, const gert::ModelExecuteArg &arg);
    virtual ge::graphStatus Erase(rtStream_t stream);
    virtual std::unique_ptr<gert::ModelV2Executor> LoadExecutorFromFile(const char *file_path, ge::graphStatus &error_code);
    virtual std::unique_ptr<gert::ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data, ge::graphStatus &error_code);
    virtual std::unique_ptr<gert::ModelV2Executor> LoadExecutorFromModelDataWithRtSession(const ge::ModelData &model_data,
                                                                                          gert::RtSession *const rt_session,
                                                                                          ge::graphStatus &error_code);
    virtual ge::graphStatus LoadDataFromFileV2(const char *path, ge::ModelData &model_data);
    virtual std::unique_ptr<gert::ModelV2Executor>
    LoadExecutorFromModelDataWithMem(const ge::ModelData &model_data, ge::graphStatus &error_code,
                                     const void *weight_ptr, const size_t weight_size);
    virtual std::unique_ptr<gert::StreamExecutor> LoadStreamExecutorFromModelData(const ge::ModelData &model_data, const void *weight_ptr,
                                                                                  const size_t weight_size, ge::graphStatus &error_code);
    virtual std::unique_ptr<gert::StreamExecutor> LoadStreamExecutorFromModelData(const ge::ModelData &model_data,
                                                                                  const gert::LoweringOption &optimize_option,
                                                                                  ge::graphStatus &error_code);
    virtual ge::graphStatus IsDynamicModel(const char *file_path, bool &is_dynamic_model);
    virtual ge::graphStatus Load();
    virtual ge::graphStatus Load(const gert::ModelExecuteArg &arg);
    virtual ge::graphStatus Load(const gert::ModelExecuteArg &arg, const gert::ModelLoadArg &load_arg);
    virtual std::unique_ptr<ge::Allocator> Create(const gert::TensorPlacement &placement);
    virtual ge::graphStatus Execute(const gert::ModelExecuteArg &arg,
                                    gert::Tensor **inputs, size_t input_num,
                                    gert::Tensor **outputs, size_t output_num);
    virtual ge::graphStatus ExecuteSync(gert::Tensor **inputs, size_t input_num,
                                        gert::Tensor **outputs, size_t output_num);
    virtual ge::graphStatus UnLoad();

    // fe function
    virtual uint32_t InitializePlatformInfo();
    virtual uint32_t GetPlatformInfos(
        const std::string SoCVersion, fe::PlatFormInfos &platformInfo, fe::OptionalInfos &optionalInfo);
    virtual uint32_t InitRuntimePlatformInfos(const std::string &SoCVersion);
    virtual uint32_t GetRuntimePlatformInfosByDevice(const uint32_t &device_id, fe::PlatFormInfos &platform_infos);
    virtual uint32_t UpdateRuntimePlatformInfosByDevice(const uint32_t &device_id,
                                                        fe::PlatFormInfos &platform_infos);
    virtual bool GetPlatformResWithLock(const std::string &label, std::map<std::string, std::string> &res);
    virtual bool GetPlatformResWithLock(const string &label, const string &key, string &val);

    // runtime function
    virtual rtError_t rtSubscribeReport(uint64_t threadId, rtStream_t stream);
    virtual rtError_t rtRegTaskFailCallbackByModule(const char *moduleName, rtTaskFailCallback callback);
    virtual rtError_t rtCallbackLaunch(rtCallback_t callBackFunc, void *fnData, rtStream_t stream, bool isBlock);
    virtual rtError_t rtProcessReport(int32_t timeout);
    virtual rtError_t rtUnSubscribeReport(uint64_t threadId, rtStream_t stream);
    virtual rtError_t rtCtxCreateEx(rtContext_t *ctx, uint32_t flags, int32_t device);
    virtual rtError_t rtSetDevice(int32_t device);
    virtual rtError_t rtDeviceReset(int32_t device);
    virtual rtError_t rtDeviceResetForce(int32_t device);
    virtual rtError_t rtSetDeviceWithoutTsd(int32_t device);
    virtual rtError_t rtDeviceResetWithoutTsd(int32_t device);
    virtual rtError_t rtDeviceSynchronize(void);
    virtual rtError_t rtDeviceSynchronizeWithTimeout(int32_t timeout);
    virtual rtError_t rtGetDevice(int32_t *device);
    virtual rtError_t rtSetTSDevice(uint32_t tsId);
    virtual rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority);
    virtual rtError_t rtStreamCreateWithFlags(rtStream_t *stream, int32_t priority, uint32_t flags);
    virtual rtError_t rtsStreamCreate(rtStream_t *stream, rtStreamCreateConfig_t *config);
    virtual rtError_t rtStreamSetMode(rtStream_t stm, const uint64_t mode);
    virtual rtError_t rtStreamDestroy(rtStream_t stream);
    virtual rtError_t rtStreamDestroyForce(rtStream_t stream);
    virtual rtError_t rtStreamSynchronize(rtStream_t stream);
    virtual rtError_t rtStreamSynchronizeWithTimeout(rtStream_t stream, const int32_t timeout);
    virtual rtError_t rtStreamQuery(rtStream_t stream);
    virtual rtError_t rtStreamWaitEvent(rtStream_t stream, rtEvent_t event);
    virtual rtError_t rtStreamWaitEventWithTimeout(rtStream_t stream, rtEvent_t event, uint32_t timeout);
    virtual rtError_t rtStreamAbort(rtStream_t stream);
    virtual rtError_t rtCtxDestroyEx(rtContext_t ctx);
    virtual rtError_t rtCtxSetCurrent(rtContext_t ctx);
    virtual rtError_t rtCtxSynchronize();
    virtual rtError_t rtCtxGetCurrent(rtContext_t *ctx);
    virtual rtError_t rtGetPriCtxByDeviceId(int32_t device, rtContext_t *ctx);
    virtual rtError_t rtEventCreateWithFlag(rtEvent_t *event_, uint32_t flag);
    virtual rtError_t rtEventCreateExWithFlag(rtEvent_t *event_, uint32_t flag);
    virtual rtError_t rtEventCreate(rtEvent_t *event);
    virtual rtError_t rtGetEventID(rtEvent_t event, uint32_t *eventId);
    virtual rtError_t rtEventDestroy(rtEvent_t event);
    virtual rtError_t rtEventRecord(rtEvent_t event, rtStream_t stream);
    virtual rtError_t rtEventReset(rtEvent_t event, rtStream_t stream);
    virtual rtError_t rtEventSynchronize(rtEvent_t event);
    virtual rtError_t rtEventSynchronizeWithTimeout(rtEvent_t event, const int32_t timeout);
    virtual rtError_t rtEventQuery(rtEvent_t event);
    virtual rtError_t rtEventQueryStatus(rtEvent_t event, rtEventStatus_t *status);
    virtual rtError_t rtEventQueryWaitStatus(rtEvent_t event, rtEventWaitStatus *status);
    virtual rtError_t rtNotifyCreate(int32_t device_id, rtNotify_t *notify_);
    virtual rtError_t rtNotifyDestroy(rtNotify_t notify_);
    virtual rtError_t rtNotifyRecord(rtNotify_t notify_, rtStream_t stream_);
    virtual rtError_t rtGetNotifyID(rtNotify_t notify_, uint32_t *notify_id);
    virtual rtError_t rtNotifyWait(rtNotify_t notify_, rtStream_t stream_);
    virtual rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type, uint16_t moduleId);
    virtual rtError_t rtMallocCached(void **devPtr, uint64_t size, rtMemType_t type, uint16_t moduleId);
    virtual rtError_t rtFlushCache(void *devPtr, size_t size);
    virtual rtError_t rtInvalidCache(void *devPtr, size_t size);
    virtual rtError_t rtFree(void *devPtr);
    virtual rtError_t rtDvppMalloc(void **devPtr, uint64_t size, uint16_t moduleId);
    virtual rtError_t rtDvppMallocWithFlag(void **devPtr, uint64_t size, uint32_t flag, uint16_t moduleId);
    virtual rtError_t rtDvppFree(void *devPtr);
    virtual rtError_t rtMallocHost(void **hostPtr, uint64_t size, uint16_t moduleId);
    virtual rtError_t rtFreeHost(void *hostPtr);
    virtual rtError_t rtMemset(void *devPtr, uint64_t destMax, uint32_t value, uint64_t count);
    virtual rtError_t rtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind);
    virtual rtError_t rtMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind,
                                    rtStream_t stream);
    virtual rtError_t rtMemcpyAsyncEx(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind,
                                    rtStream_t stream, rtMemcpyConfig_t *memcpyConfig);
    virtual rtError_t rtMemsetAsync(void *ptr, uint64_t destMax, uint32_t value, uint64_t count, rtStream_t stream);
    virtual rtError_t rtCpuKernelLaunchWithFlag(const void *soName, const void *kernelName, uint32_t blockDim,
                                                const rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm,
                                                uint32_t flags);
    virtual rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total);
    virtual rtError_t rtGetRunMode(rtRunMode *mode);
    virtual rtError_t rtGetDeviceCount(int32_t *count);
    virtual rtError_t rtEventElapsedTime(float *time, rtEvent_t start, rtEvent_t end);
    virtual rtError_t rtDevBinaryUnRegister(void *handle);
    virtual rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle);
    virtual rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char *stubName,
                                         const void *devFunc, uint32_t funcMode);
    virtual rtError_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                                     rtSmDesc_t *smDesc, rtStream_t stream);
    virtual rtError_t rtGetSocVersion(char *version, const uint32_t maxLen);
    virtual rtError_t rtGetGroupCount(uint32_t *count);
    virtual rtError_t rtGetGroupInfo(int32_t groupid, rtGroupInfo_t *groupInfo, uint32_t count);
    virtual rtError_t rtSetGroup(int32_t groupid);
    virtual rtError_t rtGetDevicePhyIdByIndex(uint32_t devIndex, uint32_t *phyId);
    virtual rtError_t rtEnableP2P(uint32_t devIdDes, uint32_t phyIdSrc, uint32_t flag);
    virtual rtError_t rtDisableP2P(uint32_t devIdDes, uint32_t phyIdSrc);
    virtual rtError_t rtDeviceCanAccessPeer(int32_t *canAccessPeer, uint32_t device, uint32_t peerDevice);
    virtual rtError_t rtGetStreamId(rtStream_t stream_, int32_t *streamId);
    virtual rtError_t rtRegDeviceStateCallback(const char *regName, rtDeviceStateCallback callback);
    virtual rtError_t rtRegDeviceStateCallbackEx(const char *regName, rtDeviceStateCallback callback,
                                                 const rtDevCallBackDir_t notifyPos);
    virtual rtError_t rtDeviceGetStreamPriorityRange(int32_t *leastPriority, int32_t *greatestPriority);
    virtual rtError_t rtGetDeviceCapability(int32_t device, int32_t moduleType, int32_t featureType, int32_t *value);
    virtual rtError_t rtSetOpWaitTimeOut(uint32_t timeout);
    virtual rtError_t rtSetOpExecuteTimeOut(uint32_t timeout);
    virtual rtError_t rtSetOpExecuteTimeOutWithMs(uint32_t timeout);
    virtual rtError_t rtCtxSetSysParamOpt(const rtSysParamOpt configOpt, const int64_t configVal);
    virtual rtError_t rtCtxGetSysParamOpt(const rtSysParamOpt configOpt, int64_t *const configVal);
    virtual rtError_t rtSetSysParamOpt(const rtSysParamOpt configOpt, const int64_t configVal);
    virtual rtError_t rtGetSysParamOpt(const rtSysParamOpt configOpt, int64_t *const configVal);
    virtual rtError_t rtGetDeviceSatStatus(void *const outputAddrPtr, const uint64_t outputSize, rtStream_t stm);
    virtual rtError_t rtCleanDeviceSatStatus(rtStream_t stm);

    virtual rtError_t rtMemQueueInitQS(int32_t devId, const char *groupName);
    virtual rtError_t rtMemQueueCreate(int32_t devId, const rtMemQueueAttr_t *queAttr, uint32_t *qid);

    virtual rtError_t rtMemQueueDestroy(int32_t devId, uint32_t qid);

    virtual rtError_t rtMemQueueInit(int32_t devId);

    virtual rtError_t rtMemQueueEnQueue(int32_t devId, uint32_t qid, void *mbuf);

    virtual rtError_t rtMemQueueDeQueue(int32_t devId, uint32_t qid, void **mbuf);

    virtual rtError_t rtMemQueueEnQueueBuff(int32_t devId, uint32_t qid, rtMemQueueBuff_t *inBuf, int32_t timeout);

    virtual rtError_t rtMemQueueDeQueueBuff(int32_t devId, uint32_t qid, rtMemQueueBuff_t *outBuf, int32_t timeout);

    virtual rtError_t rtMemQueueQuery(int32_t devId, rtMemQueueQueryCmd_t cmd, void *inBuff, uint32_t inLen,
                                      void *outBuff, uint32_t *outLen);

    virtual rtError_t rtMemQueueQueryInfo(int32_t device, uint32_t qid, rtMemQueueInfo_t *queueInfo);

    virtual rtError_t rtMemQueueGrant(int32_t devId, uint32_t qid, int32_t pid, rtMemQueueShareAttr_t *attr);

    virtual rtError_t rtMemQueueAttach(int32_t devId, uint32_t qid, int32_t timeout);

    virtual rtError_t rtEschedSubmitEventSync(int32_t devId, rtEschedEventSummary_t *event, rtEschedEventReply_t *ack);

    virtual rtError_t rtQueryDevPid(rtBindHostpidInfo_t *info, pid_t *devPid);

    virtual rtError_t rtMbufInit(rtMemBuffCfg_t *cfg);

    virtual rtError_t rtMbufAlloc(rtMbufPtr_t *mbuf, uint64_t size);

    virtual rtError_t rtMbufAllocEx(rtMbufPtr_t *mbuf, uint64_t size, uint64_t flag, int32_t grpId);

    virtual rtError_t rtMbufFree(rtMbufPtr_t mbuf);

    virtual rtError_t rtMbufGetBuffAddr(rtMbufPtr_t mbuf, void **databuf);

    virtual rtError_t rtMbufGetBuffSize(rtMbufPtr_t mbuf, uint64_t *size);

    virtual rtError_t rtMbufGetPrivInfo(rtMbufPtr_t mbuf, void **priv, uint64_t *size);

    virtual rtError_t rtMbufCopyBufRef(rtMbufPtr_t mbuf, rtMbufPtr_t *newMbuf);

    virtual rtError_t rtMemGrpCreate(const char *name, const rtMemGrpConfig_t *cfg);

    virtual rtError_t rtMemGrpAddProc(const char *name, int32_t pid, const rtMemGrpShareAttr_t *attr);

    virtual rtError_t rtMemGrpAttach(const char *name, int32_t timeout);

    virtual rtError_t rtMemGrpQuery(rtMemGrpQueryInput_t * const input, rtMemGrpQueryOutput_t *output);

    virtual rtError_t rtMemcpy2d(void *dst, uint64_t dpitch, const void *src, uint64_t spitch, uint64_t width,
                                 uint64_t height, rtMemcpyKind_t kind);
    virtual rtError_t rtMemcpy2dAsync(void *dst, uint64_t dpitch, const void *src, uint64_t spitch, uint64_t width,
                                      uint64_t height, rtMemcpyKind_t kind, rtStream_t stream);
    virtual rtError_t rtGetDevMsg(rtGetDevMsgType_t getMsgType, rtGetMsgCallback callback);
    virtual rtError_t rtGetFaultEvent(const int32_t deviceId, rtDmsEventFilter *filter, rtDmsFaultEvent *dmsEvent,
                                      uint32_t len, uint32_t *eventCount);
    virtual rtError_t rtSetDeviceSatMode(rtFloatOverflowMode_t floatOverflowMode);
    virtual rtError_t rtGetDeviceSatMode(rtFloatOverflowMode_t *floatOverflowMode);
    virtual rtError_t rtSetStreamOverflowSwitch(rtStream_t stm, uint32_t flags);
    virtual rtError_t rtGetStreamOverflowSwitch(rtStream_t stm, uint32_t *flags);
    virtual rtError_t rtGetAiCoreCount(uint32_t *aiCoreCnt);
    virtual rtError_t rtGetDeviceInfo(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t *val);
    virtual rtError_t rtGetAllUtilizations(const int32_t devId, const rtTypeUtil_t kind, uint8_t *const util);
    virtual rtError_t rtDeviceStatusQuery(const uint32_t devId, rtDeviceStatus *deviceStatus);

    virtual rtError_t rtReserveMemAddress(void **devPtr, size_t size, size_t alignment, void *devAddr, uint64_t flags);
    virtual rtError_t rtReleaseMemAddress(void *devPtr);
    virtual rtError_t rtMallocPhysical(rtDrvMemHandle *handle, size_t size, rtDrvMemProp_t *prop, uint64_t flags);
    virtual rtError_t rtFreePhysical(rtDrvMemHandle handle);
    virtual rtError_t rtMapMem(void *devPtr, size_t size, size_t offset, rtDrvMemHandle handle, uint64_t flags);
    virtual rtError_t rtUnmapMem(void *devPtr);
    virtual rtError_t rtBinaryLoadWithoutTilingKey(const void *data, const uint64_t length, rtBinHandle *binHandle);
    virtual rtError_t rtBinaryUnLoad(rtBinHandle binHandle);
    virtual rtError_t rtsFuncGetByName(const rtBinHandle binHandle, const char_t *kernelName,
                                       rtFuncHandle *funcHandle);
    virtual rtError_t rtCreateLaunchArgs(size_t argsSize, size_t hostInfoTotalSize, size_t hostInfoNum,
                                         void *argsData, rtLaunchArgsHandle *argsHandle);
    virtual rtError_t rtDestroyLaunchArgs(rtLaunchArgsHandle argsHandle);
    virtual rtError_t rtLaunchKernelByFuncHandleV3(rtFuncHandle funcHandle, uint32_t blockDim,
                                                   const rtArgsEx_t *const argsInfo, rtStream_t stm, const rtTaskCfgInfo_t *const cfgInfo);
    virtual rtError_t rtsMemExportToShareableHandle(rtDrvMemHandle handle, rtDrvMemHandleType handleType,
                                                    uint64_t flag, uint64_t *shareableHandle);
    virtual rtError_t rtMemImportFromShareableHandle(uint64_t shareableHandle, int32_t deviceId,
                                                     rtDrvMemHandle *handle);
    virtual rtError_t rtMemSetPidToShareableHandle(uint64_t shareableHandle, int pid[], uint32_t pidNum);
    virtual rtError_t rtMemGetAllocationGranularity(rtDrvMemProp_t *prop,
                                                    rtDrvMemGranularityOptions option, size_t *granularity);
    virtual rtError_t rtDeviceGetBareTgid(uint32_t *pid);
    virtual rtError_t rtGetL2CacheOffset(uint32_t deivceId, uint64_t *offset);
    virtual rtError_t rtRegKernelLaunchFillFunc(const char *symbol, rtKernelLaunchFillFunc func);
    virtual rtError_t rtUnRegKernelLaunchFillFunc(const char *symbol);
    virtual rtError_t rtGetMemUceInfo(const uint32_t deviceId, rtMemUceInfo *memUceInfo);
    virtual rtError_t rtMemUceRepair(const uint32_t deviceId, rtMemUceInfo *memUceInfo);
    virtual rtError_t rtDeviceTaskAbort(int32_t devId, uint32_t timeout);
    virtual rtError_t rtMemQueueReset(int32_t devId, uint32_t qid);
    virtual rtError_t rtSetDefaultDeviceId(int32_t deviceId);
    virtual rtError_t rtDeviceSetLimit(int32_t devId, rtLimitType_t type, uint32_t val);
    virtual rtError_t rtRegStreamStateCallback(const char *regName, rtStreamStateCallback callback);
    virtual rtError_t rtCtxGetCurrentDefaultStream(rtStream_t* stm);
    virtual rtError_t rtCmoAsync(void *srcAddrPtr, size_t srcLen, rtCmoOpCode_t cmpType, rtStream_t stm);
    virtual rtError_t rtsCmoAsync(void *srcAddrPtr, size_t srcLen, rtCmoOpCode_t cmoType, rtStream_t stm);
    virtual rtError_t rtStreamBeginCapture(rtStream_t stm, const rtStreamCaptureMode mode);
    virtual rtError_t rtStreamGetCaptureInfo(rtStream_t stm, rtStreamCaptureStatus *const status,
                                             rtModel_t *captureMdl);
    virtual rtError_t rtStreamEndCapture(rtStream_t stm, rtModel_t *captureMdl);

    virtual rtError_t rtModelDebugDotPrint(rtModel_t mdl);
    virtual rtError_t rtThreadExchangeCaptureMode(rtStreamCaptureMode *mode);
    virtual rtError_t rtModelExecute(rtModel_t mdl, rtStream_t stm, uint32_t flag);
    virtual rtError_t rtModelDestroy(rtModel_t mdl);
    virtual rtError_t rtsStreamBeginTaskGrp(rtStream_t stm);
    virtual rtError_t rtsStreamEndTaskGrp(rtStream_t stm, rtTaskGrp_t *handle);
    virtual rtError_t rtsStreamBeginTaskUpdate(rtStream_t stm, rtTaskGrp_t handle);
    virtual rtError_t rtsStreamEndTaskUpdate(rtStream_t stm);

    virtual rtError_t rtsMemcpyAsyncWithDesc(rtMemcpyDesc_t desc, rtMemcpyKind kind, rtMemcpyConfig_t *config, rtStream_t stream);
    virtual rtError_t rtsGetMemcpyDescSize(rtMemcpyKind kind, size_t *size);
    virtual rtError_t rtsSetMemcpyDesc(rtMemcpyDesc_t desc, rtMemcpyKind kind, void *srcAddr, void *dstAddr, size_t count, rtMemcpyConfig_t *config);
    virtual rtError_t rtsBinaryLoadFromFile(const char * const binPath, const rtLoadBinaryConfig_t * const optionalCfg, rtBinHandle *handle);
    virtual rtError_t rtsFuncGetByEntry(const rtBinHandle binHandle, const uint64_t funcEntry, rtFuncHandle *funcHandle);
    virtual rtError_t rtsFuncGetAddr(const rtFuncHandle funcHandle, void **aicAddr, void **aivAddr);

    virtual rtError_t rtsLaunchKernelWithConfig(rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm, rtKernelLaunchCfg_t *cfg, rtArgsHandle argsHandle, void* reserve);
    virtual rtError_t rtsKernelArgsInit(rtFuncHandle funcHandle, rtArgsHandle *handle);
    virtual rtError_t rtsKernelArgsInitByUserMem(rtFuncHandle funcHandle, rtArgsHandle argsHandle, void *userHostMem, size_t actualArgsSize);
    virtual rtError_t rtsKernelArgsFinalize(rtArgsHandle argsHandle);
    virtual rtError_t rtsKernelArgsAppend(rtArgsHandle handle, void *para, size_t paraSize, rtParaHandle *paraHandle);
    virtual rtError_t rtsKernelArgsAppendPlaceHolder(rtArgsHandle handle, rtParaHandle *paraHandle);
    virtual rtError_t rtsKernelArgsParaUpdate(rtArgsHandle argsHandle, rtParaHandle paraHandle, void *para, size_t paraSize);
    virtual rtError_t rtsKernelArgsGetMemSize(rtFuncHandle funcHandle, size_t userArgsSize, size_t *actualArgsSize);
    virtual rtError_t rtsKernelArgsGetHandleMemSize(rtFuncHandle funcHandle, size_t *memSize);
    virtual rtError_t rtsKernelArgsGetPlaceHolderBuffer(rtArgsHandle argsHandle, rtParaHandle paraHandle, size_t dataSize, void **bufferAddr);

    virtual rtError_t rtsMalloc(void **devPtr, uint64_t size, rtMallocPolicy policy, rtMallocAdvise advise, rtMallocConfig_t *cfg);
    virtual rtError_t rtsMallocHost(void **hostPtr, uint64_t size, const rtMallocConfig_t *cfg);

    virtual rtError_t rtsPointerGetAttributes(const void *ptr, rtPtrAttributes_t *attributes);
    virtual rtError_t rtsHostRegister(void *ptr, uint64_t size, rtHostRegisterType type, void **devPtr);
    virtual rtError_t rtsHostUnregister(void *ptr);
    virtual rtError_t rtsGetThreadLastTaskId(uint32_t *taskId);
    virtual rtError_t rtsStreamGetId(rtStream_t stm, int32_t *streamId);

    virtual rtError_t rtsValueWrite(const void * const devAddr, const uint64_t value, const uint32_t flag, rtStream_t stm);
    virtual rtError_t rtsValueWait(const void * const devAddr, const uint64_t value, const uint32_t flag, rtStream_t stm);

    virtual rtError_t rtsStreamGetAvailableNum(uint32_t *streamCount);
    virtual rtError_t rtsStreamSetAttribute(rtStream_t stm, rtStreamAttr stmAttrId, rtStreamAttrValue_t *attrValue);
    virtual rtError_t rtsStreamGetAttribute(rtStream_t stm, rtStreamAttr stmAttrId, rtStreamAttrValue_t *attrValue);

    virtual rtError_t rtsNotifyCreate(rtNotify_t *notify, uint64_t flag);
    virtual rtError_t rtsNotifyDestroy(rtNotify_t notify);
    virtual rtError_t rtsNotifyRecord(rtNotify_t notify, rtStream_t stream);
    virtual rtError_t rtsNotifyWaitAndReset(rtNotify_t notify, rtStream_t stream, uint32_t timeout);
    virtual rtError_t rtsNotifyGetId(rtNotify_t notify, uint32_t *notifyId);

    virtual rtError_t rtsEventGetId(rtEvent_t event, uint32_t *eventId);
    virtual rtError_t rtsEventGetAvailNum(uint32_t *eventCount);

    virtual rtError_t rtsDeviceGetInfo(uint32_t deviceId, rtDevAttr attr, int64_t *val);
    virtual rtError_t rtsDeviceGetStreamPriorityRange(int32_t *leastPriority, int32_t *greatestPriority);
    virtual rtError_t rtsDeviceGetCapability(int32_t deviceId, int32_t devFeatureType, int32_t *val);

    virtual rtError_t rtsCtxGetCurrentDefaultStream(rtStream_t *stm);
    virtual rtError_t rtsGetPrimaryCtxState(const int32_t devId, uint32_t *flags, int32_t *active);

    virtual rtError_t rtsModelCreate(rtModel_t *mdl, uint32_t flag);
    virtual rtError_t rtsModelBindStream(rtModel_t mdl, rtStream_t stm, uint32_t flag);
    virtual rtError_t rtsEndGraph(rtModel_t mdl, rtStream_t stm);
    virtual rtError_t rtsModelLoadComplete(rtModel_t mdl, void *reserve);
    virtual rtError_t rtsModelUnbindStream(rtModel_t mdl, rtStream_t stm);
    virtual rtError_t rtsModelExecute(rtModel_t mdl, int32_t timeout);

    virtual rtError_t rtsLaunchReduceAsyncTask(const rtReduceInfo_t *reduceInfo, const rtStream_t stm, const void *reserve);

    virtual rtError_t rtsGetDeviceResLimit(const int32_t deviceId, const rtDevResLimitType_t type, uint32_t *value);
    virtual rtError_t rtsSetDeviceResLimit(const int32_t deviceId, const rtDevResLimitType_t type, uint32_t value);
    virtual rtError_t rtsResetDeviceResLimit(const int32_t deviceId);

    virtual rtError_t rtsGetStreamResLimit(rtStream_t stream, const rtDevResLimitType_t type, uint32_t *value);
    virtual rtError_t rtsSetStreamResLimit(rtStream_t stream, const rtDevResLimitType_t type, uint32_t value);
    virtual rtError_t rtsResetStreamResLimit(rtStream_t stream);
    virtual rtError_t rtsUseStreamResInCurrentThread(rtStream_t stream);
    virtual rtError_t rtsNotUseStreamResInCurrentThread(rtStream_t stream);
    virtual rtError_t rtsGetResInCurrentThread(const rtDevResLimitType_t type, uint32_t *value);

    virtual rtError_t rtsLabelCreate(rtLabel_t *lbl);
    virtual rtError_t rtsLabelSet(rtLabel_t lbl, rtStream_t stm);
    virtual rtError_t rtsLabelDestroy(rtLabel_t lbl);
    virtual rtError_t rtsLabelSwitchListCreate(rtLabel_t *labels, size_t num, void **labelList);
    virtual rtError_t rtsLabelSwitchListDestroy(void *labelList);
    virtual rtError_t rtsLabelSwitchByIndex(void *ptr, uint32_t maxValue, void *labelInfoPtr, rtStream_t stm);

    virtual rtError_t rtsActiveStream(rtStream_t activeStream, rtStream_t stream);
    virtual rtError_t rtsSwitchStream(void *leftValue, rtCondition_t cond, void *rightValue, rtSwitchDataType_t dataType, rtStream_t trueStream, rtStream_t falseStream, rtStream_t stream);
    virtual rtError_t rtsFuncGetName(const rtFuncHandle funcHandle, const uint32_t maxLen, char_t * const name);
    virtual rtError_t rtsModelSetName(rtModel_t mdl, const char_t *mdlName);
    virtual rtError_t rtsModelGetName(rtModel_t mdl, const uint32_t maxLen, char_t * const mdlName);

    virtual rtError_t rtsBinaryLoadFromData(const void *const data, const uint64_t length, const rtLoadBinaryConfig_t *const optionalCfg, rtBinHandle *handle);
    virtual rtError_t rtsRegisterCpuFunc(rtBinHandle binHandle, const char_t *const funcName, const char_t *const kernelName, rtFuncHandle *funcHandle);
    virtual rtError_t rtsCmoAsyncWithBarrier(void *srcAddrPtr, size_t srcLen, rtCmoOpCode cmoType, uint32_t logicId, rtStream_t stm);
    virtual rtError_t rtsLaunchBarrierTask(rtBarrierTaskInfo_t *taskInfo, rtStream_t stm, uint32_t flag);
    virtual rtError_t rtsGetPairDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, uint64_t *val);

    virtual rtError_t rtsMemcpyBatch(void **dsts, void **srcs, size_t *sizes, size_t count, rtMemcpyBatchAttr *attrs, size_t *attrsIdxs, size_t numAttrs, size_t *failIdx);
    virtual rtError_t rtsMemcpyBatchAsync(void **dsts, size_t *destMaxs, void **srcs, size_t *sizes, size_t count,
        rtMemcpyBatchAttr *attrs, size_t *attrsIdxs, size_t numAttrs, size_t *failIdx, rtStream_t stream);

    virtual rtError_t rtsIpcMemGetExportKey(const void *ptr, size_t size, char_t *key, uint32_t len, uint64_t flags);
    virtual rtError_t rtsIpcMemClose(const char_t *key);
    virtual rtError_t rtsIpcMemImportByKey(void **ptr, const char_t *key, uint64_t flags);
    virtual rtError_t rtsIpcMemSetImportPid(const char_t *key, int32_t pid[], int num);

    virtual rtError_t rtsNotifyBatchReset(rtNotify_t *notifies, uint32_t num);
    virtual rtError_t rtsNotifyGetExportKey(rtNotify_t notify, char_t *key, uint32_t len, uint64_t flags);
    virtual rtError_t rtsNotifyImportByKey(rtNotify_t *notify, const char_t *key, uint64_t flags);
    virtual rtError_t rtsNotifySetImportPid(rtNotify_t notify, int32_t pid[], int num);
    // geterror function
    virtual rtError_t rtsGetErrorVerbose(uint32_t deviceId, rtErrorInfo* errorInfo);
    virtual rtError_t rtsRepairError(uint32_t deviceId, const rtErrorInfo* errorInfo);

    // prof function
    virtual int32_t MsprofFinalize();
    virtual int32_t MsprofInit(uint32_t aclDataType, void *data, uint32_t dataLen);
    virtual int32_t MsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char *typeName);
    // adx function
    virtual int AdxDataDumpServerInit();
    virtual int AdxDataDumpServerUnInit();
    virtual int32_t AdumpSetDumpConfig(Adx::DumpType dumpType, const Adx::DumpConfig &dumpConfig);
    virtual bool AdumpIsDumpEnable(Adx::DumpType dumpType);

    // slog function
    virtual int dlog_getlevel(int module_id, int *enable_event);

    // mmpa function
    virtual void *mmAlignMalloc(mmSize mallocSize, mmSize alignSize);
    virtual INT32 mmAccess2(const CHAR *pathName, INT32 mode);
    virtual INT32 mmDladdr(VOID *addr, mmDlInfo *info);

    // acl_rt
    virtual aclError aclrtCreateEventWithFlagImpl(aclrtEvent *event, uint32_t flag);
    virtual aclError aclrtFreeImpl(void *devPtr);
    virtual aclError aclrtMallocImpl(void **devPtr, size_t size, aclrtMemMallocPolicy policy);
    virtual aclError aclrtGetEventIdImpl(aclrtEvent event, uint32_t *eventId);
    virtual aclError aclrtResetEventImpl(aclrtEvent event, aclrtStream stream);
    virtual aclError aclrtDestroyEventImpl(aclrtEvent event);
    virtual aclError aclrtStreamWaitEventImpl(aclrtStream stream, aclrtEvent event);
    virtual aclError aclrtGetRunModeImpl(aclrtRunMode *runMode);
    virtual aclError aclrtMemcpyImpl(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind);
    virtual aclError aclrtCreateStreamImpl(aclrtStream *stream);
    virtual aclError aclrtMemcpyAsyncImpl(void *dst, size_t destMax, const void *src, size_t count,
        aclrtMemcpyKind kind, aclrtStream stream);
    virtual aclError aclrtDestroyStreamImpl(aclrtStream stream);
    virtual aclError aclrtSynchronizeStreamImpl(aclrtStream stream);
    virtual aclError aclrtFree(void *devPtr);
    virtual aclError aclrtGetNotifyIdImpl(aclrtNotify notify, uint32_t *notifyId);
    virtual aclError aclrtUnSubscribeReportImpl(uint64_t threadId, aclrtStream stream);
    virtual aclError aclrtSubscribeReportImpl(uint64_t threadId, aclrtStream stream);
    virtual aclError aclrtMemsetImpl(void *devPtr, size_t maxCount, int32_t value, size_t count);
    virtual aclError aclrtGetCurrentContextImpl(aclrtContext *context);
    virtual aclError aclrtSetCurrentContextImpl(aclrtContext context);
    virtual aclError aclrtLaunchCallbackImpl(aclrtCallback fn, void *userData,
        aclrtCallbackBlockType blockType, aclrtStream stream);
    virtual aclError aclrtGetDeviceImpl(int32_t *deviceId);
    virtual aclDataBuffer *aclCreateDataBuffer(void *data, size_t size);
    virtual void *aclGetDataBufferAddr(const aclDataBuffer *dataBuffer);
    virtual void *aclGetDataBufferAddrImpl(const aclDataBuffer *dataBuffer);
    virtual aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer);
    // virtual hi_s32 hi_mpi_vpc_equalize_hist_for_acl(hi_s32 chn, const hi_vpc_pic_info* source_pic,
    //     hi_vpc_pic_info* dest_pic, const hi_vpc_lut_remap* hist_config_ptr, DvppStream stream);
    virtual size_t aclDataTypeSize(aclDataType dataType);
    virtual aclError aclrtSynchronizeStreamWithTimeoutImpl(aclrtStream stream, int32_t timeout);
    virtual size_t aclGetDataBufferSizeV2Impl(const aclDataBuffer *dataBuffer);
    virtual aclError aclrtAllocatorGetByStreamImpl(aclrtStream stream,
                                    aclrtAllocatorDesc *allocatorDesc,
                                    aclrtAllocator *allocator,
                                    aclrtAllocatorAllocFunc *allocFunc,
                                    aclrtAllocatorFreeFunc *freeFunc,
                                    aclrtAllocatorAllocAdviseFunc *allocAdviseFunc,
                                    aclrtAllocatorGetAddrFromBlockFunc *getAddrFromBlockFunc);
    virtual aclError aclInitCallbackRegisterImpl(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc,
                                                            void *userData);
    virtual aclError aclInitCallbackUnRegisterImpl(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc);
    virtual aclError aclFinalizeCallbackRegisterImpl(aclRegisterCallbackType type,
                                                                aclFinalizeCallbackFunc cbFunc, void *userData);
    virtual aclError aclFinalizeCallbackUnRegisterImpl(aclRegisterCallbackType type,
                                                                aclFinalizeCallbackFunc cbFunc);
    virtual size_t aclGetDataBufferSizeV2(const aclDataBuffer *dataBuffer);
    virtual uint32_t aclGetDataBufferSize(const aclDataBuffer *dataBuffer);
    virtual const char *aclrtGetSocNameImpl();
    virtual aclError aclDumpSetCallbackRegister(aclDumpSetCallbackFunc cbFunc);
    virtual aclError aclDumpSetCallbackUnRegister();
    virtual aclError aclDumpUnsetCallbackRegister(aclDumpUnsetCallbackFunc cbFunc);
    virtual aclError aclDumpUnsetCallbackUnRegister();
    virtual aclError aclopSetAttrBool(aclopAttr *attr, const char *attrName, uint8_t attrValue);
    virtual aclError aclrtGetCurrentContext(aclrtContext *context);
    virtual aclError aclrtSetCurrentContext(aclrtContext context);

    // mmpa
    virtual INT32 mmGetTid();
};

class MockFunctionTest : public aclStub
{
public:
    MockFunctionTest();
    static MockFunctionTest &aclStubInstance();
    void ResetToDefaultMock();

    // error manager
    MOCK_METHOD0(GetErrMgrErrorMessage, std::unique_ptr<const char_t[]>());

    // ge function stub
    MOCK_METHOD1(SetDump, ge::Status(const ge::DumpConfig &dump_config));
    MOCK_METHOD1(GEInitialize, ge::Status(const std::map<AscendString, AscendString> &options));
    MOCK_METHOD0(Finalize, ge::Status());
    MOCK_METHOD0(GEFinalize, ge::Status());
    MOCK_METHOD6(BuildSingleOpModel, ge::Status(ge::OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                                const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                                                int32_t compile_flag, ModelBufferData &model_buff));
    MOCK_METHOD8(BuildSingleOpModel, ge::Status(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                                const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                                                int32_t compile_flag, ModelBufferData &model_buff,
                                                GraphStage graph_stage, ComputeGraphPtr &compute_graph));
    MOCK_METHOD1(SetShapeRange, graphStatus(const std::vector<std::pair<int64_t, int64_t>> &range));
    MOCK_METHOD3(ReadBytesFromBinaryFile, bool(char const *file_name, char **buffer, int &length));
    MOCK_METHOD1(Initialize, ge::Status(const std::map<std::string, std::string> &options));
    MOCK_METHOD1(GetName, graphStatus(AscendString &name));
    MOCK_METHOD2(Initialize, ge::Status(const std::map<std::string, std::string> &options, OmgContext &omgContext));
    MOCK_METHOD0(Ge_Generator_Finalize, ge::Status());
    MOCK_METHOD5(LoadSingleOpV2, ge::Status(const std::string &modelName, const ModelData &modelData, void *stream,
                                            SingleOp **single_op, const uint64_t model_id));
    MOCK_METHOD2(SetAllocator, ge::Status(void *const stream, ge::Allocator *const external_allocator));
    MOCK_METHOD5(LoadDynamicSingleOpV2, ge::Status(const std::string &model_name, const ge::ModelData &modelData, void *stream,
                                                   DynamicSingleOp **single_op, const uint64_t model_id));
    MOCK_METHOD5(ExecuteAsync, ge::Status(DynamicSingleOp *executor, const std::vector<GeTensorDesc> &input_desc,
                                          const std::vector<DataBuffer> &inputs, std::vector<GeTensorDesc> &output_desc, std::vector<DataBuffer> &outputs));
    MOCK_METHOD3(ExecuteAsync, ge::Status(SingleOp *executor, const std::vector<DataBuffer> &inputs, std::vector<DataBuffer> &outputs));
    MOCK_METHOD3(GetBool, bool(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name, bool &value));
    MOCK_METHOD3(GetInt, bool(AttrUtils::ConstAttrHolderAdapter &&obj, const std::string &name, int32_t &value));
    MOCK_METHOD3(GetListNamedAttrs, bool(ge::AttrUtils::ConstAttrHolderAdapter &&obj, std::string const &name, vector<GeAttrValue::NAMED_ATTRS> &value));
    MOCK_METHOD0(GetAllAttrs, std::map<string, AnyValue>());
    MOCK_METHOD1(RealPath, std::string(const char *path));
    MOCK_METHOD1(GetOpsTypeList, graphStatus(std::vector<ge::AscendString> &all_ops));
    MOCK_METHOD4(GetModelDescInfo, ge::Status(uint32_t modelId, std::vector<TensorDesc> &inputDesc,
                                              std::vector<TensorDesc> &outputDesc, bool new_model_desc));

    MOCK_METHOD2(GetModelDescInfoFromMem, ge::Status(const ModelData &model_data, ModelInOutInfo &info));
    MOCK_METHOD1(GetShapeRange, graphStatus(std::vector<std::pair<int64_t, int64_t>> &range));
    MOCK_METHOD0(GetFormat, Format());
    MOCK_METHOD3(GetDynamicBatchInfo, ge::Status(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type));
    MOCK_METHOD6(LoadModelFromData, ge::Status(uint32_t &model_id, const ModelData &modelData,
                                               void *dev_ptr, size_t memsize, void *weight_ptr, size_t weightsize));
    MOCK_METHOD3(LoadModelFromDataWithArgs, ge::Status(uint32_t &model_id, const ModelData &model_data, const ModelLoadArg &load_arg));
    MOCK_METHOD2(LoadDataFromFile, ge::graphStatus(std::string const &path, ModelData &modelData));
    MOCK_METHOD4(LoadModelWithQ, ge::Status(uint32_t &model_id, const ge::ModelData &ge_model_data,
                                            const std::vector<uint32_t> &input_queue_ids,
                                            const std::vector<uint32_t> &output_queue_ids));
    MOCK_METHOD3(LoadModelWithQ, ge::Status(uint32_t &model_id, const ge::ModelData &ge_model_data,
                                            const ge::ModelQueueArg &queue_arg));
    MOCK_METHOD1(UnloadModel, ge::Status(uint32_t modelId));
    MOCK_METHOD3(GetMemAndWeightSize, ge::Status(const std::string &path, size_t &mem_size, size_t &weight_size));
    MOCK_METHOD4(GetMemAndWeightSize, ge::Status(const void *model_data, size_t model_size, size_t &mem_size, size_t &weight_size));
    MOCK_METHOD7(ExecModel, ge::Status(uint32_t model_id, void *stream, const ge::RunModelData &run_input_data,
                                       const std::vector<ge::GeTensorDesc> &input_desc, ge::RunModelData &run_output_data,
                                       std::vector<ge::GeTensorDesc> &output_desc, bool async_mode));
    MOCK_METHOD4(SetDynamicBatchSize, ge::Status(uint32_t model_id, void *dynamic_input_addr, uint64_t length, uint64_t batch_size));
    MOCK_METHOD5(SetDynamicImageSize, ge::Status(uint32_t model_id, void *dynamic_input_addr, uint64_t length, uint64_t image_height, uint64_t image_width));
    MOCK_METHOD4(SetDynamicDims, ge::Status(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                            const vector<uint64_t> &dynamic_dims));
    MOCK_METHOD3(GetCurDynamicDims, ge::Status(uint32_t model_id, const vector<uint64_t> &dynamic_dims,
                                               vector<uint64_t> &cur_dynamic_dims));
    MOCK_METHOD4(GetAippType, ge::Status(uint32_t model_id, uint32_t index, ge::InputAippType &type, size_t &aippindex));
    MOCK_METHOD3(GetAippType, ge::Status(uint32_t index, ge::InputAippType &type, size_t &aippindex));
    MOCK_METHOD2(GetUserDesignateShapeOrder, ge::Status(uint32_t model_id, vector<string> &user_designate_shape_order));
    MOCK_METHOD3(GetCurShape, ge::Status(const uint32_t model_id, std::vector<int64_t> &batch_info, int32_t &dynamic_type));
    MOCK_METHOD2(GetModelAttr, ge::Status(uint32_t model_id, std::vector<std::string> &dynamic_output_shape_info));
    MOCK_METHOD4(GetOpAttr, ge::Status(uint32_t model_id, const std::string &op_name, const std::string &attr_name, std::string &attr_value));
    MOCK_METHOD3(GetAIPPInfo, ge::Status(uint32_t model_id, uint32_t index, AippConfigInfo &aipp_params));
    MOCK_METHOD2(GetAippInfo, ge::Status(const uint32_t index, ge::AippConfigInfo &aipp_info));
    MOCK_METHOD2(GetBatchInfoSize, ge::Status(uint32_t model_id, size_t &shape_count));
    MOCK_METHOD3(GetOrigInputInfo, ge::Status(uint32_t model_id, uint32_t index, OriginInputInfo &origOutputInfo));
    MOCK_METHOD2(GetOriginAippInputInfo, ge::Status(uint32_t index, OriginInputInfo &origOutputInfo));
    MOCK_METHOD4(GetAllAippInputOutputDims, ge::Status(uint32_t model_id, uint32_t index, std::vector<InputOutputDims> &input_dims, std::vector<InputOutputDims> &output_dims));
    MOCK_METHOD3(GetAllAippInputOutputDims, ge::Status(uint32_t index, std::vector<InputOutputDims> &input_dims, std::vector<InputOutputDims> &output_dims));
    MOCK_METHOD5(SetDynamicAippData, ge::Status(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                                const std::vector<kAippDynamicBatchPara> &aippBatchPara,
                                                const kAippDynamicPara &aippParms));
    MOCK_METHOD0(Init, int());
    MOCK_METHOD1(OpsProtoManager_Initialize, bool(const std::map<std::string, std::string> &options));
    MOCK_METHOD3(TransShape, ge::Status(const TensorDesc &src_desc, Format dst_format,
                                        std::vector<int64_t> &dst_shape));
    MOCK_METHOD3(Load, graphStatus(const uint8_t *data, size_t len, Model &model));
    MOCK_METHOD2(Init, ge::Status(uint8_t *model_data, const uint32_t model_data_size));
    MOCK_METHOD2(GetModelPartition, ge::Status(ModelPartitionType type, ModelPartition &partition));
    MOCK_METHOD2(HasAttr, bool(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name));
    MOCK_METHOD3(GetListTensor,
        bool(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name, vector<ConstGeTensorPtr> &value));
    MOCK_METHOD1(IsOriginShapeInRange, bool(const gert::Shape &shape));

    // RT2.0 function stub
    MOCK_METHOD2(GetOrCreateLoaded, gert::ModelV2Executor *(rtStream_t stream, const gert::ModelExecuteArg &arg));
    MOCK_METHOD2(CreateAndLoad, gert::ModelV2Executor *(rtStream_t stream, const gert::ModelExecuteArg &arg));
    MOCK_METHOD1(Erase, ge::graphStatus(rtStream_t stream));
    MOCK_METHOD2(LoadExecutorFromFile, std::unique_ptr<gert::ModelV2Executor>(const char *file_path, ge::graphStatus &error_code));
    MOCK_METHOD2(LoadExecutorFromModelData, std::unique_ptr<gert::ModelV2Executor>(const ge::ModelData &model_data,
                                                                                   ge::graphStatus &error_code));
    MOCK_METHOD3(LoadExecutorFromModelData, std::unique_ptr<gert::ModelV2Executor>(const ge::ModelData &model_data,
                                                                                   const gert::LoadExecutorArgs &args,
                                                                                   ge::graphStatus &error_code));
    MOCK_METHOD3(LoadExecutorFromModelDataWithRtSession, std::unique_ptr<gert::ModelV2Executor> (const ge::ModelData &model_data,
                                                                                  gert::RtSession *const rt_session,
                                                                                  ge::graphStatus &error_code));
    MOCK_METHOD2(LoadDataFromFileV2, ge::graphStatus(const char *path, ge::ModelData &modelData));
    MOCK_METHOD4(LoadExecutorFromModelDataWithMem, std::unique_ptr<gert::ModelV2Executor>(
                                                       const ge::ModelData &model_data, ge::graphStatus &error_code, const void *weight_ptr,
                                                       const size_t weight_size));
    MOCK_METHOD4(LoadStreamExecutorFromModelData, std::unique_ptr<gert::StreamExecutor>(
                                                      const ge::ModelData &model_data, const void *weight_ptr, const size_t weight_size, ge::graphStatus &error_code));
    MOCK_METHOD3(LoadStreamExecutorFromModelData, std::unique_ptr<gert::StreamExecutor>(
                                                      const ge::ModelData &model_data, const gert::LoweringOption &optimize_option, ge::graphStatus &error_code));
    MOCK_METHOD2(IsDynamicModel, ge::graphStatus(const char *file_path, bool &is_dynamic_model));
    MOCK_METHOD0(Load, ge::graphStatus());
    MOCK_METHOD1(Load, ge::graphStatus(const gert::ModelExecuteArg &arg));
    MOCK_METHOD2(Load, ge::graphStatus(const gert::ModelExecuteArg &arg, const gert::ModelLoadArg &load_arg));
    MOCK_METHOD1(Create, std::unique_ptr<ge::Allocator>(const gert::TensorPlacement &placement));
    MOCK_METHOD5(Execute, ge::graphStatus(const gert::ModelExecuteArg &arg,
                                          gert::Tensor **inputs, size_t input_num,
                                          gert::Tensor **outputs, size_t output_num));
    MOCK_METHOD4(ExecuteSync, ge::graphStatus(gert::Tensor **inputs, size_t input_num,
                                              gert::Tensor **outputs, size_t output_num));

    MOCK_METHOD0(UnLoad, ge::graphStatus());

    // fe function
    MOCK_METHOD0(InitializePlatformInfo, uint32_t());
    MOCK_METHOD3(GetPlatformInfos,
        uint32_t(const std::string SoCVersion, fe::PlatFormInfos &platformInfo, fe::OptionalInfos &optionalInfo));
    MOCK_METHOD1(InitRuntimePlatformInfos, uint32_t(const std::string &SoCVersion));
    MOCK_METHOD2(GetRuntimePlatformInfosByDevice, uint32_t(const uint32_t &device_id, fe::PlatFormInfos &platform_infos));
    MOCK_METHOD2(GetPlatformResWithLock, bool(const std::string &label, std::map<std::string, std::string> &res));
    MOCK_METHOD3(GetPlatformResWithLock, bool(const string &label, const string &key, string &val));
    MOCK_METHOD2(UpdateRuntimePlatformInfosByDevice, uint32_t(const uint32_t &device_id, fe::PlatFormInfos &platform_infos));

    // runtime function stub
    MOCK_METHOD2(rtSubscribeReport, rtError_t(uint64_t threadId, rtStream_t stream));
    MOCK_METHOD2(rtRegTaskFailCallbackByModule, rtError_t(const char *moduleName, rtTaskFailCallback callback));
    MOCK_METHOD4(rtCallbackLaunch, rtError_t(rtCallback_t callBackFunc, void *fnData, rtStream_t stream, bool isBlock));
    MOCK_METHOD1(rtProcessReport, rtError_t(int32_t timeout));
    MOCK_METHOD2(rtUnSubscribeReport, rtError_t(uint64_t threadId, rtStream_t stream));
    MOCK_METHOD3(rtCtxCreateEx, rtError_t(rtContext_t *ctx, uint32_t flags, int32_t device));
    MOCK_METHOD1(rtSetDevice, rtError_t(int32_t device));
    MOCK_METHOD1(rtSetDefaultDeviceId, rtError_t(int32_t device));
    MOCK_METHOD3(rtDeviceSetLimit, rtError_t(int32_t devId, rtLimitType_t type, uint32_t val));
    MOCK_METHOD1(rtDeviceReset, rtError_t(int32_t device));
    MOCK_METHOD1(rtDeviceResetForce, rtError_t(int32_t device));
    MOCK_METHOD1(rtSetDeviceWithoutTsd, rtError_t(int32_t device));
    MOCK_METHOD1(rtDeviceResetWithoutTsd, rtError_t(int32_t device));
    MOCK_METHOD0(rtDeviceSynchronize, rtError_t(void));
    MOCK_METHOD1(rtDeviceSynchronizeWithTimeout, rtError_t(int32_t timeout));
    MOCK_METHOD1(rtGetDevice, rtError_t(int32_t *device));
    MOCK_METHOD1(rtSetTSDevice, rtError_t(uint32_t tsId));
    MOCK_METHOD2(rtStreamCreate, rtError_t(rtStream_t *stream, int32_t priority));
    MOCK_METHOD3(rtStreamCreateWithFlags, rtError_t(rtStream_t *stream, int32_t priority, uint32_t flags));
    MOCK_METHOD2(rtStreamSetMode, rtError_t(rtStream_t stream, const uint64_t mode));
    MOCK_METHOD1(rtStreamDestroy, rtError_t(rtStream_t stream));
    MOCK_METHOD1(rtStreamDestroyForce, rtError_t(rtStream_t stream));
    MOCK_METHOD1(rtStreamSynchronize, rtError_t(rtStream_t stream));
    MOCK_METHOD2(rtStreamSynchronizeWithTimeout, rtError_t(rtStream_t stream, const int32_t timeout));
    MOCK_METHOD1(rtStreamQuery, rtError_t(rtStream_t stream));
    MOCK_METHOD2(rtStreamWaitEvent, rtError_t(rtStream_t stream, rtEvent_t event));
    MOCK_METHOD3(rtStreamWaitEventWithTimeout, rtError_t(rtStream_t stream, rtEvent_t event, uint32_t timeout));
    MOCK_METHOD1(rtCtxDestroyEx, rtError_t(rtContext_t ctx));
    MOCK_METHOD1(rtCtxSetCurrent, rtError_t(rtContext_t ctx));
    MOCK_METHOD0(rtCtxSynchronize, rtError_t());
    MOCK_METHOD1(rtCtxGetCurrent, rtError_t(rtContext_t *ctx));
    MOCK_METHOD2(rtGetPriCtxByDeviceId, rtError_t(int32_t device, rtContext_t *ctx));
    MOCK_METHOD2(rtEventCreateWithFlag, rtError_t(rtEvent_t *event_, uint32_t flag));
    MOCK_METHOD2(rtEventCreateExWithFlag, rtError_t(rtEvent_t *event_, uint32_t flag));
    MOCK_METHOD1(rtEventCreate, rtError_t(rtEvent_t *event));
    MOCK_METHOD2(rtGetEventID, rtError_t(rtEvent_t event, uint32_t *eventId));
    MOCK_METHOD1(rtEventDestroy, rtError_t(rtEvent_t event));
    MOCK_METHOD2(rtEventRecord, rtError_t(rtEvent_t event, rtStream_t stream));
    MOCK_METHOD2(rtEventReset, rtError_t(rtEvent_t event, rtStream_t stream));
    MOCK_METHOD1(rtEventSynchronize, rtError_t(rtEvent_t event));
    MOCK_METHOD2(rtEventSynchronizeWithTimeout, rtError_t(rtEvent_t event, const int32_t timeout));
    MOCK_METHOD1(rtEventQuery, rtError_t(rtEvent_t event));
    MOCK_METHOD2(rtEventQueryStatus, rtError_t(rtEvent_t event, rtEventStatus_t *status));
    MOCK_METHOD2(rtEventQueryWaitStatus, rtError_t(rtEvent_t event, rtEventWaitStatus *status));
    MOCK_METHOD2(rtNotifyCreate, rtError_t(int32_t device_id, rtNotify_t *notify_));
    MOCK_METHOD1(rtNotifyDestroy, rtError_t(rtNotify_t notify_));
    MOCK_METHOD2(rtNotifyRecord, rtError_t(rtNotify_t notify_, rtStream_t stream_));
    MOCK_METHOD2(rtGetNotifyID, rtError_t(rtNotify_t notify_, uint32_t *notify_id));
    MOCK_METHOD2(rtNotifyWait, rtError_t(rtNotify_t notify_, rtStream_t stream_));
    MOCK_METHOD4(rtMalloc, rtError_t(void **devPtr, uint64_t size, rtMemType_t type, uint16_t moduleId));
    MOCK_METHOD4(rtMallocCached, rtError_t(void **devPtr, uint64_t size, rtMemType_t type, uint16_t moduleId));
    MOCK_METHOD2(rtFlushCache, rtError_t(void *devPtr, size_t size));
    MOCK_METHOD2(rtInvalidCache, rtError_t(void *devPtr, size_t size));
    MOCK_METHOD1(rtFree, rtError_t(void *devPtr));
    MOCK_METHOD3(rtDvppMalloc, rtError_t(void **devPtr, uint64_t size, uint16_t moduleId));
    MOCK_METHOD4(rtDvppMallocWithFlag, rtError_t(void **devPtr, uint64_t size, uint32_t flag, uint16_t moduleId));
    MOCK_METHOD1(rtDvppFree, rtError_t(void *devPtr));
    MOCK_METHOD3(rtMallocHost, rtError_t(void **hostPtr, uint64_t size, uint16_t moduleId));
    MOCK_METHOD1(rtFreeHost, rtError_t(void *hostPtr));
    MOCK_METHOD4(rtMemset, rtError_t(void *devPtr, uint64_t destMax, uint32_t value, uint64_t count));
    MOCK_METHOD5(rtMemcpy, rtError_t(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind));
    MOCK_METHOD6(rtMemcpyAsync, rtError_t(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind,
                                          rtStream_t stream));
    MOCK_METHOD7(rtMemcpyAsyncEx, rtError_t(void *dst, uint64_t destMax, const void *src, uint64_t count,
                                            rtMemcpyKind_t kind, rtStream_t stream, rtMemcpyConfig_t *memcpyConfig));
    MOCK_METHOD5(rtMemsetAsync, rtError_t(void *ptr, uint64_t destMax, uint32_t value, uint64_t count, rtStream_t stream));
    MOCK_METHOD7(rtCpuKernelLaunchWithFlag, rtError_t(const void *soName, const void *kernelName, uint32_t blockDim,
            const rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm, uint32_t flags));
    MOCK_METHOD3(rtMemGetInfoEx, rtError_t(rtMemInfoType_t memInfoType, size_t *free, size_t *total));
    MOCK_METHOD1(rtGetRunMode, rtError_t(rtRunMode *mode));
    MOCK_METHOD1(rtGetDeviceCount, rtError_t(int32_t *count));
    MOCK_METHOD3(rtEventElapsedTime, rtError_t(float *time, rtEvent_t start, rtEvent_t end));
    MOCK_METHOD1(rtDevBinaryUnRegister, rtError_t(void *handle));
    MOCK_METHOD2(rtDevBinaryRegister, rtError_t(const rtDevBinary_t *bin, void **handle));
    MOCK_METHOD5(rtFunctionRegister, rtError_t(void *binHandle, const void *stubFunc, const char *stubName,
                                               const void *devFunc, uint32_t funcMode));
    MOCK_METHOD6(rtKernelLaunch, rtError_t(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                                           rtSmDesc_t *smDesc, rtStream_t stream));
    MOCK_METHOD2(rtGetSocVersion, rtError_t(char *version, const uint32_t maxLen));
    MOCK_METHOD1(rtGetGroupCount, rtError_t(uint32_t *count));
    MOCK_METHOD3(rtGetGroupInfo, rtError_t(int32_t groupid, rtGroupInfo_t *groupInfo, uint32_t count));
    MOCK_METHOD1(rtSetGroup, rtError_t(int32_t groupid));
    MOCK_METHOD2(rtGetDevicePhyIdByIndex, rtError_t(uint32_t devIndex, uint32_t *phyId));
    MOCK_METHOD3(rtEnableP2P, rtError_t(uint32_t devIdDes, uint32_t phyIdSrc, uint32_t flag));
    MOCK_METHOD2(rtDisableP2P, rtError_t(uint32_t devIdDes, uint32_t phyIdSrc));
    MOCK_METHOD3(rtDeviceCanAccessPeer, rtError_t(int32_t *canAccessPeer, uint32_t device, uint32_t peerDevice));
    MOCK_METHOD2(rtGetStreamId, rtError_t(rtStream_t stream_, int32_t *streamId));
    MOCK_METHOD2(rtRegDeviceStateCallback, rtError_t(const char *regName, rtDeviceStateCallback callback));
    MOCK_METHOD3(rtRegDeviceStateCallbackEx, rtError_t(const char *regName, rtDeviceStateCallback callback,
                                                       const rtDevCallBackDir_t notifyPos));
    MOCK_METHOD2(rtDeviceGetStreamPriorityRange, rtError_t(int32_t *leastPriority, int32_t *greatestPriority));
    MOCK_METHOD4(rtGetDeviceCapability, rtError_t(int32_t device, int32_t moduleType, int32_t featureType, int32_t *value));
    MOCK_METHOD1(rtSetOpWaitTimeOut, rtError_t(uint32_t timeout));
    MOCK_METHOD1(rtSetOpExecuteTimeOut, rtError_t(uint32_t timeout));
    MOCK_METHOD1(rtSetOpExecuteTimeOutWithMs, rtError_t(uint32_t timeout));
    MOCK_METHOD2(rtCtxSetSysParamOpt, rtError_t(const rtSysParamOpt configOpt, const int64_t configVal));
    MOCK_METHOD2(rtCtxGetSysParamOpt, rtError_t(const rtSysParamOpt configOpt, int64_t *const configVal));
    MOCK_METHOD2(rtSetSysParamOpt, rtError_t(const rtSysParamOpt configOpt, const int64_t configVal));
    MOCK_METHOD2(rtGetSysParamOpt, rtError_t(const rtSysParamOpt configOpt, int64_t *const configVal));
    MOCK_METHOD3(rtGetDeviceSatStatus, rtError_t(void *const outputAddrPtr, const uint64_t outputSize, rtStream_t stm));
    MOCK_METHOD1(rtCleanDeviceSatStatus, rtError_t(rtStream_t stm));

    MOCK_METHOD2(rtMemQueueInitQS, rtError_t(int32_t devId, const char *groupName));
    MOCK_METHOD3(rtMemQueueCreate, rtError_t(int32_t devId, const rtMemQueueAttr_t *queAttr, uint32_t *qid));
    MOCK_METHOD2(rtMemQueueDestroy, rtError_t(int32_t devId, uint32_t qid));
    MOCK_METHOD1(rtMemQueueInit, rtError_t(int32_t devId));
    MOCK_METHOD3(rtMemQueueEnQueue, rtError_t(int32_t devId, uint32_t qid, void *mbuf));
    MOCK_METHOD3(rtMemQueueDeQueue, rtError_t(int32_t devId, uint32_t qid, void **mbuf));
    // MOCK_METHOD4(rtMemQueuePeek, rtError_t(int32_t devId, uint32_t qid, size_t *bufLen, int32_t timeout));
    MOCK_METHOD4(rtMemQueueEnQueueBuff, rtError_t(int32_t devId, uint32_t qid, rtMemQueueBuff_t *inBuf, int32_t timeout));
    MOCK_METHOD4(rtMemQueueDeQueueBuff, rtError_t(int32_t devId, uint32_t qid, rtMemQueueBuff_t *outBuf, int32_t timeout));
    MOCK_METHOD6(rtMemQueueQuery, rtError_t(int32_t devId, rtMemQueueQueryCmd_t cmd, void *inBuff, uint32_t inLen,
                                            void *outBuff, uint32_t *outLen));

    MOCK_METHOD3(rtMemQueueQueryInfo, rtError_t(int32_t device, uint32_t qid, rtMemQueueInfo_t *queueInfo));
    MOCK_METHOD4(rtMemQueueGrant, rtError_t(int32_t devId, uint32_t qid, int32_t pid, rtMemQueueShareAttr_t *attr));
    MOCK_METHOD3(rtMemQueueAttach, rtError_t(int32_t devId, uint32_t qid, int32_t timeout));
    MOCK_METHOD3(rtEschedSubmitEventSync, rtError_t(int32_t devId, rtEschedEventSummary_t *event, rtEschedEventReply_t *ack));
    MOCK_METHOD2(rtQueryDevPid, rtError_t(rtBindHostpidInfo_t *info, pid_t *devPid));
    MOCK_METHOD1(rtMbufInit, rtError_t(rtMemBuffCfg_t *cfg));
    MOCK_METHOD2(rtMbufAlloc, rtError_t(rtMbufPtr_t *mbuf, uint64_t size));
    MOCK_METHOD4(rtMbufAllocEx, rtError_t(rtMbufPtr_t *mbuf, uint64_t size, uint64_t flag, int32_t grpId));
    MOCK_METHOD1(rtMbufFree, rtError_t(rtMbufPtr_t mbuf));
    MOCK_METHOD2(rtMbufGetBuffAddr, rtError_t(rtMbufPtr_t mbuf, void **databuf));
    MOCK_METHOD2(rtMbufGetBuffSize, rtError_t(rtMbufPtr_t mbuf, uint64_t *size));
    MOCK_METHOD3(rtMbufGetPrivInfo, rtError_t(rtMbufPtr_t mbuf, void **priv, uint64_t *size));
    MOCK_METHOD2(rtMbufCopyBufRef, rtError_t(rtMbufPtr_t mbuf, rtMbufPtr_t *newMbuf));
    MOCK_METHOD2(rtMemGrpCreate, rtError_t(const char *name, const rtMemGrpConfig_t *cfg));
    MOCK_METHOD3(rtMemGrpAddProc, rtError_t(const char *name, int32_t pid, const rtMemGrpShareAttr_t *attr));
    MOCK_METHOD2(rtMemGrpAttach, rtError_t(const char *name, int32_t timeout));
    MOCK_METHOD2(rtMemGrpQuery, rtError_t(rtMemGrpQueryInput_t * const input, rtMemGrpQueryOutput_t *output));
    MOCK_METHOD7(rtMemcpy2d, rtError_t(void *dst, uint64_t dpitch, const void *src, uint64_t spitch, uint64_t width,
                                       uint64_t height, rtMemcpyKind_t kind));
    MOCK_METHOD8(rtMemcpy2dAsync, rtError_t(void *dst, uint64_t dpitch, const void *src, uint64_t spitch,
                                            uint64_t width, uint64_t height, rtMemcpyKind_t kind, rtStream_t stream));
    MOCK_METHOD2(rtGetDevMsg, rtError_t(rtGetDevMsgType_t getMsgType, rtGetMsgCallback callback));
    MOCK_METHOD5(rtGetFaultEvent, rtError_t (const int32_t deviceId, rtDmsEventFilter *filter,
                                            rtDmsFaultEvent *dmsEvent, uint32_t len, uint32_t *eventCount));
    MOCK_METHOD1(rtSetDeviceSatMode, rtError_t(rtFloatOverflowMode_t floatOverflowMode));
    MOCK_METHOD1(rtGetDeviceSatMode, rtError_t(rtFloatOverflowMode_t *floatOverflowMode));
    MOCK_METHOD2(rtSetStreamOverflowSwitch, rtError_t(rtStream_t stm, uint32_t flags));
    MOCK_METHOD2(rtGetStreamOverflowSwitch, rtError_t(rtStream_t stm, uint32_t *flags));
    MOCK_METHOD1(rtGetAiCoreCount, rtError_t(uint32_t *aiCoreCnt));
    MOCK_METHOD4(rtGetDeviceInfo, rtError_t(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t *val));
    MOCK_METHOD3(rtGetAllUtilizations, rtError_t(const int32_t devId, const rtTypeUtil_t kind, uint8_t *const util));
    MOCK_METHOD2(rtDeviceStatusQuery, rtError_t(const uint32_t devId, rtDeviceStatus *deviceStatus));

    MOCK_METHOD5(rtReserveMemAddress, rtError_t(void **devPtr, size_t size, size_t alignment, void *devAddr, uint64_t flags));
    MOCK_METHOD1(rtReleaseMemAddress, rtError_t(void *devPtr));
    MOCK_METHOD4(rtMallocPhysical, rtError_t(rtDrvMemHandle *handle, size_t size, rtDrvMemProp_t *prop, uint64_t flags));
    MOCK_METHOD1(rtFreePhysical, rtError_t(rtDrvMemHandle handle));
    MOCK_METHOD5(rtMapMem, rtError_t(void *devPtr, size_t size, size_t offset, rtDrvMemHandle handle, uint64_t flags));
    MOCK_METHOD1(rtUnmapMem, rtError_t(void *devPtr));

    MOCK_METHOD3(rtBinaryLoadWithoutTilingKey, rtError_t(const void *data, const uint64_t length, rtBinHandle *binHandle));
    MOCK_METHOD1(rtBinaryUnLoad, rtError_t(rtBinHandle binHandle));
    MOCK_METHOD3(rtsFuncGetByName, rtError_t(const rtBinHandle binHandle, const char_t *kernelName,
                                             rtFuncHandle *funcHandle));
    MOCK_METHOD5(rtCreateLaunchArgs, rtError_t(size_t argsSize, size_t hostInfoTotalSize, size_t hostInfoNum,
                                               void *argsData, rtLaunchArgsHandle *argsHandle));
    MOCK_METHOD1(rtDestroyLaunchArgs, rtError_t(rtLaunchArgsHandle argsHandle));
    MOCK_METHOD5(rtLaunchKernelByFuncHandleV3, rtError_t(rtFuncHandle funcHandle, uint32_t blockDim,
                                                         const rtArgsEx_t *const argsInfo, rtStream_t stm,
                                                         const rtTaskCfgInfo_t *const cfgInfo));
    MOCK_METHOD4(rtsMemExportToShareableHandle, rtError_t(rtDrvMemHandle handle, rtDrvMemHandleType handleType,
            uint64_t flag, uint64_t * shareableHandle));
    MOCK_METHOD3(rtMemImportFromShareableHandle, rtError_t(uint64_t shareableHandle, int32_t deviceId,
            rtDrvMemHandle *handle));
    MOCK_METHOD3(rtMemSetPidToShareableHandle, rtError_t(uint64_t shareableHandle, int pid[], uint32_t pidNum));
    MOCK_METHOD3(rtMemGetAllocationGranularity, rtError_t(rtDrvMemProp_t * prop,
            rtDrvMemGranularityOptions option, size_t * granularity));
    MOCK_METHOD1(rtDeviceGetBareTgid, rtError_t(uint32_t * pid));
    MOCK_METHOD2(rtGetL2CacheOffset, rtError_t(uint32_t deivceId, uint64_t *offset));
    MOCK_METHOD2(rtRegKernelLaunchFillFunc, rtError_t(const char *symbol, rtKernelLaunchFillFunc func));
    MOCK_METHOD1(rtUnRegKernelLaunchFillFunc, rtError_t(const char *symbol));
    MOCK_METHOD2(rtGetMemUceInfo, rtError_t(const uint32_t, rtMemUceInfo *));
    MOCK_METHOD2(rtMemUceRepair, rtError_t(const uint32_t, rtMemUceInfo *));
    MOCK_METHOD2(rtDeviceTaskAbort, rtError_t(int32_t, uint32_t));
    MOCK_METHOD2(rtMemQueueReset, rtError_t(int32_t, uint32_t));
    MOCK_METHOD2(rtRegStreamStateCallback, rtError_t(const char *regName, rtStreamStateCallback callback));
    MOCK_METHOD1(rtCtxGetCurrentDefaultStream, rtError_t(rtStream_t* stm));
    MOCK_METHOD4(rtCmoAsync, rtError_t(void *srcAddrPtr, size_t srcLen, rtCmoOpCode_t cmpType, rtStream_t stm));
    MOCK_METHOD4(rtsCmoAsync, rtError_t(void *srcAddrPtr, size_t srcLen, rtCmoOpCode_t cmoType, rtStream_t stm));
    MOCK_METHOD1(rtStreamAbort, rtError_t(rtStream_t stm));
    MOCK_METHOD2(rtStreamBeginCapture, rtError_t(rtStream_t stm, const rtStreamCaptureMode mode));
    MOCK_METHOD3(rtStreamGetCaptureInfo, rtError_t(rtStream_t stm, rtStreamCaptureStatus *const status,
                                                   rtModel_t *captureMdl));
    MOCK_METHOD2(rtStreamEndCapture, rtError_t(rtStream_t stm, rtModel_t *captureMdl));
    MOCK_METHOD1(rtModelDebugDotPrint, rtError_t(rtModel_t mdl));
    MOCK_METHOD1(rtThreadExchangeCaptureMode, rtError_t(rtStreamCaptureMode *mode));
    MOCK_METHOD3(rtModelExecute, rtError_t(rtModel_t mdl, rtStream_t stm, uint32_t flag));
    MOCK_METHOD1(rtModelDestroy, rtError_t(rtModel_t mdl));
    MOCK_METHOD1(rtsStreamBeginTaskGrp, rtError_t(rtStream_t stm));
    MOCK_METHOD2(rtsStreamEndTaskGrp, rtError_t(rtStream_t stm, rtTaskGrp_t *handle));
    MOCK_METHOD2(rtsStreamBeginTaskUpdate, rtError_t(rtStream_t stm, rtTaskGrp_t handle));
    MOCK_METHOD1(rtsStreamEndTaskUpdate, rtError_t(rtStream_t stm));

    MOCK_METHOD4(rtsMemcpyAsyncWithDesc, rtError_t(rtMemcpyDesc_t desc, rtMemcpyKind kind, rtMemcpyConfig_t *config,
                                                   rtStream_t stream));
    MOCK_METHOD2(rtsGetMemcpyDescSize, rtError_t(rtMemcpyKind kind, size_t *size));
    MOCK_METHOD6(rtsSetMemcpyDesc, rtError_t(rtMemcpyDesc_t desc, rtMemcpyKind kind, void *srcAddr,
                                             void *dstAddr, size_t count, rtMemcpyConfig_t *config));
    MOCK_METHOD3(rtsBinaryLoadFromFile, rtError_t(const char * const binPath,
                                                  const rtLoadBinaryConfig_t * const optionalCfg, rtBinHandle *handle));
    MOCK_METHOD3(rtsFuncGetByEntry, rtError_t(const rtBinHandle binHandle, const uint64_t funcEntry,
                                              rtFuncHandle *funcHandle));
    MOCK_METHOD3(rtsFuncGetAddr, rtError_t(const rtFuncHandle funcHandle, void **aicAddr, void **aivAddr));
    MOCK_METHOD6(rtsLaunchKernelWithConfig, rtError_t(rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm,
                                                      rtKernelLaunchCfg_t *cfg, rtArgsHandle argsHandle,
                                                      void* reserve));
    MOCK_METHOD2(rtsKernelArgsInit, rtError_t(rtFuncHandle funcHandle, rtArgsHandle *handle));
    MOCK_METHOD1(rtsKernelArgsFinalize, rtError_t(rtArgsHandle argsHandle));
    MOCK_METHOD4(rtsKernelArgsAppend, rtError_t(rtArgsHandle handle, void *para, size_t paraSize,
                                                rtParaHandle *paraHandle));
    MOCK_METHOD2(rtsKernelArgsAppendPlaceHolder, rtError_t(rtArgsHandle handle, rtParaHandle *paraHandle));
    MOCK_METHOD4(rtsKernelArgsParaUpdate, rtError_t(rtArgsHandle argsHandle, rtParaHandle paraHandle, void *para,
                                                    size_t paraSize));
    MOCK_METHOD4(rtsKernelArgsInitByUserMem, rtError_t(rtFuncHandle funcHandle, rtArgsHandle argsHandle,
                                                       void *userHostMem, size_t actualArgsSize));
    MOCK_METHOD3(rtsKernelArgsGetMemSize, rtError_t(rtFuncHandle funcHandle, size_t userArgsSize,
                                                    size_t *actualArgsSize));
    MOCK_METHOD2(rtsKernelArgsGetHandleMemSize, rtError_t(rtFuncHandle funcHandle, size_t *memSize));
    MOCK_METHOD4(rtsKernelArgsGetPlaceHolderBuffer, rtError_t(rtArgsHandle argsHandle, rtParaHandle paraHandle,
                                                              uint32_t dataSize, void **bufferAddr));
    MOCK_METHOD5(rtsMalloc, rtError_t(void **devPtr, uint64_t size, rtMallocPolicy policy, rtMallocAdvise advise, rtMallocConfig_t *cfg));
    MOCK_METHOD3(rtsMallocHost, rtError_t(void **hostPtr, uint64_t size, const rtMallocConfig_t *cfg));

    MOCK_METHOD2(rtsPointerGetAttributes, rtError_t(const void *ptr, rtPtrAttributes_t *attributes));
    MOCK_METHOD4(rtsHostRegister, rtError_t(void *ptr, uint64_t size, rtHostRegisterType type, void **devPtr));
    MOCK_METHOD1(rtsHostUnregister, rtError_t(void *ptr));
    MOCK_METHOD1(rtsGetThreadLastTaskId, rtError_t(uint32_t *taskId));
    MOCK_METHOD2(rtsStreamGetId, rtError_t(rtStream_t stm, int32_t *streamId));

    MOCK_METHOD4(rtsValueWrite, rtError_t(const void * const devAddr, const uint64_t value, const uint32_t flag, rtStream_t stm));
    MOCK_METHOD4(rtsValueWait, rtError_t(const void * const devAddr, const uint64_t value, const uint32_t flag, rtStream_t stm));

    MOCK_METHOD1(rtsStreamGetAvailableNum, rtError_t(uint32_t *streamCount));
    MOCK_METHOD3(rtsStreamSetAttribute, rtError_t(rtStream_t stm, rtStreamAttr stmAttrId, rtStreamAttrValue_t *attrValue));
    MOCK_METHOD3(rtsStreamGetAttribute, rtError_t(rtStream_t stm, rtStreamAttr stmAttrId, rtStreamAttrValue_t *attrValue));

    MOCK_METHOD2(rtsNotifyCreate, rtError_t(rtNotify_t *notify, uint64_t flag));
    MOCK_METHOD1(rtsNotifyDestroy, rtError_t(rtNotify_t notify));
    MOCK_METHOD2(rtsNotifyRecord, rtError_t(rtNotify_t notify, rtStream_t stream));
    MOCK_METHOD3(rtsNotifyWaitAndReset, rtError_t(rtNotify_t notify, rtStream_t stream, uint32_t timeout));
    MOCK_METHOD2(rtsNotifyGetId, rtError_t(rtNotify_t notify, uint32_t *notifyId));

    MOCK_METHOD2(rtsEventGetId, rtError_t(rtEvent_t event, uint32_t *eventId));
    MOCK_METHOD1(rtsEventGetAvailNum, rtError_t(uint32_t *eventCount));

    MOCK_METHOD3(rtsDeviceGetInfo, rtError_t(uint32_t deviceId, rtDevAttr attr, int64_t *val));
    MOCK_METHOD2(rtsDeviceGetStreamPriorityRange, rtError_t(int32_t *leastPriority, int32_t *greatestPriority));
    MOCK_METHOD3(rtsDeviceGetCapability, rtError_t(int32_t deviceId, int32_t devFeatureType, int32_t *val));

    MOCK_METHOD1(rtsCtxGetCurrentDefaultStream, rtError_t(rtStream_t *stm));
    MOCK_METHOD3(rtsGetPrimaryCtxState, rtError_t(const int32_t devId, uint32_t *flags, int32_t *active));

    MOCK_METHOD2(rtsModelCreate, rtError_t(rtModel_t *mdl, uint32_t flag));
    MOCK_METHOD3(rtsModelBindStream, rtError_t(rtModel_t mdl, rtStream_t stm, uint32_t flag));
    MOCK_METHOD2(rtsEndGraph, rtError_t(rtModel_t mdl, rtStream_t stm));
    MOCK_METHOD2(rtsModelLoadComplete, rtError_t(rtModel_t mdl, void *reserve));
    MOCK_METHOD2(rtsModelUnbindStream, rtError_t(rtModel_t mdl, rtStream_t stm));
    MOCK_METHOD2(rtsModelExecute, rtError_t(rtModel_t mdl, int32_t timeout));

    MOCK_METHOD3(rtsLaunchReduceAsyncTask, rtError_t(const rtReduceInfo_t *reduceInfo, const rtStream_t stm, const void *reserve));

    MOCK_METHOD3(rtsGetDeviceResLimit, rtError_t(const int32_t deviceId, const rtDevResLimitType_t type, uint32_t *value));
    MOCK_METHOD3(rtsSetDeviceResLimit, rtError_t(const int32_t deviceId, const rtDevResLimitType_t type, uint32_t value));
    MOCK_METHOD1(rtsResetDeviceResLimit, rtError_t(const int32_t deviceId));

    MOCK_METHOD3(rtsGetStreamResLimit, rtError_t(rtStream_t stream, const rtDevResLimitType_t type, uint32_t *value));
    MOCK_METHOD3(rtsSetStreamResLimit, rtError_t(rtStream_t stream, const rtDevResLimitType_t type, uint32_t value));
    MOCK_METHOD1(rtsResetStreamResLimit, rtError_t(rtStream_t stream));
    MOCK_METHOD1(rtsUseStreamResInCurrentThread, rtError_t(rtStream_t stream));
    MOCK_METHOD1(rtsNotUseStreamResInCurrentThread, rtError_t(rtStream_t stream));
    MOCK_METHOD2(rtsGetResInCurrentThread, rtError_t(const rtDevResLimitType_t type, uint32_t *value));

    MOCK_METHOD1(rtsLabelCreate, rtError_t(rtLabel_t *lbl));
    MOCK_METHOD2(rtsLabelSet, rtError_t(rtLabel_t lbl, rtStream_t stm));
    MOCK_METHOD1(rtsLabelDestroy, rtError_t(rtLabel_t lbl));
    MOCK_METHOD3(rtsLabelSwitchListCreate, rtError_t(rtLabel_t *labels, size_t num, void **labelList));
    MOCK_METHOD1(rtsLabelSwitchListDestroy, rtError_t(void *labelList));
    MOCK_METHOD4(rtsLabelSwitchByIndex, rtError_t(void *ptr, uint32_t maxValue, void *labelInfoPtr, rtStream_t stm));

    MOCK_METHOD2(rtsActiveStream, rtError_t(rtStream_t activeStream, rtStream_t stream));
    MOCK_METHOD7(rtsSwitchStream, rtError_t(void *leftValue, rtCondition_t cond, void *rightValue, rtSwitchDataType_t dataType, rtStream_t trueStream, rtStream_t falseStream, rtStream_t stream));
    MOCK_METHOD3(rtsFuncGetName, rtError_t(const rtFuncHandle funcHandle, const uint32_t maxLen, char_t * const name));
    MOCK_METHOD2(rtsModelSetName, rtError_t(rtModel_t mdl, const char_t *mdlName));
    MOCK_METHOD3(rtsModelGetName, rtError_t(rtModel_t mdl, const uint32_t maxLen, char_t * const mdlName));

    MOCK_METHOD4(rtsBinaryLoadFromData, rtError_t(const void *const data, const uint64_t length, const rtLoadBinaryConfig_t *const optionalCfg, rtBinHandle *handle));
    MOCK_METHOD4(rtsRegisterCpuFunc, rtError_t(rtBinHandle binHandle, const char_t *const funcName, const char_t *const kernelName, rtFuncHandle *funcHandle));
    MOCK_METHOD5(rtsCmoAsyncWithBarrier, rtError_t(void *srcAddrPtr, size_t srcLen, rtCmoOpCode cmoType, uint32_t logicId, rtStream_t stm));
    MOCK_METHOD3(rtsLaunchBarrierTask, rtError_t(rtBarrierTaskInfo_t *taskInfo, rtStream_t stm, uint32_t flag));
    MOCK_METHOD4(rtsGetPairDevicesInfo, rtError_t(uint32_t devId, uint32_t otherDevId, int32_t infoType, uint64_t *val));

    MOCK_METHOD8(rtsMemcpyBatch, rtError_t(void **dsts, void **srcs, size_t *sizes, size_t count, rtMemcpyBatchAttr *attrs, size_t *attrsIdxs, size_t numAttrs, size_t *failIdx));
    MOCK_METHOD10(rtsMemcpyBatchAsync, rtError_t(void **dsts, size_t *destMaxs, void **srcs, size_t *sizes, size_t count,
        rtMemcpyBatchAttr *attrs, size_t *attrsIdxs, size_t numAttrs, size_t *failIdx, rtStream_t stream));

    MOCK_METHOD5(rtsIpcMemGetExportKey, rtError_t(const void *ptr, size_t size, char_t *key, uint32_t len, uint64_t flags));
    MOCK_METHOD1(rtsIpcMemClose, rtError_t(const char_t *key));
    MOCK_METHOD3(rtsIpcMemImportByKey, rtError_t(void **ptr, const char_t *key, uint64_t flags));
    MOCK_METHOD3(rtsIpcMemSetImportPid, rtError_t(const char_t *key, int32_t pid[], int num));

    MOCK_METHOD2(rtsNotifyBatchReset, rtError_t(rtNotify_t *notifies, uint32_t num));
    MOCK_METHOD4(rtsNotifyGetExportKey, rtError_t(rtNotify_t notify, char_t *key, uint32_t len, uint64_t flags));
    MOCK_METHOD3(rtsNotifyImportByKey, rtError_t(rtNotify_t *notify, const char_t *key, uint64_t flags));
    MOCK_METHOD3(rtsNotifySetImportPid, rtError_t(rtNotify_t notify, int32_t pid[], int num));

    // geterror function stub
    MOCK_METHOD2(rtsGetErrorVerbose, rtError_t(uint32_t deviceId, rtErrorInfo* errorInfo));
    MOCK_METHOD2(rtsRepairError, rtError_t(uint32_t deviceId, const rtErrorInfo* errorInfo));

    // prof function stub
    MOCK_METHOD0(MsprofFinalize, int32_t());
    MOCK_METHOD3(MsprofInit, int32_t(uint32_t aclDataType, void *data, uint32_t dataLen));
    MOCK_METHOD3(MsprofRegTypeInfo, int32_t(uint16_t level, uint32_t typeId, const char *typeName));

    // adx function stub
    MOCK_METHOD0(AdxDataDumpServerInit, int());
    MOCK_METHOD0(AdxDataDumpServerUnInit, int());
    MOCK_METHOD2(AdumpSetDumpConfig, int(Adx::DumpType dumpType, const Adx::DumpConfig &dumpConfig));
    MOCK_METHOD1(AdumpIsDumpEnable, bool(Adx::DumpType dumpType));

    // slog function stub
    MOCK_METHOD2(dlog_getlevel, int(int module_id, int *enable_event));

    // mmpa function stub
    MOCK_METHOD2(mmAlignMalloc, void *(mmSize mallocSize, mmSize alignSize));
    MOCK_METHOD2(mmAccess2, INT32(const CHAR *pathName, INT32 mode));
    MOCK_METHOD2(mmDladdr, INT32(VOID *addr, mmDlInfo *info));

    // acl_rt
    MOCK_METHOD2(aclrtCreateEventWithFlagImpl, aclError(aclrtEvent *event, uint32_t flag));
    MOCK_METHOD1(aclrtFreeImpl, aclError(void *devPtr));
    MOCK_METHOD3(aclrtMallocImpl, aclError(void **devPtr, size_t size, aclrtMemMallocPolicy policy));
    MOCK_METHOD2(aclrtGetEventIdImpl, aclError(aclrtEvent event, uint32_t *eventId));
    MOCK_METHOD2(aclrtResetEventImpl, aclError(aclrtEvent event, aclrtStream stream));
    MOCK_METHOD1(aclrtDestroyEventImpl, aclError(aclrtEvent event));
    MOCK_METHOD2(aclrtStreamWaitEventImpl, aclError(aclrtStream stream, aclrtEvent event));
    MOCK_METHOD1(aclrtGetRunModeImpl, aclError(aclrtRunMode *runMode));
    MOCK_METHOD5(aclrtMemcpyImpl, aclError(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind));
    MOCK_METHOD1(aclrtCreateStreamImpl, aclError(aclrtStream *stream));
    MOCK_METHOD6(aclrtMemcpyAsyncImpl, aclError(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind, aclrtStream stream));
    MOCK_METHOD1(aclrtDestroyStreamImpl, aclError(aclrtStream stream));
    MOCK_METHOD1(aclrtSynchronizeStreamImpl, aclError(aclrtStream stream));
    MOCK_METHOD1(aclrtFree, aclError(void *devPtr));
    MOCK_METHOD2(aclrtGetNotifyIdImpl, aclError(aclrtNotify notify, uint32_t *notifyId));
    MOCK_METHOD2(aclrtUnSubscribeReportImpl, aclError(uint64_t threadId, aclrtStream stream));
    MOCK_METHOD2(aclrtSubscribeReportImpl, aclError(uint64_t threadId, aclrtStream stream));
    MOCK_METHOD4(aclrtMemsetImpl, aclError(void *devPtr, size_t maxCount, int32_t value, size_t count));
    MOCK_METHOD1(aclrtGetCurrentContextImpl, aclError(aclrtContext *context));
    MOCK_METHOD1(aclrtSetCurrentContextImpl, aclError(aclrtContext context));
    MOCK_METHOD4(aclrtLaunchCallbackImpl, aclError(aclrtCallback fn, void *userData, aclrtCallbackBlockType blockType, aclrtStream stream));
    MOCK_METHOD1(aclrtGetDeviceImpl, aclError(int32_t *deviceId));
    MOCK_METHOD2(aclCreateDataBuffer, aclDataBuffer *(void *data, size_t size));
    MOCK_METHOD1(aclGetDataBufferAddr, void *(const aclDataBuffer *dataBuffer));
    MOCK_METHOD1(aclGetDataBufferAddrImpl, void *(const aclDataBuffer *dataBuffer));
    MOCK_METHOD1(aclDestroyDataBuffer, aclError(const aclDataBuffer *dataBuffer));
    MOCK_METHOD1(aclDataTypeSize, size_t(aclDataType dataType));
    MOCK_METHOD2(aclrtSynchronizeStreamWithTimeoutImpl, aclError(aclrtStream stream, int32_t timeout));
    MOCK_METHOD1(aclGetDataBufferSizeV2Impl, size_t(const aclDataBuffer *dataBuffer));
    MOCK_METHOD7(aclrtAllocatorGetByStreamImpl, aclError(aclrtStream stream,
                                    aclrtAllocatorDesc *allocatorDesc,
                                    aclrtAllocator *allocator,
                                    aclrtAllocatorAllocFunc *allocFunc,
                                    aclrtAllocatorFreeFunc *freeFunc,
                                    aclrtAllocatorAllocAdviseFunc *allocAdviseFunc,
                                    aclrtAllocatorGetAddrFromBlockFunc *getAddrFromBlockFunc));
    MOCK_METHOD3(aclInitCallbackRegisterImpl, aclError(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc,
                                                            void *userData));
    MOCK_METHOD2(aclInitCallbackUnRegisterImpl, aclError(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc));
    MOCK_METHOD3(aclFinalizeCallbackRegisterImpl, aclError(aclRegisterCallbackType type,
                                                                aclFinalizeCallbackFunc cbFunc, void *userData));
    MOCK_METHOD2(aclFinalizeCallbackUnRegisterImpl, aclError(aclRegisterCallbackType type,
                                                                aclFinalizeCallbackFunc cbFunc));
    MOCK_METHOD1(aclGetDataBufferSizeV2, size_t(const aclDataBuffer *dataBuffer));
    MOCK_METHOD1(aclGetDataBufferSize, uint32_t(const aclDataBuffer *dataBuffer));
    MOCK_METHOD0(aclrtGetSocNameImpl, const char *());
    MOCK_METHOD1(aclDumpSetCallbackRegister, aclError(aclDumpSetCallbackFunc cbFunc));
    MOCK_METHOD0(aclDumpSetCallbackUnRegister, aclError());
    MOCK_METHOD1(aclDumpUnsetCallbackRegister, aclError(aclDumpUnsetCallbackFunc cbFunc));
    MOCK_METHOD0(aclDumpUnsetCallbackUnRegister, aclError());
    MOCK_METHOD3(aclopSetAttrBool, aclError(aclopAttr *attr, const char *attrName, uint8_t attrValue));
    MOCK_METHOD1(aclrtGetCurrentContext, aclError(aclrtContext *context));
    MOCK_METHOD1(aclrtSetCurrentContext, aclError(aclrtContext context));

    // mmpa
    MOCK_METHOD0(mmGetTid, INT32());
};
