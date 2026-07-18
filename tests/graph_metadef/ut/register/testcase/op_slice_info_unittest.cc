/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "register/graph_optimizer/fusion_common/op_slice_info.h"

using namespace fe;

class UtestOpSliceInfo : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

// ======================== InputSplitInfo ========================

TEST_F(UtestOpSliceInfo, InputSplitInfo_default_ctor_is_ptr_null) {
  InputSplitInfo info;
  EXPECT_TRUE(info.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, InputSplitInfo_initialize_success) {
  InputSplitInfo info;
  EXPECT_TRUE(info.Initialize());
  EXPECT_FALSE(info.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, InputSplitInfo_set_get_index) {
  InputSplitInfo info;
  info.Initialize();
  size_t idx = 3;
  info.SetIndex(idx);
  EXPECT_EQ(info.GetIndex(), 3U);
}

TEST_F(UtestOpSliceInfo, InputSplitInfo_set_get_axis) {
  InputSplitInfo info;
  info.Initialize();
  std::vector<int64_t> axis = {0, 1, 2};
  info.SetAxis(axis);
  EXPECT_EQ(info.GetAxis(), axis);
}

TEST_F(UtestOpSliceInfo, InputSplitInfo_set_get_head_overlap) {
  InputSplitInfo info;
  info.Initialize();
  std::vector<int64_t> head = {10, 20};
  info.SetHeadOverLap(head);
  EXPECT_EQ(info.GetHeadOverLap(), head);
}

TEST_F(UtestOpSliceInfo, InputSplitInfo_set_get_tail_overlap) {
  InputSplitInfo info;
  info.Initialize();
  std::vector<int64_t> tail = {5, 15};
  info.SetTailOverLap(tail);
  EXPECT_EQ(info.GetTailOverLap(), tail);
}

TEST_F(UtestOpSliceInfo, InputSplitInfo_copy_ctor) {
  InputSplitInfo info;
  info.Initialize();
  size_t idx = 7;
  info.SetIndex(idx);
  std::vector<int64_t> axis = {1};
  info.SetAxis(axis);

  InputSplitInfo copy(info);
  EXPECT_FALSE(copy.IsPtrNull());
  EXPECT_EQ(copy.GetIndex(), 7U);
  EXPECT_EQ(copy.GetAxis(), axis);
}

TEST_F(UtestOpSliceInfo, InputSplitInfo_assign_op) {
  InputSplitInfo info;
  info.Initialize();
  size_t idx = 5;
  info.SetIndex(idx);

  InputSplitInfo other;
  other.Initialize();
  other = info;
  EXPECT_EQ(other.GetIndex(), 5U);
}

// ======================== OutputSplitInfo ========================

TEST_F(UtestOpSliceInfo, OutputSplitInfo_default_ctor_is_ptr_null) {
  OutputSplitInfo info;
  EXPECT_TRUE(info.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, OutputSplitInfo_initialize_success) {
  OutputSplitInfo info;
  EXPECT_TRUE(info.Initialize());
  EXPECT_FALSE(info.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, OutputSplitInfo_set_get_index) {
  OutputSplitInfo info;
  info.Initialize();
  size_t idx = 2;
  info.SetIndex(idx);
  EXPECT_EQ(info.GetIndex(), 2U);
}

TEST_F(UtestOpSliceInfo, OutputSplitInfo_set_get_axis) {
  OutputSplitInfo info;
  info.Initialize();
  std::vector<int64_t> axis = {3, 4};
  info.SetAxis(axis);
  EXPECT_EQ(info.GetAxis(), axis);
}

TEST_F(UtestOpSliceInfo, OutputSplitInfo_copy_ctor) {
  OutputSplitInfo info;
  info.Initialize();
  size_t idx = 9;
  info.SetIndex(idx);

  OutputSplitInfo copy(info);
  EXPECT_FALSE(copy.IsPtrNull());
  EXPECT_EQ(copy.GetIndex(), 9U);
}

TEST_F(UtestOpSliceInfo, OutputSplitInfo_assign_op) {
  OutputSplitInfo info;
  info.Initialize();
  size_t idx = 4;
  info.SetIndex(idx);

  OutputSplitInfo other;
  other.Initialize();
  other = info;
  EXPECT_EQ(other.GetIndex(), 4U);
}

// ======================== InputReduceInfo ========================

TEST_F(UtestOpSliceInfo, InputReduceInfo_default_ctor_is_ptr_null) {
  InputReduceInfo info;
  EXPECT_TRUE(info.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, InputReduceInfo_initialize_success) {
  InputReduceInfo info;
  EXPECT_TRUE(info.Initialize());
  EXPECT_FALSE(info.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, InputReduceInfo_set_get_index) {
  InputReduceInfo info;
  info.Initialize();
  size_t idx = 1;
  info.SetIndex(idx);
  EXPECT_EQ(info.GetIndex(), 1U);
}

TEST_F(UtestOpSliceInfo, InputReduceInfo_set_get_axis) {
  InputReduceInfo info;
  info.Initialize();
  std::vector<int64_t> axis = {0, 2, 3};
  info.SetAxis(axis);
  EXPECT_EQ(info.GetAxis(), axis);
}

TEST_F(UtestOpSliceInfo, InputReduceInfo_copy_ctor) {
  InputReduceInfo info;
  info.Initialize();
  size_t idx = 6;
  info.SetIndex(idx);

  InputReduceInfo copy(info);
  EXPECT_FALSE(copy.IsPtrNull());
  EXPECT_EQ(copy.GetIndex(), 6U);
}

TEST_F(UtestOpSliceInfo, InputReduceInfo_assign_op) {
  InputReduceInfo info;
  info.Initialize();
  size_t idx = 8;
  info.SetIndex(idx);

  InputReduceInfo other;
  other.Initialize();
  other = info;
  EXPECT_EQ(other.GetIndex(), 8U);
}

// ======================== OutputReduceInfo ========================

TEST_F(UtestOpSliceInfo, OutputReduceInfo_default_ctor_is_ptr_null) {
  OutputReduceInfo info;
  EXPECT_TRUE(info.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, OutputReduceInfo_initialize_success) {
  OutputReduceInfo info;
  EXPECT_TRUE(info.Initialize());
  EXPECT_FALSE(info.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, OutputReduceInfo_set_get_index) {
  OutputReduceInfo info;
  info.Initialize();
  size_t idx = 10;
  info.SetIndex(idx);
  EXPECT_EQ(info.GetIndex(), 10U);
}

TEST_F(UtestOpSliceInfo, OutputReduceInfo_set_get_reduce_type) {
  OutputReduceInfo info;
  info.Initialize();
  info.SetReduceType(REDUCE_ADD);
  EXPECT_EQ(info.GetReduceType(), REDUCE_ADD);

  info.SetReduceType(REDUCE_MAX);
  EXPECT_EQ(info.GetReduceType(), REDUCE_MAX);

  info.SetReduceType(REDUCE_MIN);
  EXPECT_EQ(info.GetReduceType(), REDUCE_MIN);
}

TEST_F(UtestOpSliceInfo, OutputReduceInfo_set_get_is_atomic) {
  OutputReduceInfo info;
  info.Initialize();
  EXPECT_FALSE(info.GetIsAtomic());
  info.SetIsAtomic(true);
  EXPECT_TRUE(info.GetIsAtomic());
  info.SetIsAtomic(false);
  EXPECT_FALSE(info.GetIsAtomic());
}

TEST_F(UtestOpSliceInfo, OutputReduceInfo_copy_ctor) {
  OutputReduceInfo info;
  info.Initialize();
  size_t idx = 11;
  info.SetIndex(idx);
  info.SetReduceType(REDUCE_MEAN);
  info.SetIsAtomic(true);

  OutputReduceInfo copy(info);
  EXPECT_FALSE(copy.IsPtrNull());
  EXPECT_EQ(copy.GetIndex(), 11U);
  EXPECT_EQ(copy.GetReduceType(), REDUCE_MEAN);
  EXPECT_TRUE(copy.GetIsAtomic());
}

TEST_F(UtestOpSliceInfo, OutputReduceInfo_assign_op) {
  OutputReduceInfo info;
  info.Initialize();
  info.SetReduceType(REDUCE_ADD);

  OutputReduceInfo other;
  other.Initialize();
  other = info;
  EXPECT_EQ(other.GetReduceType(), REDUCE_ADD);
}

// ======================== AxisSplitMap ========================

TEST_F(UtestOpSliceInfo, AxisSplitMap_default_ctor_is_ptr_null) {
  AxisSplitMap map;
  EXPECT_TRUE(map.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, AxisSplitMap_initialize_success) {
  AxisSplitMap map;
  EXPECT_TRUE(map.Initialize());
  EXPECT_FALSE(map.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, AxisSplitMap_copy_ctor) {
  AxisSplitMap map;
  map.Initialize();

  AxisSplitMap copy(map);
  EXPECT_FALSE(copy.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, AxisSplitMap_assign_op) {
  AxisSplitMap map;
  map.Initialize();

  AxisSplitMap other;
  other.Initialize();
  other = map;
  EXPECT_FALSE(other.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, AxisSplitMap_add_input_split_info) {
  AxisSplitMap map;
  map.Initialize();

  InputSplitInfo input_info;
  input_info.Initialize();
  size_t idx = 1;
  input_info.SetIndex(idx);
  std::vector<int64_t> axis = {0};
  input_info.SetAxis(axis);

  map.AddInputSplitInfo(input_info);
  auto infos = map.GetInputSplitInfos();
  EXPECT_EQ(infos.size(), 1U);
}

TEST_F(UtestOpSliceInfo, AxisSplitMap_add_input_split_info_uninitialized) {
  AxisSplitMap map;
  map.Initialize();

  InputSplitInfo input_info;
  map.AddInputSplitInfo(input_info);
  auto infos = map.GetInputSplitInfos();
  EXPECT_EQ(infos.size(), 1U);
}

TEST_F(UtestOpSliceInfo, AxisSplitMap_set_input_split_infos_vec) {
  AxisSplitMap map;
  map.Initialize();

  InputSplitInfo info1;
  info1.Initialize();
  size_t idx1 = 0;
  info1.SetIndex(idx1);
  InputSplitInfo info2;
  info2.Initialize();
  size_t idx2 = 1;
  info2.SetIndex(idx2);

  std::vector<InputSplitInfo> vec = {info1, info2};
  map.SetInputSplitInfos(vec);
  auto infos = map.GetInputSplitInfos();
  EXPECT_EQ(infos.size(), 2U);
}

TEST_F(UtestOpSliceInfo, AxisSplitMap_set_input_split_infos_ptr_vec) {
  AxisSplitMap map;
  map.Initialize();

  auto ptr1 = std::make_shared<InputSplitInfo>();
  ptr1->Initialize();
  auto ptr2 = std::make_shared<InputSplitInfo>();
  ptr2->Initialize();

  std::vector<InputSplitInfoPtr> vec = {ptr1, ptr2};
  map.SetInputSplitInfos(vec);
  auto infos = map.GetInputSplitInfos();
  EXPECT_EQ(infos.size(), 2U);
}

TEST_F(UtestOpSliceInfo, AxisSplitMap_get_input_split_info_vec) {
  AxisSplitMap map;
  map.Initialize();

  InputSplitInfo info;
  info.Initialize();
  size_t idx = 3;
  info.SetIndex(idx);
  map.AddInputSplitInfo(info);

  auto vec = map.GetInputSplitInfoVec();
  EXPECT_EQ(vec.size(), 1U);
  EXPECT_EQ(vec[0].GetIndex(), 3U);
}

TEST_F(UtestOpSliceInfo, AxisSplitMap_add_output_split_info) {
  AxisSplitMap map;
  map.Initialize();

  OutputSplitInfo output_info;
  output_info.Initialize();
  size_t idx = 2;
  output_info.SetIndex(idx);

  map.AddOutputSplitInfo(output_info);
  auto infos = map.GetOutputSplitInfos();
  EXPECT_EQ(infos.size(), 1U);
}

TEST_F(UtestOpSliceInfo, AxisSplitMap_add_output_split_info_uninitialized) {
  AxisSplitMap map;
  map.Initialize();

  OutputSplitInfo output_info;
  map.AddOutputSplitInfo(output_info);
  auto infos = map.GetOutputSplitInfos();
  EXPECT_EQ(infos.size(), 1U);
}

TEST_F(UtestOpSliceInfo, AxisSplitMap_set_output_split_infos_vec) {
  AxisSplitMap map;
  map.Initialize();

  OutputSplitInfo info1;
  info1.Initialize();
  OutputSplitInfo info2;
  info2.Initialize();

  std::vector<OutputSplitInfo> vec = {info1, info2};
  map.SetOutputSplitInfos(vec);
  auto infos = map.GetOutputSplitInfos();
  EXPECT_EQ(infos.size(), 2U);
}

TEST_F(UtestOpSliceInfo, AxisSplitMap_set_output_split_infos_ptr_vec) {
  AxisSplitMap map;
  map.Initialize();

  auto ptr1 = std::make_shared<OutputSplitInfo>();
  ptr1->Initialize();

  std::vector<OutputSplitInfoPtr> vec = {ptr1};
  map.SetOutputSplitInfos(vec);
  auto infos = map.GetOutputSplitInfos();
  EXPECT_EQ(infos.size(), 1U);
}

TEST_F(UtestOpSliceInfo, AxisSplitMap_get_output_split_info_vec) {
  AxisSplitMap map;
  map.Initialize();

  OutputSplitInfo info;
  info.Initialize();
  size_t idx = 5;
  info.SetIndex(idx);
  map.AddOutputSplitInfo(info);

  auto vec = map.GetOutputSplitInfoVec();
  EXPECT_EQ(vec.size(), 1U);
  EXPECT_EQ(vec[0].GetIndex(), 5U);
}

// ======================== AxisReduceMap ========================

TEST_F(UtestOpSliceInfo, AxisReduceMap_default_ctor_is_ptr_null) {
  AxisReduceMap map;
  EXPECT_TRUE(map.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, AxisReduceMap_initialize_success) {
  AxisReduceMap map;
  EXPECT_TRUE(map.Initialize());
  EXPECT_FALSE(map.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, AxisReduceMap_copy_ctor) {
  AxisReduceMap map;
  map.Initialize();

  AxisReduceMap copy(map);
  EXPECT_FALSE(copy.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, AxisReduceMap_assign_op) {
  AxisReduceMap map;
  map.Initialize();

  AxisReduceMap other;
  other.Initialize();
  other = map;
  EXPECT_FALSE(other.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, AxisReduceMap_add_input_reduce_info) {
  AxisReduceMap map;
  map.Initialize();

  InputReduceInfo info;
  info.Initialize();
  size_t idx = 0;
  info.SetIndex(idx);
  std::vector<int64_t> axis = {1};
  info.SetAxis(axis);

  map.AddInputReduceInfo(info);
  auto infos = map.GetInputReduceInfos();
  EXPECT_EQ(infos.size(), 1U);
}

TEST_F(UtestOpSliceInfo, AxisReduceMap_add_input_reduce_info_uninitialized) {
  AxisReduceMap map;
  map.Initialize();

  InputReduceInfo info;
  map.AddInputReduceInfo(info);
  auto infos = map.GetInputReduceInfos();
  EXPECT_EQ(infos.size(), 1U);
}

TEST_F(UtestOpSliceInfo, AxisReduceMap_set_input_reduce_infos_vec) {
  AxisReduceMap map;
  map.Initialize();

  InputReduceInfo info1;
  info1.Initialize();
  InputReduceInfo info2;
  info2.Initialize();

  std::vector<InputReduceInfo> vec = {info1, info2};
  map.SetInputReduceInfos(vec);
  auto infos = map.GetInputReduceInfos();
  EXPECT_EQ(infos.size(), 2U);
}

TEST_F(UtestOpSliceInfo, AxisReduceMap_set_input_reduce_infos_ptr_vec) {
  AxisReduceMap map;
  map.Initialize();

  auto ptr1 = std::make_shared<InputReduceInfo>();
  ptr1->Initialize();

  std::vector<InputReduceInfoPtr> vec = {ptr1};
  map.SetInputReduceInfos(vec);
  auto infos = map.GetInputReduceInfos();
  EXPECT_EQ(infos.size(), 1U);
}

TEST_F(UtestOpSliceInfo, AxisReduceMap_get_input_reduce_info_vec) {
  AxisReduceMap map;
  map.Initialize();

  InputReduceInfo info;
  info.Initialize();
  size_t idx = 4;
  info.SetIndex(idx);
  map.AddInputReduceInfo(info);

  auto vec = map.GetInputReduceInfoVec();
  EXPECT_EQ(vec.size(), 1U);
  EXPECT_EQ(vec[0].GetIndex(), 4U);
}

TEST_F(UtestOpSliceInfo, AxisReduceMap_add_output_reduce_info) {
  AxisReduceMap map;
  map.Initialize();

  OutputReduceInfo info;
  info.Initialize();
  size_t idx = 1;
  info.SetIndex(idx);
  info.SetReduceType(REDUCE_ADD);

  map.AddOutputReduceInfo(info);
  auto infos = map.GetOutputReduceInfos();
  EXPECT_EQ(infos.size(), 1U);
}

TEST_F(UtestOpSliceInfo, AxisReduceMap_add_output_reduce_info_uninitialized) {
  AxisReduceMap map;
  map.Initialize();

  OutputReduceInfo info;
  map.AddOutputReduceInfo(info);
  auto infos = map.GetOutputReduceInfos();
  EXPECT_EQ(infos.size(), 1U);
}

TEST_F(UtestOpSliceInfo, AxisReduceMap_set_output_reduce_infos_vec) {
  AxisReduceMap map;
  map.Initialize();

  OutputReduceInfo info1;
  info1.Initialize();
  OutputReduceInfo info2;
  info2.Initialize();

  std::vector<OutputReduceInfo> vec = {info1, info2};
  map.SetOutputReduceInfos(vec);
  auto infos = map.GetOutputReduceInfos();
  EXPECT_EQ(infos.size(), 2U);
}

TEST_F(UtestOpSliceInfo, AxisReduceMap_set_output_reduce_infos_ptr_vec) {
  AxisReduceMap map;
  map.Initialize();

  auto ptr1 = std::make_shared<OutputReduceInfo>();
  ptr1->Initialize();

  std::vector<OutputReduceInfoPtr> vec = {ptr1};
  map.SetOutputReduceInfos(vec);
  auto infos = map.GetOutputReduceInfos();
  EXPECT_EQ(infos.size(), 1U);
}

TEST_F(UtestOpSliceInfo, AxisReduceMap_get_output_reduce_info_vec) {
  AxisReduceMap map;
  map.Initialize();

  OutputReduceInfo info;
  info.Initialize();
  size_t idx = 7;
  info.SetIndex(idx);
  map.AddOutputReduceInfo(info);

  auto vec = map.GetOutputReduceInfoVec();
  EXPECT_EQ(vec.size(), 1U);
  EXPECT_EQ(vec[0].GetIndex(), 7U);
}

// ======================== OpCalcInfo ========================

TEST_F(UtestOpSliceInfo, OpCalcInfo_default_ctor_is_ptr_null) {
  OpCalcInfo info;
  EXPECT_TRUE(info.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, OpCalcInfo_initialize_success) {
  OpCalcInfo info;
  EXPECT_TRUE(info.Initialize());
  EXPECT_FALSE(info.IsPtrNull());
}

TEST_F(UtestOpSliceInfo, OpCalcInfo_l1_fusion_default) {
  OpCalcInfo info;
  info.Initialize();
  EXPECT_EQ(info.GetL1FusionEnable(), L1FUSION_DISABLE);
}

TEST_F(UtestOpSliceInfo, OpCalcInfo_set_get_l1_fusion) {
  OpCalcInfo info;
  info.Initialize();
  info.SetL1FusionEnable(L1FUSION_BASIC);
  EXPECT_EQ(info.GetL1FusionEnable(), L1FUSION_BASIC);

  info.SetL1FusionEnable(L1FUSION_INPUT_CTR);
  EXPECT_EQ(info.GetL1FusionEnable(), L1FUSION_INPUT_CTR);
}

TEST_F(UtestOpSliceInfo, OpCalcInfo_set_get_min_tbe_l1_space) {
  OpCalcInfo info;
  info.Initialize();
  EXPECT_EQ(info.GetMinTbeL1Space(), 0);
  info.SetMinTbeL1Space(1024);
  EXPECT_EQ(info.GetMinTbeL1Space(), 1024);
}

TEST_F(UtestOpSliceInfo, OpCalcInfo_add_axis_split_map) {
  OpCalcInfo info;
  info.Initialize();

  AxisSplitMap split_map;
  split_map.Initialize();
  info.AddAxisSplitMap(split_map);

  auto maps = info.GetAxisSplitMaps();
  EXPECT_EQ(maps.size(), 1U);
}

TEST_F(UtestOpSliceInfo, OpCalcInfo_set_axis_split_maps_vec) {
  OpCalcInfo info;
  info.Initialize();

  AxisSplitMap map1;
  map1.Initialize();
  AxisSplitMap map2;
  map2.Initialize();

  std::vector<AxisSplitMap> vec = {map1, map2};
  info.SetAxisSplitMaps(vec);
  auto maps = info.GetAxisSplitMaps();
  EXPECT_EQ(maps.size(), 2U);
}

TEST_F(UtestOpSliceInfo, OpCalcInfo_set_axis_split_maps_ptr_vec) {
  OpCalcInfo info;
  info.Initialize();

  auto ptr1 = std::make_shared<AxisSplitMap>();
  ptr1->Initialize();

  std::vector<AxisSplitMapPtr> vec = {ptr1};
  info.SetAxisSplitMaps(vec);
  auto maps = info.GetAxisSplitMaps();
  EXPECT_EQ(maps.size(), 1U);
}

TEST_F(UtestOpSliceInfo, OpCalcInfo_get_axis_split_map_vec) {
  OpCalcInfo info;
  info.Initialize();

  AxisSplitMap split_map;
  split_map.Initialize();
  info.AddAxisSplitMap(split_map);

  auto vec = info.GetAxisSplitMapVec();
  EXPECT_EQ(vec.size(), 1U);
}

TEST_F(UtestOpSliceInfo, OpCalcInfo_add_axis_reduce_map) {
  OpCalcInfo info;
  info.Initialize();

  AxisReduceMap reduce_map;
  reduce_map.Initialize();
  info.AddAxisReduceMap(reduce_map);

  auto maps = info.GetAxisReduceMaps();
  EXPECT_EQ(maps.size(), 1U);
}

TEST_F(UtestOpSliceInfo, OpCalcInfo_set_axis_reduce_maps_vec) {
  OpCalcInfo info;
  info.Initialize();

  AxisReduceMap map1;
  map1.Initialize();
  AxisReduceMap map2;
  map2.Initialize();

  std::vector<AxisReduceMap> vec = {map1, map2};
  info.SetAxisReduceMaps(vec);
  auto maps = info.GetAxisReduceMaps();
  EXPECT_EQ(maps.size(), 2U);
}

TEST_F(UtestOpSliceInfo, OpCalcInfo_set_axis_reduce_maps_ptr_vec) {
  OpCalcInfo info;
  info.Initialize();

  auto ptr1 = std::make_shared<AxisReduceMap>();
  ptr1->Initialize();

  std::vector<AxisReduceMapPtr> vec = {ptr1};
  info.SetAxisReduceMaps(vec);
  auto maps = info.GetAxisReduceMaps();
  EXPECT_EQ(maps.size(), 1U);
}

TEST_F(UtestOpSliceInfo, OpCalcInfo_get_axis_reduce_map_vec) {
  OpCalcInfo info;
  info.Initialize();

  AxisReduceMap reduce_map;
  reduce_map.Initialize();
  info.AddAxisReduceMap(reduce_map);

  auto vec = info.GetAxisReduceMapVec();
  EXPECT_EQ(vec.size(), 1U);
}

TEST_F(UtestOpSliceInfo, OpCalcInfo_del_axis_split_map_base_axis_found) {
  OpCalcInfo info;
  info.Initialize();

  AxisSplitMap split_map;
  split_map.Initialize();

  InputSplitInfo input_info;
  input_info.Initialize();
  std::vector<int64_t> axis = {0, 1};
  input_info.SetAxis(axis);
  split_map.AddInputSplitInfo(input_info);

  info.AddAxisSplitMap(split_map);
  EXPECT_EQ(info.GetAxisSplitMaps().size(), 1U);

  std::vector<int64_t> del_axis = {0, 1};
  info.DelAxisSplitMapBaseAxis(del_axis);
  EXPECT_EQ(info.GetAxisSplitMaps().size(), 0U);
}

TEST_F(UtestOpSliceInfo, OpCalcInfo_del_axis_split_map_base_axis_not_found) {
  OpCalcInfo info;
  info.Initialize();

  AxisSplitMap split_map;
  split_map.Initialize();

  InputSplitInfo input_info;
  input_info.Initialize();
  std::vector<int64_t> axis = {0, 1};
  input_info.SetAxis(axis);
  split_map.AddInputSplitInfo(input_info);

  info.AddAxisSplitMap(split_map);

  std::vector<int64_t> del_axis = {2, 3};
  info.DelAxisSplitMapBaseAxis(del_axis);
  EXPECT_EQ(info.GetAxisSplitMaps().size(), 1U);
}

TEST_F(UtestOpSliceInfo, OpCalcInfo_del_axis_split_map_empty) {
  OpCalcInfo info;
  info.Initialize();

  std::vector<int64_t> del_axis = {0};
  info.DelAxisSplitMapBaseAxis(del_axis);
  EXPECT_EQ(info.GetAxisSplitMaps().size(), 0U);
}
