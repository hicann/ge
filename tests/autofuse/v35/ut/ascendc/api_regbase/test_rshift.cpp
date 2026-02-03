/**
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/**
 * test_rshift.cpp
 */

#include <cmath>
#include <random>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"

using namespace AscendC;

namespace ge {

template <typename T>
struct RShiftInputParam {
  T *y{};
  T *x1{};
  T *x2{};
  T *exp{};
  uint32_t size{};
};

class TestRegbaseApiRShiftUT : public testing::Test {
 protected:
  // Tensor - Tensor 场景
  template <typename T>
  static void InvokeTensorTensorKernel(RShiftInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf;
    tpipe.InitBuffer(x1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(x2buf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));

    LocalTensor<T> l_x1 = x1buf.Get<T>();
    LocalTensor<T> l_x2 = x2buf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();

    GmToUb(l_x1, param.x1, param.size);
    GmToUb(l_x2, param.x2, param.size);
    ShiftRight(l_y, l_x1, l_x2, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorInput(RShiftInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x2 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);

    for (uint32_t i = 0; i < param.size; i++) {
      // x1: 数据值，范围较小以避免右移后溢出
      // x2: 移位值，范围 [0, 31] 对于32位整数
      if constexpr (std::is_same_v<T, int64_t>) {
        std::uniform_int_distribution<int64_t> data_distr(-10000, 10000);
        std::uniform_int_distribution<int> shift_distr(0, 63);
        param.x1[i] = data_distr(eng);
        param.x2[i] = static_cast<T>(shift_distr(eng));
        param.exp[i] = param.x1[i] >> param.x2[i];
      } else {
        std::uniform_int_distribution<int> data_distr(-10000, 10000);
        std::uniform_int_distribution<int> shift_distr(0, 31);
        param.x1[i] = static_cast<T>(data_distr(eng));
        param.x2[i] = static_cast<T>(shift_distr(eng));
        param.exp[i] = param.x1[i] >> param.x2[i];
      }
    }
  }

  template <typename T>
  static uint32_t Valid(RShiftInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      if (param.y[i] != param.exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  // Tensor - Tensor 测试
  template <typename T>
  static void RShiftTensorTensorTest(uint32_t size) {
    RShiftInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);
  }

};

// ============ Tensor - Tensor 测试 ============
TEST_F(TestRegbaseApiRShiftUT, RShift_TensorTensor_Test) {
  // int8
  RShiftTensorTensorTest<int8_t>(ONE_BLK_SIZE / sizeof(int8_t));
  RShiftTensorTensorTest<int8_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  RShiftTensorTensorTest<int8_t>((ONE_BLK_SIZE - sizeof(int8_t)) / sizeof(int8_t));
  RShiftTensorTensorTest<int8_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int8_t));
  RShiftTensorTensorTest<int8_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int8_t));
  RShiftTensorTensorTest<int8_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int8_t));

  // int16
  RShiftTensorTensorTest<int16_t>(ONE_BLK_SIZE / sizeof(int16_t));
  RShiftTensorTensorTest<int16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  RShiftTensorTensorTest<int16_t>((ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t));
  RShiftTensorTensorTest<int16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t));
  RShiftTensorTensorTest<int16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int16_t));
  RShiftTensorTensorTest<int16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int16_t));

  // int32
  RShiftTensorTensorTest<int>(ONE_BLK_SIZE / sizeof(int));
  RShiftTensorTensorTest<int>(ONE_REPEAT_BYTE_SIZE / sizeof(int));
  RShiftTensorTensorTest<int>((ONE_BLK_SIZE - sizeof(int)) / sizeof(int));
  RShiftTensorTensorTest<int>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int));
  RShiftTensorTensorTest<int>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int));
  RShiftTensorTensorTest<int>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int));

  // int64
  RShiftTensorTensorTest<int64_t>(ONE_BLK_SIZE / sizeof(int64_t));
  RShiftTensorTensorTest<int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t));
  RShiftTensorTensorTest<int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t));
  RShiftTensorTensorTest<int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t));
  RShiftTensorTensorTest<int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int64_t));
  RShiftTensorTensorTest<int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int64_t));
}

}  // namespace ge