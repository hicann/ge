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
 * test_compare.cpp
 */

#include <cmath>
#include "gtest/gtest.h"
#include "test_api_utils.h"
#include "tikicpulib.h"
#include "utils.h"
// 保持在utils.h之后
#include "duplicate.h"
// 保持在duplicate.h之后
#include "api_regbase/compare.h"

using namespace AscendC;

namespace ge {
template <typename O, typename I, uint8_t dim, CMPMODE mode>
struct CompareInputParam {
  O *y{};
  bool *exp{};
  I *src0{};
  I src1;
  CMPMODE cmpmode{CMPMODE::EQ};
  uint16_t size{0};
  uint16_t out_size{0};
  BinaryRepeatParams a;
};

template <typename O, typename I, uint8_t dim, CMPMODE mode>
struct TensorCompareInputParam {
  O *y{};
  bool *exp{};
  I *src0{};
  I *src1{};
  CMPMODE cmpmode{CMPMODE::EQ};
  uint16_t size{0};
  uint16_t out_size{0};
  uint16_t first_axis{0};
  uint16_t last_axis{0};
  BinaryRepeatParams a;
};

class TestApiCompareUT : public testing::Test {
protected:
    template <typename O, typename I, uint8_t dim, CMPMODE mode>
    static void CreateInput(CompareInputParam<O, I, dim, mode> &param, float def_src1) 
    {
        // 构造测试输入和预期结果
        param.y = static_cast<O *>(AscendC::GmAlloc(sizeof(O) * param.size));
        param.exp = static_cast<bool *>(AscendC::GmAlloc(sizeof(bool) * param.size));
        param.src0 = static_cast<I *>(AscendC::GmAlloc(sizeof(I) * param.size));

        if constexpr (std::is_same_v<I, int64_t>) {
            param.src1 = 0xAAAAAAAABBBBBBBB;
        } else {
            param.src1 = def_src1;
        }

        int input_range = 10;
        std::mt19937 eng(1);                                         // Seed the generator
        std::uniform_int_distribution distr(0, input_range);  // Define the range

        for (int i = 0; i < param.size; i++) {
        auto input = distr(eng);  // Use the secure random number generator
        param.src0[i] = input;
        switch (param.cmpmode) {
            case CMPMODE::EQ:
            if (input > 5 || i == param.size - 1) {
                param.src0[i] = param.src1;
                param.exp[i] = true;
            } else {
                param.exp[i] = DefaultCompare(input, param.src1);
            }
            break;
            case CMPMODE::NE:
            if (input > 5 || i == param.size - 1) {
                param.src0[i] = param.src1;
                param.exp[i] = false;
            } else {
                param.exp[i] = !DefaultCompare(param.src0[i], param.src1);
            }
            break;
            case CMPMODE::GE:
            if constexpr (std::is_same_v<I, half>) {
                param.exp[i] = static_cast<half>(input) >= param.src1;
            } else {
                param.exp[i] = input >= param.src1;
            }
            break;
            case CMPMODE::LE:
            if constexpr (std::is_same_v<I, half>) {
                param.exp[i] = static_cast<half>(input) <= param.src1;
            } else {
                param.exp[i] = input <= param.src1;
            }
            break;
            case CMPMODE::GT:
            if constexpr (std::is_same_v<I, half>) {
                param.exp[i] = static_cast<half>(input) > param.src1;
            } else {
                param.exp[i] = input > param.src1;
            }
            break;
            default:
            break;
        }
        }
    }

    template <typename O, typename I, uint8_t dim, CMPMODE mode>
    static uint32_t Valid(CompareInputParam<O, I, dim, mode> &param) 
    {
        uint32_t diff_count = 0;

        for (uint32_t i = 0; i < param.size; i++) {
            if (static_cast<bool>(param.y[i]) != param.exp[i]) {
                diff_count++;
            }
        }
        return diff_count;
    }

    template <typename O, typename I, uint8_t dim, CMPMODE mode>
    static void InvokeKernel(CompareInputParam<O, I, dim, mode> &param) 
    {
        TPipe tpipe;
        TBuf<TPosition::VECCALC> x1buf, ybuf;
        tpipe.InitBuffer(x1buf, sizeof(I) * param.size);
        tpipe.InitBuffer(ybuf, sizeof(O) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(O)));
    
        LocalTensor<I> l_x1 = x1buf.Get<I>();
        LocalTensor<O> l_y = ybuf.Get<O>();

        GmToUb(l_x1, param.src0, param.size);
        const uint16_t output_dims[1] = {param.size};
        const uint16_t output_stride[1] = {1};
        const uint16_t input_stride[1] = {1};
        CompareScalarExtend<I, dim, mode>(l_y, l_x1, param.src1, output_dims, output_stride, input_stride);
        UbToGm(param.y, l_y, param.size);
    }

    template <typename O, typename I, uint8_t dim, CMPMODE mode>
    static void CompareTest(uint16_t size, float def_src1 = 4.5) 
    {
        CompareInputParam<O, I, dim, mode> param{};
        param.size = size;
        param.cmpmode = mode;

        CreateInput(param, def_src1);

        // 构造Api调用函数
        auto kernel = [&param] { InvokeKernel(param); };

        // 调用kernel
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(kernel, 1);

        // 验证结果
        uint32_t diff_count = Valid(param);
        EXPECT_EQ(diff_count, 0);
    }


  /* -------------------- 输入是两个tensor相关的测试基础方法定义(count+normal)-------------------- */

    template <typename O, typename I, uint8_t dim, CMPMODE mode>
    static void InvokeKernelWithTwoTensorInput(TensorCompareInputParam<O, I, dim, mode> &param) 
    {
        TPipe tpipe;
        TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf;
        if constexpr (dim == 1) {
            tpipe.InitBuffer(x1buf, sizeof(I) * param.size);
            tpipe.InitBuffer(x2buf, sizeof(I) * param.size);
            tpipe.InitBuffer(ybuf, sizeof(O) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(O)));

            LocalTensor<I> l_x1 = x1buf.Get<I>();
            LocalTensor<I> l_x2 = x2buf.Get<I>();

            LocalTensor<O> l_y = ybuf.Get<O>();

            GmToUb(l_x1, param.src0, param.size);
            GmToUb(l_x2, param.src1, param.size);
            const uint16_t output_dims[dim] = {param.size};
            const uint16_t output_stride[dim] = {1};
            const uint16_t input_stride[dim] = {1};
            CompareExtend<I, dim, mode>(l_y, l_x1, l_x2, output_dims, output_stride, input_stride);
            UbToGm(param.y, l_y, param.size);
        } else if constexpr (dim == 2) {
            constexpr auto alignInput = ONE_BLK_SIZE / sizeof(I);
            constexpr auto alignOutput = ONE_BLK_SIZE / sizeof(O);
            uint16_t inputStride = CeilDivision(param.last_axis, alignInput) * alignInput;
            uint16_t outputStride = CeilDivision(param.last_axis, alignOutput) * alignOutput;

            uint32_t inputSize = param.first_axis * inputStride;
            uint32_t outputSize = param.first_axis * outputStride;

            tpipe.InitBuffer(x1buf, sizeof(I) * inputSize);
            tpipe.InitBuffer(x2buf, sizeof(I) * inputSize);
            tpipe.InitBuffer(ybuf, sizeof(O) * AlignUp(outputSize, ONE_BLK_SIZE / sizeof(O)));

            LocalTensor<I> l_x1 = x1buf.Get<I>();
            LocalTensor<I> l_x2 = x2buf.Get<I>();
            LocalTensor<O> l_y = ybuf.Get<O>();

            GmToUbNormal(l_x1, param.src0, param.first_axis, param.last_axis, inputStride);
            GmToUbNormal(l_x2, param.src1, param.first_axis, param.last_axis, inputStride);
            const uint16_t output_dims[dim] = {param.first_axis, param.last_axis};
            const uint16_t output_stride[dim] = {outputStride, 1};
            const uint16_t input_stride[dim] = {inputStride, 1};
            CompareExtend<I, dim, mode>(l_y, l_x1, l_x2, output_dims, output_stride, input_stride);
            UbToGmNormal(param.y, l_y, param.first_axis, param.last_axis, outputStride);
        }
    }

    template <typename O, typename I, uint8_t dim, CMPMODE mode>
    static void CreateTensorInput(TensorCompareInputParam<O, I, dim, mode> &param, float def_src1) 
    {
        // 构造测试输入和预期结果
        param.y = static_cast<O *>(AscendC::GmAlloc(sizeof(O) * param.size));
        param.exp = static_cast<bool *>(AscendC::GmAlloc(sizeof(bool) * param.size));
        param.src0 = static_cast<I *>(AscendC::GmAlloc(sizeof(I) * param.size));
        param.src1 = static_cast<I *>(AscendC::GmAlloc(sizeof(I) * param.size));
        I src1_val;
        if constexpr (std::is_same_v<I, int64_t>) {
            src1_val = 0xAAAAAAAABBBBBBBB;
        } else {
            src1_val = def_src1;
        }
        (void)src1_val;
        int input_range = 10;
        // 构造src0的随机生成器
        std::mt19937 eng(1);
        std::uniform_int_distribution distr(0, input_range);  // Define the range

        // 构造src1的随机生成器
        std::mt19937 eng1(3);                                  // Seed the generator
        std::uniform_int_distribution distr1(0, input_range);  // Define the range
        bool src1IsBlkTensor = false;
        if (param.size * sizeof(I) == ONE_BLK_SIZE) {
            src1IsBlkTensor = true;
            //应用场景：外部调用者会把一个标量scalar广播为一个blk大小的tensor作为src1传入compare接口
        }
        for (int i = 0; i < param.size; i++) {
            auto input = distr(eng);  // Use the secure random number generator
            auto input1 = distr1(eng1);
            param.src0[i] = input;
            if (i > 0) {
                if (src1IsBlkTensor) {
                    param.src1[i] = param.src1[0];
                } else {
                    param.src1[i] = input1;
                }
            } else {
                param.src1[i] = input1;
            }

            switch (param.cmpmode) {
                case CMPMODE::EQ:
                    if (input > 5 || i == param.size - 1) {
                        param.src0[i] = param.src1[i];
                        param.exp[i] = true;
                    } else {
                        param.exp[i] = DefaultCompare(param.src0[i], param.src1[i]);
                    }
                    break;
                case CMPMODE::NE:
                    if (input > 5 || i == param.size - 1) {
                        param.src0[i] = param.src1[i];
                        param.exp[i] = false;
                    } else {
                        param.exp[i] = !DefaultCompare(param.src0[i], param.src1[i]);
                    }
                    break;
                case CMPMODE::GE:
                    if constexpr (std::is_same_v<I, half>) {
                        param.exp[i] = static_cast<half>(param.src0[i]) >= param.src1[i];
                    } else {
                        param.exp[i] = param.src0[i] >= param.src1[i];
                    }
                    break;
                case CMPMODE::LE:
                    if constexpr (std::is_same_v<I, half>) {
                        param.exp[i] = static_cast<half>(param.src0[i]) <= param.src1[i];
                    } else {
                        param.exp[i] = param.src0[i] <= param.src1[i];
                    }
                    break;
                case CMPMODE::GT:
                    param.exp[i] = param.src0[i] > param.src1[i];
                    break;
                default:
                    break;
            }
        }
    }

    template <typename O>
    static uint32_t Valid(O *y, bool *exp, size_t comp_size) 
    {
        uint32_t diff_count = 0;
        for (uint32_t i = 0; i < comp_size; i++) {
            if (static_cast<bool>(y[i]) != exp[i]) {
                diff_count++;
            }
        }
        return diff_count;
    }

    template <typename O, typename I, uint8_t dim, CMPMODE mode>
    static void TensorCompareTestNormal(uint16_t first_axis, uint16_t last_axis, float def_src1 = 4.5) 
    {
        TensorCompareInputParam<O, I, dim, mode> param{};
        param.first_axis = first_axis;
        param.last_axis = last_axis;
        param.size = first_axis * last_axis;
        param.cmpmode = mode;

        CreateTensorInput(param, def_src1);

        // 构造Api调用函数
        auto kernel = [&param] { InvokeKernelWithTwoTensorInput(param); };

        // 调用kernel
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(kernel, 1);

        // 验证结果
        uint32_t diff_count = Valid(param.y, param.exp, param.size);
        EXPECT_EQ(diff_count, 0);
    }
    template <typename O, typename I, uint8_t dim, CMPMODE mode>
    static void TensorCompareTest(uint16_t size, float def_src1 = 4.5) 
    {
        TensorCompareInputParam<O, I, dim, mode> param{};
        param.size = size;
        param.cmpmode = mode;

        CreateTensorInput(param, def_src1);

        // 构造Api调用函数
        auto kernel = [&param] { InvokeKernelWithTwoTensorInput(param); };

        // 调用kernel
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(kernel, 1);

        // 验证结果
        uint32_t diff_count = Valid(param.y, param.exp, param.size);
        EXPECT_EQ(diff_count, 0);
    }
};

//comparescalar的count模式
TEST_F(TestApiCompareUT, CompareScalar_Eq_float_uint8) 
{
    CompareTest<uint8_t, float, 1, CMPMODE::EQ>(ONE_BLK_SIZE / sizeof(float));
    CompareTest<uint8_t, float, 1, CMPMODE::EQ>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
    CompareTest<uint8_t, float, 1, CMPMODE::EQ>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float));
    CompareTest<uint8_t, float, 1, CMPMODE::EQ>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(float)) / sizeof(float));
    CompareTest<uint8_t, float, 1, CMPMODE::EQ>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float));
    CompareTest<uint8_t, float, 1, CMPMODE::EQ>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                (ONE_BLK_SIZE - sizeof(float))) /sizeof(float));
}

//compare的count模式
TEST_F(TestApiCompareUT, CompareCount_Eq_float_uint8) 
{
    TensorCompareTest<uint8_t, float, 1, CMPMODE::EQ>(ONE_BLK_SIZE / sizeof(float));
    TensorCompareTest<uint8_t, float, 1, CMPMODE::EQ>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
    TensorCompareTest<uint8_t, float, 1, CMPMODE::EQ>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float));
    TensorCompareTest<uint8_t, float, 1, CMPMODE::EQ>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(float)) / sizeof(float));
    TensorCompareTest<uint8_t, float, 1, CMPMODE::EQ>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float));
    TensorCompareTest<uint8_t, float, 1, CMPMODE::EQ>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                (ONE_BLK_SIZE - sizeof(float))) /sizeof(float));
}

//compare的normal模式
TEST_F(TestApiCompareUT, CompareNormal_Eq_float_uint8) 
{
    TensorCompareTestNormal<uint8_t, float, 2, CMPMODE::EQ>(2, ONE_BLK_SIZE / sizeof(float));
    TensorCompareTestNormal<uint8_t, float, 2, CMPMODE::EQ>(3, ONE_REPEAT_BYTE_SIZE / sizeof(float));
    TensorCompareTestNormal<uint8_t, float, 2, CMPMODE::EQ>(4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float) / 4);
    TensorCompareTestNormal<uint8_t, float, 2, CMPMODE::EQ>(5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(float)) / sizeof(float));
    TensorCompareTestNormal<uint8_t, float, 2, CMPMODE::EQ>(6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float) / 6);
    TensorCompareTestNormal<uint8_t, float, 2, CMPMODE::EQ>(7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                (ONE_BLK_SIZE - sizeof(float))) /sizeof(float) / 8);
}
}  // namespace ge
