#include <gtest/gtest.h>
#include <vector>
#include "graph/tensor.h"
#include "graph/ge_tensor.h"
#include "graph/utils/tensor_value_utils.h"
#include "graph/utils/tensor_adapter.h"

namespace ge {

class TensorValueUtilsTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TensorValueUtilsTest, CovConvertFloat) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  TensorDesc desc(Shape({3}), FORMAT_NCHW, DT_FLOAT);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(data.data()), data.size() * sizeof(float));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_FLOAT, ",", true);
  EXPECT_NE(result.find("1"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertInt32) {
  std::vector<int32_t> data = {10, 20, 30};
  TensorDesc desc(Shape({3}), FORMAT_NCHW, DT_INT32);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(data.data()), data.size() * sizeof(int32_t));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_INT32, ",", true);
  EXPECT_NE(result.find("10"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertInt64) {
  std::vector<int64_t> data = {100, 200};
  TensorDesc desc(Shape({2}), FORMAT_NCHW, DT_INT64);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(data.data()), data.size() * sizeof(int64_t));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_INT64, ",", true);
  EXPECT_NE(result.find("100"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertDouble) {
  std::vector<double> data = {1.5, 2.5};
  TensorDesc desc(Shape({2}), FORMAT_NCHW, DT_DOUBLE);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(data.data()), data.size() * sizeof(double));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_DOUBLE, ",", true);
  EXPECT_NE(result.find("1.5"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertBool) {
  bool data[] = {true, false};
  TensorDesc desc(Shape({2}), FORMAT_NCHW, DT_BOOL);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(data), 2U * sizeof(bool));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_BOOL, ",", true);
  EXPECT_NE(result.find("true"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertFp16) {
  uint16_t fp16_data[2] = {0x3C00, 0x4000};
  TensorDesc desc(Shape({2}), FORMAT_NCHW, DT_FLOAT16);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(fp16_data), 2U * sizeof(uint16_t));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_FLOAT16, ",", true);
  EXPECT_NE(result.find("1"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertFp16NoSkip) {
  uint16_t fp16_data[3] = {0x3C00, 0x4000, 0x4200};
  TensorDesc desc(Shape({3}), FORMAT_NCHW, DT_FLOAT16);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(fp16_data), 3U * sizeof(uint16_t));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_FLOAT16, ",", false);
  EXPECT_NE(result.find("1"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertFp16Infinity) {
  uint16_t fp16_data[2] = {0x7C00, 0xFC00};
  TensorDesc desc(Shape({2}), FORMAT_NCHW, DT_FLOAT16);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(fp16_data), 2U * sizeof(uint16_t));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_FLOAT16, ",", true);
  EXPECT_NE(result.find("inf"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertFp16NaN) {
  uint16_t fp16_data[1] = {0x7E00};
  TensorDesc desc(Shape({1}), FORMAT_NCHW, DT_FLOAT16);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(fp16_data), 1U * sizeof(uint16_t));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_FLOAT16, ",", true);
  EXPECT_NE(result.find("nan"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertFp16Denormalized) {
  uint16_t fp16_data[2] = {0x0001, 0x0200};
  TensorDesc desc(Shape({2}), FORMAT_NCHW, DT_FLOAT16);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(fp16_data), 2U * sizeof(uint16_t));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_FLOAT16, ",", true);
  EXPECT_NE(result.find("["), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertFp16Zero) {
  uint16_t fp16_data[2] = {0x0000, 0x8000};
  TensorDesc desc(Shape({2}), FORMAT_NCHW, DT_FLOAT16);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(fp16_data), 2U * sizeof(uint16_t));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_FLOAT16, ",", true);
  EXPECT_NE(result.find("0"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertUnsupportedType) {
  TensorDesc desc(Shape({1}), FORMAT_NCHW, DT_STRING);
  Tensor tensor(desc);
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_STRING, ",", true);
  EXPECT_EQ(result, "<not_supported>");
}

TEST_F(TensorValueUtilsTest, CovConvertEmptyTensor) {
  TensorDesc desc(Shape({0}), FORMAT_NCHW, DT_FLOAT);
  Tensor tensor(desc);
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_FLOAT, ",", true);
  EXPECT_EQ(result, "<empty>");
}

TEST_F(TensorValueUtilsTest, CovConvertNullData) {
  GeTensor ge_tensor(GeTensorDesc(GeShape({2}), FORMAT_NCHW, DT_FLOAT));
  ge_tensor.SetData(std::shared_ptr<AlignedPtr>(), 10U);
  Tensor tensor = TensorAdapter::AsTensor(ge_tensor);
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_FLOAT, ",", true);
  EXPECT_EQ(result, "<invalid>");
}

TEST_F(TensorValueUtilsTest, CovConvertUnalignedData) {
  uint8_t data[3] = {1, 2, 3};
  TensorDesc desc(Shape({3}), FORMAT_NCHW, DT_UINT8);
  Tensor tensor(desc, data, 3U);
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_FLOAT, ",", true);
  EXPECT_EQ(result, "<invalid>");
}

TEST_F(TensorValueUtilsTest, CovConvertUint8) {
  std::vector<uint8_t> data = {1, 2, 3};
  TensorDesc desc(Shape({3}), FORMAT_NCHW, DT_UINT8);
  Tensor tensor(desc, data.data(), data.size() * sizeof(uint8_t));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_UINT8, ",", true);
  EXPECT_NE(result.find("1"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertInt8) {
  std::vector<int8_t> data = {1, 2, 3};
  TensorDesc desc(Shape({3}), FORMAT_NCHW, DT_INT8);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(data.data()), data.size() * sizeof(int8_t));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_INT8, ",", true);
  EXPECT_NE(result.find("1"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertInt16) {
  std::vector<int16_t> data = {1, 2};
  TensorDesc desc(Shape({2}), FORMAT_NCHW, DT_INT16);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(data.data()), data.size() * sizeof(int16_t));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_INT16, ",", true);
  EXPECT_NE(result.find("1"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertUint16) {
  std::vector<uint16_t> data = {1, 2};
  TensorDesc desc(Shape({2}), FORMAT_NCHW, DT_UINT16);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(data.data()), data.size() * sizeof(uint16_t));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_UINT16, ",", true);
  EXPECT_NE(result.find("1"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertUint32) {
  std::vector<uint32_t> data = {1, 2};
  TensorDesc desc(Shape({2}), FORMAT_NCHW, DT_UINT32);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(data.data()), data.size() * sizeof(uint32_t));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_UINT32, ",", true);
  EXPECT_NE(result.find("1"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertUint64) {
  std::vector<uint64_t> data = {1, 2};
  TensorDesc desc(Shape({2}), FORMAT_NCHW, DT_UINT64);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(data.data()), data.size() * sizeof(uint64_t));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_UINT64, ",", true);
  EXPECT_NE(result.find("1"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertSingleElement) {
  std::vector<float> data = {42.0f};
  TensorDesc desc(Shape({1}), FORMAT_NCHW, DT_FLOAT);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(data.data()), data.size() * sizeof(float));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_FLOAT, ",", true);
  EXPECT_NE(result.find("42"), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertManyElementsSkip) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  TensorDesc desc(Shape({8}), FORMAT_NCHW, DT_FLOAT);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(data.data()), data.size() * sizeof(float));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_FLOAT, ",", true);
  EXPECT_NE(result.find("..."), std::string::npos);
}

TEST_F(TensorValueUtilsTest, CovConvertNoSkipFloat) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  TensorDesc desc(Shape({3}), FORMAT_NCHW, DT_FLOAT);
  Tensor tensor(desc, reinterpret_cast<const uint8_t *>(data.data()), data.size() * sizeof(float));
  std::string result = TensorValueUtils::ConvertTensorValue(tensor, DT_FLOAT, ",", false);
  EXPECT_NE(result.find("1"), std::string::npos);
  EXPECT_NE(result.find("3"), std::string::npos);
}

}  // namespace ge
