#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include "graph/utils/graph_thread_pool.h"
#include "graph/utils/multi_thread_graph_builder.h"
#include "graph/utils/type_utils.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"

namespace ge {

class CovUtilsMiscTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(CovUtilsMiscTest, CovGraphThreadPoolConstructorAndDestructor) {
  GraphThreadPool pool(2U);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovGraphThreadPoolZeroSize) {
  GraphThreadPool pool(0U);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovGraphThreadPoolSubmitTask) {
  GraphThreadPool pool(2U);
  std::atomic<int> counter(0);
  pool.commit([&counter]() { counter++; });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(counter.load(), 1);
}

TEST_F(CovUtilsMiscTest, CovMultiThreadGraphBuilderConstructor) {
  MultiThreadGraphBuilder builder(2);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovMultiThreadGraphBuilderSingleThread) {
  MultiThreadGraphBuilder builder(1);
  ge::Graph graph("test");
  std::vector<ge::Operator> inputs;
  auto &result = builder.SetInputs(inputs, graph);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovMultiThreadGraphBuilderMultiThread) {
  MultiThreadGraphBuilder builder(2);
  ge::Graph graph("test");
  std::vector<ge::Operator> inputs;
  auto &result = builder.SetInputs(inputs, graph);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovTypeUtilsAscendStringToDataType) {
  ge::AscendString str("DT_FLOAT");
  DataType dt = TypeUtils::AscendStringToDataType(str);
  EXPECT_EQ(dt, DT_FLOAT);
}

TEST_F(CovUtilsMiscTest, CovTypeUtilsFormatToAscendString) {
  AscendString result = TypeUtils::FormatToAscendString(FORMAT_NCHW);
  EXPECT_NE(result.GetString(), nullptr);
}

TEST_F(CovUtilsMiscTest, CovTypeUtilsAscendStringToFormat) {
  AscendString str("NCHW");
  Format fmt = TypeUtils::AscendStringToFormat(str);
  EXPECT_EQ(fmt, FORMAT_NCHW);
}

TEST_F(CovUtilsMiscTest, CovTypeUtilsDataFormatToFormat) {
  AscendString str("NCHW");
  Format fmt = TypeUtils::DataFormatToFormat(str);
  SUCCEED();
}

}  // namespace ge
