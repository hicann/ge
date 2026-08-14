# 总体约束

在进行模型转换前，请务必查看如下约束要求：

- 如果要将Faster RCNN等网络模型转成适配AI处理器的离线模型，则务必参见[定制网络修改（Caffe）](../custom_network_modify_caffe/README.md)先修改prototxt模型文件。

- 支持原始框架类型为Caffe、TensorFlow、MindSpore、ONNX的模型转换：
  - 当原始框架类型为Caffe、MindSpore、ONNX时，输入数据类型为FP32、FP16、UINT8（通过配置数据预处理[--insert\_op\_conf](--insert_op_conf.md)实现）。
  - 当原始框架类型为TensorFlow时，输入数据类型为FP16、FP32、UINT8、INT32、INT64、BOOL。

- 对于Caffe框架网络模型：输入数据最大支持四维，转维算子（Reshape、ExpandDims等）不能输出五维。

- 模型中的所有层算子除const算子外，输入和输出需要满足dim!=0。

- 只支持《[算子库](https://gitcode.com/cann/docs/blob/master/docs/zh/ops-lib/0_README.md)》\>“Ascend IR算子规格说明”中的算子，并需满足算子限制条件。

- 由于软件约束（动态shape场景下暂不支持输入数据为DT\_INT8），量化后的部署模型使用ATC工具进行模型转换时，不能使用动态shape相关参数，例如[--dynamic\_batch\_size](--dynamic_batch_size.md)和[--dynamic\_image\_size](--dynamic_image_size.md)等，否则模型转换会失败。

- 使用AMCT工具量化后的部署模型，使用ATC工具进行模型转换时，不能再使用高精度特性，比如不能再通过[--precision\_mode](--precision_mode.md)参数配置**force\_fp32**或**must\_keep\_origin\_dtype（原图fp32输入）**；不能再通过[--precision\_mode\_v2](--precision_mode_v2.md)参数配置**origin**；不能通过[--op\_precision\_mode](--op_precision_mode.md)配置**high\_precision**参数等。在高精度模式下设置量化参数，既拿不到量化的性能收益，也拿不到高精度模式的精度收益。
