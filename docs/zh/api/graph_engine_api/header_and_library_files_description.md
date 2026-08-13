# 头文件和库文件说明

GE图引擎接口头文件在如下目录：

- 编译类接口：
  - $\{INSTALL\_DIR\}/include/ge/
  - $\{INSTALL\_DIR\}/include/parser/
  - $\{INSTALL\_DIR\}/include/acl/

- 执行类接口：
  - $\{INSTALL\_DIR\}/include/ge/
  - $\{INSTALL\_DIR\}/include/graph/
  - $\{INSTALL\_DIR\}/include/external/
  - $\{INSTALL\_DIR\}/include/acl/
  - $\{INSTALL\_DIR\}/include/register/
  - $\{INSTALL\_DIR\}/include/transformer/
  - $\{INSTALL\_DIR\}/pkg\_inc/register

库文件在`${INSTALL_DIR}/lib64/`目录下，`${INSTALL_DIR}`请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。

## ge-compiler编译类接口

| 定义接口的头文件 | 用途 | 对应的库文件 |
| --- | --- | --- |
| ge/hcom_gradient_split_tune.h | HCOM梯度切分调优指标、输入和结果接口。 | libhcom_gradient_split_tune.so |
| ge/hcom_ops_stores.h | HCOM插件初始化及 OpsKernelInfoStore/GraphOptimizer 查询接口。 | libhcom_graph_adaptor.so |
| ge/ge_ir_build.h | aclgrphBuild*、模型编译、保存及权重可刷新图接口。 | libge_compiler.so |
| ge/ge_utils.h | GE改图、Shape 推导、节点支持校验等工具接口。 | libge_compiler.so |
| ge/fusion/graph_rewriter.h | 子图替换和改图接口。 | libge_compiler.so |
| ge/fusion/match_result.h | Pattern匹配结果与捕获节点/张量访问。 | libge_compiler.so |
| ge/fusion/graph_fuse_inspector_utils.h | 融合前合法性检查及融合结果上报。 | libge_compiler.so |
| ge/fusion/pattern.h | 定义待匹配图模式及捕获张量。 | libge_compiler.so |
| ge/fusion/pattern_matcher.h | 在目标图上执行Pattern匹配。 | libge_compiler.so |
| ge/fusion/subgraph_boundary.h | 描述子图输入、输出和边界。 | libge_compiler.so |
| ge/fusion/pattern_matcher_config.h | 配置PatternMatcher匹配行为。 | libge_compiler.so |
| ge/fusion/infer_shape_util.h | 融合/替换图的Shape推导辅助接口。 | libge_compiler.so |
| ge/fusion/pass/decompose_pass.h | 一对多算子分解Pass接口及注册宏。 | libge_compiler.so |
| ge/fusion/pass/fusion_base_pass.h | 自定义融合Pass纯抽象基类。 | Header-only接口；具体Pass框架消费libge_compiler.so |
| ge/fusion/pass/fusion_pass_reg.h | 融合Pass注册数据和注册器。 | libge_compiler.so |
| ge/fusion/pass/pattern_fusion_pass.h | Pattern融合Pass基类和替换流程。 | libge_compiler.so |
| parser/caffe_parser.h | Caffe 模型解析为GE Graph。 | libfmk_parser.so |
| parser/onnx_parser.h | ONNX 模型解析为GE Graph。 | libfmk_onnx_parser.so |
| parser/tensorflow_parser.h | TensorFlow模型解析为GE Graph。 | libfmk_parser.so |
| ge/ge_api.h | GE初始化、Session、Graph添加/编译/执行等V1 API。 | libge_runner.so |
| ge/ge_api_v2.h | GESession、V2图加载与执行接口。 | libge_runner_v2.so |
| ge/ge_feature_memory.h | 编译结果的Feature Memory描述。 | libge_common_base.so（实际符号归属；上层编译流程同时使用libge_compiler.so） |
| ge/ge_graph_compile_summary.h | 编译结果、流分配、外置权重摘要。 | libge_compiler.so |
| ge/esb_funcs.h | ES构图器C ABI。 | libeager_style_graph_builder_base.so |
| ge/compliant_node_builder.h | 按IR规范构造节点的链式Builder。 | libeager_style_graph_builder_base.so |
| ge/es_c_graph_builder.h | ES C Graph Builder封装。 | libeager_style_graph_builder_base.so |
| ge/es_c_tensor_holder.h | ES C TensorHolder封装。 | libeager_style_graph_builder_base.so |
| ge/es_graph_builder.h | C++模板化ES Graph Builder门面。 | Header-only门面；调用 libeager_style_graph_builder_base.so的C ABI |
| ge/es_tensor_holder.h | C++ ES TensorHolder接口。 | libeager_style_graph_builder_base.so |
| ge/es_tensor_like.h | ES TensorLike描述与转换接口。 | libeager_style_graph_builder_base.so |
| acl/acl_op_compiler.h | 单算子编译、编译并执行及编译选项接口。 | libacl_op_compiler.so |

## ge-executor执行类接口

| 定义接口的头文件 | 用途 | 对应的库文件 |
| --- | --- | --- |
| ge/ge_api_types.h | GE公共状态、张量、图和Session相关类型的聚合入口。 | Header-only聚合头；按实际接口链接 libgraph.so/libgraph_base.so等 |
| ge/ge_api_error_codes.h | GEAPI 错误码注册与字符串转换接口。 | libge_common_base.so |
| ge/ge_external_weight_desc.h | 外置权重位置、长度和属性描述接口。 | libge_common_base.so |
| external/ge_common/ge_api_types.h | 兼容旧路径的GE API类型聚合入口。 | Header-only兼容头；按实际接口链接对应库 |
| external/ge_common/ge_common_api_types.h | GE公共枚举、结构体和基础API类型定义。 | Header-only定义头 |
| acl/acl_mdl.h | ACL模型加载、执行、动态Shape和模型配置接口。 | libacl_mdl.so（公开门面；运行时内部还会依赖模型执行实现库） |
| acl/acl_base_mdl.h | ACL模型基础类型、数据集和描述接口。 | libacl_mdl.so（公开门面） |
| acl/acl_op.h | ACL单算子描述、执行及算子属性接口。 | libacl_op_executor.so（公开门面；内部实现依赖不作为应用直链接口） |
| acl/ops/acl_cblas.h | ACL CBLAS/矩阵计算接口。 | libacl_cblas.so |
| exe_graph/runtime/eager_op_execution_context.h | Eager算子执行时的输入、输出、Stream和Workspace上下文。 | liblowering.so |
| exe_graph/runtime/op_compile_context.h | 算子编译、Tiling和Shape推导上下文。 | liblowering.so |
| graph/graph.h | GE Graph创建、增删节点、输入输出与图属性接口。 | libgraph.so |
| graph/ct_infer_shape_range_context.h | 编译期Shape Range推导上下文。 | Header-only接口；主要消费库libgraph.so |
| graph/ct_infer_shape_context.h | 编译期Shape推导上下文。 | Header-only接口；主要消费库libgraph.so |
| graph/operator_reg.h | 算子原型、输入输出、属性和推导函数注册宏。 | libgraph.so为主要注册实现，并使用libgraph_base.so基础能力 |
| graph/gnode.h | Graph节点查询、连接关系和属性访问接口。 | libgraph.so |
| graph/graph_buffer.h | 图序列化Buffer的创建、读写和生命周期接口。 | libgraph_base.so |
| graph/inference_context.h | Shape/Value推导的上下文、Marks和资源管理接口。 | libgraph.so |
| graph/attr_value.h | Graph属性值类型及序列化访问接口。 | libgraph.so |
| graph/operator.h | GE Operator创建、连接、属性和描述访问接口。 | libgraph_base.so |
| graph/operator_factory.h | 按类型创建Operator及查询算子注册信息。 | libgraph.so |
| graph/resource_context.h | 图资源上下文的抽象接口和资源键。 | Header-only接口；主要消费库libgraph.so |
| graph/kernel_launch_info.h | Kernel Launch参数、句柄和注册信息描述。 | libregister.so |
| graph/arg_desc_info.h | Kernel参数描述信息。 | libgraph.so |
| graph/custom_op.h | 自定义算子注册、推导和验证宏。 | libgraph.so（主要公开注册符号） |
| graph/hcom_executor.h | HCOM图执行器初始化、Finalize和执行接口。 | libhcom_executor.so |
| register/register_base.h | GE注册机制基础类、宏和优先级定义。 | libregister.so |
| register/op_lib_register.h | 算子库、OpInfo与OpsKernelInfoStore注册接口。 | libregister.so |
| register/register_custom_pass.h | 自定义图Pass注册及执行时机配置。 | libregister.so |
| register/op_binary_resource_manager.h | 算子二进制资源注册、查询和生命周期管理。 | libregister.so |
| register/scope/scope_fusion_pass_register.h | Scope Fusion Pass注册、规则和结果接口。 | libregister.so |
| transformer/transfer_def.h | Format/Shape转换公共结构和枚举定义。 | Header-only定义头；主要消费库libgraph.so |
| transformer/transfer_shape_according_to_format_ext.h | 按源/目标Format转换Shape。 | libgraph.so |
| pkg_inc/register/amct_interface.h | AMCT插件回调及量化相关抽象接口。 | Header-only插件接口；宿主侧主要消费库libregister.so |
| pkg_inc/register/amct_registry.h | AMCT插件注册、查询和实例创建。 | libregister.so |
