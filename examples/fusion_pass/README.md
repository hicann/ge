## 融合Pass样例

本目录提供了融合Pass相关的样例：

| 样例                               | 样例链接                                                         |
|----------------------------------|--------------------------------------------------------------|
| MatMul+Add融合为GEMM自定义pass样例       | [README](1_fuse_matmul_add_pass/README.md)                   |
| 删除加零操作的自定义pass样例                 | [README](2_add_zero_pass/README.md)                          |
| 拆分分组卷积的自定义pass样例                 | [README](3_decompose_grouped_conv_to_splited_pass/README.md) |
| 移动Concat后ReLu至Concat前的自定义pass样例 | [README](7_move_relu_before_concat_pass/README.md) |

## 开发指南

更多关于融合Pass开发的信息，请参考：[融合Pass开发指南](融合Pass开发指南.md)
