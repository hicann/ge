# Ascend C 自定义算子声明式地址刷新样例

本目录提供基于 `AnnotatedArgsOp::DeclareLaunchArgs` 的自定义算子声明式地址刷新在线与离线样例：

- [online](./online/README.md)：对比声明式地址刷新算子与非地址刷新算子的在线执行性能。
- [offline](./offline/README.md)：演示 AIR/OM 生成、`PortableOp` 序列化与反序列化、`DeclareLaunchArgs` 参数声明，以及通过 ACL 加载并执行 OM。
- [python](./python/README.md)：Python 构图 + ATC 编译 + ACL 两轮 NPU 地址刷新验证。
