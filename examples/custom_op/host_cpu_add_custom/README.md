# HostCpu Add 自定义算子样例

本目录按场景和语言组织样例，C++ 与 Python 分别提供独立实现：

- [constant_folding/cpp](./constant_folding/cpp/README.md)：C++ ES 构图 + 常量折叠，验证编译期 HostCpu 执行。
- [constant_folding/python](./constant_folding/python/README.md)：Python ES 构图 + 常量折叠，验证编译期调用 Python HostCpu 回调。
- [host_scheduling/cpp](./host_scheduling/cpp/README.md)：C++ 动态 shape + 小 shape，验证 `HostcpuEngineUpdatePass` 将内置算子调度到 HostCpu。
- [host_scheduling/python](./host_scheduling/python/README.md)：Python 动态 shape + 小 shape，验证内置算子被调度到 Python HostCpu 回调。
- [offline/cpp](./offline/cpp/README.md)：C++ ES 构图 + ATC 转 OM + ACL 加载执行，演示离线编译和部署全流程。

当前 Python 自定义算子尚不支持离线打包（OM 不包含 Python 插件源码），因此离线场景仅提供 C++ 实现。
