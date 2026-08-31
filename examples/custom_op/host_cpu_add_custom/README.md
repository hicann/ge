# HostCpu Add 自定义算子样例

本目录包含三个子场景：

- [offline](./offline/README.md)：ES 构图 + ATC 转 OM + ACL 加载执行，演示离线编译和部署全流程。
- [constant_folding](./constant_folding/README.md)：ES 构图 + 常量折叠，验证编译期 HostCpu 执行。
- [host_scheduling](./host_scheduling/README.md)：动态 shape + 小 shape，验证 `HostcpuEngineUpdatePass` 将内置算子调度到 HostCpu。
