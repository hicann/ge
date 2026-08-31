# HostCpu Add Custom Op Samples

This directory contains three sub-scenarios that share the `AddCustom` operator definition:

- [offline](./offline/README_en.md): ES graph construction + ATC to OM + ACL load and execute, demonstrating the full offline compilation and deployment flow.
- [constant_folding](./constant_folding/README_en.md): ES graph construction with constant folding, so HostCpu runs during compilation.
- [host_scheduling](./host_scheduling/README_en.md): dynamic shape with a small shape, so `HostcpuEngineUpdatePass` schedules the built-in op to HostCpu.
