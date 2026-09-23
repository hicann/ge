# HostCpu Add Custom Op Samples

This directory is organized by scenario and language, with independent C++ and Python implementations:

- [constant_folding/cpp](./constant_folding/cpp/README_en.md): C++ ES graph construction with constant folding, so HostCpu runs during compilation.
- [constant_folding/python](./constant_folding/python/README_en.md): Python ES graph construction with constant folding, so the Python HostCpu callback runs during compilation.
- [host_scheduling/cpp](./host_scheduling/cpp/README_en.md): C++ dynamic shape with a small shape, so `HostcpuEngineUpdatePass` schedules the built-in op to HostCpu.
- [host_scheduling/python](./host_scheduling/python/README_en.md): Python dynamic shape with a small shape, so the built-in op is scheduled to the Python HostCpu callback.
- [offline/cpp](./offline/cpp/README_en.md): C++ ES graph construction + ATC to OM + ACL load and execute, demonstrating the full offline compilation and deployment flow.

Python custom ops do not yet support offline packaging (the OM does not contain the Python plugin source), so the offline scenario only provides a C++ implementation.
