# RT2 Runtime Constraints Document


## General Business Rules

**Design Philosophy:** Rules distilled from business areas prone to errors, ensuring correctness and consistency during development.

**Specific Principles:**

**During Loading:**
- When calling RT2 load and execute interfaces, must ensure loading and execution use the same stream and allocator. If using different stream and allocator, need to call stream synchronization interface after loading completes to ensure loading is complete.
- During RT2 executor construction, should not modify computational graph. Otherwise, may cause multiple executor construction failures.
- RT2 Lowering generated execution node order does not represent executor's execution node order. When writing Lowering functions, must pay attention to node dependencies to ensure execution timing correctness.
- `OfflineOptimizer` must always register `CopyFlowLaunchFuse`, and it may register the two once passes only when `IsEnableRt2MultiThread()` is `true`, in `SplitMixedLaunchMemory -> RemoveLaunchFreeEdge` order. Single-thread execution retains Legacy `CopyFlowLaunch`. Dynamic-shape multi-stream must force the check to return `false`, perform no split or relation optimization, and retain original Launch-to-Free edges.
- The multi-thread H2D split must be `CalcDeviceCopySizes -> AllocMemHbm -> ShareH2DCopyResult -> LaunchH2DCopy`. `CalcDeviceCopySizes` keeps the original tensor size as the allocation basis and calculates actual copy size from shape/type; both sizes are zero when no copy is required or the Tensor is empty, and `DT_STRING` size is calculated only after copy is known to be required. `ShareH2DCopyResult` produces the owning/shared Consumer/Free output on the memory worker. `LaunchH2DCopy` is output-free and only submits asynchronous memcpy.
- The multi-thread CopyFlow split must be `CalcCopyFlowAllocSizes -> AllocCopyFlowHbm -> PrepareCopyFlowResult -> LaunchCopyFlowH2D`. All four nodes carry the same nonzero static `copy_flow_count`; runtime input count, output count, allocated-address count, and input-index CVV outer cardinality must agree. Prepare owns all results on the memory worker and updates merged args, host args, and alignment without memcpy; the output-free Launch submits only independently allocated, nonzero H2D copies.

**During Execution:**
- When involving resource handling, need to consider resource specifications and limits, as well as resource lifecycle.
- Asynchronous H2D copy must be paired with HOST_TO_DEVICE_EX option, otherwise may cause host memory premature destruction, leading to issues.
- A direct Consumer that is either a semantic Launch, including HCOM, or registered with `kKernelLaunch` must wait for the corresponding output-free Copy Launch. `BuildRefTensor` and ACLNN `ExecuteOpPrepare` must not wait for `LaunchH2DCopy`; the following ACLNN or CustomOp semantic Launch or `kKernelLaunch` Consumer must wait and must control the corresponding Free before relation removal. Consumers, Free nodes, and Guarders must read data from `ShareH2DCopyResult`/`PrepareCopyFlowResult`, and Guarders may copy only explicit output mappings.

---

## Performance Rules

**Design Philosophy:** Ensure system can still run efficiently under high load, avoid performance bottlenecks and degradation.

**Specific Principles:**

- During execution should try to avoid dynamic memory allocation, which may cause random performance degradation.
- When adding new kernels or modifying kernel implementations, need to evaluate impact on execution performance. When adding kernels, should clarify performance impact in design phase and confirm performance specifications. When modifying kernels, performance degradation should not exceed 100ns.
- Current exact-relation measurements come only from an **O0/GCOV + ASan/LSan stub build** and must not be presented as Release latency. The median paired successful-Malloc delta is about `+0.243%` (noise), an unrelated node event about `33 ns`, an immediate wait about `44 ns`, and one Free/Launch occurrence about `146 ns`. For 100,000 nodes and 1,000,000 relations, median `Prepare` time is `6.313 ms` and structural memory is `11,300,008 B`. In the no-task measurement, reset-array capacity did not grow and there was no net live allocation across the measured call sequence; this must not be generalized to zero allocation for every `Schedule` path or used as liveness evidence.

---

## Compatibility Rules

**Design Philosophy:** Ensure system compatibility across different versions and environments, avoid compatibility issues due to changes.

**Specific Principles:**

- When involving changes to external options, environment variables, interfaces, data structures etc., may affect compatibility. These changes need to go through review and obtain passing conclusion before implementation.
- This optimization must not change public APIs, the OM schema, v1 Known Shape/DavinciModel interfaces, or Adapter/Session contracts. Dynamic-shape static subgraphs receive no new v2-to-v1 data, and Free-to-Launch relations must not cross owner graphs.

---

## Concurrency Handling Rules

**Design Philosophy:** Ensure system stability and performance in multi-threaded environments, avoid resource competition and exceptions.

**Specific Principles:**

- When adding new kernel implementations, need to support multi-threaded calling. If involving handling critical resources (e.g., memory-related kernels), need to mark that kernel (through ConcurrentCriticalSectionKey interface during kernel registration) to avoid resource competition and exceptions in multi-threaded execution scenarios.
- When the multi-thread executor has a total of three threads, usually configured as `MAX_RUNTIME_CORE_NUMBER=3` on the Hybrid RT2 path, `AllocHostCpuOutputMemory`, `SplitDataTensor`, `IdentityAddr`, `IdentityShapeAndAddr`, and `AccessMemCrossStream` retain their registered `kKernelUseMemory` classification and execute on the sole MEMORY worker. `AccessMemCrossStream` changes shared TensorData, reference-count, or cross-stream MIF state through `ShareFrom`/`WanderFrom` and must be serialized with other shared-memory state changes. Because its placement is determined at runtime, `RemoveLaunchFreeEdge` does not treat it as an eligible direct producer and preserves the original Launch-to-Free edge. `BuildTensor`, `BuildTensorStorage`, and `BuildTensorPureShape` remain the only NORMAL-worker exceptions, preserving the existing three-thread pipeline optimization. Allocator hot paths add no lock; execution-graph scheduling serializes Host allocator and wrapper-pool state changes, trading parallelism between those state changes for lock-free hot paths and safe lifecycle transitions.
- RT2 multi-threading must separate owning-result materialization from asynchronous RTS/ACL submission. A Kernel that produces results or changes memory state must be registered with `kKernelUseMemory`; an output-free submission Kernel must be registered with `kKernelLaunch`. The removed batch ready-Launch count, TLS batch threshold, mixed-Kernel pre-execution barrier, and fabricated completion values must not be reintroduced.
- `RemoveLaunchFreeEdge` may process only a device Free with a hold-address variant whose direct `ReleaseResourceIndex` producer is on a non-Host memory path and registered with `kKernelUseMemory`. A runtime-placement-dependent `AccessMemCrossStream` producer is conservatively ineligible and retains its original Launch-to-Free edge. The pass must first complete the Cartesian ordering between every eligible producer and every eligible Free for the Launch, then record and deduplicate exact `(Free*, Launch*)` relations on the owner graph, replace the Free with its hold-address variant, and finally remove the edge. Host, normal, launch, unregistered, or missing producers and `FreeTensorMemory` must retain their original Free type, all Launch-to-Free edges, and existing scheduling attributes.
- ExecutionData must build a deterministic execution-ID CSR after final node mapping, owned by `MultiThreadResourceGuard`. Nested owner graphs may be derived only from mapped nodes. Owner, mapped node, Free/Launch type, ID, offsets, and relation cardinality must be validated; an unprotected edge-removal result must be rejected. The execution hot path must not scan the graph or allocate relation containers.
- Every `Schedule` must isolate its epoch, submitted/free/required generations, relation-Launch membership, unmet Launch count, and original abort status. Only a successful Free activates a requirement, and only a successful relation Launch advances submitted generation. Multiple Free nodes referencing one Launch occurrence count as one unmet Launch. Failure and EOS do not advance generations and must preserve the original status and wake all waiters.
- `CachingMemAllocator` and `L2MemPool` may wait only before physical stream synchronization/recycling, and only for requirements activated by executed Free nodes. A successful Malloc fast path must not call the scheduler; unexecuted relations and unrelated Launches must not block. Recycling must retain `exact wait -> stream sync -> recycle -> retry`.
- When source `kComputeNodeIndex` exists, `LaunchH2DCopy` and `LaunchCopyFlowH2D` must inherit it and remain outside the compute Launches defined by `IsLaunchNode`. They must register with `kKernelLaunch` so that `IsLaunchOrHasSubGraphNode` includes them in priority calculation and strict ordering. DataDump compute-op range FSM and launch-name collection must directly reuse `IsLaunchNode`, naturally skipping these internal helpers so that the real Consumer Launch maintains the operator boundary.
- `ENABLE_DYNAMIC_SHAPE_MULTI_STREAM=1` is mutually exclusive with this split/relation optimization; Launch-to-Free edges must not be removed on the multi-stream path.
- **Liveness with a sole memory worker is not solved.** If an activated Launch still has an unfinished memory-worker predecessor queued behind a blocked memory task, the blocking wait can form a cycle. Tasks 1.1, 1.2, 4.6, and 7.4 remain deferred by user choice, and the current implementation must not claim liveness for this scenario. Production acceptance requires either a conservative proof gate that retains the original edge whenever safety cannot be proven, or a non-blocking continuation that does not occupy the sole memory worker.
- Existing `EnsureNodeExeInOrderInSubgraph`, `IsCopyAsyncNode`, zero-copy, and address-reuse ordering semantics must not be changed.

---

## Debuggability & Maintainability Principles

**Design Philosophy:** Ensure system is easy to debug and maintain during development and operation, reduce time for problem localization and resolution.

**Specific Principles:**

- Be cautious when adding logs, avoid high-frequency log printing. Log content should be concise and clear, contain necessary context information for quick problem localization.
- Recommend registering KernelTrace function to print Kernel debugging info during execution. Design necessary and effective positioning information to improve debugging efficiency and accuracy.

---
