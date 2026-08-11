# RT2 运行时约束文档


## 通用业务规则

**设计思想：** 针对业务上容易犯错的一些问题点提炼的规则，确保开发过程中的正确性和一致性。

**具体原则：**

**加载时：**
- 调用RT2的加载和执行接口时，必须确保加载和执行使用相同的stream和allocator。如果使用不同的stream和allocator，需要在加载完成后调用流同步接口，以确保加载完成。
- 在RT2执行器的构造过程中，不应修改计算图。否则，可能会导致多次构造执行器失败。
- RT2 Lowering生成的执行节点顺序并不代表执行器执行节点的顺序。编写Lowering函数时，务必注意节点依赖关系，以确保执行时序的正确性。
- `OfflineOptimizer` 必须始终注册 `CopyFlowLaunchFuse`，并且只能在 `IsEnableRt2MultiThread()` 为 `true` 时按 `SplitMixedLaunchMemory -> RemoveLaunchFreeEdge` 顺序注册两个 once pass。单线程保留 Legacy `CopyFlowLaunch`；动态 shape 多流必须强制返回 `false`，不拆分、不生成关系并保留原 Launch-to-Free 边。
- H2D 多线程拆分必须为 `CalcDeviceCopySizes -> AllocMemHbm -> ShareH2DCopyResult -> LaunchH2DCopy`。`CalcDeviceCopySizes` 保留原 tensor size 作为申请基准并按 shape/type 计算实际 copy size；无需拷贝或空 Tensor 时两个 size 均为 0，`DT_STRING` 只在确定需要拷贝后计算 string size。`ShareH2DCopyResult` 在 memory worker 产生 Consumer/Free 使用的 owning/shared 输出，`LaunchH2DCopy` 无输出且只提交异步 memcpy。
- CopyFlow 多线程拆分必须为 `CalcCopyFlowAllocSizes -> AllocCopyFlowHbm -> PrepareCopyFlowResult -> LaunchCopyFlowH2D`。四个节点携带相同的非零静态 `copy_flow_count`；运行时输入数、输出数、allocated-address 数和输入索引 CVV 外层基数必须一致。Prepare 在 memory worker 拥有全部结果并完成 merged args/host args/对齐，不提交 memcpy；output-free Launch 只提交独立分配的非零 H2D。

**执行时：**
- 涉及资源处理时，需要考虑资源规格和限制，以及资源的生命周期。
- 异步H2D拷贝必须配合HOST_TO_DEVICE_EX选项，否则可能导致host内存提前销毁，引发问题。
- 全局语义上的 Launch（包括 HCOM）或注册为 `kKernelLaunch` 的直接 Consumer 必须等待对应 output-free Copy Launch。`BuildRefTensor` 和 ACLNN `ExecuteOpPrepare` 不得等待 `LaunchH2DCopy`；其后的 ACLNN/CustomOp 语义 Launch 或 `kKernelLaunch` Consumer 必须等待，并在关系删除前控制对应 Free。Consumer、Free 和 Guarder 的数据必须来自 `ShareH2DCopyResult`/`PrepareCopyFlowResult`，Guarder 只复制显式 output mapping。

---

## 性能规则

**设计思想：** 确保系统在高负载下仍能保持高效运行，避免性能瓶颈和劣化。

**具体原则：**

- 执行时应尽量避免动态申请内存，这可能导致随机性能劣化。
- 新增kernel或修改kernel实现时，需要评估对执行性能的影响。新增kernel时，应在设计环节明确性能影响并确认性能规格。修改kernel时，性能劣化不应超过100ns。
- 当前精确关系数据仅来自 **O0/GCOV + ASan/LSan stub build**，不得当作 Release 延迟：成功 Malloc paired delta 中位数约 `+0.243%`（噪声），无关节点事件约 `33 ns`，立即等待约 `44 ns`，一组 Free/Launch occurrence 约 `146 ns`；100,000 节点/1,000,000 关系的 `Prepare` 中位数为 `6.313 ms`，结构内存为 `11,300,008 B`。无任务测量中 reset 数组 capacity 未增长且前后无净 live allocation，但不得外推为所有 `Schedule` 路径零分配，也不得据此推导活性结论。

---

## 兼容性规则

**设计思想：** 确保系统在不同版本和环境下的兼容性，避免因变更导致的兼容性问题。

**具体原则：**

- 涉及对外的option、环境变量、接口、数据结构等变更时，可能会影响兼容性。这些变更需要经过评审并获得通过结论后才能实施。
- 本优化不得修改外部 API、OM schema、v1 Known Shape/DavinciModel 接口或 Adapter/Session 契约。动态 shape 静态子图不新增 v2 到 v1 的传递数据，Free-to-Launch 关系不得跨 owner graph。

---

## 并发处理规则

**设计思想：** 确保系统在多线程环境下的稳定性和性能，避免资源竞争和异常。

**具体原则：**

- 新增kernel实现时，需要支持多线程调用。如果涉及处理临界资源（如内存相关kernel），需要给该kernel打上标记（通过kernel注册时注册ConcurrentCriticalSectionKey接口），以避免多线程执行场景下的资源竞争和异常。
- 多线程执行器的总线程数为 3 时（Hybrid RT2 路径通常由 `MAX_RUNTIME_CORE_NUMBER=3` 配置），`AllocHostCpuOutputMemory`、`SplitDataTensor`、`IdentityAddr`、`IdentityShapeAndAddr` 和 `AccessMemCrossStream` 保留注册的 `kKernelUseMemory` 分类，并在唯一的 MEMORY worker 上执行。`AccessMemCrossStream` 的 `ShareFrom`/`WanderFrom` 会修改共享 TensorData、引用计数或跨流 MIF 状态，必须与其它共享内存状态变更串行化。由于其 placement 在运行时才确定，`RemoveLaunchFreeEdge` 不将它作为可删边的直接 producer，保留原 Launch-to-Free 边。`BuildTensor`、`BuildTensorStorage` 和 `BuildTensorPureShape` 是仅有的 NORMAL worker 例外，从而保留既有三线程流水优化。Allocator 热路径不增加锁；执行图调度串行化 Host allocator 与 wrapper pool 的状态变更，以牺牲这些状态变更的并行度换取无锁热路径和安全的生命周期转换。
- RT2 多线程必须将 owning 结果物化与异步 RTS/ACL 提交拆开：产生结果或修改内存状态的 Kernel 注册为 `kKernelUseMemory`，output-free 提交 Kernel 注册为 `kKernelLaunch`。旧批次 ready Launch 计数、TLS 批次阈值、混合 Kernel 预执行屏障和伪造完成值不得恢复。
- `RemoveLaunchFreeEdge` 只可处理具有 hold-address 变体的设备 Free，且 `ReleaseResourceIndex` 直接 producer 必须位于非 Host 内存路径并注册为 `kKernelUseMemory`。Pass 必须先补齐同一 Launch 下全部 eligible producer 与全部 eligible Free 的笛卡尔积顺序，再按 owner graph 记录并去重精确 `(Free*, Launch*)` 关系、替换 hold-address Free，最后删除边。Host、normal、launch、未注册或缺失 producer 和 `FreeTensorMemory` 必须保留原 Free 类型、全部 Launch-to-Free 边及既有调度属性。
- ExecutionData 必须在最终节点映射后构建按 execution ID 确定排序的 CSR，由 `MultiThreadResourceGuard` 持有。嵌套 owner graph 只能从 mapped nodes 推导；必须校验 owner、mapped node、Free/Launch 类型、ID、offset 和关系基数，失败时拒绝执行无保护的删边结果。执行热路径不得扫描图或动态分配关系容器。
- 每次 `Schedule` 必须隔离 epoch、submitted/free/required generation、relation-Launch membership、unmet Launch 数和 abort 原始状态。成功 Free 才激活 requirement，成功 relation Launch 才推进 submitted generation；多个 Free 对同一 Launch occurrence 只计一个 unmet。失败和 EOS 不推进 generation，必须保存原状态并唤醒全部等待者。
- `CachingMemAllocator` 和 `L2MemPool` 只能在物理 stream 同步/回收前等待已执行 Free 激活的 requirement。成功 Malloc fast path 不得调用 scheduler，未执行关系和无关 Launch 不得阻塞；回收顺序必须保持 `exact wait -> stream sync -> recycle -> retry`。
- 当源 `kComputeNodeIndex` 存在时，`LaunchH2DCopy`、`LaunchCopyFlowH2D` 必须继承该索引，并保持在 `IsLaunchNode` 定义的计算 Launch 之外；它们必须注册为 `kKernelLaunch`，由 `IsLaunchOrHasSubGraphNode` 纳入 priority/strict-order。DataDump 的 compute-op range FSM 和 launch-name 收集必须直接复用 `IsLaunchNode`，自然跳过这两类内部 helper，由真实 Consumer Launch 维护算子边界。
- 动态 shape 多流开关 `ENABLE_DYNAMIC_SHAPE_MULTI_STREAM=1` 与上述拆分/关系优化互斥；不得在多流路径删除 Launch-to-Free 边。
- **唯一 memory worker 活性尚未解决。** 如果已激活 Launch 仍有排在阻塞 memory task 之后的未完成 memory-worker 前驱，当前阻塞等待可能形成闭环。任务 1.1、1.2、4.6、7.4 由用户选择延期，当前实现不得宣称保证该场景活性。生产验收前必须增加保守证明门禁，对无法证明安全的关系保留原边，或采用不占用唯一 memory worker 的非阻塞 continuation。
- 不得修改 `EnsureNodeExeInOrderInSubgraph`、`IsCopyAsyncNode`、zero-copy 和地址复用的既有保序语义。

---

## 可调试性&可维护性原则

**设计思想：** 确保系统在开发和运维过程中易于调试和维护，减少问题定位和解决的时间。

**具体原则：**

- 谨慎添加日志，避免高频次日志打印。日志内容应简洁明了，包含必要的上下文信息，以便快速定位问题。
- 建议通过注册KernelTrace函数来打印执行时Kernel的维测信息。设计必要且有效的定位信息，以提高调试的效率和准确性。

---
