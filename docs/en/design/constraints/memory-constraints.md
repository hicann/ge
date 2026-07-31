# GE Memory Constraints Document

# Static Memory Reuse

**Code Location:** `compiler/graph/build/memory/`

## Constraint 1: Graph compilation module memory reuse phase forbids graph modification

Precise boundary:

- **Scope of graph modification prohibition:** `BlockMemAssigner::AssignMemoryWithReuse` implementation and all functions it triggers
- **Multi-threaded entry:** `HybridMemAssigner::Assign` starts multiple threads, concurrently calling AssignMemoryWithReuse
- **Explicitly prohibited:** Adding, modifying, deleting attributes on ComputeGraph's Node
- **Safe operations:** Reading attributes/OpDesc is safe (system reads OpDesc extensively during traversal to judge memory allocation strategy)

## Constraint 2: Dynamic multi-batch scenario impact analysis

- **Code entity:** `DynamicBatchMemAssigner` (`dynamic_batch_mem_assigner.h`)
- **Meaning:** System identifies different batches through `batch_label_` (set by user or GE upper framework), supports different memory reuse strategies between different batches
- **Conflict with static memory reuse:**
  - Continuous input memory in different batches will be merged into one large block for alignment
  - No reuse within/between batches, but alignment strategy between batches exists, leading to possibly lower memory usage efficiency
  - Maximum split size limit: `kMaxSplitSizeForDynamicBatch = 400MB` (`dynamic_batch_mem_assigner.h`)

## Constraint 3: Scenarios to consider for static graph memory new features

| Scenario | Code marker/basis | Impact on reuse |
| ------ | ------------- | ------------- |
| **Continuous memory** | `continuous_block_` (`block_mem_assigner.h`), `ContinuousMemMng` (`continuous_mem.cc`) | Supports memory merging for continuous input nodes; continuous memory in different batches can merge and reuse |
| **Atomic centralized zero-clear** | `atomic_addr_clean_id_` (`block_mem_assigner.h`) | Memory blocks needing atomic zero-clear cannot be reused by other nodes; if node has no related attribute, skip zero-clear |
| **Zero copy** | `is_zero_copy_` (`block_mem_assigner.h`), `IsNodeAndPeerNodeTaskSupportZeroCopy` (`block_mem_assigner.cc`) | Zero copy blocks can be reused across nodes (`IsRealSizeReuseBlock`); zero copy memory cannot merge (multiple user input addresses may be discontinuous) |
| **Immutable address output** | `is_fixed_addr_prior_` (`block_mem_assigner.h`) | Output addresses of constant/const/variable/fileconstant/constplaceholder type operators are fixed at compile time, fixed address priority memory blocks can be reused but addresses are immutable |
| **Operators not supporting address refresh** | HCOM/rtsStreamSwitchByIndex etc | Input/output addresses of these operators must be stable, cannot use zero copy |
| **P2P memory type** | `RT_MEMORY_P2P_DDR` (`block_mem_assigner.cc`) | P2P memory cannot merge zero-clear with other memory types (`graph_mem_assigner.cc`) |

## Constraint 4: Particularity of HCOM operators

- **Meaning of "continuous":** Logically continuous, not physically continuous. Outputs of multiple HCOM operators form continuous memory region logically, managed through `ContinuousMemMng` manager for allocation and reuse. — `continuous_mem.cc`
- **featureBaseRefreshable configuration:**
  - Get method: `ge::GetContext().GetOption(ge::OPTION_FEATURE_BASE_REFRESHABLE, refreshable)` — `block_mem_assigner.cc`
  - Member variable: `is_feature_map_refreshable_` (`block_mem_assigner.h`)
  - Default value: `false`, set to `true` when configuration value is "1"
  - Effect: Controls whether feature map is refreshable, affects `IsNoNeedAssignMemory` judgment

## Constraint 5: Other constraints

1. **PreAssign/SetOpMemOffset not thread-safe:** Can only be called by single thread, other concurrent operations need attention. — `block_mem_assigner.h`

2. **Alignment strategy difference:** Zero copy memory uses 32-byte alignment, others use 512-byte alignment. — `graph_mem_assigner.h`

3. **Subgraph NETOUTPUT special handling:** NETOUTPUT nodes in subgraphs cannot perform zero copy. — `block_mem_assigner.cc`

4. **Multi-batch shape data node constraint:** Multi-batch shape data nodes do not support zero copy. — `block_mem_assigner.cc`

5. **Suspended memory block management:** Suspended memory blocks are released during next node allocation, lifecycle managed through `life_time_begin_` and `life_time_end_`, cannot be modified once set. — `block_mem_assigner.h`

6. **Reuse strategy configurability:** Supports dynamic configuration through parameters like `use_range_`, `ascending_sort_`, `reuse_first_release_`, `memory_priority_mode_`. — `block_mem_assigner.h`

---

# Dynamic Memory Reuse

**Code Location:**

- v2 layer: `runtime/v2/kernel/memory/allocator/` (ScalableAllocator, MemoryPool)
- v1 layer: `runtime/v1/graph/manager/active_memory_allocator.h` (ActiveMemoryAllocator, ExpandableActiveMemoryAllocator, PhysicalMemoryAllocator)
- Bridge layer: `runtime/v2/kernel/memory/device/device_allocator.h` (DeviceAllocator)

## Constraint 1: ScalableAllocator does not support multi-threaded concurrency

- **Code location:** `runtime/v2/kernel/memory/allocator/scalable_allocator.h`
- **Lock-free design basis:** Class internally has no `std::mutex` or `std::recursive_mutex`, only `static std::atomic_size_t global_allocator_id_` for generating unique ID (`scalable_allocator.h`)
- **Safety guarantee method:** Guaranteed by `aclmdlExecute` call constraint for single-threaded calling (see `docs/graph_engine_api/aclmdlExecute.md` for details), underlying allocator guarantees thread safety through recursive_mutex
- **Underlying has lock protection:** v1 layer's PhysicalMemoryAllocator uses `std::recursive_mutex` (`active_memory_allocator.h`), ExpandableActiveMemoryAllocatorImp also uses `std::recursive_mutex` (`active_memory_allocator.h`)

## Constraint 2: ActiveMemoryAllocator/ExpandableActiveMemoryAllocator/PhysicalMemoryAllocator support multi-threading

- **Thread safety mechanism:** Uses `std::recursive_mutex` to protect shared resources
- **New code requirements:** Must lock when accessing shared resources, follow existing lock usage patterns

---

# Memory Management

**Code Location:** `runtime/v2/kernel/memory/` (excluding allocator subdirectory)

## Constraint 1: Device id correctness

- **Code location:** `memory_kernel.cc` uses `aclrtGetDevice` to get device_id
- **Requirement:** When calling rts interfaces, device id must be explicitly passed correct value, avoid using default parameters (default is 0), need to verify multi-device scenario test cases

## Constraint 2: Memory release timing

- **Order:** First stream synchronization → then release memory → finally destroy device
- **Code association:** `caching_mem_allocator.cc`, `AllocateWithTryRecycle` method ensures synchronization before release

## Constraint 3: Virtual memory compatibility design

- **rtReserveMemAddress purpose:** Virtual address reservation, used for dynamic shape pre-allocated address space
- **Actual call location:** `runtime/v1/graph/manager/active_memory_allocator.cc`
- **Fallback path:** When `rtReserveMemAddress` fails, mark as not supporting virtual address reservation, fallback to physical address allocation mode. — `runtime/v1/graph/manager/active_memory_allocator.cc` ("Maybe not support rtReserveMemAddress.")
- **Requirement:** Need to ensure business flow is normal, no ERROR logs

## Constraint 4: caching_mem_allocator process exit timing

- **Code location:** `caching_mem_allocator.h` — `static std::vector<CachingMemAllocator *> all_caching_mem_allocators_`
- **Mechanism:** Global variable stores all allocator instance pointers, GE finalize actively calls all allocators' finalize method
- **Reason:** When allocator destructs, if memory is still unreleased, it will call rts interfaces to release, but at this point rts so may have been unloaded, causing core dump
- **Finalize entry:** `rts_caching_mem_allocator.cc` traverses `all_caching_mem_allocators_` and finalizes each one

## Constraint 5: Singleton memory pool must also be destructed when model is unloaded or session is destructed

For example, `SessionMemAllocator<ExpandableActiveMemoryAllocator>` is a singleton that internally stores objects based on `session id`. When user creates a new session, or calls interfaces starting with `aclmdl` for offline inference, a new `session id` will be generated, which also creates new objects. Therefore, `SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().RemoveAllocator` must be called at the following two locations to release object memory. Otherwise, in scenarios where user creates multiple sessions or repeatedly calls interfaces starting with `aclmdl` for offline inference, unused objects will occupy extra host memory or device memory resources, causing "memory leak" phenomenon.

- GeExecutor::UnloadModel — `aclmdl` offline inference scenario model unloading
- InnerSession::Finalize — InnerSession destruction

## CachingMemAllocator Complete Relationship Diagram

```
┌──────────────────────────────────────────────────────────────┐
│  CachingMemAllocator                                         │
│  runtime/v2/kernel/memory/caching_mem_allocator.h            │
│  - all_caching_mem_allocators_ (global allocator registry)    │
│  - rts_mem_allocator_ (RtsFirstLevelPool)                    │
│  - memory_pool_ → ScalableAllocator                          │
│  - mutex_ (static, protects allocator creation/destruction)   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌── RtsCachingMemAllocator                                  │
│  │   runtime/v2/kernel/memory/rts_caching_mem_allocator.h    │
│  │   - device_id_to_allocators_ (device → allocator mapping) │
│  │   - Responsible for finalize flow, traverses              │
│  │     all_caching_mem_allocators_                           │
│  │                                                           │
│  └── ScalableAllocator (through memory_pool_ pointer)        │
│       runtime/v2/kernel/memory/allocator/scalable_allocator  │
│       - Lock-free design, single-threaded calling            │
│       - Bridges to v1 allocator through DeviceAllocator      │
│                                                              │
│       └── DeviceAllocator                                    │
│            runtime/v2/kernel/memory/device/device_allocator   │
│            - active_memory_allocator_ (Expandable...Imp, v1)  │
│                                                              │
│            └── PhysicalMemoryAllocator (v1)                  │
│                 runtime/v1/graph/manager/active_mem...h      │
│                 - recursive_mutex protection                 │
│                 - Ultimately calls rtMalloc / rtFree         │
└──────────────────────────────────────────────────────────────┘
```

# Compile-time Memory Layout Conflict

**Code Location:** `compiler/graph/optimize/mem_layout_conflict_optimize/`

## Debugging Log Constraints

- Conflict found: `[MemConflict][Conflict] type: [%s, %s], anchor: [%s, %s], will insert %s %s`
- Insert Identity: `[MemConflict][INSERT][NODE]`
- Logs use `[MemConflict]` keyword, if only printed once per graph, use `LOGI`

# Cross-module Constraint Summary

The following constraints span multiple modules and must be considered synchronously when modifying:

1. **Address refresh capability (Module 1 + Module 2):** The conflict detection in mem_layout_conflict_optimize determines the "not supporting address refresh" operator list, which must remain consistent with the handling logic in static memory reuse.

2. **v1/v2 allocator boundary (Module 3 + Module 4):** v2's ScalableAllocator bridges to v1's PhysicalMemoryAllocator through DeviceAllocator. Modifying any layer's interface requires considering the impact on the other layer.

3. **caching_mem_allocator lifecycle (Module 3 + Module 4):** CachingMemAllocator's finalize flow depends on the `all_caching_mem_allocators_` global registry, new allocator types must be correctly registered in this table.
