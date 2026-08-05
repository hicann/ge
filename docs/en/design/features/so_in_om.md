# SO in OM Feature Description

## 1. Feature Overview

In the Ascend AI processor ecosystem, operators are implemented as dynamic link libraries (.so files). In the traditional deployment mode, users need to:

1. **Install the complete OPP (Operator Primitive Package) on the target machine**, which contains hundreds of .so files
2. **Ensure the operator package versions match exactly between the compilation and runtime environments**
3. **Manage complex environment variable paths** (`ASCEND_OPP_PATH`) that point to the operator library location

The above approach has the following limitations:

- **High deployment complexity**: Each inference node requires installing a large operator package, resulting in huge container images during containerized deployment
- **Difficult version matching**: Inconsistent operator versions between compilation and runtime leads to inference failures, and errors are difficult to locate
- **Runtime compilation dependency in dynamic shape scenarios**: Models with unknown shapes require dynamically generating tiling parameters at runtime, which depends on the operator implementation .so from the compilation phase

### 1.2 Feature Introduction

The **SO in OM** feature packs the operator .so files that the model depends on directly into the .om (Offline Model) file. This makes the model file self-contained with all operator code needed at runtime, allowing it to load and execute without external operator packages.

This feature uses an **on-demand packaging** strategy. Through dependency analysis during compilation, only the operators actually used by the model are packaged, avoiding unnecessary size overhead.

## 2. Overall Architecture

```mermaid
graph TB
    subgraph "Compilation Phase (ATC/GeGenerator)"
        A[User Graph] --> B[Graph Compilation Pipeline]
        B --> C[Operator Compilation]
        C --> D[Generate TaskDef]
        D --> E{Check Dependent SO}
        E -->|SpaceRegistry| F[Collect tiling/infer shape so]
        E -->|OpMasterDevice| G[Collect op_master_device so]
        E -->|Autofuse| H[Collect autofuse so]
        E -->|CustomOp| H2[Collect custom op so]
        F --> I[OpSoStore Packaging]
        G --> I
        H --> I
        H2 --> I
        I --> J[Write to OM file SO_BINS partition]
    end

    subgraph "OM File Structure"
        J --> K[ModelFileHeader]
        K --> L[MODEL_DEF partition]
        K --> M[WEIGHTS_DATA partition]
        K --> N[TBE_KERNELS partition]
        K --> O[SO_BINS partition]
        K --> P[TILING_DATA partition]
        K --> P2[CUSTOM_OPS partition]
    end

    subgraph "Runtime (ModelManager)"
        O --> Q[ModelHelper::LoadOpSoBin]
        Q --> R[OpSoStore::Load parsing]
        R --> S{Classify by SoBinType}
        S -->|SpaceRegistry| T[Register to OpImplSpaceRegistry]
        S -->|OpMasterDevice| U[Load to built_in/cust_op_master<br/>_so_names_to_bin_]
        S -->|Autofuse| V[guard_check.so restored to _guard_check_so_data<br/>others written to bin_file_buffer ext attribute]
        S -->|CustomOp| V2[Load to CustomOpSoLoader]
        T --> W[Dynamic invocation during model execution]
        U --> W
        V --> W
        V2 --> W
    end
```

### 2.1 OM File Partition Structure

The SO in OM feature adds a new partition type `SO_BINS` to the OM file format:

| Partition Type | Purpose | Relationship with SO in OM |
|---------|------|-------------------|
| `MODEL_DEF` | Model definition (graph structure, operator attributes) | Contains `so_in_om_flag` marker |
| `WEIGHTS_DATA` | Model weight data | Independent |
| `TBE_KERNELS` | TBE operator binary | Complementary to SO_BINS |
| `SO_BINS` | Operator .so file collection | Core carrier of SO in OM |
| `TILING_DATA` | Precomputed tiling parameters | Used together with SpaceRegistry SO |
| `CUSTOM_OPS` | Custom operator instance serialized data | Used together with kCustomOp type SO in SO_BINS |

### 2.2 Four SO Types

GE categorizes the SO files to be packaged into four types, each corresponding to different use cases and lifecycle:

```mermaid
graph LR
    subgraph "SoBinType Enumeration"
        A[kSpaceRegistry = 0] -->|bit 15| B[0x8000]
        C[kOpMasterDevice = 1] -->|bit 14| D[0x4000]
        E[kAutofuse = 2] -->|bit 13| F[0x2000]
        E2[kCustomOp = 3] -->|bit 12| F2[0x1000]
    end

    subgraph "Purpose"
        B --> G[RT2 dynamic shape<br/>infer shape / tiling so]
        D --> H[Device-side tiling so<br/>op_master_device]
        F --> I[Autofuse fusion operator so<br/>offline save and load]
        F2 --> I2[PortableOp custom operator so<br/>offline save and load]
    end

    subgraph "Trigger Condition"
        G --> J[Dynamic shape model<br/>or _static_to_dynamic_softsync_op]
        H --> K[TaskDef contains<br/>PREPROCESS_KERNEL type task]
        I --> L[Graph node contains<br/>bin_file_path attribute<br/>or _guard_check_so_data is non-empty]
        I2 --> L2[Graph contains PortableOp custom operators<br/>recognizable by CustomOpRegistry]
    end
```

`so_in_om_flag` is a `uint16_t`, where each bit indicates whether a SO type is enabled. Multiple SO types can be combined (for example, `0xC000` indicates both SpaceRegistry and OpMasterDevice are included).

## 3. Compilation Phase: SO Packaging Process

### 3.1 Trigger Point

SO packaging occurs at the end of the `GeGenerator` offline model generation process. Key entry point:

**File Path**: `compiler/api/generator/ge_generator.cc`

```
GenerateOfflineModel()
  └── GenerateModel()
        └── impl_->SaveRootModel()
              └── ModelHelper::SaveToOmRootModel()            (model_helper.cc)
                    └── SaveRootModelPartitions()             (model_helper.cc)
                          ├── SaveSoStoreModelPartitionInfo()  ← SO packaging entry (model_helper.cc)
                          └── SaveCustomOpsPartition()         ← Custom operator partition (model_helper.cc)
```

### 3.2 Detection Phase: CheckAndSetNeedSoInOM

Before packaging, the system needs to determine whether the model requires SO packaging and which types of SO need to be packaged.

**File Path**: `base/common/model/ge_root_model.cc`

The detection logic consists of four independent check functions:

#### 3.2.1 CheckAndSetSpaceRegistry

**Trigger Conditions**:
- Model contains dynamic shape (`ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED` is true or `GetGraphUnknownFlag()` is true)
- Model contains `_static_to_dynamic_softsync_op` type operators

**Description**: Dynamic shape models require dynamically computing tensor memory layout and tiling parameters at runtime. These computation logic is provided by .so files in SpaceRegistry. After packaging, the runtime does not need to load from external OPP paths.

#### 3.2.2 CheckAndSetOpMasterDevice

**Trigger Conditions**: Traverse all TaskDefs. If a `MODEL_TASK_PREPROCESS_KERNEL` type task is found and its `kernel().so_name()` is non-empty.

**Description**: `PREPROCESS_KERNEL` is the preprocessing logic executed on the device side before the operator executes (such as tiling calculation), which requires corresponding .so files to provide implementation. These .so files are typically located in `/op_impl/ai_core/tbe/op_master_device/lib/` path.

#### 3.2.3 CheckAndSetAutofuseSo

**Trigger Conditions**:
- Graph node contains `bin_file_path` attribute
- Root graph `_guard_check_so_data` attribute is non-empty (ge_root_model.cc)

**Description**: Autofuse is GE's operator automatic fusion optimization feature. Fused operators generate independent .so files that need to be distributed with the model. In addition to checking `bin_file_path`, `CheckAndSetAutofuseSo()` also checks the root graph's `_guard_check_so_data` attribute — when this attribute is non-empty, the kAutofuse flag is also set, ensuring guard check data is saved and loaded with the model.

#### 3.2.4 CheckAndSetCustomOpSo

**Trigger Conditions**: The graph contains `PortableOp` custom operators recognizable by the `CustomOpRegistry` held by the current `GeRootModel`.

**Description**: `GraphManager::PreRun()` explicitly binds the process-level global `CustomOpRegistry` to the current `GeRootModel` after `BuildModel()` returns. Subsequent custom operator SO collection and `CUSTOM_OPS` partition serialization both access custom operators through `ge_root_model->GetCustomOpRegistry()`, and the save process no longer directly accesses `CustomOpFactory`. When repackaging an existing OM, if the model does not carry a custom op registry, only the custom op partition processing is skipped, without falling back to the process-level global registry.

**Cross-compilation scenario** (ge_root_model.cc): `CheckAndSetCustomOpSo()` uses `IsCrossCompileTarget()` to determine whether the target environment differs from the compilation environment (comparing OS and CPU architecture). In non-cross-compilation scenarios, `dladdr` is used to resolve the actual SO path from the `PortableOp` vtable, and `CheckSoArchMatchesTarget()` performs ELF architecture validation; in cross-compilation scenarios, local SO collection is skipped, and `CollectCustomOpSoFromCustomOppPath()` is called to collect SOs from the target environment operator package directory pointed to by the `ASCEND_CUSTOM_OPP_PATH` environment variable, also validated by `CheckSoArchMatchesTarget()` to ensure the ELF architecture matches the target CPU.

### 3.3 Collection Phase: LoadAndStoreOppSo

After determining the SO types to be packaged, `ModelHelper` calls `LoadAndStoreOppSo()` to load .so files from disk into the `OpSoStore` object in memory.

**File Path**: `base/common/helper/model_helper.cc`

```
SaveSpaceRegistrySoBin()
  └── GetSoBinData(cpu, os)  ← Get corresponding so based on compilation host environment
  └── LoadAndStoreOppSo()

SaveOpMasterDeviceSoBin()
  └── LoadAndStoreOppSo(ge_root_model->GetOpMasterDeviceSoSet())

SaveAutofuseSoBin()  (model_helper.cc)
  ├── Process _guard_check_so_data attribute → generate guard_check.so OpSoBin
  ├── Process bin_file_buffer ext attribute → sync existing OpSoBin
  └── LoadAndStoreOppSo(ge_root_model->GetAutofuseSoSet())

SaveCustomOpSoBin()  (model_helper.cc)
  └── LoadAndStoreOppSo(ge_root_model->GetCustomOpSoSet(), SoBinType::kCustomOp)
```

SpaceRegistry SO file names embed the compilation host's OS and CPU information (such as `_linux_x86_64` suffix), because tiling/infer shape logic executes on the host side and needs to match the compilation environment.

In addition to the regular `LoadAndStoreOppSo()` loading, `SaveAutofuseSoBin()` has two extra steps: first, it wraps the root graph's `_guard_check_so_data` attribute content as an OpSoBin named `guard_check.so` and adds it to OpSoStore; then it checks the root graph's `bin_file_buffer` ext attribute (present when repackaging an existing OM), and if non-empty, directly syncs the OpSoBins within it to OpSoStore without repeatedly loading from disk.

### 3.4 Serialization Phase: OpSoStore::Build

**File Path**: `base/common/op_so_store/op_so_store.cc`

`OpSoStore` serializes multiple .so files into a contiguous memory block and writes it to the `SO_BINS` partition of the OM file. The binary format is as follows:

```
┌─────────────────────────────────────────┐
│ SoStoreHead (4 bytes)                   │
│   so_num: uint32                        │  ← Total SO file count
├─────────────────────────────────────────┤
│ SoStoreItemHead (16 bytes)              │  ← Header of 1st SO
│   magic:       0x5D776EFD               │
│   so_name_len: uint16                   │
│   so_bin_type: uint16                   │  ← SpaceRegistry/OpMasterDevice/Autofuse/CustomOp
│   vendor_name_len: uint32               │
│   bin_len:     uint32                   │
├─────────────────────────────────────────┤
│ so_name (so_name_len bytes)             │
├─────────────────────────────────────────┤
│ vendor_name (vendor_name_len bytes)     │
├─────────────────────────────────────────┤
│ so binary data (bin_len bytes)          │
├─────────────────────────────────────────┤
│ SoStoreItemHead (16 bytes)              │  ← Header of 2nd SO
│ ...                                     │
└─────────────────────────────────────────┘
```

Format description:

- **Magic number validation**: Each item contains a magic number (`0x5D776EFD`), used to validate data integrity during loading
- **Variable-length strings**: so_name and vendor_name use length-prefixed variable-length encoding, avoiding space waste from fixed-length fields
- **Type marker**: Each item independently records `so_bin_type`, and during loading, items are distributed to different caches by type

### 3.5 Environment Information Recording

**File Path**: `base/common/helper/model_helper.cc`

While packaging SO, the system records compilation environment information to the `SoInOmInfo` structure, including compilation host CPU architecture, operating system, OPP operator package version, and compiler version. This information is used for compatibility validation during runtime loading, ensuring the SO in the OM file is compatible with the current runtime environment.

## 4. Runtime: SO Loading and Execution

### 4.1 Loading Entry

**File Path**: `base/common/helper/model_custom_kernels_helper.cc`

During model loading, `ModelHelper` processes the SO_BINS partition in the following order. Note: `LoadModel()` (model_helper.cc) does not process SO_BINS; the SO loading entry is in the `LoadRootModel()` (model_helper.cc) flow:

```
ModelHelper::LoadRootModel()                         (model_helper.cc)
  └── GenerateGeRootModel()                           (model_helper.cc)
        └── LoadCustomOpRegistry()                    (model_custom_kernels_helper.cc)
              └── LoadOpSoBin()                       (model_custom_kernels_helper.cc)
                    └── GeRootModel::LoadSoBinData()  (ge_root_model.cc)
                          └── OpSoStore::Load(data, len)  ← Deserialize SO_BINS partition
                                └── Parse SoStoreHead and each SoStoreItemHead
                                └── Create OpSoBin objects and add to kernels_ list
```

### 4.2 Loading by Type Distribution

**File Path**: `base/common/helper/model_custom_kernels_helper.cc`, `runtime/v1/graph/load/model_manager/model_manager.cc`

After `LoadOpSoBin()` completes, SO files are distributed to different processing paths by type:

#### 4.2.1 SpaceRegistry SO Loading

SpaceRegistry SO is registered to `OpImplSpaceRegistryV2Array`, which is the core data structure used by RT2 (Runtime V2) executor to manage dynamic shape operator implementations. During inference, the executor finds and loads corresponding tiling/infer shape functions through the registry.

#### 4.2.2 OpMasterDevice SO Loading

**File Path**: `runtime/v1/graph/load/model_manager/model_manager.cc`

OpMasterDevice SO loading (`InitOpMasterDeviceSo`, model_manager.cc) uses two deduplication strategies, stored in different maps:

- **Built-in SO**: Stored in `built_in_op_master_so_names_to_bin_` (model_manager.cc), deduplicated by SO name (type + version number ensures uniqueness). Only one copy of SO with the same name is retained
- **Custom SO**: Stored in `cust_op_master_so_names_to_bin_` (model_manager.cc), deduplicated by binary content — the complete SO data is used as the key to establish mapping in `cust_op_master_so_datas_to_name_` (model_manager.cc). When multiple models reference custom operators with the same content but different file names, the system can identify and reuse existing SO, avoiding repeated loading

#### 4.2.3 Autofuse SO Loading

**File Path**: `base/common/helper/model_custom_kernels_helper.cc`

When `LoadOpSoBin()` iterates over all OpSoBins, kAutofuse type SOs are not simply cached but processed by content separately (model_custom_kernels_helper.cc):

- **`guard_check.so`**: Its binary content is restored as the root graph's `_guard_check_so_data` string attribute, for runtime guard check logic to use
- **Other Autofuse SOs**: Stored in the `bin_file_buffer` mapping with `vendor_name/so_name` as the key, and set as the root graph's ext attribute for on-demand loading at runtime

#### 4.2.4 CUSTOM_OPS Partition Loading

**File Path**: `base/common/helper/model_custom_kernels_helper.cc`

The `CUSTOM_OPS` partition in the offline OM carries custom operator instance serialized data. When loading the root model offline, even if the OM does not carry custom operator SOs or a non-empty `CUSTOM_OPS` partition, a model-level empty `CustomOpRegistry` is created and injected into the `GeRootModel`, to identify the model's custom operator lookup scope. When loading a non-empty `CUSTOM_OPS` partition, it must be written to the `CustomOpRegistry` held by the current model, and must not fall back to the process-level global `CustomOpFactory`, to prevent multi-model private custom operator state from polluting each other. RT2 `ModelConverter::ConvertGeModelToExecuteGraph()` only consumes the registry already injected into `GeRootModel`; if the registry is empty, it is treated as an upstream construction anomaly, and the global registry is not used as fallback during the Convert phase.

#### 4.2.5 Compatibility Validation

**File Path**: `base/common/helper/model_helper.cc`

During loading, the system validates whether the OPP version and compiler version recorded in the OM file are compatible with the current runtime environment, detecting incompatibility issues early.

### 4.3 Execution Invocation

After SO is loaded into memory, it is invoked during model execution through the following path:

```
Model execution request
  └── StreamExecutor::Execute()
        └── HybridModelExecutor::Execute()
              └── NodeExecutor::Execute()
                    └── OpImplSpaceRegistry::GetFunction()  ← Find loaded SO function
                          └── dlsym() to get function pointer
                                └── Call tiling/infer shape function
```

For Single Op scenarios (`runtime/v1/single_op/`), the execution flow is `SingleOpModel::BuildOp()` → `BuildTaskList()` → `BuildTEKernelAndTask()`, using the kernel implementations in the loaded SO.

## 5. Single Op Scenario

### 5.1 Single Op Compilation Process

**File Path**: `api/acl/acl_op_compiler/single_op/compile/local_compiler.cpp`

Single Op compilation is an important application scenario of the SO in OM feature. Users compile individual operators to OM files through ACL API:

```
aclopCompileOp()
  └── OpCompiler::CompileOp()
        └── LocalCompiler::DoCompile()
              └── OnlineCompileAndDump()
                    └── GeGenerator::BuildSingleOpModel()
                          └── BuildSingleOp()
                                └── Compile operator → Generate OM → Package SO
```

### 5.2 Single Op Execution

**File Path**: `runtime/v1/single_op/single_op_model.cc`

After the Single Op OM file is loaded, it is parsed and executed through the `SingleOpModel` class, completing input/output tensor description parsing, device memory allocation, address mapping setup, TaskDef list parsing, and execution task chain construction.

The `SingleOpModelParam` structure contains the `space_registries_` field, used to pass SpaceRegistry SO registration information, ensuring Single Op execution can also access tiling functions needed for dynamic shape.

## 6. Data Structures

### 6.1 SoInOmFlag Bit Flags

**File Path**: `base/common/op_so_store/op_so_store_utils.h`

Bit flags implement type judgment and setting through bit shift operations. From high bit to low bit: SpaceRegistry(15), OpMasterDevice(14), Autofuse(13), CustomOp(12).

**Bit Flag Values**:
- `kSpaceRegistry` (0): `0x8000`
- `kOpMasterDevice` (1): `0x4000`
- `kAutofuse` (2): `0x2000`
- `kCustomOp` (3): `0x1000`

Combination example: `0xC000` = SpaceRegistry + OpMasterDevice

### 6.2 OpSoBin Object

**File Path**: `inc/graph_metadef/graph/op_so_bin.h`

`OpSoBin` encapsulates the metadata and binary content of a single SO file, containing SO file name, vendor name (built-in / vendors/xxx), binary data, data size, and SO type.

### 6.3 SoStoreHead and SoStoreItemHead

**File Path**: `base/common/op_so_store/op_so_store.h`

`SoStoreHead` records the total SO file count. `SoStoreItemHead` contains magic number (0x5D776EFD), SO name length, SO type enumeration value, vendor name length, and binary data length.

## 7. Key File Index

| File Path | Responsibility |
|---------|------|
| `inc/graph_metadef/graph/op_so_bin.h` | `OpSoBin`, `SoBinType`, `SoInOmInfo` definitions |
| `base/common/op_so_store/op_so_store.h` | `OpSoStore` class definition, SO serialization container |
| `base/common/op_so_store/op_so_store.cc` | `OpSoStore::Build/Load` implementation |
| `base/common/op_so_store/op_so_store_utils.h` | `OpSoStoreUtils` bit flag operation utilities |
| `base/common/model/ge_root_model.cc` | `CheckAndSetNeedSoInOM` detection logic |
| `base/common/helper/model_helper.cc` | Core process of SO packaging and loading |
| `base/common/helper/model_custom_kernels_helper.cc` | `LoadOpSoBin`, `LoadCustomOpRegistry`, `SaveCustomOpsPartition` implementation |
| `compiler/api/generator/ge_generator.cc` | `BuildSingleOpModel` compilation entry |
| `runtime/v1/graph/load/model_manager/model_manager.cc` | `InitOpMasterDeviceSo` runtime loading |
| `runtime/v1/single_op/single_op_model.cc` | Single Op model parsing and execution |
| `api/acl/acl_op_compiler/single_op/compile/local_compiler.cpp` | ACL Single Op compilation implementation |
| `tests/ge/st/testcase/fast_runtime_v2/so_in_om_system_test.cc` | SO in OM system test cases |
