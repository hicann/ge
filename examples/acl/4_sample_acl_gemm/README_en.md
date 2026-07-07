# ACL GEMM API Migration Sample

## Function Description

Demonstrates how to **replace deprecated ACLBLAS and ACLOP APIs with ACLNN APIs**. Uses the GEMM (General Matrix Multiply) operator to compare equivalent implementations across three ACL API families, helping developers understand the migration workflow.

| API | Status | Invocation |
|-----|--------|------------|
| `aclblasGemmEx` | Deprecated | Execute GEMM via ACLBLAS interface |
| `aclopExecuteV2("GEMM")` | Deprecated | Invoke GEMM operator via ACLOP interface |
| `aclnnMatmul` + `aclnnMuls` + `aclnnAdd` | Recommended | Compose atomic ACLNN operations |

The sample runs the same input data through all three APIs sequentially, automatically compares precision, and outputs `max_error`.

### GEMM Formula

```
C = α·A·B + β·C
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| M | 64 | Rows of A / Rows of C |
| N | 64 | Columns of B / Columns of C |
| K | 64 | Columns of A / Rows of B |
| alpha | 2.0 | Scale factor for matmul result |
| beta | 2.0 | Scale factor for original C |
| Data type | FP16 | All matrices and scalars use FP16 |

**ACLNN API combination**: ACLNN APIs are closer to atomic operator operations, requiring GEMM to be decomposed into:
1. `aclnnMatmul` → `A @ B`
2. `aclnnMuls` → `alpha × (A @ B)` and `beta × C`
3. `aclnnAdd` → sum the two parts

## Quick Start

```bash
bash scripts/build.sh && bash scripts/run.sh
```

## Directory Structure

```
4_sample_acl_gemm/
├── CMakeLists.txt              # Top-level build script
├── scripts/
│   ├── build.sh                # Build script
│   └── run.sh                  # Run script
└── src/
    ├── CMakeLists.txt          # Build configuration
    └── sample_acl_gemm.cpp     # Main program (three-API comparison)
```

## Environment Requirements

- [ ] Install `toolkit` and `ops` packages per [Environment Preparation](../../../docs/en/quick_install.md)
- [ ] Set environment variables: `source /usr/local/Ascend/cann/set_env.sh` (adjust path to your installation)
- [ ] (Optional) Specify chip version: `export SOC_VERSION=Ascend910B3` (defaults to `Ascend910B3` if not set)

## Detailed Steps

### Step 1: Build

```bash
bash scripts/build.sh
```

### Step 2: Run

```bash
bash scripts/run.sh
```

## Expected Output

```
[INFO] ACL GEMM sample starts
[INFO] Running ACLBLAS GEMM...
[INFO] Running ACLOP GEMM...
[INFO] Running ACLNN GEMM...
[INFO] Comparing results...
max_error: 0
[INFO] VERIFICATION PASSED
[INFO] SAMPLE PASSED
```

## FAQ

**Q: Build fails with missing ACLNN header files**

Make sure you have run `source /usr/local/Ascend/cann/set_env.sh` and that your installed toolkit version includes ACLNN APIs.

**Q: max_error is not zero**

Small errors may occur under FP16 precision. If max_error is small (e.g., < 1e-3), it is a normal floating-point error. If it is large, check whether `SOC_VERSION` matches your current chip.

**Q: How to migrate my own ACLOP code to ACLNN?**

Follow the ACLNN implementation pattern in `sample_acl_gemm.cpp`: first call `aclnnXxxGetWorkspaceSize` to obtain the workspace size, then call `aclnnXxx` to execute the operation.
