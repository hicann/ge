# ES (Eager Style) Generator

## Prerequisites

1. Correctly install the `toolkit` package through the [installation guide](../../../quick_install.md), and **correctly configure environment variables** as instructed
2. Correctly install the operator `ops` package through the [installation guide](../../../quick_install.md) (ES depends on operator prototypes for API generation), and **correctly configure environment variables** as instructed

## Environment Variable Requirements

Environment variable list required by gen_esb:

- ASCEND_OPP_PATH: Points to the opp path under the installation directory
- LD_LIBRARY_PATH: Environment variable specifying dynamic link library search paths

Note: The above environment variables do not need to be configured separately and it is not recommended to do so. The environment variables already configured in the [Prerequisites](#prerequisites) section by default satisfy the requirements.

## Functionality Description

### This program supports two generation modes

1. Code generation mode
  Generates ES graph builder C, C++, and Python code, including:

- C interfaces for all supported operators (ops)
- C++ interfaces for all supported operators
- Python interfaces for all supported operators
- Aggregated header file for users to include all operators at once
- Aggregated Python file for users to import all operators at once

2. Historical prototype library generation mode
  Generates historical prototype structured data, including:

- Version index
- Version metadata
- Operator prototype data for that version

## Usage Methods

### Code Generation Mode

```bash
gen_esb [--output_dir=DIR] [--module_name=NAME] [--h_guard_prefix=PREFIX] [--exclude_ops=OP_TYPE1,OP_TYPE2] [--history_registry=PKG_DIR] [--release_version=VER]
```

### Historical Prototype Library Generation Mode

```bash
gen_esb --es_mode=extract_history --release_version=VER [--output_dir=DIR] [--release_date=YYYY-MM-DD] [--branch_name=BRANCH]
```

Note: Because environment variables have already been configured in the [Prerequisites](#prerequisites) section, `gen_esb` has been added to the `PATH` environment variable at this point, so it can be executed directly.

### Parameter Description

- --es_mode: Optional parameter, specifies the generation mode, supports `codegen` and `extract_history`
  If not specified, defaults to codegen
- --output_dir: Optional parameter, specifies the target output directory
  If not specified, defaults to outputting to the current directory
- --module_name: Optional parameter, controls the naming of aggregated header files
  - "math" -> es_math_ops_c.h, es_math_ops.h, es_math_ops.py
  - "all" -> es_all_ops_c.h, es_all_ops.h, es_all_ops.py
  - Not passed -> defaults to "all"
- --h_guard_prefix: Optional parameter, controls the generated header file guard macro prefix, used for distinguishing possible name conflicts between internal and external operators
  - If not specified, uses the default prefix
  - When specified, concatenates with the default prefix
  - Python files are not aware of this parameter; same-name scenarios are avoided through different paths
- --exclude_ops: Optional parameter, controls which operators to exclude from code generation
  - Separate operator names by `,`
- --history_registry: Optional parameter, specifies the historical prototype library directory for code production
  - If not specified, the historical prototype library is not enabled by default
  - When specified, generated C++ interfaces will include compatible version information from the historical prototype library
- --release_version:
  - Code generation mode: Optional parameter, used with `--history_registry`, specifies the current version number; generated C++ interfaces include compatible version information for that version; if not specified, generates historical versions compatible with the current date as baseline
  - Historical prototype library generation mode: Required parameter, specifies the version number corresponding to the current historical prototype data
- --release_date: Optional parameter, controls the release date of historical prototype structured data, format `YYYY-MM-DD`
  - If not specified, uses the current date
- --branch_name: Optional parameter, controls the release branch name of historical prototype structured data

### Output File Description

#### Code Generation Mode Output

- es_\<module\>_ops_c.h: C interface aggregated header file
- es_\<module\>_ops.h: C++ interface aggregated header file
- es_\<module\>_ops.py: Python interface aggregated file
- es_\<op_type>_c.h: Single operator C interface header file
- es_\<op_type>.cpp: Single operator C interface implementation file
- es_\<op_type>.h: Single operator C++ interface header file
- es_\<op_type>.py: Single operator Python interface file

#### Historical Prototype Library Generation Mode Output

- index.json: Version index
- registry/<ver>/metadata.json: Version metadata
- registry/<ver>/operators.json: Operator prototype data for that version

## Usage Examples

### Generate code to current directory, use default module name "all", default guard macro prefix

`gen_esb`

### Generate code to specified directory, use default module name "all", default guard macro prefix

`gen_esb --output_dir=./output`

### Generate code to specified directory, use "math" module name, default guard macro prefix

`gen_esb --output_dir=./output --module_name=math`

### Generate code to specified directory, use "all" module name, default guard macro prefix

`gen_esb --output_dir=./output --module_name=all`

### Generate code to specified directory, use "math" module name, custom guard macro prefix "MY_CUSTOM"

`gen_esb --output_dir=./output --module_name=math --h_guard_prefix=MY_CUSTOM`

### Generate code to specified directory, use "math" module name, custom guard macro prefix "MY_CUSTOM", and exclude Add operator generation

`gen_esb --output_dir=./output --module_name=math --h_guard_prefix=MY_CUSTOM --exclude_ops=Add`

### Generate code to specified directory, use "math" module name, default guard macro prefix, generated C++ interfaces will include compatible version information filtered based on current date from math historical prototype directory

`./gen_esb --output_dir=./output --module_name=math --history_registry=/${CANN_INSTALL_PATH}/cann/opp/history_registry/math`

### Generate code to specified directory, use "math" module name, default guard macro prefix, generated C++ interfaces will include historical version information compatible with "8.0.RC2" version from math historical prototype directory

`./gen_esb --output_dir=./output --module_name=math --history_registry=/${CANN_INSTALL_PATH}/cann/opp/history_registry/math --release_version=8.0.RC2`

### Generate historical prototype structured data to current directory, release version "8.0.RC1", default release date as current date

`./gen_esb --es_mode=extract_history --release_version=8.0.RC1`

### Generate historical prototype structured data to specified directory, release version "8.0.RC1", default release date as current date

`./gen_esb --es_mode=extract_history --release_version=8.0.RC1 --output_dir=/${CANN_INSTALL_PATH}/cann/opp/history_registry/math`

### Generate historical prototype structured data to specified directory, release version "8.0.RC1", custom release date "2024-09-30", branch name "master"

`./gen_esb --es_mode=extract_history --release_version=8.0.RC1 --output_dir=/${CANN_INSTALL_PATH}/cann/opp/history_registry/math --release_date=2024-09-30 --branch_name=master`

## Precautions

1. Ensure [environment variables](#environment-variable-requirements) are correctly set
2. Ensure sufficient disk space for storing generated code files
3. The number of generated code files depends on the number of operators registered in the system
4. The guard macro prefix should consist of uppercase letters and underscores to avoid conflicts with C++ keywords

## Error Handling

- If environment variables are not set, the program will prompt an error and exit
- If the output directory creation fails, it will fall back to the current directory
- Unsupported operators will be recorded in the generated code comments
