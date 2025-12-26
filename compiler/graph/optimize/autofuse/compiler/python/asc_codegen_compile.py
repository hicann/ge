#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#-------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import subprocess
import shutil
import tempfile
import platform
import json
import time

from tbe.common.buildcfg import get_current_build_config
from tbe.tikcpp.compile_op import CommonUtility, AscendCLogLevel
from tbe.common.platform.platform_info import get_soc_spec, set_soc_spec
import tbe.common.utils.log as logger
# Python3 lib pyautofuse.so
from .pyautofuse import Schedule, CodeGen, ascir
from .ascbc_kernel_compile import ascbc_kernel_compile, camel_to_snake
from .compile_adapter import get_pgo_env_flag, get_pgo_topn
from tbe.tikcpp.get_op_tiling import TilingInfo, _change_param_name_to_name, gen_static_shape_v2
from tbe.common.utils.op_tiling import do_op_tiling
from tbe.common.context import get_context
from tbe.tikcpp import OpInfo

PYF_PATH = os.path.dirname(os.path.realpath(__file__))
ASCEND_PATH = os.path.join(PYF_PATH, "..", "..", "..")
timestamp_list = []


def generate_cmake_lists(asc_graph_name, kernel_name, host_build_dir, is_last_compile, is_static_shape):
    source = f"################ {kernel_name}.so ################\n"

    machine = platform.machine()
    source += f"set(CMAKE_CXX_STANDARD 17)\n"
    source += f"link_directories({ASCEND_PATH}/{machine}-linux/lib64/\n"
    source += f")\n\n"

    source += f"file(GLOB ALL_CPP_SRCS\n"
    source += f"    *tiling_func*.cpp\n"
    source += f"    *infershape*.cpp\n"
    if is_last_compile:
        source += f"    *get_kernel*.cpp\n"
    source += ")\n\n"
    source += f"add_library({kernel_name} SHARED ${{ALL_CPP_SRCS}})\n\n"

    source += f"target_compile_options({kernel_name} PRIVATE\n"
    if is_static_shape:
        source += ("    -O0 -fno-common -Werror -Wextra -Wfloat-equal -fvisibility=default -DLOG_CPP"
                   " -ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/=\n")
    else:
        source += ("    -O2 -fno-common -Werror -Wextra -Wfloat-equal -fvisibility=default -DLOG_CPP"
                   " -ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/=\n")

    source += ")\n\n"
    source += f"message(STATUS \"Using environment variable ASCEND_INSTALL_PATH: {ASCEND_PATH}\")\n"
    source += (f"target_link_libraries({kernel_name} PRIVATE  c_sec ascendalog platform"
               f" tiling_api graph_base register)\n")
    source += f"target_include_directories({kernel_name} PRIVATE\n"
    source += f"    {ASCEND_PATH}/include\n"
    source += f"    {ASCEND_PATH}/pkg_inc/base\n"
    source += f"    {ASCEND_PATH}/include/experiment\n"
    source += f"    {ASCEND_PATH}/{machine}-linux/ascendc/include/highlevel_api/tiling/platform\n"
    source += f"    {ASCEND_PATH}/{machine}-linux/ascendc/include/highlevel_api\n"
    source += "    )\n\n"

    with open(host_build_dir + "/CMakeLists.txt", "w") as cmake_file:
        cmake_file.write(source)


def generate_file(dst_dir, file_name, text):
    os.makedirs(dst_dir, exist_ok=True)
    file_path = os.path.join(dst_dir, file_name)
    with open(file_path, "w") as file:
        file.write(text)


def ascbc_host_compile(graph_name, kernel_name, host_build_dir, is_last_compile, is_static_shape):
    generate_cmake_lists(graph_name, kernel_name, host_build_dir, is_last_compile, is_static_shape)
    ori_directory = os.getcwd()
    # 切换到临时工作目录
    os.chdir(host_build_dir)

    cmake_command = ["cmake", "-S", ".", "-B", "./", "-DCMAKE_C_COMPILER=gcc", "-DCMAKE_CXX_COMPILER=g++"]

    # 运行CMake
    cmake_ret = subprocess.run(cmake_command, capture_output=True, text=True)
    if cmake_ret.returncode != 0:
        os.chdir(ori_directory)
        CommonUtility.print_compile_log("", f"camke fail: {cmake_ret.stderr}", AscendCLogLevel.LOG_ERROR)
        error_msg = f"execute cmake failed with return code {cmake_ret.returncode}\n"
        error_msg += f"Standard Output:\n{cmake_ret.stdout}\n"
        error_msg += f"Standard Error:\n{cmake_ret.stderr}\n"
        raise Exception(error_msg)

    # 定义 CMake 编译命令
    make_command = ["make", "-C", "./", '-j']
    # 编译生成so
    make_ret = subprocess.run(make_command, capture_output=True, text=True)
    if make_ret.returncode != 0:
        os.chdir(ori_directory)
        CommonUtility.print_compile_log("", f"make fail: {make_ret.stderr}", AscendCLogLevel.LOG_ERROR)
        error_msg = f"execute cmake failed with return code {make_ret.returncode}\n"
        error_msg += f"Standard Output:\n{make_ret.stdout}\n"
        error_msg += f"Standard Error:\n{make_ret.stderr}\n"
        raise Exception(error_msg)
    # 回退到原始工作目录
    os.chdir(ori_directory)


def gen_valid_name(t_name):
    invalid_chars = ["/", ".", "+", "-", "*", "%"]
    ret_name = t_name
    for char in invalid_chars:
        ret_name = ret_name.replace(char, "_")

    if ret_name and ret_name[0].isdigit():
        ret_name = "t_" + ret_name

    return ret_name


def is_valid_path(path):
    """
    判断路径是否合法。
    
    :param path: 路径字符串
    :return: 如果路径合法返回 True，否则返回 False
    """
    try:
        # 尝试获取路径的绝对路径
        os.path.abspath(path)
        return True
    except (TypeError, ValueError):
        # 如果路径无效，会抛出异常
        return False


def check_keys_in_dict(d, keys):
    """
    检查字典中是否缺少指定的键。如果所有指定的键都存在，返回True,
    
    :param d: 字典
    :param keys: 需要检查的键列表
    """
    # 检查每个键是否在字典中
    for key in keys:
        if key not in d:
            return False  # 如果任何一个键不存在，直接返回false

    # 如果所有键都存在
    return True


def modify_json_file(json_file, host_so):
    """
    修改json文件中的binFileName/binFileSuffix
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    kernenl_file_name = data['binFileName']
    lib_file_name = "lib" + kernenl_file_name
    lib_file_suffix = ".so"
    data['binFileName'] = lib_file_name
    data['binFileSuffix'] = lib_file_suffix
    import ctypes
    # 加载共享库
    lib = ctypes.CDLL(host_so)

    # 定义函数参数和返回类型
    lib.GetTilingDataSize.argtypes = []
    lib.GetTilingDataSize.restype = ctypes.c_size_t

    # 调用函数
    data['opParaSize'] = int(lib.GetTilingDataSize())
    CommonUtility.print_compile_log("", f"{kernenl_file_name} tiling size: {data['opParaSize']}",
                                    AscendCLogLevel.LOG_INFO)
    # 自动融合 workspace 默认只有一个 
    data['workspace']['num'] = int(1)
    data['workspace']['size'] = list([-1])
    data['workspace']['type'] = list([0])
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


def timestamp_set(is_start_stamp, graph_name, stage, need_report=False):
    """
    设置时间戳列表
    """
    timestamp = time.time_ns()
    if is_start_stamp:
        timestamp_list.append(timestamp)
    else:
        start = timestamp_list[-1]
        cost_time = (timestamp - start)
        timestamp_list[-1] = cost_time
        ascir.utils.duration_record([stage, graph_name], int(start), int(cost_time))
    if need_report:
        ascir.utils.report_durations()

def ascendc_clean(temp_dir):
    src_directory = os.getcwd()
    keep_dirs = {'host', 'device'}
    # 遍历目录中的所有条目
    for entry in os.listdir(temp_dir):
        entry_path = os.path.join(temp_dir, entry)
        # 检查是否是文件或目录
        if os.path.isfile(entry_path):
            # 如果是文件，直接删除
            os.remove(entry_path)
            CommonUtility.print_compile_log("", f"delete file: {entry_path}", AscendCLogLevel.LOG_INFO)
        else:
            # 如果是目录，检查是否需要保留
            if entry not in keep_dirs:
                shutil.rmtree(entry_path)
                CommonUtility.print_compile_log("", f"delete dir: {entry_path}", AscendCLogLevel.LOG_INFO)
    os.chdir(src_directory)


def static_shape_kernel_proc(kernel_src, temp_dir, kernel_type):
    import re
    base_device_files = os.path.basename(kernel_src)
    kernel_file = os.path.join(temp_dir, "device", base_device_files)

    with open(kernel_file, 'r', encoding='utf-8') as file:
        code = file.read()

    # 使用正则表达式进行替换
    code = re.sub(r'GET_TILING_DATA\(t, gm_tiling_data\);', 'const AutofuseTilingData t;', code)
    code = re.sub(r'GM_ADDR workspace, AutofuseTilingData& t', 'GM_ADDR workspace, const AutofuseTilingData& t', code)
    if kernel_type is not None:
        code = re.sub(r'KERNEL_TASK_TYPE_DEFAULT\(.*\)', f'KERNEL_TASK_TYPE_DEFAULT({kernel_type})', code)

    # 将修改后的内容写回文件
    with open(kernel_file, 'w', encoding='utf-8') as file:
        file.write(code)


def static_shape_compile(kernel_name, temp_dir, graph_name):
    ori_directory = os.getcwd()
    host_build_dir = os.path.join(temp_dir, "host")
    ascbc_host_compile(graph_name, kernel_name, host_build_dir, False, True)

    kernel_src = graph_name + "_op_kernel.cpp"
    host_so = os.path.join(host_build_dir, f"lib{kernel_name}.so")
    pgo_config_path = os.path.abspath(
        os.path.join(temp_dir, "..", "..", "pgo", f"{graph_name}_config.txt")
    )

    import ctypes
    lib = ctypes.CDLL(host_so)

    CommonUtility.print_compile_log("", f"static shape compile", AscendCLogLevel.LOG_INFO)
    ascendc_clean(temp_dir)
    lib.GenConstTilingData.argtypes = []
    lib.GenConstTilingData.restype = ctypes.c_char_p
    result = lib.GenConstTilingData(ctypes.c_char_p(pgo_config_path.encode('utf-8')),
                                    ctypes.c_int(int(get_soc_spec('vector_core_cnt'))),
                                    ctypes.c_int(int(get_soc_spec('ub_size'))))

    const_tiling_data = result.decode('utf-8')
    tiling_data = os.path.join(temp_dir, "device", "autofuse_tiling_data.h")
    tiling_data_bak = os.path.join(temp_dir, "device", "autofuse_tiling_data_bak.h")
    # 备份原tilingdata文件
    shutil.copy(tiling_data, tiling_data_bak)
    with open(tiling_data, "w") as file:
        file.write(const_tiling_data)

    # 静态shape可以在编译期间获取kernel_type
    kernel_type = None
    if hasattr(lib, 'GetTilingKeyKernelTypeForStatic'):
        lib.GetTilingKeyKernelTypeForStatic.argtypes = []
        lib.GetTilingKeyKernelTypeForStatic.restype = ctypes.c_char_p
        kernel_type_result = lib.GetTilingKeyKernelTypeForStatic()
        kernel_type = kernel_type_result.decode('utf-8')

    # 修改kernel文件
    static_shape_kernel_proc(kernel_src, temp_dir, kernel_type)

def is_static_compile(params, tiling_func_srcs):
    import re
    force_unknown = params.get('force_unknown')
    if force_unknown:
        return False;

    static_shape = False;
    pattern = r'AutofuseIsStaticShape\s*\(\s*\)\s*\{[^{}]*return\s+(true|false)\s*;'
    for tiling_func_src in tiling_func_srcs.values():
        match = re.search(pattern, tiling_func_src, re.IGNORECASE)
        if match:
            return_value = match.group(1).lower()
            static_shape = (return_value == "true")
    return static_shape


def generate_device_and_host_code(graph_name, temp_dir, params, code_gen, pgo_dir):
    #生成device代码
    timestamp_set(True, graph_name, "GenDevice")
    schedule_results = params['schedule_results']
    vector_core_num = "0" if params['vector_core_num'] is None else str(params['vector_core_num'])
    tiling_data_src, op_kernel_src = code_gen.device_code_generator(schedule_results)
    kernel_build_dir = os.path.join(temp_dir, "device")
    generate_file(kernel_build_dir, "autofuse_tiling_data.h", tiling_data_src)
    generate_file(kernel_build_dir, graph_name + "_op_kernel.cpp", op_kernel_src)
    timestamp_set(False, graph_name, "GenDevice")

    #生成host代码
    if not check_keys_in_dict(params, ['output_symbol_shape']):
        CommonUtility.print_compile_log("", f"output_symbol_shape is not exist", AscendCLogLevel.LOG_ERROR)
        raise Exception("An error occurred autofuse compile for check extra_params")
    # output_symbol_shape stub
    CommonUtility.print_compile_log("", f"params output_symbol_shape : {params['output_symbol_shape']}",
                                    AscendCLogLevel.LOG_INFO)
    timestamp_set(True, graph_name, "GenHost")
    shape_info = params.get('symbol_source_info')
    output_symbol = json.loads(params.get('output_symbol_shape'))
    tiling_func_srcs, infershape_src =\
        code_gen.host_code_generator(
        schedule_results,
        shape_info,
        output_symbol,
        pgo_dir,
        vector_core_num)
    host_build_dir = os.path.join(temp_dir, "host")
    generate_file(host_build_dir, "autofuse_tiling_data.h", tiling_data_src)

    for key, value in tiling_func_srcs.items():
        if key == "TilingHead":
            generate_file(host_build_dir, "autofuse_tiling_func_common.h", value)
        elif "TilingData" not in key:
            generate_file(host_build_dir, graph_name + "_tiling_func_" + key + ".cpp", value)
    generate_file(host_build_dir, graph_name + "_infershape.cpp", infershape_src)
    timestamp_set(False, graph_name, "GenHost")

    return op_kernel_src, tiling_func_srcs


def pgo_kernel_compile(*args, temp_dir, params, op_kernel_src, code_gen):
    """PGO依赖的动态so编译"""
    schedule_results = params['schedule_results']
    kernel_name = args[-1]
    input_num = schedule_results.get_input_num()
    output_num = schedule_results.get_output_num()
    kernel_build_dir = os.path.join(temp_dir, "device")
    host_build_dir = os.path.join(temp_dir, "host")
    use_list_tensor_desc = op_kernel_src.find('kernel_operator_list_tensor_intf.h') > 0
    enable_parallel_compile = op_kernel_src.rfind('void fake_tiling_ids()') > 0
    pgo_dir = os.path.abspath(os.path.join(temp_dir, "..", "..", "pgo"))
    graph_name = gen_valid_name(schedule_results.get_name())
    graph_name = camel_to_snake(graph_name)

    kernel_file, json_file = ascbc_kernel_compile(args, graph_name=graph_name, kernel_name=kernel_name,
        input_num=input_num, output_num=output_num, temp_build_dir=kernel_build_dir,
        impl_mode=params.get('impl_mode'), use_list_tensor_desc=use_list_tensor_desc,
        enable_parallel_compile=enable_parallel_compile)
    get_kernel_src = code_gen.get_kernel_and_json_generator(kernel_file, json_file)
    generate_file(host_build_dir, graph_name + "_get_kernel.cpp", get_kernel_src)
    ascbc_host_compile(graph_name, kernel_name, host_build_dir, True, False)
    source_lib_path = os.path.join(host_build_dir, f"lib{kernel_name}.so")
    target_lib_path = os.path.join(pgo_dir, f"lib{graph_name}.so")
    shutil.copy(source_lib_path, target_lib_path)


def check_dir_permissions(path):
    import stat
    # 检查是否为软链接
    if path.is_symlink():
        raise Exception(f"Warning: {path} is a symbolic link")

    # 检查目录是否存在
    if not path.exists():
        raise FileNotFoundError(f"Directory {path} does not exist")

    # 检查读写执行权限
    mode = path.stat().st_mode
    if not (mode & stat.S_IRUSR):
        raise PermissionError(f"No read permission for {path}")
    if not (mode & stat.S_IWUSR):
        raise PermissionError(f"No write permission for {path}")
    if not (mode & stat.S_IXUSR):
        raise PermissionError(f"No execute permission for {path}")


def replace_kernel(kernel_build_dir, graph_name):
    import re
    from pathlib import Path
    autofuse_dfx_flags_env = os.getenv("AUTOFUSE_DFX_FLAGS", "")
    if not autofuse_dfx_flags_env or "replace_kernel=" not in autofuse_dfx_flags_env:
        return
    pattern = r'replace_kernel=([^";]+)'
    match = re.search(pattern, autofuse_dfx_flags_env)
    if not match:
        logger.info("match env replace_kernel failed. AUTOFUSE_DFX_FLAGS is %s: ", autofuse_dfx_flags_env)
        return
    replace_kernel_path = match.group(1)
    kernel_path = Path(replace_kernel_path).expanduser().absolute()
    check_dir_permissions(kernel_path)
    src_kernel_path = os.path.join(kernel_path, f"{graph_name}_op_kernel.cpp")
    if not os.path.exists(src_kernel_path):
        return
    dst_kernel_path = os.path.join(kernel_build_dir, f"{graph_name}_op_kernel.cpp")
    shutil.copy(src_kernel_path, dst_kernel_path)
    logger.info("replace kernel: %s to %s", src_kernel_path, dst_kernel_path)


def generate_pgo_code(params, code_gen, pgo_dir, host_build_dir):
    """生成PGO代码"""
    schedule_results = params['schedule_results']
    device_id = params['device_id']
    device_id = "0" if device_id is None or not device_id.isdigit() else device_id
    graph_name = gen_valid_name(schedule_results.get_name())
    graph_name = camel_to_snake(graph_name)
    soc_vector_core_cnt = int(get_soc_spec('vector_core_cnt'))
    param_vector_core_num = soc_vector_core_cnt if params['vector_core_num'] is None else int(params['vector_core_num'])
    vector_core_num = str(min(param_vector_core_num, soc_vector_core_cnt))
    msg = f"[PGO] device_id:{device_id}, graph_name:{graph_name}, vector_core_num:{vector_core_num}"
    CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_DEBUG)

    pgo_src = code_gen.pgo_code_generator(schedule_results, pgo_dir, vector_core_num, str(get_soc_spec('ub_size')), device_id)
    generate_file(host_build_dir, graph_name + "_pgo.cpp", pgo_src)


def build_pgo_compile_command(source_file, output_file):
    """构建PGO编译命令"""
    mspti_dir = f"{ASCEND_PATH}/tools/mspti"
    base_cmd = ["g++", "-std=c++17", "-O2", "-fPIC"]
    includes = [
        f"-I{ASCEND_PATH}/include/",
        f"-I{ASCEND_PATH}/pkg_inc/base",
        f"-I{ASCEND_PATH}/include/experiment/runtime",
        f"-I{ASCEND_PATH}/include/experiment/msprof",
        f"-I{mspti_dir}/include"
    ]
    libs = [
        f"-L{ASCEND_PATH}/lib64", "-lascendcl", "-lruntime",
        f"-L{mspti_dir}/lib64", "-lmspti",
        "-ldl"
    ]
    cmd = base_cmd + [source_file, "-o", output_file] + includes + libs
    return cmd


def pgo_compile(graph_name, host_build_dir):
    """编译PGO代码"""
    pgo_source_file = os.path.join(host_build_dir, f"{graph_name}_pgo.cpp")
    output_file = os.path.join(host_build_dir, "pgo")

    cmd = build_pgo_compile_command(pgo_source_file, output_file)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = "cmd = " + ' '.join(cmd) + "\n"
        msg += f"stdout: {result.stdout}\n"
        msg += f"stderr: {result.stderr}\n"
        CommonUtility.print_compile_log("", f"pgo compile fail: {msg}", AscendCLogLevel.LOG_ERROR)
        return None
    return output_file


def pgo_program_exec(temp_dir, exec_type=0):
    """执行PGO, 0表示首次动态调优，1表示二次调优"""
    host_build_dir = os.path.join(temp_dir, "host")
    pgo_file = os.path.join(host_build_dir, "pgo")
    mspti_dir = f"{ASCEND_PATH}/tools/mspti"
    try:
        logger.debug("start pgo exec")
        result = subprocess.run(
            [
                pgo_file,
                str(exec_type)
            ],
            text=True,
            stderr=subprocess.PIPE,
            env={
                "LD_PRELOAD": mspti_dir + "/lib64/libmspti.so",
                **dict(os.environ)
            },
            timeout=600 # 暂定，设置超时时间为600秒
        )
        # pgo目前仍有部分日志通过stderr输出，后续统一整改为slog
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.debug(line)
        logger.debug("end pgo exec")
        if result.returncode != 0:
            msg = f"pgo exec fail, return code: {result.returncode}\n"
            # 执行失败时，回退原始解，不打断模型正常执行流程
            CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_ERROR)
            return False
        return True
    except Exception as e:
        CommonUtility.print_compile_log("", f"pgo exec fail: {str(e)}", AscendCLogLevel.LOG_ERROR)
        return False


def pgo_second_optimization(*args, temp_dir, params, op_kernel_src, code_gen):
    """静态tiling的二次调优"""
    from autofuse.compile_adapter import pgo_get_top_result, pgo_write_config, pgo_generate_config

    kernel_name = args[-1]
    schedule_results = params['schedule_results']
    graph_name = gen_valid_name(schedule_results.get_name())
    graph_name = camel_to_snake(graph_name)
    input_num = schedule_results.get_input_num()
    output_num = schedule_results.get_output_num()
    use_list_tensor_desc = op_kernel_src.find('kernel_operator_list_tensor_intf.h') > 0
    enable_parallel_compile = op_kernel_src.rfind('void fake_tiling_ids()') > 0
    kernel_build_dir = os.path.join(temp_dir, "device")
    host_build_dir = os.path.join(temp_dir, "host")
    pgo_dir = os.path.abspath(os.path.join(temp_dir, "..", "..", "pgo"))
    config_path = os.path.join(pgo_dir, f"{graph_name}_config.txt")
    search_path = os.path.join(pgo_dir, f"{graph_name}_search.txt")
    top_n = get_pgo_topn()

    top_lines, origin_line, top_n = pgo_get_top_result(search_path, top_n)
    if top_lines is None or origin_line is None or top_n is None:
        return False

    def comile_and_exec(config_line):
        pgo_write_config(config_path, config_line)
        static_shape_compile(kernel_name=kernel_name, temp_dir=temp_dir, graph_name=graph_name)
        kernel_file, json_file = ascbc_kernel_compile(args, graph_name=graph_name, kernel_name=kernel_name,
            input_num=input_num, output_num=output_num, temp_build_dir=kernel_build_dir,
            impl_mode=params.get('impl_mode'), use_list_tensor_desc=use_list_tensor_desc,
            enable_parallel_compile=enable_parallel_compile)
        get_kernel_src = code_gen.get_kernel_and_json_generator(kernel_file, json_file)
        generate_file(host_build_dir, graph_name + "_get_kernel.cpp", get_kernel_src)
        ascbc_host_compile(graph_name, kernel_name, host_build_dir, True, True)

        source_lib_path = os.path.join(host_build_dir, f"lib{kernel_name}.so")
        target_lib_path = os.path.join(pgo_dir, f"lib{graph_name}.so")
        shutil.copy(source_lib_path, target_lib_path)

        result = pgo_program_exec(temp_dir=temp_dir, exec_type=1)
        return result

    for line in top_lines:
        result = comile_and_exec(line)
        if result is False:
            return False

    result = comile_and_exec(origin_line)
    if result is False:
        return False

    pgo_generate_config(search_path, config_path, top_n)


def asc_pgo_exec(*args, temp_dir, params, op_kernel_src, code_gen):
    """执行PGO"""
    schedule_results = params['schedule_results']
    graph_name = gen_valid_name(schedule_results.get_name())
    graph_name = camel_to_snake(graph_name)
    pgo_dir = os.path.abspath(os.path.join(temp_dir, "..", "..", "pgo"))
    host_build_dir = os.path.join(temp_dir, "host")
    os.makedirs(pgo_dir, exist_ok=True)

    config_path = os.path.join(pgo_dir, f"{graph_name}_config.txt")
    if os.path.exists(config_path):
        CommonUtility.print_compile_log("", f"{config_path} exist, skip pgo tuning", AscendCLogLevel.LOG_DEBUG)
        return

    timestamp_set(True, graph_name, "CompileKernelForPGO")
    pgo_kernel_compile(*args, temp_dir=temp_dir, params=params, op_kernel_src=op_kernel_src, code_gen=code_gen)
    timestamp_set(False, graph_name, "CompileKernelForPGO", True)

    timestamp_set(True, graph_name, "GenerateForPGO")
    generate_pgo_code(params=params, code_gen=code_gen, pgo_dir=pgo_dir, host_build_dir=host_build_dir)
    timestamp_set(False, graph_name, "GenerateForPGO", True)

    timestamp_set(True, graph_name, "CompilePGO")
    output = pgo_compile(graph_name=graph_name, host_build_dir=host_build_dir)
    timestamp_set(False, graph_name, "CompilePGO", True)

    if output is None:
        return
    timestamp_set(True, graph_name, "RunForPGO")
    result = pgo_program_exec(temp_dir=temp_dir)
    timestamp_set(False, graph_name, "RunForPGO", True)
    if result is False:
        if os.path.exists(config_path):
            os.remove(config_path)
        return

    timestamp_set(True, graph_name, "SecondOptimizeForPGO")
    result = pgo_second_optimization(*args, temp_dir=temp_dir, params=params, op_kernel_src=op_kernel_src, code_gen=code_gen)
    timestamp_set(False, graph_name, "SecondOptimizeForPGO", True)
    if result is False:
        if os.path.exists(config_path):
            os.remove(config_path)
        return

def _build_args(args_list, input_num, output_num):
    _inputs_ = []
    _outputs_ = []
    _origin_inputs_ = []
    _origin_outputs_ = []
    # 遍历args中的每个元素
    for i, arg in enumerate(args_list[:(input_num + output_num)]):
        if i < input_num:
            msg = "Processing input " + str(i) + ":" + str(arg)
            CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)
            _origin_inputs_.append(arg)
            if arg is not None:
                if isinstance(arg, (list, tuple)):
                    if len(arg) == 0:
                        continue
                    _inputs_.append(arg[0])
                else:
                    _inputs_.append(arg)
            else:
                _inputs_.append(arg)
            _inputs_[-1]["param_name"] = "input" + str(i)
            shape = _inputs_[-1]["shape"]
            _inputs_[-1]["shape"] = shape
            _inputs_[-1]["ori_shape"] = shape
        else:
            msg = "Processing output " + str(i - input_num) + ":" + str(arg)
            CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)
            _origin_outputs_.append(arg)
            if arg is not None:
                if isinstance(arg, (list, tuple)):
                    if len(arg) == 0:
                        continue
                    _outputs_.append(arg[0])
                else:
                    _outputs_.append(arg)
            else:
                _outputs_.append(arg)
            _outputs_[-1]["param_name"] = "output" + str(i - input_num)
            shape = _outputs_[-1]["shape"]
            _outputs_[-1]["shape"] = shape
            _outputs_[-1]["ori_shape"] = shape
    return _origin_inputs_, _origin_outputs_, _inputs_, _outputs_

def ascbc_cube_kernel_tiling_pro(
    *args,
    temp_dir,
    graph_name, 
    kernel_name, 
    input_num, 
    output_num, 
    use_list_tensor_desc,
    cube_attrs,
    tiling_key_list
):
    graph_name = camel_to_snake(graph_name)
    args_list = args[0]
    if use_list_tensor_desc:
        inputs = args_list[:input_num]
        outputs = args_list[input_num: input_num + output_num]
        args_list = [inputs, outputs]
        input_num = 1
        output_num = 1

    _origin_inputs_, _origin_outputs_, _inputs_, _outputs_ = \
        _build_args(args_list, input_num=input_num, output_num=output_num)

    if cube_attrs is None:
        logger.error("kernel_name=[%s] can't find cube attrs", kernel_name)
        return
    cube_attributes = cube_attrs.get("cube_attributes", {})
    if not cube_attributes:
        logger.error("kernel_name=[%s] can't find cube attributes", kernel_name)
        return None
    mm_attr1 = {"name" : "transpose_x1", "dtype" : "bool", "value" : cube_attributes.get("transpose_x1", False)}
    mm_attr2 = {"name" : "transpose_x2", "dtype" : "bool", "value" : cube_attributes.get("transpose_x2", False)}
    mm_attr3 = {"name" : "offset_x", "dtype" : "int", "value" : cube_attributes.get("offset_x", 0)}
    mm_attr4 = {"name" : "enable_hf32", "dtype" : "bool", "value" : cube_attributes.get("enable_hf32", False)}

    attrs = [mm_attr1, mm_attr2, mm_attr3, mm_attr4]
    tiling_info = TilingInfo()
    context = get_context()
    _change_param_name_to_name(_inputs_)
    _change_param_name_to_name(_origin_inputs_)
    compile_info = context.get_compile_info()
    tiling_config = {"name" : "ascendc_op_para_size", "dtype" : "int", "value" : 2 * 1024 * 1024}
    attrs.append(tiling_config)
    if cube_attributes.get("is_batch", False):
        run_info = do_op_tiling("BatchMatMulV3", compile_info, _origin_inputs_, _origin_outputs_, None, None, attrs)
        tiling_info.tiling_data = run_info["tiling_data"]
        tiling_info.tiling_key = run_info['tiling_key']
        tiling_key_list[0] = tiling_info.tiling_key
        tiling_info.file_content = gen_static_shape_v2(
            "BatchMatMulV3", "BatchMatMulV3TilingData", run_info["tiling_data"])
    else:
        run_info = do_op_tiling("MatMulV3", compile_info, _origin_inputs_, _origin_outputs_, None, None, attrs)
        tiling_info.tiling_data = run_info["tiling_data"]
        tiling_info.tiling_key = run_info['tiling_key']
        tiling_key_list[0] = tiling_info.tiling_key
        tiling_info.file_content = gen_static_shape_v2(
            "MatMulV3", "MatMulV3TilingData", run_info["tiling_data"])

    logger.info("kernel_name=[%s], cube tiling_key[%s], tiling_data=[%s], tiling_file_context=[%s]", kernel_name, str(
        tiling_info.tiling_key), str(tiling_info.tiling_data), str(tiling_info.file_content))
    
    tiling_data = os.path.join(temp_dir, "device", "autofuse_cube_tiling_data.h")
    with open(tiling_data, "w") as file:
        file.write(tiling_info.file_content)

def asc_graph_compile(*args, temp_dir, params):
    """入口为asc graph场景"""
    #获取ascgraph信息
    schedule_results = params['schedule_results']
    graph_name = gen_valid_name(schedule_results.get_name())
    graph_name = camel_to_snake(graph_name)
    input_num = schedule_results.get_input_num()
    output_num = schedule_results.get_output_num()
    is_cube = schedule_results.is_cube_type()
    tiling_key_list = [-1]
    cube_attrs = schedule_results.get_cube_attributes()
    kernel_name = args[-1]
    vector_core_num = params['vector_core_num']
    msg = f"graph_name:{graph_name}, input_num:{input_num}, output_num:{output_num}," \
          f"kernel_name:{kernel_name}, vector_core_num:{vector_core_num}, is_cube:{is_cube}"
    CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_DEBUG)
    graph_name = camel_to_snake(graph_name)
    code_gen = CodeGen()

    kernel_build_dir = os.path.join(temp_dir, "device")
    os.makedirs(kernel_build_dir, exist_ok=True)
    host_build_dir = os.path.join(temp_dir, "host")
    os.makedirs(host_build_dir, exist_ok=True)
    pgo_dir = os.path.abspath(os.path.join(temp_dir, "..", "..", "pgo"))
    os.makedirs(pgo_dir, exist_ok=True)

    #生成device和host代码
    op_kernel_src, tiling_func_srcs = generate_device_and_host_code(graph_name=graph_name, temp_dir=temp_dir,
        params=params, code_gen=code_gen, pgo_dir=pgo_dir)
    static_compile_flag = is_static_compile(params, tiling_func_srcs)

    use_list_tensor_desc = op_kernel_src.find('kernel_operator_list_tensor_intf.h') > 0
    enable_parallel_compile = op_kernel_src.rfind('void fake_tiling_ids()') > 0

    if (is_cube and static_compile_flag):
        ascbc_cube_kernel_tiling_pro(args, temp_dir=temp_dir, graph_name=graph_name, kernel_name=kernel_name,
            input_num=input_num, output_num=output_num,
            use_list_tensor_desc=use_list_tensor_desc, cube_attrs=cube_attrs, tiling_key_list=tiling_key_list)
    else:
        # 静态shape场景编译host so，并修改kernel代码和tiling_data代码
        if static_compile_flag:
            #PGO需要使用非const tiling编译的kernel进行调优
            pgo_env = get_pgo_env_flag()
            if pgo_env:
                asc_pgo_exec(*args, temp_dir=temp_dir, params=params, op_kernel_src=op_kernel_src, code_gen=code_gen)
            timestamp_set(True, graph_name, "CompileHost")
            static_shape_compile(kernel_name=kernel_name, temp_dir=temp_dir, graph_name=graph_name)
            timestamp_set(False, graph_name, "CompileHost")
    replace_kernel(kernel_build_dir=kernel_build_dir, graph_name=graph_name)

    # 编译device代码
    timestamp_set(True, graph_name, "CompileDevice")
    kernel_file, json_file = ascbc_kernel_compile(args, graph_name=graph_name, kernel_name=kernel_name,
                                                  input_num=input_num, output_num=output_num,
                                                  temp_build_dir=kernel_build_dir,
                                                  impl_mode=params.get('impl_mode'),
                                                  use_list_tensor_desc=use_list_tensor_desc,
                                                  enable_parallel_compile=enable_parallel_compile, tiling_key=tiling_key_list[0])
    timestamp_set(False, graph_name, "CompileDevice")

    # 生成get_kernel.cpp代码
    timestamp_set(True, graph_name, "GenGetKernel")
    get_kernel_src = code_gen.get_kernel_and_json_generator(kernel_file, json_file)
    generate_file(host_build_dir, graph_name + "_get_kernel.cpp", get_kernel_src)
    timestamp_set(False, graph_name, "GenGetKernel")

    #重新编译host so
    timestamp_set(True, graph_name, "CompileSecondHost")
    ascbc_host_compile(graph_name, kernel_name, host_build_dir, True, static_compile_flag)
    timestamp_set(False, graph_name, "CompileSecondHost", True)

    #拷贝so和json
    host_so = os.path.join(host_build_dir, f"lib{kernel_name}.so")
    kernel_meta_dir = get_current_build_config("kernel_meta_parent_dir")
    shutil.copy(host_so, os.path.join(kernel_meta_dir, "kernel_meta"))
    modify_json_file(json_file, host_so)
    if os.path.exists(kernel_file):
        os.remove(kernel_file)

def compute_graph_compile(*args, temp_dir, params, vector_core_num, device_id):
    """入口为compute graph场景"""
    timestamp_set(True, "start", "Schedule")
    compute_graph = ascir.utils.deserialize("compute_graph", params['compute_graph'])
    logger.info("[AUTOFUSE_BACKEND]compile start graph[%s], build dir[%s], kernel_name[%s], "
                "symbol_source_info[%s], output_symbol_shape[%s]",
                compute_graph.get_name(), temp_dir, args[-1], params.get('symbol_source_info', "None"),
                params.get('output_symbol_shape', "[]"))
    value = params.get('symbol_source_info')
    CommonUtility.print_compile_log("", f"symbol_source_info is {type(value)}, {value}", AscendCLogLevel.LOG_DEBUG)
    if value is None or value == 'null':
        CommonUtility.print_compile_log("", "maybe a static shape", AscendCLogLevel.LOG_WARNING)
        symbol_source_info = None
    else:
        symbol_source_info = ascir.utils.deserialize("symbol_source_info", value)
    #Do schedule
    sched = Schedule()
    schedule_results = sched.scheduleV2(compute_graph)
    timestamp_set(False, gen_valid_name(compute_graph.get_name()), "Schedule")
    #构造ascgraph参数
    asc_param = {}
    asc_param['schedule_results'] = schedule_results
    asc_param['symbol_source_info'] = symbol_source_info
    asc_param['output_symbol_shape'] = params.get('output_symbol_shape', "[]")
    asc_param['vector_core_num'] = vector_core_num
    asc_param['device_id'] = device_id
    asc_graph_compile(*args, temp_dir=temp_dir, params=asc_param)


def asc_codegen_compile(*args, **kwargs):
    """自动融合算子注册到tefusion的统一编译入口，编译完成后，将编译后的`so`和`json`写入`kernel meta`目录
    **args**
    * inputs：包含输入shape、dtype
    * outputs：包含输出shape、dtype
    * attrs：对于自动融合节点，不存在原型属性，因此该字段不存在
    * kernel_name: kernel name用于生成so名称 确保在最后一个参数中

    **op_info.extra_params**
    * fused_graph: compute graph序列化后的字符串 <string>
    * symbol_source_info: shape的符号化信息，存储了如`s0`是第0个输入的第2个维度信息 <dict>
    * output_symbol_shape: infer shape所需符号化表达 <list<str>>
    """

    def asc_codegen_compile_with_tmpdir(*args, temp_dir, **kwargs):
        import tbe.common.context.op_context as op_context
        context = op_context.get_context()
        op_info = context.get_op_info()
        extra_params = op_info[0].extra_params
        vector_core_num = op_context.get_context().get_addition("_op_vectorcore_num")
        device_id = op_context.get_context().get_addition("device_id")
        CommonUtility.print_compile_log("", f"params type: {type(extra_params)}, params: {extra_params}",
                                            AscendCLogLevel.LOG_DEBUG)
        #反序列化graph和symbol_source_info
        if not check_keys_in_dict(extra_params, ['compute_graph', 'symbol_source_info']):
            CommonUtility.print_compile_log("", f"compute_graph and symbol_source_info do not exist",
                                            AscendCLogLevel.LOG_ERROR)
            raise Exception("An error occurred autofuse compile for check extra_params")
        compute_graph_compile(*args, temp_dir=temp_dir, params=extra_params, vector_core_num=vector_core_num, device_id=device_id)


    kernel_meta_dir = get_current_build_config("kernel_meta_parent_dir")
    if not os.path.isabs(kernel_meta_dir):
        kernel_meta_dir = os.path.abspath(kernel_meta_dir)

    if not is_valid_path(kernel_meta_dir):
        CommonUtility.print_compile_log("", f"invalid kernel meta path : `{kernel_meta_dir}' ",
                                        AscendCLogLevel.LOG_ERROR)
        raise Exception("An error occurred autofuse compile for check kernel_meta path")

    if not os.path.exists(kernel_meta_dir):
        CommonUtility.print_compile_log("", f"kernel meta parent dir is not exist : `{kernel_meta_dir}' ",
                                        AscendCLogLevel.LOG_ERROR)
        raise Exception("An error occurred autofuse compile for check kernel_meta path")

    kernel_name = args[-1]
    from autofuse.compile_adapter import get_debug_flag
    compile_debug = get_debug_flag()
    if not compile_debug:
        with tempfile.TemporaryDirectory(prefix=kernel_name, dir=kernel_meta_dir) as temp_dir:
            msg = f"kernel name:{kernel_name}, kernel_meta_dir:{kernel_meta_dir}, tmp:{temp_dir}"
            CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)
            asc_codegen_compile_with_tmpdir(*args, temp_dir=temp_dir, **kwargs)
    else:
        temp_dir = tempfile.mkdtemp(prefix=kernel_name, dir=kernel_meta_dir)
        msg = f"kernel name:{kernel_name}, kernel_meta_dir:{kernel_meta_dir}, tmp:{temp_dir}"
        CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)
        asc_codegen_compile_with_tmpdir(*args, temp_dir=temp_dir, **kwargs)

    CommonUtility.print_compile_log("", "asc_codegen_compile finish", AscendCLogLevel.LOG_INFO)
