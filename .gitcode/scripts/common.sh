#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Color
Red='\e[0;31m'          # Red
Green='\e[0;32m'        # Green
BRed='\e[1;31m'         # Red
BGreen='\e[1;32m'       # Green
BCyan='\e[1;36m'        # Cyan
Purple='\e[0;35m'       # Purple
BPurple='\e[1;35m'      # Bold Purple
Color_Off='\e[0m'       # Text Reset
Now=`date +"%Y-%m-%d %H:%M:%S"`

function LOG_DO() {
    local date_time
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BPurple}[Command]${Color_Off} ${date_time} ${Purple}$*${Color_Off}"
    "$@"
}

function LOG_DO_TEE() {
    local log_file="$1"
    shift
    local date_time
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BPurple}[Command]${Color_Off} ${date_time} ${Purple}$*${Color_Off}"
    local old_pipefail=$(set -o | grep pipefail | awk '{print $2}')
    set -o pipefail
    "$@" 2>&1 | tee "$log_file"
    local ret=$?
    [ "$old_pipefail" = "off" ] && set +o pipefail
    return $ret
}

function check_slow_tests() {
    local log_file="$1"
    local threshold_ms="${2:-1000}"
    local repo_dir="${3:-${WORKSPACE}}"

    if [ ! -f "$log_file" ]; then
        echo "[check_slow_tests] Log file not found: $log_file, skipping"
        return 0
    fi

    cd "$repo_dir" || return 0

    local target_ref="origin/${GIT_TARGET_BRANCH}"
    git fetch origin "refs/heads/${GIT_TARGET_BRANCH}:refs/remotes/origin/${GIT_TARGET_BRANCH}" --unshallow 2>/dev/null \
        || git fetch origin "refs/heads/${GIT_TARGET_BRANCH}:refs/remotes/origin/${GIT_TARGET_BRANCH}" 2>/dev/null || true

    local merge_base
    merge_base=$(git merge-base "$target_ref" HEAD 2>/dev/null) || merge_base="$target_ref"

    local diff_output
    diff_output=$(git diff "$merge_base" HEAD --unified=0 -- '*.cc' '*.cpp' 2>/dev/null) || true

    if [ -z "$diff_output" ]; then
        echo "[check_slow_tests] No code changes found"
        return 0
    fi

    declare -A file_ranges
    local current_file=""

    while IFS= read -r line; do
        if [[ "$line" =~ ^\+\+\+\ b/(.*) ]]; then
            current_file="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^@@.*\+([0-9]+)(,([0-9]+))?\ @@ ]] && [ -n "$current_file" ] && [ -f "$current_file" ]; then
            local start="${BASH_REMATCH[1]}"
            local count="${BASH_REMATCH[3]:-1}"
            local end=$((start + count - 1))
            if [ -n "${file_ranges[$current_file]}" ]; then
                file_ranges["$current_file"]="${file_ranges[$current_file]} ${start}-${end}"
            else
                file_ranges["$current_file"]="${start}-${end}"
            fi
        fi
    done <<< "$diff_output"

    if [ ${#file_ranges[@]} -eq 0 ]; then
        echo "[check_slow_tests] No code changes found"
        return 0
    fi

    declare -A modified_tests
    for file in "${!file_ranges[@]}"; do
        local ranges="${file_ranges[$file]}"
        local test_names
        test_names=$(awk -v ranges="$ranges" '
            BEGIN {
                n = split(ranges, range_arr, " ")
                for (i = 1; i <= n; i++) {
                    split(range_arr[i], bounds, "-")
                    range_start[i] = bounds[1]
                    range_end[i] = bounds[2]
                }
                num_ranges = n
            }
            /^TEST_F\(/ {
                match($0, /TEST_F\([[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*,[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)/, arr)
                if (arr[1] != "" && arr[2] != "") {
                    test_suite = arr[1]
                    test_case = arr[2]
                    test_start = NR
                    brace_count = 0
                    in_test = 1
                }
            }
            in_test {
                for (i = 1; i <= length($0); i++) {
                    c = substr($0, i, 1)
                    if (c == "{") brace_count++
                    else if (c == "}") {
                        brace_count--
                        if (brace_count == 0) {
                            test_end = NR
                            for (r = 1; r <= num_ranges; r++) {
                                if (range_start[r] <= test_end && range_end[r] >= test_start) {
                                    print test_suite "." test_case
                                    break
                                }
                            }
                            in_test = 0
                            break
                        }
                    }
                }
            }
        ' "$file" 2>/dev/null) || true

        if [ -n "$test_names" ]; then
            while IFS= read -r name; do
                [ -n "$name" ] && modified_tests["$name"]=1
            done <<< "$test_names"
        fi
    done

    if [ ${#modified_tests[@]} -eq 0 ]; then
        echo "[check_slow_tests] No modified TEST_F found in diff"
        return 0
    fi

    echo "[check_slow_tests] Found ${#modified_tests[@]} modified TEST_F cases"

    declare -A whitelist_limits
    local whitelist_file="${repo_dir}/.gitcode/scripts/slow_test_whitelist.txt"
    if [ -f "$whitelist_file" ]; then
        while IFS= read -r line; do
            line=$(echo "$line" | sed 's/#.*//' | xargs)
            [ -z "$line" ] && continue
            local wl_name wl_limit
            wl_name=$(echo "$line" | awk '{print $1}')
            wl_limit=$(echo "$line" | awk '{print $2}')
            if [ -n "$wl_name" ] && [ -n "$wl_limit" ]; then
                whitelist_limits["$wl_name"]="$wl_limit"
            fi
        done < "$whitelist_file"
        echo "[check_slow_tests] Loaded whitelist: ${#whitelist_limits[@]} entries"
    fi

    declare -A test_times
    while IFS= read -r line; do
        if [[ "$line" =~ \[.*OK.*\].*\(.*ms\) ]]; then
            local test_name time_ms
            test_name=$(echo "$line" | sed 's/.*\[       OK \] //' | sed 's/ (.*//')
            time_ms=$(echo "$line" | grep -oE '\([0-9]+ ms\)' | grep -oE '[0-9]+')
            if [ -n "$test_name" ] && [ -n "$time_ms" ]; then
                test_times["$test_name"]="$time_ms"
            fi
        fi
    done < <(grep '\[       OK \]' "$log_file" 2>/dev/null)

    echo ""
    echo "=========================================="
    echo "Test Case Performance Summary"
    echo "=========================================="
    printf "%-60s %10s %12s\n" "Test Case" "Time (ms)" "Status"
    echo "------------------------------------------------------------------------------------"

    local found_slow=0
    local sorted_tests
    sorted_tests=$(printf "%s\n" "${!modified_tests[@]}" | sort)

    while IFS= read -r test_name; do
        [ -z "$test_name" ] && continue
        local time_ms="${test_times[$test_name]}"

        if [ -n "$time_ms" ]; then
            local effective_limit="$threshold_ms"
            local in_whitelist=0
            if [ -n "${whitelist_limits[$test_name]}" ]; then
                effective_limit="${whitelist_limits[$test_name]}"
                in_whitelist=1
            fi

            if [ "$time_ms" -gt "$effective_limit" ]; then
                printf "%-60s %10s %12s\n" "$test_name" "$time_ms" "FAIL"
                found_slow=1
            else
                if [ "$in_whitelist" -eq 1 ]; then
                    printf "%-60s %10s %12s\n" "$test_name" "$time_ms" "PASS(WL)"
                else
                    printf "%-60s %10s %12s\n" "$test_name" "$time_ms" "PASS"
                fi
            fi
        else
            printf "%-60s %10s %12s\n" "$test_name" "--" "SKIP"
        fi
    done <<< "$sorted_tests"

    echo "=========================================="

    if [ "$found_slow" -eq 1 ]; then
        echo -e "${BRed}[check_slow_tests] Some modified test cases exceed threshold. Please optimize.${Color_Off}"
        echo -e "${BRed}error 1${Color_Off}"
        return 1
    fi

    echo "[check_slow_tests] All modified test cases within threshold"
    return 0
}

# Log error
function LOG_ERROR() {
    local date_time
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BRed}[ERROR] ${date_time} ${1}${Color_Off}"
}

# Log info
function LOG_INFO() {
    local date_time
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BGreen}[INFO] ${date_time} ${1}${Color_Off}"
}

function DP_ASSERT_EQUAL() {
    local actual_value=${1}
    local expect_value=${2}
    local assert_msg=${3}
    local log_flag=${4:-"true"}
    local log_path=${5}
    if [ "${actual_value}" != "${expect_value}" ]; then
        if [ -n "${log_path}" ] && [ -f "${log_path}" ]; then
            cat "${log_path}"
        fi
        LOG_ERROR "${assert_msg} is failed."
        exit 1
    else
        if [ "${log_flag}" = "true" ]; then
            echo "${assert_msg} is success."
        fi
    fi
}

function DP_ASSERT_NOT_EQUAL() {
    local actual_value=${1}
    local expect_value=${2}
    local assert_msg=${3}
    local log_flag=${4:-"true"}
    local log_path=${5}
    if [ "${actual_value}" = "${expect_value}" ]; then
        if [ -n "${log_path}" ] && [ -f "${log_path}" ]; then
            cat ${log_path}
        fi
        LOG_ERROR "${assert_msg} is failed."
        exit 1
    else
        if [ "${log_flag}" = "true" ]; then
            LOG_INFO "${assert_msg} is success."
        fi
    fi
}

function GEN_PYTHON_COVERAGE() {
    LOG_DO git fetch origin "refs/heads/${GIT_TARGET_BRANCH}:refs/remotes/origin/${GIT_TARGET_BRANCH}" --unshallow 2>/dev/null || git fetch origin "refs/heads/${GIT_TARGET_BRANCH}:refs/remotes/origin/${GIT_TARGET_BRANCH}"
    echo "=== Remote branches ===" && git branch -r && echo "========================"
    local coveragePath
    coveragePath=$(find "${WORKSPACE}" -name "*.coverage" | head -n1)
    if [ "${coveragePath}x" == "x" ]; then
        echo "No coverage file found"
        exit 0
    fi
    local coverageDir
    coverageDir=$(dirname ${coveragePath})
    cd ${coverageDir} || exit
    echo "coverage is exist"
    coverage html -i -d cov_report
    if [ -d cov_report ]; then
        if [ "${COV_PREFIX:-st}" != "st" ]; then
            tar -zcf ut_cov_python.tar.gz cov_report
        else
            tar -zcf st_cov_python.tar.gz cov_report
        fi
    fi
    cd ${coverageDir} || exit
    coverage xml -i
    local coverage_file="${coverageDir}/coverage.xml"
    /opt/buildtools/python-3.10.2/bin/diff-cover --compare-branch=origin/${GIT_TARGET_BRANCH} "${coverage_file}" --fail-under=80
    if [ $? -ne 0 ]; then
        echo "Coverage less than 80%, please check"
        exit 1
    fi
}
