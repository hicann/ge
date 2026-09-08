---
name: cann-pr-log-check
description: |
  对 PR 或工作区改动的文件进行日志规范全量检查。自动获取变更文件列表，
  调用 cann-log-evaluation 对变更文件中的全部日志语句进行检查，
  输出带严重级、代码位置和修复建议的表格报告（Markdown + Excel）。
  规则每次运行与 cann-log-evaluation 远程条件拉取同步：远端未变更时不下载（304），变更时自动更新本地。
  **必须触发此 skill 的场景**（用户提到以下任何内容时使用）：
  - 检查PR日志、PR日志检查、日志规范检查、log check、检查日志
  - 检查改动日志、检查修改的日志、改了日志检查一下
  - 一键日志检查、日志检查、log evaluation
  - 检查这个PR的日志、看看日志有没有问题
  - cann-pr-log-check
---

# PR 日志规范检查

对 PR 或工作区改动的文件，调用 cann-log-evaluation 进行日志规范检查，
生成带严重级、代码位置和修复建议的表格报告。

**本 skill 不复制规则，始终使用 cann-log-evaluation 的最新规则和评测协议。**

## 依赖：cann-log-evaluation

规则来源和评测协议的唯一权威是 cann-log-evaluation 仓库：

```
https://gitcode.com/cann-tools/skills/tree/main/cann-log-evaluation
```

关键文件（运行时拉取）：

| 文件 | 远程 raw 地址 | 用途 |
|------|--------------|------|
| `references/log-rules.md` | `https://raw.gitcode.com/cann-tools/skills/raw/main/cann-log-evaluation/references/log-rules.md` | 12 条规则、严重级、判定尺度 |
| `SKILL.md` | `https://raw.gitcode.com/cann-tools/skills/raw/main/cann-log-evaluation/SKILL.md` | 评测协议、日志识别方法 |

### 步骤 0: 确保已安装 cann-log-evaluation 并加载最新规则

#### 0.1 查找已安装的 cann-log-evaluation

先检查本地是否已安装，路径逻辑与 `install_skill.py` 的 `target_dir()` 一致：

```bash
CANN_LOG_EVAL_DIR=""
for d in \
  "${OPENCODE_CONFIG_DIR:-$HOME/.config/opencode}/skill/cann-log-evaluation" \
  "${CLAUDE_HOME:-$HOME/.claude}/skills/cann-log-evaluation" \
  "${CODEX_HOME:-$HOME/.codex}/skills/cann-log-evaluation"; do
  if [ -f "$d/references/log-rules.md" ]; then
    CANN_LOG_EVAL_DIR="$d"
    break
  fi
done
```

#### 0.2 未安装则安装

```bash
if [ -z "$CANN_LOG_EVAL_DIR" ]; then
  curl -fsSL https://raw.gitcode.com/cann-tools/skills/raw/main/cann-log-evaluation/scripts/bootstrap.sh | bash -s -- --target all
  # 安装后重新查找
  for d in \
    "${OPENCODE_CONFIG_DIR:-$HOME/.config/opencode}/skill/cann-log-evaluation" \
    "${CLAUDE_HOME:-$HOME/.claude}/skills/cann-log-evaluation" \
    "${CODEX_HOME:-$HOME/.codex}/skills/cann-log-evaluation"; do
    if [ -f "$d/references/log-rules.md" ]; then
      CANN_LOG_EVAL_DIR="$d"
      break
    fi
  done
fi
```

#### 0.3 条件拉取同步规则（每次运行都执行）

无论本地是否已安装，每次运行都执行条件拉取：`curl -z` 会用本地文件 mtime
发 `If-Modified-Since` 请求，远端未变更时返回 304、不下载任何内容（零开销），
变更时才更新本地副本。网络失败也降级使用本地副本：

```bash
RULES_URL="https://raw.gitcode.com/cann-tools/skills/raw/main/cann-log-evaluation/references/log-rules.md"
SKILL_URL="https://raw.gitcode.com/cann-tools/skills/raw/main/cann-log-evaluation/SKILL.md"

curl -fsSL -z "$CANN_LOG_EVAL_DIR/references/log-rules.md" "$RULES_URL" -o "$CANN_LOG_EVAL_DIR/references/log-rules.md" 2>/dev/null || true
curl -fsSL -z "$CANN_LOG_EVAL_DIR/SKILL.md" "$SKILL_URL" -o "$CANN_LOG_EVAL_DIR/SKILL.md" 2>/dev/null || true
```

#### 0.4 读取规则和评测协议

读取以下两个文件，掌握规则和评测协议：

| 文件 | 路径 | 用途 |
|------|------|------|
| `references/log-rules.md` | `$CANN_LOG_EVAL_DIR/references/log-rules.md` | 12 条规则、严重级、判定尺度、黑名单 |
| `SKILL.md` | `$CANN_LOG_EVAL_DIR/SKILL.md` | 评测协议、日志识别方法 |

完整读取这两个文件，掌握：
- 12 条规则及其判定尺度
- 4 级严重级定义（致命/严重/中等/提示）
- 全局判定原则（宁缺毋滥、只报影响阅读/排障/安全的问题）
- 日志识别方法（不使用接口白名单）
- 内部敏感词黑名单

## 工作流程

### 步骤 1: 获取变更文件列表

按以下优先级自动判断：

1. **用户消息中包含 PR 编号或 GitCode PR 链接** → 用 GitCode API 获取
2. **用户消息中包含文件路径** → 直接使用该文件列表
3. **以上都不满足**（用户只说了"日志检查"等） → 从 git diff 自动获取

#### GitCode API（用户提供了 PR 编号或链接）

```bash
# 从 git remote 获取 owner/repo
repo_url=$(git remote get-url origin)
# 解析 owner 和 repo（参考 gitcode-pr skill 的方法）

# 获取 PR 变更文件列表
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files.json?access_token=$GITCODE_API_TOKEN"
```

从返回的 `diffs[].statistic.new_path` 提取文件路径列表。

#### git diff（默认）

用户没有提供 PR 编号或文件路径时，自动从 git diff 获取当前分支的变更：

```bash
# 优先：获取当前分支相对于目标分支的改动
git diff --name-only origin/HEAD...HEAD

# 如果上面为空，可能已经 push，尝试获取最近一次提交的改动
git diff --name-only HEAD~1...HEAD
# 此时告知用户：未检测到未推送的改动，已改为检查最近一次提交的改动

# 如果仍为空，提示用户
echo "未检测到变更，请提供 PR 编号或文件路径"
```

### 步骤 2: 过滤源码文件

根据步骤 0 加载的 `log-rules.md` 中的扫描范围，只保留扩展名为
`.cpp .cc .cxx .c .h .hpp .py .java .js .ts .go .rs` 的文件，
排除 `build/ output/ third_party/` 等目录。

```bash
git diff --name-only origin/HEAD...HEAD | \
  grep -E '\.(cpp|cc|cxx|c|h|hpp|py|java|js|ts|go|rs)$' | \
  grep -vE '(build/|output/|third_party/|3rdparty/|vendor/|__pycache__/|node_modules/)'
```

**如果过滤后没有文件，直接输出"本次改动不涉及源码文件，无需日志检查"并停止。**

### 步骤 3: 执行日志规范检查

#### 路径 A: Agent 自行评测（默认路径）

本 skill 只安装 cann-log-evaluation 的规则和评测协议，不配置模型 API，
因此默认由 Agent 自行完成评测。按步骤 0 读取的 `SKILL.md` 中的评测协议和
`log-rules.md` 中的规则执行：

1. **逐文件识别日志语句** — 对每个变更文件，读取完整文件内容，识别所有运行时日志/打印语句。识别范围和方法严格遵循 cann-log-evaluation 的 `SKILL.md` 中"日志识别"章节，不使用接口白名单。

2. **逐条日志按 12 条规则检查** — 对识别到的每条日志语句，按 `log-rules.md` 中的 12 条规则逐条检查，判定尺度严格遵循"规则补充尺度"表。

3. **判定原则**（来自 `log-rules.md` 的全局判定尺度）：
   - 只报会影响他人阅读、搜索、理解，或误导排障，或造成安全风险的问题
   - 宁缺毋滥：宁可少报几条准确问题，也不要增加误报
   - 同一日志优先报更具体的问题，不叠加多条规则
   - 每条问题必须能定位到真实源码行号

### 步骤 4: 生成报告

#### 4.1 输出 Markdown 表格

在终端输出以下内容：

**检查概览**：

| 项目 | 值 |
|------|-----|
| 检查范围 | PR #xxx 或 git diff |
| 规则版本 | cann-log-evaluation@<commit或latest> |
| 变更文件数 | N |
| 涉及日志语句数 | M |
| 发现问题数 | K |
| 致命 | x |
| 严重 | x |
| 中等 | x |
| 提示 | x |

**问题明细表格**（按严重级排序：致命 > 严重 > 中等 > 提示）：

| 序号 | 严重级 | 文件 | 行号 | 规则 | 问题描述 | 修复建议 |
|------|--------|------|------|------|----------|----------|
| 1 | 严重 | runtime/v1/executor.cc | 125 | 拼写错误 | "bufer" 应为 "buffer" | 改为 buffer |
| 2 | 中等 | compiler/graph/pass.cc | 88 | 禁止中文 | 日志含"失败" | 改为 failed |

**如果没有发现问题**，输出：
> 本次改动未发现日志规范问题。共检查 N 个文件、M 条日志语句。

#### 4.2 生成 Excel 报告

将检查结果整理为 JSON 数组，调用本 skill 自带的报告生成脚本：

```bash
python3 .claude/skills/cann-pr-log-check/scripts/gen_report.py \
  --findings <findings_json> \
  --output pr-log-check-report.xlsx
```

`findings_json` 格式为 JSON 数组，每个元素：
```json
{
  "severity": "严重",
  "file": "runtime/v1/executor.cc",
  "line": 125,
  "rule": "拼写错误",
  "description": "\"bufer\" 应为 \"buffer\"",
  "suggestion": "改为 buffer",
  "code_snippet": "GELOGI(\"bufer data from device\");"
}
```

如果 `openpyxl` 未安装，先 `pip install openpyxl`。

Excel 包含两张表：
- **概览**：统计信息（文件数、日志数、各级别问题数）
- **问题明细**：按严重级配色，带筛选

## 重要约束

1. **规则始终与远程同步** — 每次运行条件拉取（步骤 0.3），远端未变更时不下载，变更时自动更新本地
2. **只检查变更文件**，不做全仓扫描
3. **只发现问题并给建议，不修改被检代码**
4. **每条问题必须能定位到真实源码行号**
5. **宁缺毋滥，不增加误报**
6. **按规则表判定，不凭记忆** — 先完成步骤 0 再开始检查
