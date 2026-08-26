# DUMP\_GRAPH\_LEVEL

## 功能描述

把整个图编译流程中各个阶段的图描述信息打印到文件中。

此环境变量支持如下两种配置方式，两种方式均是控制图落盘的个数，用户可以按需使用，注意两种配置方式不支持混合使用：

- 配置数值，取值如下：
  - 1：dump所有阶段的图。
  - 2：dump白名单阶段的图。具体白名单图请参见[表1](#table1)中的“是否白名单”列。
  - 3：dump最后的生成图，即经过GE（Graph Engine，图引擎）优化、编译后的图。
  - 4：dump最早的生成图，即经过GE解析映射算子后，给到软件栈的编译入口图，此时图结构尚未经过GE的编译优化。

- 配置按照“|”分隔的字符串，配置如下：

    例如配置为"aa|bb"，则表示dump出名称包含aa和bb的图，aa和bb需要指定为图编译流程中的合法字符串，合法字符串的获取可以从全量的dump图得到。

DUMP\_GRAPH\_LEVEL环境变量只有在[DUMP\_GE\_GRAPH](DUMP_GE_GRAPH.md)开启时才生效，默认值为2。

## 配置示例

- 配置为数值：

    ```bash
    export DUMP_GRAPH_LEVEL=1
    ```

- 配置为按照|分隔的字符串：

    ```bash
    export DUMP_GRAPH_LEVEL="PreRunBegin|AfterInfershape"
    ```

## 使用约束

- 如果此环境变量设置了其他非法值，可能会导致未定义的行为发生。
- 如果开启了采集算子dump数据功能，可以参考[ge.exec.enableDump](../../api/graph_engine_api/cpp/ge/options_params/precision_comparison.md#geexecenabledump)参数，即使不配置DUMP\_GRAPH\_LEVEL环境变量，或者配置export DUMP\_GRAPH\_LEVEL="PreRunBegin|AfterInfershape"但不包括“Build”字符串，最终都会dump子图ge\_proto\_xxxx\_Build.txt。
- 此环境变量需要配合[DUMP\_GE\_GRAPH](DUMP_GE_GRAPH.md)使用，即开启[DUMP\_GE\_GRAPH](DUMP_GE_GRAPH.md)的场景下，可通过DUMP\_GRAPH\_LEVEL控制生成的dump图信息。详情请参见[dump图详细信息](../atc_tools/references/dump_graph_details.md)。

## 产品支持情况

全量芯片支持
