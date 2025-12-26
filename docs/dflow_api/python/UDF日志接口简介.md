# UDF日志接口简介<a name="ZH-CN_TOPIC_0000001976840886"></a>

UDF Python开放了日志记录接口，使用时导入flow\_func模块。使用其中定义的logger对象，调用logger对象封装的不同级别的日志接口。

**表 1**  日志分类

<a name="zh-cn_topic_0000001568715673_table03481516152310"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001568715673_row183481516172313"><th class="cellrowborder" valign="top" width="11.450000000000001%" id="mcps1.2.5.1.1"><p id="zh-cn_topic_0000001568715673_p1434814165237"><a name="zh-cn_topic_0000001568715673_p1434814165237"></a><a name="zh-cn_topic_0000001568715673_p1434814165237"></a>日志类型</p>
</th>
<th class="cellrowborder" valign="top" width="40.89%" id="mcps1.2.5.1.2"><p id="zh-cn_topic_0000001568715673_p12348171612319"><a name="zh-cn_topic_0000001568715673_p12348171612319"></a><a name="zh-cn_topic_0000001568715673_p12348171612319"></a>使用场景</p>
</th>
<th class="cellrowborder" valign="top" width="14.399999999999999%" id="mcps1.2.5.1.3"><p id="zh-cn_topic_0000001568715673_p23485169237"><a name="zh-cn_topic_0000001568715673_p23485169237"></a><a name="zh-cn_topic_0000001568715673_p23485169237"></a>日志级别</p>
</th>
<th class="cellrowborder" valign="top" width="33.26%" id="mcps1.2.5.1.4"><p id="zh-cn_topic_0000001568715673_p1234813165234"><a name="zh-cn_topic_0000001568715673_p1234813165234"></a><a name="zh-cn_topic_0000001568715673_p1234813165234"></a>对应的日志宏</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001568715673_row6348616202319"><td class="cellrowborder" rowspan="2" valign="top" width="11.450000000000001%" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000001568715673_p1434811632315"><a name="zh-cn_topic_0000001568715673_p1434811632315"></a><a name="zh-cn_topic_0000001568715673_p1434811632315"></a>运行日志</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" width="40.89%" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0000001568715673_p873284102513"><a name="zh-cn_topic_0000001568715673_p873284102513"></a><a name="zh-cn_topic_0000001568715673_p873284102513"></a>系统运行过程中的异常状态、异常动作、系统进程运行过程中的关键事件和系统资源占用的相关信息等需要记录运行日志。</p>
</td>
<td class="cellrowborder" valign="top" width="14.399999999999999%" headers="mcps1.2.5.1.3 "><p id="zh-cn_topic_0000001568715673_p1634816161233"><a name="zh-cn_topic_0000001568715673_p1634816161233"></a><a name="zh-cn_topic_0000001568715673_p1634816161233"></a>ERROR</p>
</td>
<td class="cellrowborder" valign="top" width="33.26%" headers="mcps1.2.5.1.4 "><p id="p2474457134720"><a name="p2474457134720"></a><a name="p2474457134720"></a><a href="运行日志Error级别日志宏.md">运行日志Error级别日志宏</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001568715673_row234821632314"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000001568715673_p6348191662318"><a name="zh-cn_topic_0000001568715673_p6348191662318"></a><a name="zh-cn_topic_0000001568715673_p6348191662318"></a>INFO</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p147315579474"><a name="p147315579474"></a><a name="p147315579474"></a><a href="运行日志Info级别日志宏.md">运行日志Info级别日志宏</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001568715673_row19348131616236"><td class="cellrowborder" rowspan="4" valign="top" width="11.450000000000001%" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000001568715673_p15436163692710"><a name="zh-cn_topic_0000001568715673_p15436163692710"></a><a name="zh-cn_topic_0000001568715673_p15436163692710"></a>调试日志</p>
</td>
<td class="cellrowborder" rowspan="4" valign="top" width="40.89%" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0000001568715673_p19618948162918"><a name="zh-cn_topic_0000001568715673_p19618948162918"></a><a name="zh-cn_topic_0000001568715673_p19618948162918"></a>以下场景（包括但不限于这些场景）需要记录为调试级别的日志：</p>
<p id="zh-cn_topic_0000001568715673_p36182487297"><a name="zh-cn_topic_0000001568715673_p36182487297"></a><a name="zh-cn_topic_0000001568715673_p36182487297"></a>- 接口调用、函数调用等所有调用的入口、出口</p>
<p id="zh-cn_topic_0000001568715673_p261834818294"><a name="zh-cn_topic_0000001568715673_p261834818294"></a><a name="zh-cn_topic_0000001568715673_p261834818294"></a>- 操作入口处和设置预置条件</p>
<p id="zh-cn_topic_0000001568715673_p19618124872919"><a name="zh-cn_topic_0000001568715673_p19618124872919"></a><a name="zh-cn_topic_0000001568715673_p19618124872919"></a>- 定时器启动、超时</p>
<p id="zh-cn_topic_0000001568715673_p66181848102919"><a name="zh-cn_topic_0000001568715673_p66181848102919"></a><a name="zh-cn_topic_0000001568715673_p66181848102919"></a>- 状态设置、状态迁移条件判断前后</p>
<p id="zh-cn_topic_0000001568715673_p1361824822911"><a name="zh-cn_topic_0000001568715673_p1361824822911"></a><a name="zh-cn_topic_0000001568715673_p1361824822911"></a>- 业务相关资源统计、业务处理出入口、性能计算统计</p>
<p id="zh-cn_topic_0000001568715673_p16618164819292"><a name="zh-cn_topic_0000001568715673_p16618164819292"></a><a name="zh-cn_topic_0000001568715673_p16618164819292"></a>- 所有处理失败、异常等</p>
<p id="zh-cn_topic_0000001568715673_p10618144812917"><a name="zh-cn_topic_0000001568715673_p10618144812917"></a><a name="zh-cn_topic_0000001568715673_p10618144812917"></a>调试级别日志记录的是代码级的信息，用于开发人员定位问题。</p>
</td>
<td class="cellrowborder" valign="top" width="14.399999999999999%" headers="mcps1.2.5.1.3 "><p id="zh-cn_topic_0000001568715673_p13349131620233"><a name="zh-cn_topic_0000001568715673_p13349131620233"></a><a name="zh-cn_topic_0000001568715673_p13349131620233"></a>ERROR</p>
</td>
<td class="cellrowborder" valign="top" width="33.26%" headers="mcps1.2.5.1.4 "><p id="p154723577472"><a name="p154723577472"></a><a name="p154723577472"></a><a href="调试日志Error级别日志宏.md">调试日志Error级别日志宏</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001568715673_row123495166237"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000001568715673_p10349141616232"><a name="zh-cn_topic_0000001568715673_p10349141616232"></a><a name="zh-cn_topic_0000001568715673_p10349141616232"></a>WARN</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p64721157204719"><a name="p64721157204719"></a><a name="p64721157204719"></a><a href="调试日志Warn级别日志宏.md">调试日志Warn级别日志宏</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001568715673_row1934931682317"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000001568715673_p2349121620233"><a name="zh-cn_topic_0000001568715673_p2349121620233"></a><a name="zh-cn_topic_0000001568715673_p2349121620233"></a>INFO</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p2471155754719"><a name="p2471155754719"></a><a name="p2471155754719"></a><a href="调试日志Error级别日志宏.md">调试日志Error级别日志宏</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001568715673_row36999752712"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000001568715673_p1769913772718"><a name="zh-cn_topic_0000001568715673_p1769913772718"></a><a name="zh-cn_topic_0000001568715673_p1769913772718"></a>DEBUG</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p847015711475"><a name="p847015711475"></a><a name="p847015711475"></a><a href="调试日志Debug级别日志宏.md">调试日志Debug级别日志宏</a></p>
</td>
</tr>
</tbody>
</table>

日志使用样例：

```
import dataflow.flow_func as ff
ff.logger.info("This is a test info log :%s %d %f.", "test_str", 100, 0.1)
```

日志级别修改请参考《环境变量参考》中ASCEND\_GLOBAL\_LOG\_LEVEL及ASCEND\_MODULE\_LOG\_LEVEL的使用，用户UDF日志对应的模块为APP，可以根据模块单独控制APP日志级别。

例如：用户想开启自定义UDF的Info级别日志，可以使用下面命令单独打开APP模块的info级别日志

```
export ASCEND_MODULE_LOG_LEVEL=APP=1
```

也可以使用下方命令打开所有模块的info级别日志

```
export ASCEND_GLOBAL_LOG_LEVEL=1
```

UDF执行过程中，会对APP模块日志进行流控，以防止过多太多影响框架日志以及执行性能，限制规则如下：

**表 2**  限制规则

<a name="table942192421818"></a>
<table><thead align="left"><tr id="row1390603965910"><th class="cellrowborder" valign="top" id="mcps1.2.8.1.1"><p id="p19906203915596"><a name="p19906203915596"></a><a name="p19906203915596"></a><span>接口类型</span></p>
</th>
<th class="cellrowborder" colspan="4" valign="top" id="mcps1.2.8.1.2"><p id="p2906123915917"><a name="p2906123915917"></a><a name="p2906123915917"></a><span>调试日志</span></p>
</th>
<th class="cellrowborder" colspan="2" valign="top" id="mcps1.2.8.1.3"><p id="p209061392597"><a name="p209061392597"></a><a name="p209061392597"></a><span>运行</span><span>日志</span></p>
</th>
</tr>
</thead>
<tbody><tr id="row25070247181"><td class="cellrowborder" valign="top" width="14.472894578915785%" headers="mcps1.2.8.1.1 "><p id="p0240105207"><a name="p0240105207"></a><a name="p0240105207"></a><strong id="b67688541605"><a name="b67688541605"></a><a name="b67688541605"></a>级别</strong></p>
</td>
<td class="cellrowborder" valign="top" width="14.262852570514106%" headers="mcps1.2.8.1.2 "><p id="p1350815243189"><a name="p1350815243189"></a><a name="p1350815243189"></a>debug</p>
</td>
<td class="cellrowborder" valign="top" width="14.252850570114026%" headers="mcps1.2.8.1.2 "><p id="p1050872417187"><a name="p1050872417187"></a><a name="p1050872417187"></a>info</p>
</td>
<td class="cellrowborder" valign="top" width="14.252850570114026%" headers="mcps1.2.8.1.2 "><p id="p35081124181814"><a name="p35081124181814"></a><a name="p35081124181814"></a>warn</p>
</td>
<td class="cellrowborder" valign="top" width="14.252850570114026%" headers="mcps1.2.8.1.2 "><p id="p1850812419182"><a name="p1850812419182"></a><a name="p1850812419182"></a>error</p>
</td>
<td class="cellrowborder" valign="top" width="14.252850570114026%" headers="mcps1.2.8.1.3 "><p id="p9508142414185"><a name="p9508142414185"></a><a name="p9508142414185"></a>run_info</p>
</td>
<td class="cellrowborder" valign="top" width="14.252850570114026%" headers="mcps1.2.8.1.3 "><p id="p155081024141814"><a name="p155081024141814"></a><a name="p155081024141814"></a>run_error</p>
</td>
</tr>
<tr id="row050852414186"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p13508182411189"><a name="p13508182411189"></a><a name="p13508182411189"></a><span>DEBUG</span></p>
</td>
<td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.8.1.2 "><p id="p6508724131814"><a name="p6508724131814"></a><a name="p6508724131814"></a><span>不限流</span></p>
</td>
<td class="cellrowborder" rowspan="4" colspan="2" valign="top" headers="mcps1.2.8.1.3 "><p id="p250852420183"><a name="p250852420183"></a><a name="p250852420183"></a><span>50 / 400</span></p>
</td>
</tr>
<tr id="row17508172419189"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p0508924191811"><a name="p0508924191811"></a><a name="p0508924191811"></a><span>INFO</span></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p2508122412189"><a name="p2508122412189"></a><a name="p2508122412189"></a><span>不打印</span></p>
</td>
<td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.8.1.2 "><p id="p175081324121810"><a name="p175081324121810"></a><a name="p175081324121810"></a><span>100 / 1000</span></p>
</td>
</tr>
<tr id="row350822418188"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p550818246188"><a name="p550818246188"></a><a name="p550818246188"></a><span>WARN</span></p>
</td>
<td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.8.1.2 "><p id="p3508224171819"><a name="p3508224171819"></a><a name="p3508224171819"></a><span>不打印</span></p>
</td>
<td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.8.1.2 "><p id="p450812414183"><a name="p450812414183"></a><a name="p450812414183"></a><span>50 / 400</span></p>
</td>
</tr>
<tr id="row1508122441815"><td class="cellrowborder" valign="top" headers="mcps1.2.8.1.1 "><p id="p2050882417184"><a name="p2050882417184"></a><a name="p2050882417184"></a><span>ERROR</span></p>
</td>
<td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.8.1.2 "><p id="p205081824171813"><a name="p205081824171813"></a><a name="p205081824171813"></a><span>不打印</span></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.8.1.2 "><p id="p17508142481810"><a name="p17508142481810"></a><a name="p17508142481810"></a><span>50 / 400</span></p>
</td>
</tr>
</tbody>
</table>

>![](public_sys-resources/icon-note.gif) **说明：** 
>每个独立进程A条/秒，累积上限B，上表中显示为\[A/B\]，累积上限是指一段时间进程不打印日志后，最多瞬时可输出的日志。

