# UDF日志接口简介<a name="ZH-CN_TOPIC_0000001977316906"></a>

flow\_func\_log.h提供了日志接口，方便flowfunc开发中进行日志记录，日志相关场景如下。

**表 1**  日志分类

<a name="table03481516152310"></a>
<table><thead align="left"><tr id="row183481516172313"><th class="cellrowborder" valign="top" width="11.450000000000001%" id="mcps1.2.5.1.1"><p id="p1434814165237"><a name="p1434814165237"></a><a name="p1434814165237"></a>日志类型</p>
</th>
<th class="cellrowborder" valign="top" width="40.89%" id="mcps1.2.5.1.2"><p id="p12348171612319"><a name="p12348171612319"></a><a name="p12348171612319"></a>使用场景</p>
</th>
<th class="cellrowborder" valign="top" width="14.399999999999999%" id="mcps1.2.5.1.3"><p id="p23485169237"><a name="p23485169237"></a><a name="p23485169237"></a>日志级别</p>
</th>
<th class="cellrowborder" valign="top" width="33.26%" id="mcps1.2.5.1.4"><p id="p1234813165234"><a name="p1234813165234"></a><a name="p1234813165234"></a>对应的日志宏</p>
</th>
</tr>
</thead>
<tbody><tr id="row6348616202319"><td class="cellrowborder" rowspan="2" valign="top" width="11.450000000000001%" headers="mcps1.2.5.1.1 "><p id="p1434811632315"><a name="p1434811632315"></a><a name="p1434811632315"></a>运行日志</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" width="40.89%" headers="mcps1.2.5.1.2 "><p id="p873284102513"><a name="p873284102513"></a><a name="p873284102513"></a>系统运行过程中的异常状态、异常动作、系统进程运行过程中的关键事件和系统资源占用的相关信息等需要记录运行日志。</p>
</td>
<td class="cellrowborder" valign="top" width="14.399999999999999%" headers="mcps1.2.5.1.3 "><p id="p1634816161233"><a name="p1634816161233"></a><a name="p1634816161233"></a>ERROR</p>
</td>
<td class="cellrowborder" valign="top" width="33.26%" headers="mcps1.2.5.1.4 "><p id="p18394102055419"><a name="p18394102055419"></a><a name="p18394102055419"></a><a href="运行日志Error级别日志宏.md">运行日志Error级别日志宏</a></p>
</td>
</tr>
<tr id="row234821632314"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p6348191662318"><a name="p6348191662318"></a><a name="p6348191662318"></a>INFO</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p19393620205414"><a name="p19393620205414"></a><a name="p19393620205414"></a><a href="运行日志Info级别日志宏.md">运行日志Info级别日志宏</a></p>
</td>
</tr>
<tr id="row19348131616236"><td class="cellrowborder" rowspan="4" valign="top" width="11.450000000000001%" headers="mcps1.2.5.1.1 "><p id="p15436163692710"><a name="p15436163692710"></a><a name="p15436163692710"></a>调试日志</p>
</td>
<td class="cellrowborder" rowspan="4" valign="top" width="40.89%" headers="mcps1.2.5.1.2 "><p id="p19618948162918"><a name="p19618948162918"></a><a name="p19618948162918"></a>以下场景（包括但不限于这些场景）需要记录为调试级别的日志：</p>
<p id="p36182487297"><a name="p36182487297"></a><a name="p36182487297"></a>- 接口调用、函数调用等所有调用的入口、出口</p>
<p id="p261834818294"><a name="p261834818294"></a><a name="p261834818294"></a>- 操作入口处和设置预置条件</p>
<p id="p19618124872919"><a name="p19618124872919"></a><a name="p19618124872919"></a>- 定时器启动、超时</p>
<p id="p66181848102919"><a name="p66181848102919"></a><a name="p66181848102919"></a>- 状态设置、状态迁移条件判断前后</p>
<p id="p1361824822911"><a name="p1361824822911"></a><a name="p1361824822911"></a>- 业务相关资源统计、业务处理出入口、性能计算统计</p>
<p id="p16618164819292"><a name="p16618164819292"></a><a name="p16618164819292"></a>- 所有处理失败、异常等</p>
<p id="p10618144812917"><a name="p10618144812917"></a><a name="p10618144812917"></a>调试级别日志记录的是代码级的信息，用于开发人员定位问题。</p>
</td>
<td class="cellrowborder" valign="top" width="14.399999999999999%" headers="mcps1.2.5.1.3 "><p id="p13349131620233"><a name="p13349131620233"></a><a name="p13349131620233"></a>ERROR</p>
</td>
<td class="cellrowborder" valign="top" width="33.26%" headers="mcps1.2.5.1.4 "><p id="p139382013540"><a name="p139382013540"></a><a name="p139382013540"></a><a href="调试日志Error级别日志宏.md">调试日志Error级别日志宏</a></p>
</td>
</tr>
<tr id="row123495166237"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p10349141616232"><a name="p10349141616232"></a><a name="p10349141616232"></a>WARN</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p16392720185411"><a name="p16392720185411"></a><a name="p16392720185411"></a><a href="调试日志Warn级别日志宏.md">调试日志Warn级别日志宏</a></p>
</td>
</tr>
<tr id="row1934931682317"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p2349121620233"><a name="p2349121620233"></a><a name="p2349121620233"></a>INFO</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p113922206548"><a name="p113922206548"></a><a name="p113922206548"></a><a href="调试日志Info级别日志宏.md">调试日志Info级别日志宏</a></p>
</td>
</tr>
<tr id="row36999752712"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1769913772718"><a name="p1769913772718"></a><a name="p1769913772718"></a>DEBUG</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p139132075410"><a name="p139132075410"></a><a name="p139132075410"></a><a href="调试日志Debug级别日志宏.md">调试日志Debug级别日志宏</a></p>
</td>
</tr>
</tbody>
</table>

