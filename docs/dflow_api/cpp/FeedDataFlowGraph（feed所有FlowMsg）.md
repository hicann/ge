# FeedDataFlowGraph（feed所有FlowMsg）<a name="ZH-CN_TOPIC_0000002094263878"></a>

## 产品支持情况<a name="section8178181118225"></a>

<a name="zh-cn_topic_0000002013832557_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002013832557_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002013832557_p1883113061818"><a name="zh-cn_topic_0000002013832557_p1883113061818"></a><a name="zh-cn_topic_0000002013832557_p1883113061818"></a><span id="zh-cn_topic_0000002013832557_ph20833205312295"><a name="zh-cn_topic_0000002013832557_ph20833205312295"></a><a name="zh-cn_topic_0000002013832557_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002013832557_p783113012187"><a name="zh-cn_topic_0000002013832557_p783113012187"></a><a name="zh-cn_topic_0000002013832557_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002013832557_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002013832557_p48327011813"><a name="zh-cn_topic_0000002013832557_p48327011813"></a><a name="zh-cn_topic_0000002013832557_p48327011813"></a><span id="zh-cn_topic_0000002013832557_ph583230201815"><a name="zh-cn_topic_0000002013832557_ph583230201815"></a><a name="zh-cn_topic_0000002013832557_ph583230201815"></a><term id="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term12835255145414"><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term12835255145414"></a><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term12835255145414"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002013832557_p7948163910184"><a name="zh-cn_topic_0000002013832557_p7948163910184"></a><a name="zh-cn_topic_0000002013832557_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002013832557_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002013832557_p14832120181815"><a name="zh-cn_topic_0000002013832557_p14832120181815"></a><a name="zh-cn_topic_0000002013832557_p14832120181815"></a><span id="zh-cn_topic_0000002013832557_ph1483216010188"><a name="zh-cn_topic_0000002013832557_ph1483216010188"></a><a name="zh-cn_topic_0000002013832557_ph1483216010188"></a><term id="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term1551319498507"><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term1551319498507"></a><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term1551319498507"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002013832557_p19948143911820"><a name="zh-cn_topic_0000002013832557_p19948143911820"></a><a name="zh-cn_topic_0000002013832557_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 函数功能<a name="zh-cn_topic_0000001365193284_section44282627"></a>

将数据输入到Graph图。

## 函数原型<a name="zh-cn_topic_0000001365193284_section1831611148519"></a>

```
Status FeedDataFlowGraph(uint32_t graph_id, const std::vector<FlowMsgPtr> &inputs, int32_t timeout)
```

## 参数说明<a name="zh-cn_topic_0000001365193284_section62999330"></a>

<a name="zh-cn_topic_0000001365193284_table228212484916"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001365193284_row628220488915"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001365193284_p528218487919"><a name="zh-cn_topic_0000001365193284_p528218487919"></a><a name="zh-cn_topic_0000001365193284_p528218487919"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.469999999999999%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001365193284_p102825481094"><a name="zh-cn_topic_0000001365193284_p102825481094"></a><a name="zh-cn_topic_0000001365193284_p102825481094"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.9%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001365193284_p122828481097"><a name="zh-cn_topic_0000001365193284_p122828481097"></a><a name="zh-cn_topic_0000001365193284_p122828481097"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001365193284_row152825488914"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p13200733388"><a name="p13200733388"></a><a name="p13200733388"></a>graph_id</p>
</td>
<td class="cellrowborder" valign="top" width="14.469999999999999%" headers="mcps1.1.4.1.2 "><p id="p5200931382"><a name="p5200931382"></a><a name="p5200931382"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.9%" headers="mcps1.1.4.1.3 "><p id="p1220012319387"><a name="p1220012319387"></a><a name="p1220012319387"></a>要执行图对应的ID。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001365193284_row250143872"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p520010383810"><a name="p520010383810"></a><a name="p520010383810"></a>inputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.469999999999999%" headers="mcps1.1.4.1.2 "><p id="p1720013383817"><a name="p1720013383817"></a><a name="p1720013383817"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.9%" headers="mcps1.1.4.1.3 "><p id="p152001737385"><a name="p152001737385"></a><a name="p152001737385"></a>计算图输入FlowMsg的指针，为Host上分配的共享内存。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001365193284_row32824481295"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p1920018383812"><a name="p1920018383812"></a><a name="p1920018383812"></a>timeout</p>
</td>
<td class="cellrowborder" valign="top" width="14.469999999999999%" headers="mcps1.1.4.1.2 "><p id="p1420011313380"><a name="p1420011313380"></a><a name="p1420011313380"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.9%" headers="mcps1.1.4.1.3 "><p id="p11200173103813"><a name="p11200173103813"></a><a name="p11200173103813"></a>数据输入超时时间，单位：ms，取值为-1时表示从不超时。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001365193284_section30123063"></a>

函数状态结果如下。

<a name="zh-cn_topic_0000001365193284_table2601186"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001365193284_row1832323"><th class="cellrowborder" valign="top" width="32.65%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001365193284_p14200498"><a name="zh-cn_topic_0000001365193284_p14200498"></a><a name="zh-cn_topic_0000001365193284_p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="24.36%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001365193284_p9389685"><a name="zh-cn_topic_0000001365193284_p9389685"></a><a name="zh-cn_topic_0000001365193284_p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="42.99%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001365193284_p22367029"><a name="zh-cn_topic_0000001365193284_p22367029"></a><a name="zh-cn_topic_0000001365193284_p22367029"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001365193284_row66898905"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001365193284_p50102218"><a name="zh-cn_topic_0000001365193284_p50102218"></a><a name="zh-cn_topic_0000001365193284_p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="24.36%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001365193284_p31747823"><a name="zh-cn_topic_0000001365193284_p31747823"></a><a name="zh-cn_topic_0000001365193284_p31747823"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="42.99%" headers="mcps1.1.4.1.3 "><a name="ul145462154538"></a><a name="ul145462154538"></a><ul id="ul145462154538"><li>SUCCESS：数据输入成功。</li><li>FAILED：数据输入失败。</li><li>其他错误码请参考<a href="UDF错误码.md">UDF错误码</a>。</li></ul>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001365193284_section24049039"></a>

无

