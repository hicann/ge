# LlmDataDist构造函数<a name="ZH-CN_TOPIC_0000002407583129"></a>

## 产品支持情况<a name="section8178181118225"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414"><a name="zh-cn_topic_0000001312391781_term12835255145414"></a><a name="zh-cn_topic_0000001312391781_term12835255145414"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p7948163910184"><a name="p7948163910184"></a><a name="p7948163910184"></a>√</p>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph980713477118"><a name="ph980713477118"></a><a name="ph980713477118"></a><term id="zh-cn_topic_0000001312391781_term454024162214"><a name="zh-cn_topic_0000001312391781_term454024162214"></a><a name="zh-cn_topic_0000001312391781_term454024162214"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p19948143911820"><a name="p19948143911820"></a><a name="p19948143911820"></a>√</p>
</td>
</tr>
<tr id="row15882185517522"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1682479135314"><a name="p1682479135314"></a><a name="p1682479135314"></a><span id="ph14880920154918"><a name="ph14880920154918"></a><a name="ph14880920154918"></a><term id="zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000001312391781_term16184138172215"></a>Atlas A2 训练系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p578615025316"><a name="p578615025316"></a><a name="p578615025316"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 函数功能<a name="zh-cn_topic_0000001461072801_zh-cn_topic_0000001265240866_section3729174918713"></a>

创建LLM-DataDist对象。

## 函数原型<a name="zh-cn_topic_0000001461072801_zh-cn_topic_0000001265240866_section84161445741"></a>

```
LlmDataDist(uint64_t cluster_id, LlmRole role)
```

## 参数说明<a name="zh-cn_topic_0000001461072801_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_section63604780"></a>

<a name="table1072194315344"></a>
<table><thead align="left"><tr id="row272174315340"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="p17721243153415"><a name="p17721243153415"></a><a name="p17721243153415"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.84%" id="mcps1.1.4.1.2"><p id="p1973194312342"><a name="p1973194312342"></a><a name="p1973194312342"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.53%" id="mcps1.1.4.1.3"><p id="p14731434345"><a name="p14731434345"></a><a name="p14731434345"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row127384363416"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p573184319342"><a name="p573184319342"></a><a name="p573184319342"></a>cluster_id</p>
</td>
<td class="cellrowborder" valign="top" width="14.84%" headers="mcps1.1.4.1.2 "><p id="p97316437346"><a name="p97316437346"></a><a name="p97316437346"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.53%" headers="mcps1.1.4.1.3 "><p id="p157364353411"><a name="p157364353411"></a><a name="p157364353411"></a>集群ID。LlmDataDist标识，在所有参与建链的范围内需要确保唯一。</p>
</td>
</tr>
<tr id="row287273317215"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p7873123392118"><a name="p7873123392118"></a><a name="p7873123392118"></a>role</p>
</td>
<td class="cellrowborder" valign="top" width="14.84%" headers="mcps1.1.4.1.2 "><p id="p178731033142118"><a name="p178731033142118"></a><a name="p178731033142118"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.53%" headers="mcps1.1.4.1.3 "><p id="p887323392114"><a name="p887323392114"></a><a name="p887323392114"></a>类型是<a href="LlmRole.md">LlmRole</a>，用于标识当前角色。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001461072801_zh-cn_topic_0000001265240866_section413535858"></a>

无

## 异常处理<a name="zh-cn_topic_0000001461072801_zh-cn_topic_0000001265240866_section1548781517515"></a>

无

## 约束说明<a name="zh-cn_topic_0000001461072801_zh-cn_topic_0000001265240866_section2021419196520"></a>

无

