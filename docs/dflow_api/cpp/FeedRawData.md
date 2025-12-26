# FeedRawData<a name="ZH-CN_TOPIC_0000002013792109"></a>

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

## 函数功能<a name="zh-cn_topic_0000001415994657_section44282627"></a>

将原始数据输入到Graph图。

## 函数原型<a name="zh-cn_topic_0000001415994657_section1831611148519"></a>

```
struct RawData {
  const void *addr;
  size_t len;
};
```

```
Status FeedRawData(uint32_t graph_id, const std::vector<RawData> &raw_data_list, const uint32_t index,  const DataFlowInfo &info, int32_t timeout);
```

## 参数说明<a name="zh-cn_topic_0000001415994657_section62999330"></a>

<a name="zh-cn_topic_0000001415994657_table10309404"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001415994657_row47530006"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001415994657_p24725298"><a name="zh-cn_topic_0000001415994657_p24725298"></a><a name="zh-cn_topic_0000001415994657_p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.469999999999999%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001415994657_p56592155"><a name="zh-cn_topic_0000001415994657_p56592155"></a><a name="zh-cn_topic_0000001415994657_p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.9%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001415994657_p54897010"><a name="zh-cn_topic_0000001415994657_p54897010"></a><a name="zh-cn_topic_0000001415994657_p54897010"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001415994657_row17472816"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001415994657_p6011995"><a name="zh-cn_topic_0000001415994657_p6011995"></a><a name="zh-cn_topic_0000001415994657_p6011995"></a>graph_id</p>
</td>
<td class="cellrowborder" valign="top" width="14.469999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001415994657_p17209562"><a name="zh-cn_topic_0000001415994657_p17209562"></a><a name="zh-cn_topic_0000001415994657_p17209562"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.9%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001415994657_p34829302"><a name="zh-cn_topic_0000001415994657_p34829302"></a><a name="zh-cn_topic_0000001415994657_p34829302"></a>要执行图对应的ID。</p>
</td>
</tr>
<tr id="row46685552617"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p10225123974911"><a name="p10225123974911"></a><a name="p10225123974911"></a>raw_data_list</p>
</td>
<td class="cellrowborder" valign="top" width="14.469999999999999%" headers="mcps1.1.4.1.2 "><p id="p15225139104913"><a name="p15225139104913"></a><a name="p15225139104913"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.9%" headers="mcps1.1.4.1.3 "><p id="p1222553913497"><a name="p1222553913497"></a><a name="p1222553913497"></a>由输入数据指针和长度组成的数组，可以是1个也可以是多个，如果是多个<span id="ph5911440537"><a name="ph5911440537"></a><a name="ph5911440537"></a>，</span>框架将自动把多个数据合并成一份数据传递给DataFlow图。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001415994657_row0115745088"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001415994657_p611624515811"><a name="zh-cn_topic_0000001415994657_p611624515811"></a><a name="zh-cn_topic_0000001415994657_p611624515811"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="14.469999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001415994657_p1211615453816"><a name="zh-cn_topic_0000001415994657_p1211615453816"></a><a name="zh-cn_topic_0000001415994657_p1211615453816"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.9%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001415994657_p6626238163513"><a name="zh-cn_topic_0000001415994657_p6626238163513"></a><a name="zh-cn_topic_0000001415994657_p6626238163513"></a>对应的DataFlow图的某个输入。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001415994657_row7157114920813"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001415994657_p141580499817"><a name="zh-cn_topic_0000001415994657_p141580499817"></a><a name="zh-cn_topic_0000001415994657_p141580499817"></a>info</p>
</td>
<td class="cellrowborder" valign="top" width="14.469999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001415994657_p11158649084"><a name="zh-cn_topic_0000001415994657_p11158649084"></a><a name="zh-cn_topic_0000001415994657_p11158649084"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.9%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001415994657_p111581549686"><a name="zh-cn_topic_0000001415994657_p111581549686"></a><a name="zh-cn_topic_0000001415994657_p111581549686"></a>输入数据流标志（flow flag）。具体请参考<a href="DataFlowInfo数据类型.md">DataFlowInfo数据类型</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001415994657_row87868364213"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001415994657_p47861838425"><a name="zh-cn_topic_0000001415994657_p47861838425"></a><a name="zh-cn_topic_0000001415994657_p47861838425"></a>timeout</p>
</td>
<td class="cellrowborder" valign="top" width="14.469999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001415994657_p47864315425"><a name="zh-cn_topic_0000001415994657_p47864315425"></a><a name="zh-cn_topic_0000001415994657_p47864315425"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.9%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001415994657_p177864313423"><a name="zh-cn_topic_0000001415994657_p177864313423"></a><a name="zh-cn_topic_0000001415994657_p177864313423"></a>数据输入超时时间，单位：ms，取值为-1时表示从不超时。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001415994657_section30123063"></a>

<a name="zh-cn_topic_0000001415994657_table2601186"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001415994657_row1832323"><th class="cellrowborder" valign="top" width="32.65%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001415994657_p14200498"><a name="zh-cn_topic_0000001415994657_p14200498"></a><a name="zh-cn_topic_0000001415994657_p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="24.33%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001415994657_p9389685"><a name="zh-cn_topic_0000001415994657_p9389685"></a><a name="zh-cn_topic_0000001415994657_p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="43.02%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001415994657_p22367029"><a name="zh-cn_topic_0000001415994657_p22367029"></a><a name="zh-cn_topic_0000001415994657_p22367029"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001415994657_row66898905"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001415994657_p50102218"><a name="zh-cn_topic_0000001415994657_p50102218"></a><a name="zh-cn_topic_0000001415994657_p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="24.33%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001415994657_p31747823"><a name="zh-cn_topic_0000001415994657_p31747823"></a><a name="zh-cn_topic_0000001415994657_p31747823"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="43.02%" headers="mcps1.1.4.1.3 "><a name="ul6910933155319"></a><a name="ul6910933155319"></a><ul id="ul6910933155319"><li>SUCCESS：数据输入成功。</li><li>FAILED：数据输入失败。</li><li>其他错误码请参考<a href="UDF错误码.md">UDF错误码</a>。</li></ul>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001415994657_section24049039"></a>

无

