# FlowData的构造函数和析构函数<a name="ZH-CN_TOPIC_0000002013832557"></a>

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
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507"><a name="zh-cn_topic_0000001312391781_term1551319498507"></a><a name="zh-cn_topic_0000001312391781_term1551319498507"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p19948143911820"><a name="p19948143911820"></a><a name="p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 函数功能<a name="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_section3729174918713"></a>

FlowData构造函数和析构函数，构造函数会返回一个FlowData节点。

## 函数原型<a name="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_section84161445741"></a>

```
FlowData(const char *name, int64_t index)
~FlowData() override
```

## 参数说明<a name="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_section63604780"></a>

<a name="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_table47561922"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_row29169897"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p13951479"><a name="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p13951479"></a><a name="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p13951479"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="27.900000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p56327989"><a name="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p56327989"></a><a name="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p56327989"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="44.47%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p66531170"><a name="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p66531170"></a><a name="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p66531170"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_row20315681"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001461072793_p18595429205213"><a name="zh-cn_topic_0000001461072793_p18595429205213"></a><a name="zh-cn_topic_0000001461072793_p18595429205213"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="27.900000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001461072793_p17595429125214"><a name="zh-cn_topic_0000001461072793_p17595429125214"></a><a name="zh-cn_topic_0000001461072793_p17595429125214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="44.47%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001461072793_p25951729105217"><a name="zh-cn_topic_0000001461072793_p25951729105217"></a><a name="zh-cn_topic_0000001461072793_p25951729105217"></a>输入节点名称，需要全图唯一。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001461072793_row153691343525"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001461072793_p1265404112528"><a name="zh-cn_topic_0000001461072793_p1265404112528"></a><a name="zh-cn_topic_0000001461072793_p1265404112528"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="27.900000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001461072793_p165474118522"><a name="zh-cn_topic_0000001461072793_p165474118522"></a><a name="zh-cn_topic_0000001461072793_p165474118522"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="44.47%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001461072793_p1965416418523"><a name="zh-cn_topic_0000001461072793_p1965416418523"></a><a name="zh-cn_topic_0000001461072793_p1965416418523"></a>输入节点index，全图中输入index需要从0开始累加，表示图中输入的顺序。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_section413535858"></a>

返回一个FlowData节点。

## 异常处理<a name="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_section1548781517515"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001461072793_zh-cn_topic_0000001265240866_section2021419196520"></a>

无。

