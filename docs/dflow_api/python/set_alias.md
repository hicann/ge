# set\_alias<a name="ZH-CN_TOPIC_0000002187954554"></a>

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

## 函数功能<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section3729174918713"></a>

设置节点别名，使用option:ge.experiment.data\_flow\_deploy\_info\_path指定节点部署位置时，flow\_node\_list字段可使用别名进行指定。

## 函数原型<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section84161445741"></a>

```
def set_alias(self, name) -> FlowNode
```

## 参数说明<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_section63604780"></a>

<a name="table1615311505910"></a>
<table><thead align="left"><tr id="row17154215155910"><th class="cellrowborder" valign="top" width="22.220000000000002%" id="mcps1.1.4.1.1"><p id="p19154915205919"><a name="p19154915205919"></a><a name="p19154915205919"></a><strong id="b31541159593"><a name="b31541159593"></a><a name="b31541159593"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="22.07%" id="mcps1.1.4.1.2"><p id="p9154115125919"><a name="p9154115125919"></a><a name="p9154115125919"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="55.71%" id="mcps1.1.4.1.3"><p id="p1815421575920"><a name="p1815421575920"></a><a name="p1815421575920"></a><strong id="b615491513592"><a name="b615491513592"></a><a name="b615491513592"></a>取值说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row415416154598"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p14154181517590"><a name="p14154181517590"></a><a name="p14154181517590"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="22.07%" headers="mcps1.1.4.1.2 "><p id="p1915421595910"><a name="p1915421595910"></a><a name="p1915421595910"></a>str</p>
</td>
<td class="cellrowborder" valign="top" width="55.71%" headers="mcps1.1.4.1.3 "><p id="p191541415145917"><a name="p191541415145917"></a><a name="p191541415145917"></a>FlowNode的别名。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section413535858"></a>

返回当前FlowNode。

## 调用示例<a name="section17821439839"></a>

```
import dataflow as df
flow_node = df.FlowNode(...)
flow_node.set_alias(name="hello")
```

## 约束说明<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section2021419196520"></a>

无

