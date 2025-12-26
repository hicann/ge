# map\_input<a name="ZH-CN_TOPIC_0000001976834046"></a>

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

给FlowNode映射输入，表示将FlowNode的第node\_input\_index个输入给到ProcessPoint的第pp\_input\_index个输入，并且给ProcessPoint的该输入设置上attr里的所有属性，返回映射好的FlowNode节点。该函数可选，不被调用时会默认按顺序去映射FlowNode和ProcessPoint的输入。

## 函数原型<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section84161445741"></a>

```
map_input(node_input_index, pp, pp_input_index, input_attrs=[])
```

## 参数说明<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_section63604780"></a>

<a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_table2051894852017"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row4558174815206"><th class="cellrowborder" valign="top" width="22.220000000000002%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"></a><strong id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="22.07%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="55.71%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"></a><strong id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"></a>取值说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row35581048202018"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p985264745719"><a name="p985264745719"></a><a name="p985264745719"></a>node_input_index</p>
</td>
<td class="cellrowborder" valign="top" width="22.07%" headers="mcps1.1.4.1.2 "><p id="p18851247165716"><a name="p18851247165716"></a><a name="p18851247165716"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="55.71%" headers="mcps1.1.4.1.3 "><p id="p168519478576"><a name="p168519478576"></a><a name="p168519478576"></a>FlowNode节点输入index，小于等于输入个数。</p>
</td>
</tr>
<tr id="row1731114205714"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p10731124235710"><a name="p10731124235710"></a><a name="p10731124235710"></a>pp</p>
</td>
<td class="cellrowborder" valign="top" width="22.07%" headers="mcps1.1.4.1.2 "><p id="p1773112423576"><a name="p1773112423576"></a><a name="p1773112423576"></a>Union[GraphProcessPoint, FuncProcessPoint]</p>
</td>
<td class="cellrowborder" valign="top" width="55.71%" headers="mcps1.1.4.1.3 "><p id="p5820317133910"><a name="p5820317133910"></a><a name="p5820317133910"></a>FlowNode节点映射的pp。可以是<a href="dataflow-GraphProcessPoint.md">GraphProcessPoint</a>或者<a href="dataflow-FuncProcessPoint.md">FuncProcessPoint</a>。</p>
</td>
</tr>
<tr id="row1780104195710"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p58074113572"><a name="p58074113572"></a><a name="p58074113572"></a>pp_input_index</p>
</td>
<td class="cellrowborder" valign="top" width="22.07%" headers="mcps1.1.4.1.2 "><p id="p7801341115713"><a name="p7801341115713"></a><a name="p7801341115713"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="55.71%" headers="mcps1.1.4.1.3 "><p id="p336133718391"><a name="p336133718391"></a><a name="p336133718391"></a>pp的输入index。</p>
</td>
</tr>
<tr id="row8570544115720"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p65705444578"><a name="p65705444578"></a><a name="p65705444578"></a>input_attrs</p>
</td>
<td class="cellrowborder" valign="top" width="22.07%" headers="mcps1.1.4.1.2 "><p id="p185706448574"><a name="p185706448574"></a><a name="p185706448574"></a>List[Union[TimeBatch, CountBatch]]</p>
</td>
<td class="cellrowborder" valign="top" width="55.71%" headers="mcps1.1.4.1.3 "><p id="p165701344115716"><a name="p165701344115716"></a><a name="p165701344115716"></a>属性集，当前支持<a href="dataflow-TimeBatch.md">TimeBatch</a>和<a href="dataflow-CountBatch.md">CountBatch</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section413535858"></a>

正常场景下返回None。

异常情况如下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例<a name="section17821439839"></a>

```
import dataflow as df
pp = df.FuncProcessPoint(...)
flow_node = df.FlowNode(input_num=2, output_num=1)
flow_node.add_process_point(pp)
flow_node.map_input(0, pp, 1)
flow_node.map_input(1, pp, 0)
flow_node_out = flow_node(data0, data1)
```

## 约束说明<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section2021419196520"></a>

无

