# map\_output<a name="ZH-CN_TOPIC_0000002013513637"></a>

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

给FlowNode映射输出，表示将pp的第pp\_output\_index个输出给到FlowNode的第node\_output\_index个输出，返回映射好的FlowNode节点。

可选调方法，不调用会默认按顺序去映射FlowNode和pp的输出。

## 函数原型<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section84161445741"></a>

```
map_output(node_output_index, pp, pp_output_index)
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
<tbody><tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row35581048202018"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001461432453_p13100113718595"><a name="zh-cn_topic_0000001461432453_p13100113718595"></a><a name="zh-cn_topic_0000001461432453_p13100113718595"></a>node_output_index</p>
</td>
<td class="cellrowborder" valign="top" width="22.07%" headers="mcps1.1.4.1.2 "><p id="p18851247165716"><a name="p18851247165716"></a><a name="p18851247165716"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="55.71%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001461432453_p16100203714595"><a name="zh-cn_topic_0000001461432453_p16100203714595"></a><a name="zh-cn_topic_0000001461432453_p16100203714595"></a>FlowNode输出index。</p>
</td>
</tr>
<tr id="row1731114205714"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001461432453_p16100193745910"><a name="zh-cn_topic_0000001461432453_p16100193745910"></a><a name="zh-cn_topic_0000001461432453_p16100193745910"></a>pp</p>
</td>
<td class="cellrowborder" valign="top" width="22.07%" headers="mcps1.1.4.1.2 "><p id="p1773112423576"><a name="p1773112423576"></a><a name="p1773112423576"></a>Union[GraphProcessPoint, FuncProcessPoint]</p>
</td>
<td class="cellrowborder" valign="top" width="55.71%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001461432453_p1710016377592"><a name="zh-cn_topic_0000001461432453_p1710016377592"></a><a name="zh-cn_topic_0000001461432453_p1710016377592"></a>FlowNode节点映射的pp。可以是<a href="dataflow-GraphProcessPoint.md">GraphProcessPoint</a>或者<a href="dataflow-FuncProcessPoint.md">FuncProcessPoint</a>。</p>
</td>
</tr>
<tr id="row1780104195710"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001461432453_p31001037155910"><a name="zh-cn_topic_0000001461432453_p31001037155910"></a><a name="zh-cn_topic_0000001461432453_p31001037155910"></a>pp_output_index</p>
</td>
<td class="cellrowborder" valign="top" width="22.07%" headers="mcps1.1.4.1.2 "><p id="p7801341115713"><a name="p7801341115713"></a><a name="p7801341115713"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="55.71%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001461432453_p16100143735912"><a name="zh-cn_topic_0000001461432453_p16100143735912"></a><a name="zh-cn_topic_0000001461432453_p16100143735912"></a>pp的输出index。</p>
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
flow_node = df.FlowNode(input_num=2, output_num=2) 
flow_node.add_process_point(pp)
flow_node.map_input(0, pp, 0) 
flow_node.map_input(1, pp, 1) 
flow_node.map_output(0, pp, 1)  
flow_node.map_output(1, pp, 0) 
# 构建连边关系 
flow_node_out = flow_node(data, data1)
```

## 约束说明<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section2021419196520"></a>

无

