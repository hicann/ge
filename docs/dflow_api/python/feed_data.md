# feed\_data<a name="ZH-CN_TOPIC_0000002013513621"></a>

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

将数据输入到Graph。

## 函数原型<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section84161445741"></a>

```
feed_data(feed_dict, flow_info=None, timeout=-1, partial_inputs = False) -> int
```

## 参数说明<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_section63604780"></a>

<a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_table2051894852017"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row4558174815206"><th class="cellrowborder" valign="top" width="16.009999999999998%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"></a><strong id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="34.449999999999996%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="49.54%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"></a><strong id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"></a>取值说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row35581048202018"><td class="cellrowborder" valign="top" width="16.009999999999998%" headers="mcps1.1.4.1.1 "><p id="p845914589265"><a name="p845914589265"></a><a name="p845914589265"></a>feed_dict</p>
</td>
<td class="cellrowborder" valign="top" width="34.449999999999996%" headers="mcps1.1.4.1.2 "><p id="p2458258142614"><a name="p2458258142614"></a><a name="p2458258142614"></a>Dict[FlowData, Union["numpy.ndarray", Tensor, List]]</p>
</td>
<td class="cellrowborder" valign="top" width="49.54%" headers="mcps1.1.4.1.3 "><p id="p67192539717"><a name="p67192539717"></a><a name="p67192539717"></a>key为FlowData节点，value是可转换成numpy.ndarray的任意输入，或者dataflow.tensor。</p>
<p id="p12298229101215"><a name="p12298229101215"></a><a name="p12298229101215"></a>当feed_dict为空时，flow_info必须的flow_flags必须包含DATA_FLOW_FLAG_EOS，此时partial_inputs不起作用。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row6558548162013"><td class="cellrowborder" valign="top" width="16.009999999999998%" headers="mcps1.1.4.1.1 "><p id="p194561458152620"><a name="p194561458152620"></a><a name="p194561458152620"></a>flow_info</p>
</td>
<td class="cellrowborder" valign="top" width="34.449999999999996%" headers="mcps1.1.4.1.2 "><p id="p15456658182620"><a name="p15456658182620"></a><a name="p15456658182620"></a>FlowInfo</p>
</td>
<td class="cellrowborder" valign="top" width="49.54%" headers="mcps1.1.4.1.3 "><p id="p945518584264"><a name="p945518584264"></a><a name="p945518584264"></a>按需设置FlowInfo，具体请参见<a href="dataflow-FlowInfo.md">dataflow.FlowInfo</a>。</p>
</td>
</tr>
<tr id="row1061956145719"><td class="cellrowborder" valign="top" width="16.009999999999998%" headers="mcps1.1.4.1.1 "><p id="p2454958172617"><a name="p2454958172617"></a><a name="p2454958172617"></a>timeout</p>
</td>
<td class="cellrowborder" valign="top" width="34.449999999999996%" headers="mcps1.1.4.1.2 "><p id="p1454358142617"><a name="p1454358142617"></a><a name="p1454358142617"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="49.54%" headers="mcps1.1.4.1.3 "><p id="p3453175816269"><a name="p3453175816269"></a><a name="p3453175816269"></a>数据输入超时时间，单位：ms，取值范围[0, 2147483647), 取值为-1时表示从不超时。</p>
</td>
</tr>
<tr id="row11202635494"><td class="cellrowborder" valign="top" width="16.009999999999998%" headers="mcps1.1.4.1.1 "><p id="p1820320351192"><a name="p1820320351192"></a><a name="p1820320351192"></a>partial_inputs</p>
</td>
<td class="cellrowborder" valign="top" width="34.449999999999996%" headers="mcps1.1.4.1.2 "><p id="p13203203517916"><a name="p13203203517916"></a><a name="p13203203517916"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="49.54%" headers="mcps1.1.4.1.3 "><p id="p1520316355910"><a name="p1520316355910"></a><a name="p1520316355910"></a>每次调用feed_data接口时，feed_dict是否支持模型的部分输入，取值如下。</p>
<a name="ul1534316147384"></a><a name="ul1534316147384"></a><ul id="ul1534316147384"><li>True：feed_dict中可以只包含模型的部分输入。</li><li>False：feed_dict中必须包含模型所有的输入。</li></ul>
<p id="p9742174810373"><a name="p9742174810373"></a><a name="p9742174810373"></a>默认为False。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section413535858"></a>

正常场景下返回0。

异常场景下返回具体的错误码，并打印错误日志。

## 调用示例<a name="section17821439839"></a>

```
import dataflow as df
graph = df.FlowGraph(...)
graph.feed_data(...)
```

## 约束说明<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section2021419196520"></a>

无

