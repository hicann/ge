# CountBatch<a name="ZH-CN_TOPIC_0000002013832601"></a>

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

## 功能介绍<a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_section147991226121118"></a>

CountBatch功能是指基于UDF为计算处理点将多个数据按batch\_size组成batch。

该功能应用于DataFlow异步场景，具体如下。

-   长时间没有数据输入时，可以通过CountBatch功能设置超时时间，如果没有设置padding，超时后取当前已有数据送计算处理点处理。
-   设置超时时间后，如果数据不满batch\_size时，可以通过CountBatch功能设置padding属性，计算点根据padding设置对数据进行填充到batch\_size后输出。

## 使用方法<a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_section0138175792819"></a>

用户在DataFlow构图时通过给FlowNode的输入设置属性来添加CountBatch功能。示例如下。

```
CountBatch count_batch = {};
// 按需求设置count_batch中各属性的值
count_batch.timeout = 10;
count_batch.batch_size = 300;
count_batch.padding = true;
count_batch.slide_stride = 5; 
DataFlowInputAttr flow_attr = {DataFlowAttrType::COUNT_BATCH, &count_batch};
std::vector<DataFlowInputAttr> flow_attrs = {flow_attr};
// 通过FlowNode的MapInput设置
FlowNode::MapInput(xx, xx, xx, flow_attrs);
```

<a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_table2051894852017"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row4558174815206"><th class="cellrowborder" valign="top" width="22.220000000000002%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"></a><strong id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"></a>属性名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="15.06%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="62.72%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"></a><strong id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"></a>取值说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row35581048202018"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p18123151553317"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p18123151553317"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p18123151553317"></a>batch_size</p>
</td>
<td class="cellrowborder" valign="top" width="15.06%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p3465132124816"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p3465132124816"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p3465132124816"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p17031651153315"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p17031651153315"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p17031651153315"></a>组batch大小。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row6558548162013"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p181221715103318"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p181221715103318"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p181221715103318"></a>timeout</p>
</td>
<td class="cellrowborder" valign="top" width="15.06%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p109811824194811"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p109811824194811"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p109811824194811"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p9878226323"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p9878226323"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p9878226323"></a>只有设置了batch_size时，该参数才生效。</p>
<p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p77031251123310"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p77031251123310"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p77031251123310"></a>组batch等待时间，单位（ms），取值范围[0,4294967295)，默认值是0，表示一直等待直到满batch。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row9558748152018"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p201171215183315"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p201171215183315"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p201171215183315"></a>padding</p>
</td>
<td class="cellrowborder" valign="top" width="15.06%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p10105142717472"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p10105142717472"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p10105142717472"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p12270204883510"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p12270204883510"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p12270204883510"></a>只有设置了batch_size和timeout时，该参数才生效。</p>
<p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p13704165118335"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p13704165118335"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p13704165118335"></a>不足batch时，是否padding。默认值false，表示不padding。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row143181035125"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p1031811351128"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p1031811351128"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p1031811351128"></a>slide_stride</p>
</td>
<td class="cellrowborder" valign="top" width="15.06%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p158013775517"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p158013775517"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p158013775517"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p66662516355"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p66662516355"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p66662516355"></a>只有设置了batch_size时，该参数才生效。</p>
<p id="p87011417191213"><a name="p87011417191213"></a><a name="p87011417191213"></a>滑窗步长，取值范围[0,batch_size]。</p>
<a name="ul981653361219"></a><a name="ul981653361219"></a><ul id="ul981653361219"><li>&gt;0且&lt;batch_size时表示启用滑窗方式组batch。</li><li>不设置，等于0。</li><li>等于batch_size时按照未设置滑窗步长方式组batch。</li><li>&gt;batch_size报错。</li></ul>
</td>
</tr>
<tr id="zh-cn_topic_0000001417673572_row5363174715589"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001417673572_p1661195616583"><a name="zh-cn_topic_0000001417673572_p1661195616583"></a><a name="zh-cn_topic_0000001417673572_p1661195616583"></a>batch_dim</p>
</td>
<td class="cellrowborder" valign="top" width="15.06%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001417673572_p18661115675815"><a name="zh-cn_topic_0000001417673572_p18661115675815"></a><a name="zh-cn_topic_0000001417673572_p18661115675815"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001417673572_p1136354718589"><a name="zh-cn_topic_0000001417673572_p1136354718589"></a><a name="zh-cn_topic_0000001417673572_p1136354718589"></a>未使能。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001417673572_row1797849205815"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001417673572_p151821334597"><a name="zh-cn_topic_0000001417673572_p151821334597"></a><a name="zh-cn_topic_0000001417673572_p151821334597"></a>flag</p>
</td>
<td class="cellrowborder" valign="top" width="15.06%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001417673572_p11824312599"><a name="zh-cn_topic_0000001417673572_p11824312599"></a><a name="zh-cn_topic_0000001417673572_p11824312599"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001417673572_p1897144915811"><a name="zh-cn_topic_0000001417673572_p1897144915811"></a><a name="zh-cn_topic_0000001417673572_p1897144915811"></a>未使能。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001417673572_row2719164514580"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001417673572_p117211975592"><a name="zh-cn_topic_0000001417673572_p117211975592"></a><a name="zh-cn_topic_0000001417673572_p117211975592"></a>drop_remainder</p>
</td>
<td class="cellrowborder" valign="top" width="15.06%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001417673572_p47217717595"><a name="zh-cn_topic_0000001417673572_p47217717595"></a><a name="zh-cn_topic_0000001417673572_p47217717595"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001417673572_p5719645155816"><a name="zh-cn_topic_0000001417673572_p5719645155816"></a><a name="zh-cn_topic_0000001417673572_p5719645155816"></a>未使能。</p>
</td>
</tr>
</tbody>
</table>

## 使用注意事项<a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_section643763016162"></a>

当前Batch特性无法做负荷分担，因此如果使用2P环境，需要在ge初始化时添加\{"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"\}, \{"ge.exec.logicalDeviceId", "\[0:0\]"\}。

