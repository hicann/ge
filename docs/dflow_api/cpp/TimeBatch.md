# TimeBatch<a name="ZH-CN_TOPIC_0000001977312238"></a>

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

## 功能介绍<a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_section147991226121118"></a>

TimeBatch功能是基于UDF为前提的。

正常模型每次处理一个数据，当需要一次处理一批数据时，就需要将这批数据组成一个batch，最基本的batch方式是将这批N个数据直接拼接，然后shape前加N，而某些场景需要将某段或者某几段时间数据组成一个batch，并且按特定的维度拼接，则可以通过使用TimeBatch功能来组batch。

在ASR\(Automatic Speech Recognition\)自动语音识别场景下，存在按定长时间段组batch或按时间分段（时间不连续）组整批batch两种诉求，可以通过TimeBatch实现。

## 使用方法<a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_section0138175792819"></a>

用户在DataFlow构图时通过给FlowNode的输入设置属性来添加TimeBatch功能。示例如下。

```
TimeBatch time_batch = {};
// 按需求设置time_batch中各属性的值
time_batch.time_window = 10;
time_batch.batch_dim = 5;
time_batch.drop_remainder = true;
DataFlowInputAttr flow_attr = {DataFlowAttrType::TIME_BATCH, &time_batch};
std::vector<DataFlowInputAttr> flow_attrs = {flow_attr};
// 然后通过FlowNode的MapInput设置
FlowNode::MapInput(xx, xx, xx, flow_attrs);
```

<a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_table568013275512"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_row8680133235515"><th class="cellrowborder" valign="top" width="23.32%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p168033275520"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p168033275520"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p168033275520"></a><strong id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_b11680183211553"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_b11680183211553"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_b11680183211553"></a>属性名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="14.399999999999999%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p537710614477"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p537710614477"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p537710614477"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="62.28%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p19680143205514"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p19680143205514"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p19680143205514"></a><strong id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_b568018329555"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_b568018329555"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_b568018329555"></a>取值说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_row868013216553"><td class="cellrowborder" valign="top" width="23.32%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p9680123295518"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p9680123295518"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p9680123295518"></a>time_window</p>
</td>
<td class="cellrowborder" valign="top" width="14.399999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p1537716174710"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p1537716174710"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p1537716174710"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" width="62.28%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p7681143275512"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p7681143275512"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p7681143275512"></a>整型（单位ms），当值&gt;0时表示按该时间窗来组batch，当值为-1时表示按时间分段来组batch，其他值报错。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_row17681193295511"><td class="cellrowborder" valign="top" width="23.32%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p14681123205513"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p14681123205513"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p14681123205513"></a>batch_dim</p>
</td>
<td class="cellrowborder" valign="top" width="14.399999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p16409224155910"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p16409224155910"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p16409224155910"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" width="62.28%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p9878226323"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p9878226323"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p9878226323"></a>只有设置了time_window时，该参数才生效。取值范围[-1,shape维度]。</p>
<a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_ul14214947443"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_ul14214947443"></a><ul id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_ul14214947443"><li>默认为-1表示数据输出shape会在第0维添加一个batch维。</li><li>shape维度&gt;batch_dim&gt;=0时表示按某个维度组batch。</li><li>batch_dim&gt;shape维度或者&lt;-1时报错。</li></ul>
</td>
</tr>
<tr id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_row1068117323553"><td class="cellrowborder" valign="top" width="23.32%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p4681332195517"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p4681332195517"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p4681332195517"></a>drop_remainder</p>
</td>
<td class="cellrowborder" valign="top" width="14.399999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p23773634718"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p23773634718"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p23773634718"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="62.28%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p198907204349"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p198907204349"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p198907204349"></a>只有设置了time_window时，该参数才生效。</p>
<p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p4681143205520"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p4681143205520"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p4681143205520"></a>仅在time_window&gt;0时生效，选择不足time_window时是否丢弃，默认false不丢弃。true则丢弃。举例如下：</p>
<p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p196991458941"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p196991458941"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p196991458941"></a>假如time_window=5ms，输入数据时长为3ms，则：</p>
<a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_ul136019596817"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_ul136019596817"></a><ul id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_ul136019596817"><li>drop_remainder不配置或者配置为false时，不丢弃输入数据。</li><li>drop_remainder配置为true时<a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_ul1762204218919"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_ul1762204218919"></a><ul id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_ul1762204218919"><li>如果输入数据未携带EOS或者SEG，会一直等待，不丢弃数据。</li><li>如果输入数据只携带了SEG，则丢弃数据。</li><li>如果输入数据携带了EOS标记，则丢弃输入数据，只传递EOS标记。</li></ul>
</li></ul>
</td>
</tr>
<tr id="zh-cn_topic_0000001417515936_row44431820205517"><td class="cellrowborder" valign="top" width="23.32%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001417515936_p0584713145718"><a name="zh-cn_topic_0000001417515936_p0584713145718"></a><a name="zh-cn_topic_0000001417515936_p0584713145718"></a>time_interval</p>
</td>
<td class="cellrowborder" valign="top" width="14.399999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001417515936_p1144362013553"><a name="zh-cn_topic_0000001417515936_p1144362013553"></a><a name="zh-cn_topic_0000001417515936_p1144362013553"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" width="62.28%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001417515936_p1244319205557"><a name="zh-cn_topic_0000001417515936_p1244319205557"></a><a name="zh-cn_topic_0000001417515936_p1244319205557"></a>未使能</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001417515936_row6781171815552"><td class="cellrowborder" valign="top" width="23.32%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001417515936_p7781121813558"><a name="zh-cn_topic_0000001417515936_p7781121813558"></a><a name="zh-cn_topic_0000001417515936_p7781121813558"></a>timeout</p>
</td>
<td class="cellrowborder" valign="top" width="14.399999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001417515936_p1978114185552"><a name="zh-cn_topic_0000001417515936_p1978114185552"></a><a name="zh-cn_topic_0000001417515936_p1978114185552"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" width="62.28%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001417515936_p1878191845514"><a name="zh-cn_topic_0000001417515936_p1878191845514"></a><a name="zh-cn_topic_0000001417515936_p1878191845514"></a>未使能</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001417515936_row17781017165519"><td class="cellrowborder" valign="top" width="23.32%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001417515936_p147951795510"><a name="zh-cn_topic_0000001417515936_p147951795510"></a><a name="zh-cn_topic_0000001417515936_p147951795510"></a>flag</p>
</td>
<td class="cellrowborder" valign="top" width="14.399999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001417515936_p127916179552"><a name="zh-cn_topic_0000001417515936_p127916179552"></a><a name="zh-cn_topic_0000001417515936_p127916179552"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="62.28%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001417515936_p1879617185510"><a name="zh-cn_topic_0000001417515936_p1879617185510"></a><a name="zh-cn_topic_0000001417515936_p1879617185510"></a>未使能</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001417515936_row122801734125712"><td class="cellrowborder" valign="top" width="23.32%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001417515936_p12280534145710"><a name="zh-cn_topic_0000001417515936_p12280534145710"></a><a name="zh-cn_topic_0000001417515936_p12280534145710"></a>padding</p>
</td>
<td class="cellrowborder" valign="top" width="14.399999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001417515936_p10281183475720"><a name="zh-cn_topic_0000001417515936_p10281183475720"></a><a name="zh-cn_topic_0000001417515936_p10281183475720"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="62.28%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001417515936_p528113435719"><a name="zh-cn_topic_0000001417515936_p528113435719"></a><a name="zh-cn_topic_0000001417515936_p528113435719"></a>未使能</p>
</td>
</tr>
</tbody>
</table>

## 使用注意事项<a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_section643763016162"></a>

当前Batch特性无法做负荷分担，因此如果使用2P环境，需要在ge初始化时添加\{"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"\}, \{"ge.exec.logicalDeviceId", "\[0:0\]"\}。其中logicalDeviceId可以是\[0:0\]，也可以是\[0:1\]，详细介绍如下。

logicalDeviceClusterDeployMode为SINGLE时，用于指定模型部署在某个指定的设备上。

配置格式：\[node\_id:device\_id\]

-   node\_id：昇腾AI处理器逻辑ID，从0开始，表示资源配置文件中第几个设备。
-   device\_id：昇腾AI处理器物理ID。

