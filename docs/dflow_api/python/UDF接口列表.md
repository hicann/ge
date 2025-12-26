# UDF接口列表<a name="ZH-CN_TOPIC_0000002013400925"></a>

本文档主要描述UDF（User Defined Function）模块对外提供的接口，用户可以调用这些接口进行自定义处理函数的开发，然后通过DataFlow构图在CPU上执行该处理函数。

您可以在CANN软件安装后文件存储路径下的“python/site-packages/dataflow/flow\_func/flow\_func.py”查看对应接口的实现。接口列表如下。

## FlowMsg类<a name="section7170195113211"></a>

用于处理FlowFunc输入输出的相关操作。

**表 1**  FlowMsg类接口

<a name="table1354843712162"></a>
<table><thead align="left"><tr id="row1154919370163"><th class="cellrowborder" valign="top" width="33.47%" id="mcps1.2.3.1.1"><p id="p1354963712161"><a name="p1354963712161"></a><a name="p1354963712161"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.53%" id="mcps1.2.3.1.2"><p id="p354917378162"><a name="p354917378162"></a><a name="p354917378162"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row1454912374161"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p65680920227"><a name="p65680920227"></a><a name="p65680920227"></a><a href="FlowMsg构造函数.md">FlowMsg构造函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001489029853_zh-cn_topic_0000001359708654_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001489029853_zh-cn_topic_0000001359708654_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001489029853_zh-cn_topic_0000001359708654_zh-cn_topic_0000001264921066_p36600850"></a>FlowMsg的构造函数。</p>
</td>
</tr>
<tr id="row135493379161"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p20567397226"><a name="p20567397226"></a><a name="p20567397226"></a><a href="get_msg_type（UDF）.md">get_msg_type（UDF）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001438993190_zh-cn_topic_0000001409268917_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001438993190_zh-cn_topic_0000001409268917_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001438993190_zh-cn_topic_0000001409268917_zh-cn_topic_0000001264921066_p36600850"></a>获取FlowMsg的消息类型。</p>
</td>
</tr>
<tr id="row185491737161616"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p20565109192215"><a name="p20565109192215"></a><a name="p20565109192215"></a><a href="get_tensor.md">get_tensor</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001439153146_zh-cn_topic_0000001409388717_p148801903014"><a name="zh-cn_topic_0000001439153146_zh-cn_topic_0000001409388717_p148801903014"></a><a name="zh-cn_topic_0000001439153146_zh-cn_topic_0000001409388717_p148801903014"></a>获取FlowMsg中的tensor对象。</p>
</td>
</tr>
<tr id="row15549637201617"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p556410912224"><a name="p556410912224"></a><a name="p556410912224"></a><a href="set_ret_code.md">set_ret_code</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001488949577_zh-cn_topic_0000001359389150_p489119523119"><a name="zh-cn_topic_0000001488949577_zh-cn_topic_0000001359389150_p489119523119"></a><a name="zh-cn_topic_0000001488949577_zh-cn_topic_0000001359389150_p489119523119"></a>设置FlowMsg消息中的错误码。</p>
</td>
</tr>
<tr id="row5549143710169"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p9562198229"><a name="p9562198229"></a><a name="p9562198229"></a><a href="get_ret_code.md">get_ret_code</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001489269369_zh-cn_topic_0000001408829017_p254635918010"><a name="zh-cn_topic_0000001489269369_zh-cn_topic_0000001408829017_p254635918010"></a><a name="zh-cn_topic_0000001489269369_zh-cn_topic_0000001408829017_p254635918010"></a>获取输入FlowMsg消息中的错误码。</p>
</td>
</tr>
<tr id="row0550113771615"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1656113914224"><a name="p1656113914224"></a><a name="p1656113914224"></a><a href="set_start_time.md">set_start_time</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001439470042_zh-cn_topic_0000001359548690_p1562613391532"><a name="zh-cn_topic_0000001439470042_zh-cn_topic_0000001359548690_p1562613391532"></a><a name="zh-cn_topic_0000001439470042_zh-cn_topic_0000001359548690_p1562613391532"></a>设置FlowMsg消息头中的开始时间戳。</p>
</td>
</tr>
<tr id="row15550193712163"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p055910912221"><a name="p055910912221"></a><a name="p055910912221"></a><a href="get_start_time.md">get_start_time</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001489189321_zh-cn_topic_0000001409228617_p198905321146"><a name="zh-cn_topic_0000001489189321_zh-cn_topic_0000001409228617_p198905321146"></a><a name="zh-cn_topic_0000001489189321_zh-cn_topic_0000001409228617_p198905321146"></a>获取FlowMsg消息中的开始时间戳。</p>
</td>
</tr>
<tr id="row2055023701613"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p355819962214"><a name="p355819962214"></a><a name="p355819962214"></a><a href="set_end_time.md">set_end_time</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001439310790_zh-cn_topic_0000001359708658_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001439310790_zh-cn_topic_0000001359708658_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001439310790_zh-cn_topic_0000001359708658_zh-cn_topic_0000001264921066_p36600850"></a>设置FlowMsg消息头中的结束时间戳。</p>
</td>
</tr>
<tr id="row2055043761616"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1455619952218"><a name="p1455619952218"></a><a name="p1455619952218"></a><a href="get_end_time.md">get_end_time</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001489029857_zh-cn_topic_0000001409268921_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001489029857_zh-cn_topic_0000001409268921_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001489029857_zh-cn_topic_0000001409268921_zh-cn_topic_0000001264921066_p36600850"></a>获取FlowMsg消息中的结束时间戳。</p>
</td>
</tr>
<tr id="row10550123731611"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p105544916222"><a name="p105544916222"></a><a name="p105544916222"></a><a href="set_flow_flags.md">set_flow_flags</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001438993194_zh-cn_topic_0000001409388721_p186651645278"><a name="zh-cn_topic_0000001438993194_zh-cn_topic_0000001409388721_p186651645278"></a><a name="zh-cn_topic_0000001438993194_zh-cn_topic_0000001409388721_p186651645278"></a>设置FlowMsg消息头中的flags。</p>
</td>
</tr>
<tr id="row19550173791615"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p15529992218"><a name="p15529992218"></a><a name="p15529992218"></a><a href="get_flow_flags.md">get_flow_flags</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001439153150_zh-cn_topic_0000001408829021_p204566483916"><a name="zh-cn_topic_0000001439153150_zh-cn_topic_0000001408829021_p204566483916"></a><a name="zh-cn_topic_0000001439153150_zh-cn_topic_0000001408829021_p204566483916"></a>获取FlowMsg消息头中的flags。</p>
</td>
</tr>
<tr id="row455083719162"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p155119982213"><a name="p155119982213"></a><a name="p155119982213"></a><a href="set_route_label.md">set_route_label</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001600564720_zh-cn_topic_0000001409388721_p186651645278"><a name="zh-cn_topic_0000001600564720_zh-cn_topic_0000001409388721_p186651645278"></a><a name="zh-cn_topic_0000001600564720_zh-cn_topic_0000001409388721_p186651645278"></a>设置路由的标签<strong id="zh-cn_topic_0000001600564720_b151154742613"><a name="zh-cn_topic_0000001600564720_b151154742613"></a><a name="zh-cn_topic_0000001600564720_b151154742613"></a>。</strong></p>
</td>
</tr>
<tr id="row5551113710166"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p12550591221"><a name="p12550591221"></a><a name="p12550591221"></a><a href="get_transaction_id.md">get_transaction_id</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p114114156254"><a name="p114114156254"></a><a name="p114114156254"></a>获取FlowMsg消息中的事务ID，事务Id从1开始计数，每feed一批数据，事务Id会加一，可用于识别哪一批数据<strong id="b174171517252"><a name="b174171517252"></a><a name="b174171517252"></a>。</strong></p>
</td>
</tr>
<tr id="row27596132343"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p11759013203417"><a name="p11759013203417"></a><a name="p11759013203417"></a><a href="set_msg_type.md">set_msg_type</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p5550951103417"><a name="p5550951103417"></a><a name="p5550951103417"></a>设置FlowMsg的消息类型<strong id="b7551145111342"><a name="b7551145111342"></a><a name="b7551145111342"></a>。</strong></p>
</td>
</tr>
<tr id="row17229161503419"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1222991593412"><a name="p1222991593412"></a><a name="p1222991593412"></a><a href="get_raw_data.md">get_raw_data</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p129225714346"><a name="p129225714346"></a><a name="p129225714346"></a>获取rawdata类型的数据。</p>
</td>
</tr>
<tr id="row1711721223418"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1117101273417"><a name="p1117101273417"></a><a name="p1117101273417"></a><a href="set_transaction_id.md">set_transaction_id</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p119271221352"><a name="p119271221352"></a><a name="p119271221352"></a>设置DataFlow数据传输使用的事务ID<strong id="b122516178424"><a name="b122516178424"></a><a name="b122516178424"></a>。</strong></p>
</td>
</tr>
</tbody>
</table>

## Tensor类<a name="section1040402383518"></a>

用于执行Tensor的相关操作。这里获取的Tensor是dataflow.Tensor。

**表 2**  Tensor类接口

<a name="table728210919335"></a>
<table><thead align="left"><tr id="row6283169113319"><th class="cellrowborder" valign="top" width="33.47%" id="mcps1.2.3.1.1"><p id="p62837912338"><a name="p62837912338"></a><a name="p62837912338"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.53%" id="mcps1.2.3.1.2"><p id="p4283149133319"><a name="p4283149133319"></a><a name="p4283149133319"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row628316920333"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p167134652212"><a name="p167134652212"></a><a name="p167134652212"></a><a href="Tensor构造函数.md">Tensor构造函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001439470046_zh-cn_topic_0000001357504881_zh-cn_topic_0000001264921070_zh-cn_topic_0204328222_zh-cn_topic_0182636384_p13843256"><a name="zh-cn_topic_0000001439470046_zh-cn_topic_0000001357504881_zh-cn_topic_0000001264921070_zh-cn_topic_0204328222_zh-cn_topic_0182636384_p13843256"></a><a name="zh-cn_topic_0000001439470046_zh-cn_topic_0000001357504881_zh-cn_topic_0000001264921070_zh-cn_topic_0204328222_zh-cn_topic_0182636384_p13843256"></a>Tensor构造函数。</p>
</td>
</tr>
<tr id="row928317913314"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1271219612227"><a name="p1271219612227"></a><a name="p1271219612227"></a><a href="get_shape.md">get_shape</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001489189325_zh-cn_topic_0000001304385288_zh-cn_topic_0000001265240890_p12001237"><a name="zh-cn_topic_0000001489189325_zh-cn_topic_0000001304385288_zh-cn_topic_0000001265240890_p12001237"></a><a name="zh-cn_topic_0000001489189325_zh-cn_topic_0000001304385288_zh-cn_topic_0000001265240890_p12001237"></a>获取Tensor的Shape。</p>
</td>
</tr>
<tr id="row1128349123312"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1271036152214"><a name="p1271036152214"></a><a name="p1271036152214"></a><a href="get_data_type.md">get_data_type</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001439310794_zh-cn_topic_0000001304225448_zh-cn_topic_0000001265240878_p1589323202015"><a name="zh-cn_topic_0000001439310794_zh-cn_topic_0000001304225448_zh-cn_topic_0000001265240878_p1589323202015"></a><a name="zh-cn_topic_0000001439310794_zh-cn_topic_0000001304225448_zh-cn_topic_0000001265240878_p1589323202015"></a>获取Tensor中的数据类型。</p>
</td>
</tr>
<tr id="row728339183319"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p67099622217"><a name="p67099622217"></a><a name="p67099622217"></a><a href="get_data_size.md">get_data_size</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001438993202_zh-cn_topic_0000001357384993_zh-cn_topic_0000001265399802_p30971651"><a name="zh-cn_topic_0000001438993202_zh-cn_topic_0000001357384993_zh-cn_topic_0000001265399802_p30971651"></a><a name="zh-cn_topic_0000001438993202_zh-cn_topic_0000001357384993_zh-cn_topic_0000001265399802_p30971651"></a>获取Tensor中的数据大小。</p>
</td>
</tr>
<tr id="row12838993315"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p167071652220"><a name="p167071652220"></a><a name="p167071652220"></a><a href="get_element_cnt.md">get_element_cnt</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001439153154_zh-cn_topic_0000001357345205_p1390215711126"><a name="zh-cn_topic_0000001439153154_zh-cn_topic_0000001357345205_p1390215711126"></a><a name="zh-cn_topic_0000001439153154_zh-cn_topic_0000001357345205_p1390215711126"></a>获取Tensor中的元素的个数。</p>
</td>
</tr>
<tr id="row42831899338"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1170612619229"><a name="p1170612619229"></a><a name="p1170612619229"></a><a href="reshape.md">reshape</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p108882036192619"><a name="p108882036192619"></a><a name="p108882036192619"></a>对tensor进行Reshape操作，不改变tensor的内容。</p>
</td>
</tr>
</tbody>
</table>

## MetaParams类<a name="section20904915437"></a>

使用该类获取共享的变量信息。

**表 3**  MetaParams类接口

<a name="table16541353174018"></a>
<table><thead align="left"><tr id="row1754111532406"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p254245319403"><a name="p254245319403"></a><a name="p254245319403"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p15421253194012"><a name="p15421253194012"></a><a name="p15421253194012"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row19542253164018"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p3616243221"><a name="p3616243221"></a><a name="p3616243221"></a><a href="PyMetaParams构造函数.md">PyMetaParams构造函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001481244594_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_p13843256"><a name="zh-cn_topic_0000001481244594_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_p13843256"></a><a name="zh-cn_topic_0000001481244594_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_p13843256"></a>PyMetaParams构造函数。</p>
</td>
</tr>
<tr id="row8542553174015"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p12615114122211"><a name="p12615114122211"></a><a name="p12615114122211"></a><a href="get_name.md">get_name</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001481564190_zh-cn_topic_0000001304385296_p735218494917"><a name="zh-cn_topic_0000001481564190_zh-cn_topic_0000001304385296_p735218494917"></a><a name="zh-cn_topic_0000001481564190_zh-cn_topic_0000001304385296_p735218494917"></a>获取Flowfunc的实例名。</p>
</td>
</tr>
<tr id="row65421853174019"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1061484192218"><a name="p1061484192218"></a><a name="p1061484192218"></a><a href="get_attr_int.md">get_attr_int</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"><a name="zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a><a name="zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a>获取指定名称的int类型属性值。</p>
</td>
</tr>
<tr id="row1954265316403"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p2061224132210"><a name="p2061224132210"></a><a name="p2061224132210"></a><a href="get_attr_bool_list.md">get_attr_bool_list</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p1819913554347"><a name="p1819913554347"></a><a name="p1819913554347"></a>获取指定名称的bool数组类型属性值。</p>
</td>
</tr>
<tr id="row122631438103219"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p82631238143214"><a name="p82631238143214"></a><a name="p82631238143214"></a><a href="get_attr_int_list.md">get_attr_int_list</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p1964012063517"><a name="p1964012063517"></a><a name="p1964012063517"></a>获取指定名称的int数组类型属性值。</p>
</td>
</tr>
<tr id="row7263123814323"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p182636384324"><a name="p182636384324"></a><a name="p182636384324"></a><a href="get_attr_int_list_list.md">get_attr_int_list_list</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p34300518359"><a name="p34300518359"></a><a name="p34300518359"></a>获取指定名称的int二维数组类型属性值。</p>
</td>
</tr>
<tr id="row1326373823210"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p0264193817327"><a name="p0264193817327"></a><a name="p0264193817327"></a><a href="get_attr_bool.md">get_attr_bool</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p75781983519"><a name="p75781983519"></a><a name="p75781983519"></a>获取指定名称的bool类型属性值。</p>
</td>
</tr>
<tr id="row13264163833218"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p17264143812325"><a name="p17264143812325"></a><a name="p17264143812325"></a><a href="get_attr_float_list.md">get_attr_float_list</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p18802101343511"><a name="p18802101343511"></a><a name="p18802101343511"></a>获取指定名称的float数组类型属性值。</p>
</td>
</tr>
<tr id="row6542195344016"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1361112419224"><a name="p1361112419224"></a><a name="p1361112419224"></a><a href="get_attr_tensor_dtype.md">get_attr_tensor_dtype</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p18274218193511"><a name="p18274218193511"></a><a name="p18274218193511"></a>获取指定名称的numpy dtype类型的属性值。</p>
</td>
</tr>
<tr id="row1854212531406"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p861012411229"><a name="p861012411229"></a><a name="p861012411229"></a><a href="get_attr_tensor_dtype_list.md">get_attr_tensor_dtype_list</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p1487114239356"><a name="p1487114239356"></a><a name="p1487114239356"></a>获取指定名称的numpy dtype数组类型的属性值。</p>
</td>
</tr>
<tr id="row6903174214120"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p4608946229"><a name="p4608946229"></a><a name="p4608946229"></a><a href="get_attr_str.md">get_attr_str</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p64162918357"><a name="p64162918357"></a><a name="p64162918357"></a>获取指定名称的string类型的属性值。</p>
</td>
</tr>
<tr id="row157951318334"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p479533193315"><a name="p479533193315"></a><a name="p479533193315"></a><a href="get_attr_str_list.md">get_attr_str_list</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p1295673316354"><a name="p1295673316354"></a><a name="p1295673316354"></a>获取指定名称的string数组类型的属性值。</p>
</td>
</tr>
<tr id="row6795431173312"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p7795113113333"><a name="p7795113113333"></a><a name="p7795113113333"></a><a href="get_attr_float.md">get_attr_float</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p1810140143514"><a name="p1810140143514"></a><a name="p1810140143514"></a>获取指定名称的float类型属性值。</p>
</td>
</tr>
<tr id="row1479573143314"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p079543113315"><a name="p079543113315"></a><a name="p079543113315"></a><a href="get_input_num.md">get_input_num</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001481739176_zh-cn_topic_0000001532019181_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001481739176_zh-cn_topic_0000001532019181_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001481739176_zh-cn_topic_0000001532019181_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a>获取Flowfunc的输入个数。</p>
</td>
</tr>
<tr id="row13795183103319"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p137951531153313"><a name="p137951531153313"></a><a name="p137951531153313"></a><a href="get_output_num.md">get_output_num</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001481259728_zh-cn_topic_0000001532259553_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001481259728_zh-cn_topic_0000001532259553_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001481259728_zh-cn_topic_0000001532259553_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a>获取Flowfunc的输出个数。</p>
</td>
</tr>
<tr id="row102131227143312"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p3213182716337"><a name="p3213182716337"></a><a name="p3213182716337"></a><a href="get_work_path.md">get_work_path</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001532179937_zh-cn_topic_0000001532260413_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001532179937_zh-cn_topic_0000001532260413_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001532179937_zh-cn_topic_0000001532260413_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a>获取Flowfunc的工作路径。</p>
</td>
</tr>
<tr id="row172138279331"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p19213127133319"><a name="p19213127133319"></a><a name="p19213127133319"></a><a href="get_running_device_id.md">get_running_device_id</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001650406017_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001650406017_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001650406017_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a>获取正在运行的设备ID。</p>
</td>
</tr>
</tbody>
</table>

## MetaRunContext类<a name="section154853131118"></a>

用于FlowFunc处理函数的上下文信息相关处理，如申请Tensor、设置输出、运行FlowModel等操作。

**表 4**  MetaRunContext类接口

<a name="table126251824204317"></a>
<table><thead align="left"><tr id="row362582464314"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p86251124134316"><a name="p86251124134316"></a><a name="p86251124134316"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p4626202434316"><a name="p4626202434316"></a><a name="p4626202434316"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row17626152444310"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p122037242217"><a name="p122037242217"></a><a name="p122037242217"></a><a href="MetaRunContext构造函数.md">MetaRunContext构造函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001481724054_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_p13843256"><a name="zh-cn_topic_0000001481724054_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_p13843256"></a><a name="zh-cn_topic_0000001481724054_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_p13843256"></a>MetaRunContext构造函数。</p>
</td>
</tr>
<tr id="row186260242438"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1520214213227"><a name="p1520214213227"></a><a name="p1520214213227"></a><a href="alloc_tensor_msg.md">alloc_tensor_msg</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_p1476471614810"><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_p1476471614810"></a><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_p1476471614810"></a>根据shape、data type以及对齐大小申请tensor类型的FlowMsg。</p>
</td>
</tr>
<tr id="row1062682454317"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1820162162217"><a name="p1820162162217"></a><a name="p1820162162217"></a><a href="set_output.md">set_output</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312640889_p4089365"><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312640889_p4089365"></a><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312640889_p4089365"></a>设置指定index的output的tensor。</p>
</td>
</tr>
<tr id="row186271424104316"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p31991122221"><a name="p31991122221"></a><a name="p31991122221"></a><a href="set_multi_outputs.md">set_multi_outputs</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p191251316143713"><a name="p191251316143713"></a><a name="p191251316143713"></a>批量设置指定index的output的tensor。</p>
</td>
</tr>
<tr id="row206281724164310"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p21981127225"><a name="p21981127225"></a><a name="p21981127225"></a><a href="run_flow_model.md">run_flow_model</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a>同步执行指定的模型。</p>
</td>
</tr>
<tr id="row9628192414317"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p20197152102218"><a name="p20197152102218"></a><a name="p20197152102218"></a><a href="alloc_empty_data_msg.md">alloc_empty_data_msg</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001481728758_p1281314307502"><a name="zh-cn_topic_0000001481728758_p1281314307502"></a><a name="zh-cn_topic_0000001481728758_p1281314307502"></a>申请空数据的MsgType类型的message。</p>
</td>
</tr>
<tr id="row462932418439"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p161961124223"><a name="p161961124223"></a><a name="p161961124223"></a><a href="get_user_data（UDF）.md">get_user_data（UDF）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p1933933218373"><a name="p1933933218373"></a><a name="p1933933218373"></a>获取用户定义数据。</p>
</td>
</tr>
<tr id="row1659416406543"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1594640115410"><a name="p1594640115410"></a><a name="p1594640115410"></a><a href="raise_exception.md">raise_exception</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p6678174395611"><a name="p6678174395611"></a><a name="p6678174395611"></a>UDF主动上报异常。</p>
</td>
</tr>
<tr id="row15235134245418"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p12353427545"><a name="p12353427545"></a><a name="p12353427545"></a><a href="get_exception.md">get_exception</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p1873611562563"><a name="p1873611562563"></a><a name="p1873611562563"></a>UDF获取其他UDF节点上报的异常。</p>
</td>
</tr>
<tr id="row1452817293328"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1952882911322"><a name="p1952882911322"></a><a name="p1952882911322"></a><a href="alloc_raw_data_msg.md">alloc_raw_data_msg</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p554313472329"><a name="p554313472329"></a><a name="p554313472329"></a>根据输入的size申请一块连续内存，用于承载raw data类型的FlowMsg。</p>
</td>
</tr>
<tr id="row10417103123215"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p141753113326"><a name="p141753113326"></a><a name="p141753113326"></a><a href="to_flow_msg.md">to_flow_msg</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p8331754183218"><a name="p8331754183218"></a><a name="p8331754183218"></a>将dataflow Tensor转换成FlowMsg。</p>
</td>
</tr>
</tbody>
</table>

## AffinityPolicy类<a name="section201762039162118"></a>

**表 5**  AffinityPolicy类接口

<a name="table9851722202117"></a>
<table><thead align="left"><tr id="row18520229210"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p138510224217"><a name="p138510224217"></a><a name="p138510224217"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p48582213211"><a name="p48582213211"></a><a name="p48582213211"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row28515227214"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p579865942114"><a name="p579865942114"></a><a name="p579865942114"></a><a href="AffinityPolicy类.md">AffinityPolicy类</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p14797175920215"><a name="p14797175920215"></a><a name="p14797175920215"></a>亲和策略枚举定义。</p>
</td>
</tr>
</tbody>
</table>

## BalanceConfig类<a name="section9368163171611"></a>

当需要均衡分发时，需要设置输出数据标识和权重矩阵相关配置信息，根据配置调度模块可以完成多实例之间的均衡分发。

**表 6**  BalanceConfig类接口

<a name="table92314512441"></a>
<table><thead align="left"><tr id="row192312517445"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p6238594413"><a name="p6238594413"></a><a name="p6238594413"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p62310511442"><a name="p62310511442"></a><a name="p62310511442"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row19241656443"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p103801852192116"><a name="p103801852192116"></a><a name="p103801852192116"></a><a href="BalanceConfig构造函数.md">BalanceConfig构造函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p628113114387"><a name="p628113114387"></a><a name="p628113114387"></a>BalanceConfig构造函数。</p>
</td>
</tr>
<tr id="row122416554413"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p103783528214"><a name="p103783528214"></a><a name="p103783528214"></a><a href="set_data_pos.md">set_data_pos</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p7489141573820"><a name="p7489141573820"></a><a name="p7489141573820"></a>设置输出数据对应权重矩阵中的位置。</p>
</td>
</tr>
<tr id="row1424758443"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p7376195252110"><a name="p7376195252110"></a><a name="p7376195252110"></a><a href="get_inner_config.md">get_inner_config</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p11333122433817"><a name="p11333122433817"></a><a name="p11333122433817"></a>获取内部配置对象，被<a href="set_output.md">set_output</a>或者<a href="set_multi_outputs.md">set_multi_outputs</a>调用。</p>
</td>
</tr>
</tbody>
</table>

## FlowMsgQueue类<a name="section211018485275"></a>

流式输入场景下（即flow func函数入参为队列时），用于flow func的输入队列，队列中的FlowMsg出队后会根据MsgType转换为对应的数据类型返回给用户。

**表 7**  FlowMsgQueue类接口

<a name="table51101048112710"></a>
<table><thead align="left"><tr id="row1411084832715"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p14110848162719"><a name="p14110848162719"></a><a name="p14110848162719"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p911014810271"><a name="p911014810271"></a><a name="p911014810271"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row91101148172713"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p91804131286"><a name="p91804131286"></a><a name="p91804131286"></a><a href="FlowMsgQueue构造函数.md">FlowMsgQueue构造函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p16831741172914"><a name="p16831741172914"></a><a name="p16831741172914"></a>FlowMsgQueue构造函数和析构函数。</p>
</td>
</tr>
<tr id="row61100485274"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p417981319280"><a name="p417981319280"></a><a name="p417981319280"></a><a href="get.md">get</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p19392144813299"><a name="p19392144813299"></a><a name="p19392144813299"></a>获取队列中的元素。</p>
</td>
</tr>
<tr id="row756285622811"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p317791320286"><a name="p317791320286"></a><a name="p317791320286"></a><a href="get_nowait.md">get_nowait</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p393391683017"><a name="p393391683017"></a><a name="p393391683017"></a>无等待地获取队列中的元素，功能等同于get(block=False)。</p>
</td>
</tr>
<tr id="row1311014489273"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p267718318295"><a name="p267718318295"></a><a name="p267718318295"></a><a href="full.md">full</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p992182317308"><a name="p992182317308"></a><a name="p992182317308"></a>判断队列是否满。</p>
</td>
</tr>
<tr id="row1783515415283"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p2836185482812"><a name="p2836185482812"></a><a name="p2836185482812"></a><a href="full.md">full</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p182512304306"><a name="p182512304306"></a><a name="p182512304306"></a>判断队列是否为空。</p>
</td>
</tr>
<tr id="row8856052917"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p18590122915"><a name="p18590122915"></a><a name="p18590122915"></a><a href="qsize.md">qsize</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p15411036193014"><a name="p15411036193014"></a><a name="p15411036193014"></a>获取队列中当前元素的个数。</p>
</td>
</tr>
</tbody>
</table>

## UDF日志接口<a name="section663155413194"></a>

UDF Python开放了日志记录接口，使用时导入flow\_func模块。使用其中定义的logger对象，调用logger对象封装的不同级别的日志接口。

**表 8**  UDF日志接口

<a name="table675418912447"></a>
<table><thead align="left"><tr id="row167548918448"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p17541190448"><a name="p17541190448"></a><a name="p17541190448"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p27541799441"><a name="p27541799441"></a><a name="p27541799441"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row17541493448"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p871413544216"><a name="p871413544216"></a><a name="p871413544216"></a><a href="FlowFuncLogger构造函数.md">FlowFuncLogger构造函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001567398973_zh-cn_topic_0000001304225452_p1476471614810"><a name="zh-cn_topic_0000001567398973_zh-cn_topic_0000001304225452_p1476471614810"></a><a name="zh-cn_topic_0000001567398973_zh-cn_topic_0000001304225452_p1476471614810"></a>FlowFuncLogger构造函数。</p>
</td>
</tr>
<tr id="row37541493449"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p137126546214"><a name="p137126546214"></a><a name="p137126546214"></a><a href="get_log_header.md">get_log_header</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001516238800_zh-cn_topic_0000001304225452_p1476471614810"><a name="zh-cn_topic_0000001516238800_zh-cn_topic_0000001304225452_p1476471614810"></a><a name="zh-cn_topic_0000001516238800_zh-cn_topic_0000001304225452_p1476471614810"></a>获取日志扩展头信息。</p>
</td>
</tr>
<tr id="row57541290441"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p12711854192111"><a name="p12711854192111"></a><a name="p12711854192111"></a><a href="is_log_enable.md">is_log_enable</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_p1476471614810"><a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_p1476471614810"></a><a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_p1476471614810"></a>查询对应级别和类型的日志是否开启。</p>
</td>
</tr>
<tr id="row6337191318513"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p37101054152115"><a name="p37101054152115"></a><a name="p37101054152115"></a><a href="运行日志Error级别日志宏.md">运行日志Error级别日志宏</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001567398977_zh-cn_topic_0000001307722046_p179521451181018"><a name="zh-cn_topic_0000001567398977_zh-cn_topic_0000001307722046_p179521451181018"></a><a name="zh-cn_topic_0000001567398977_zh-cn_topic_0000001307722046_p179521451181018"></a>运行日志Error级别日志宏。</p>
</td>
</tr>
<tr id="row10337213355"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p970935415213"><a name="p970935415213"></a><a name="p970935415213"></a><a href="运行日志Info级别日志宏.md">运行日志Info级别日志宏</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001567079081_zh-cn_topic_0000001307722046_p179521451181018"><a name="zh-cn_topic_0000001567079081_zh-cn_topic_0000001307722046_p179521451181018"></a><a name="zh-cn_topic_0000001567079081_zh-cn_topic_0000001307722046_p179521451181018"></a>运行日志Info级别日志宏。</p>
</td>
</tr>
<tr id="row1233712131556"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p27084548213"><a name="p27084548213"></a><a name="p27084548213"></a><a href="调试日志Error级别日志宏.md">调试日志Error级别日志宏</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001516238804_zh-cn_topic_0000001307722046_p179521451181018"><a name="zh-cn_topic_0000001516238804_zh-cn_topic_0000001307722046_p179521451181018"></a><a name="zh-cn_topic_0000001516238804_zh-cn_topic_0000001307722046_p179521451181018"></a>调试日志Error级别日志宏。</p>
</td>
</tr>
<tr id="row13337413350"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p107071954122114"><a name="p107071954122114"></a><a name="p107071954122114"></a><a href="调试日志Warn级别日志宏.md">调试日志Warn级别日志宏</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001516398768_zh-cn_topic_0000001307722046_p179521451181018"><a name="zh-cn_topic_0000001516398768_zh-cn_topic_0000001307722046_p179521451181018"></a><a name="zh-cn_topic_0000001516398768_zh-cn_topic_0000001307722046_p179521451181018"></a>调试日志Warn级别日志宏。</p>
</td>
</tr>
<tr id="row1333715134519"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p570635419213"><a name="p570635419213"></a><a name="p570635419213"></a><a href="调试日志Info级别日志宏.md">调试日志Info级别日志宏</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001567198809_zh-cn_topic_0000001307722046_p179521451181018"><a name="zh-cn_topic_0000001567198809_zh-cn_topic_0000001307722046_p179521451181018"></a><a name="zh-cn_topic_0000001567198809_zh-cn_topic_0000001307722046_p179521451181018"></a>调试日志Info级别日志宏。</p>
</td>
</tr>
<tr id="row87552964420"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p117040547217"><a name="p117040547217"></a><a name="p117040547217"></a><a href="调试日志Debug级别日志宏.md">调试日志Debug级别日志宏</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001516692242_zh-cn_topic_0000001307722046_p179521451181018"><a name="zh-cn_topic_0000001516692242_zh-cn_topic_0000001307722046_p179521451181018"></a><a name="zh-cn_topic_0000001516692242_zh-cn_topic_0000001307722046_p179521451181018"></a>调试日志Debug级别日志宏。</p>
</td>
</tr>
</tbody>
</table>

