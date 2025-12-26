# DataFlow接口列表<a name="ZH-CN_TOPIC_0000001985738776"></a>

您可以在“$\{INSTALL\_DIR\}/include/flow\_graph”路径下查看对应接口的头文件。

$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。若安装的Ascend-cann-toolkit软件包，以root安装举例，则安装后文件存储路径为：/usr/local/Ascend/ascend-toolkit/latest。

<a name="zh-cn_topic_0000001411352680_zh-cn_topic_0000001312720969_table6861546172820"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001411352680_zh-cn_topic_0000001312720969_row986112467281"><th class="cellrowborder" valign="top" width="25.070000000000004%" id="mcps1.1.5.1.1"><p id="zh-cn_topic_0000001411352680_zh-cn_topic_0000001312720969_p3861046142817"><a name="zh-cn_topic_0000001411352680_zh-cn_topic_0000001312720969_p3861046142817"></a><a name="zh-cn_topic_0000001411352680_zh-cn_topic_0000001312720969_p3861046142817"></a>接口分类</p>
</th>
<th class="cellrowborder" valign="top" width="22.540000000000003%" id="mcps1.1.5.1.2"><p id="zh-cn_topic_0000001411352680_zh-cn_topic_0000001312720969_p17861134612813"><a name="zh-cn_topic_0000001411352680_zh-cn_topic_0000001312720969_p17861134612813"></a><a name="zh-cn_topic_0000001411352680_zh-cn_topic_0000001312720969_p17861134612813"></a>头文件路径</p>
</th>
<th class="cellrowborder" valign="top" width="28.870000000000008%" id="mcps1.1.5.1.3"><p id="p79991615201413"><a name="p79991615201413"></a><a name="p79991615201413"></a>简介</p>
</th>
<th class="cellrowborder" valign="top" width="23.520000000000003%" id="mcps1.1.5.1.4"><p id="p16862164158"><a name="p16862164158"></a><a name="p16862164158"></a>对应的库文件</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001411352680_row1223525510164"><td class="cellrowborder" valign="top" width="25.070000000000004%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001411352680_p6235105561612"><a name="zh-cn_topic_0000001411352680_p6235105561612"></a><a name="zh-cn_topic_0000001411352680_p6235105561612"></a>FlowOperator类</p>
</td>
<td class="cellrowborder" valign="top" width="22.540000000000003%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001411352680_p62351055111616"><a name="zh-cn_topic_0000001411352680_p62351055111616"></a><a name="zh-cn_topic_0000001411352680_p62351055111616"></a>flow_graph.h</p>
</td>
<td class="cellrowborder" valign="top" width="28.870000000000008%" headers="mcps1.1.5.1.3 "><p id="p599941521410"><a name="p599941521410"></a><a name="p599941521410"></a>FlowOperator类是DataFlow Graph的节点基类，继承于GE的Operator。不支持在外部单独构造使用。</p>
</td>
<td class="cellrowborder" valign="top" width="23.520000000000003%" headers="mcps1.1.5.1.4 "><p id="p38661619153"><a name="p38661619153"></a><a name="p38661619153"></a>NA</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352680_zh-cn_topic_0000001312720969_row17163247142218"><td class="cellrowborder" valign="top" width="25.070000000000004%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001411352680_p17523155924311"><a name="zh-cn_topic_0000001411352680_p17523155924311"></a><a name="zh-cn_topic_0000001411352680_p17523155924311"></a>FlowData类</p>
</td>
<td class="cellrowborder" valign="top" width="22.540000000000003%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001411352680_p590394814320"><a name="zh-cn_topic_0000001411352680_p590394814320"></a><a name="zh-cn_topic_0000001411352680_p590394814320"></a>flow_graph.h</p>
</td>
<td class="cellrowborder" valign="top" width="28.870000000000008%" headers="mcps1.1.5.1.3 "><p id="p1099914153141"><a name="p1099914153141"></a><a name="p1099914153141"></a>继承于FlowOperator类，为DataFlow Graph中的数据节点，每个FlowData对应一个输入。</p>
</td>
<td class="cellrowborder" valign="top" width="23.520000000000003%" headers="mcps1.1.5.1.4 "><p id="p486121661518"><a name="p486121661518"></a><a name="p486121661518"></a>libflow_graph.so</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352680_zh-cn_topic_0000001312720969_row208611046142817"><td class="cellrowborder" valign="top" width="25.070000000000004%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001411352680_p052219592434"><a name="zh-cn_topic_0000001411352680_p052219592434"></a><a name="zh-cn_topic_0000001411352680_p052219592434"></a>FlowNode类</p>
</td>
<td class="cellrowborder" valign="top" width="22.540000000000003%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001411352680_p2169202317525"><a name="zh-cn_topic_0000001411352680_p2169202317525"></a><a name="zh-cn_topic_0000001411352680_p2169202317525"></a>flow_graph.h</p>
</td>
<td class="cellrowborder" valign="top" width="28.870000000000008%" headers="mcps1.1.5.1.3 "><p id="p19999131521415"><a name="p19999131521415"></a><a name="p19999131521415"></a>继承于FlowOperator类，DataFlow Graph中的计算节点。</p>
</td>
<td class="cellrowborder" valign="top" width="23.520000000000003%" headers="mcps1.1.5.1.4 "><p id="p7861516121518"><a name="p7861516121518"></a><a name="p7861516121518"></a>libflow_graph.so</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352680_row205662013717"><td class="cellrowborder" valign="top" width="25.070000000000004%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001411352680_p11521125954319"><a name="zh-cn_topic_0000001411352680_p11521125954319"></a><a name="zh-cn_topic_0000001411352680_p11521125954319"></a>FlowGraph类</p>
</td>
<td class="cellrowborder" valign="top" width="22.540000000000003%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001411352680_p1861462325212"><a name="zh-cn_topic_0000001411352680_p1861462325212"></a><a name="zh-cn_topic_0000001411352680_p1861462325212"></a>flow_graph.h</p>
</td>
<td class="cellrowborder" valign="top" width="28.870000000000008%" headers="mcps1.1.5.1.3 "><p id="p2999191591415"><a name="p2999191591415"></a><a name="p2999191591415"></a>DataFlow的graph，由输入节点FlowData和计算节点FlowNode构成。</p>
</td>
<td class="cellrowborder" valign="top" width="23.520000000000003%" headers="mcps1.1.5.1.4 "><p id="p178681691517"><a name="p178681691517"></a><a name="p178681691517"></a>libflow_graph.so</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352680_row161411167314"><td class="cellrowborder" valign="top" width="25.070000000000004%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001411352680_p3141416193110"><a name="zh-cn_topic_0000001411352680_p3141416193110"></a><a name="zh-cn_topic_0000001411352680_p3141416193110"></a>ProcessPoint类</p>
</td>
<td class="cellrowborder" valign="top" width="22.540000000000003%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001411352680_p101514161314"><a name="zh-cn_topic_0000001411352680_p101514161314"></a><a name="zh-cn_topic_0000001411352680_p101514161314"></a>process_point.h</p>
</td>
<td class="cellrowborder" valign="top" width="28.870000000000008%" headers="mcps1.1.5.1.3 "><p id="p170161661420"><a name="p170161661420"></a><a name="p170161661420"></a>ProcessPoint是一个虚基类，无法实例化对象。</p>
</td>
<td class="cellrowborder" valign="top" width="23.520000000000003%" headers="mcps1.1.5.1.4 "><p id="p11863168156"><a name="p11863168156"></a><a name="p11863168156"></a>NA</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352680_zh-cn_topic_0000001312720969_row11861146142810"><td class="cellrowborder" valign="top" width="25.070000000000004%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001411352680_p115208597436"><a name="zh-cn_topic_0000001411352680_p115208597436"></a><a name="zh-cn_topic_0000001411352680_p115208597436"></a>FunctionPp类</p>
</td>
<td class="cellrowborder" valign="top" width="22.540000000000003%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001411352680_p1190194824320"><a name="zh-cn_topic_0000001411352680_p1190194824320"></a><a name="zh-cn_topic_0000001411352680_p1190194824320"></a>process_point.h</p>
</td>
<td class="cellrowborder" valign="top" width="28.870000000000008%" headers="mcps1.1.5.1.3 "><p id="p12061631417"><a name="p12061631417"></a><a name="p12061631417"></a>继承于<a href="ProcessPoint类.md">ProcessPoint类</a>，用来表示Function的计算处理点。</p>
</td>
<td class="cellrowborder" valign="top" width="23.520000000000003%" headers="mcps1.1.5.1.4 "><p id="p98661618150"><a name="p98661618150"></a><a name="p98661618150"></a>libflow_graph.so</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352680_zh-cn_topic_0000001312720969_row2903258114615"><td class="cellrowborder" valign="top" width="25.070000000000004%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001411352680_p12519159104319"><a name="zh-cn_topic_0000001411352680_p12519159104319"></a><a name="zh-cn_topic_0000001411352680_p12519159104319"></a>GraphPp类</p>
</td>
<td class="cellrowborder" valign="top" width="22.540000000000003%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001411352680_p910942865312"><a name="zh-cn_topic_0000001411352680_p910942865312"></a><a name="zh-cn_topic_0000001411352680_p910942865312"></a>process_point.h</p>
</td>
<td class="cellrowborder" valign="top" width="28.870000000000008%" headers="mcps1.1.5.1.3 "><p id="p1001612141"><a name="p1001612141"></a><a name="p1001612141"></a>继承自<a href="ProcessPoint类.md">ProcessPoint类</a>，用来表示Graph的计算处理点。</p>
</td>
<td class="cellrowborder" valign="top" width="23.520000000000003%" headers="mcps1.1.5.1.4 "><p id="p168641601517"><a name="p168641601517"></a><a name="p168641601517"></a>libflow_graph.so</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352680_row56011573265"><td class="cellrowborder" valign="top" width="25.070000000000004%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001411352680_p125182599437"><a name="zh-cn_topic_0000001411352680_p125182599437"></a><a name="zh-cn_topic_0000001411352680_p125182599437"></a>DataFlowInputAttr结构体</p>
</td>
<td class="cellrowborder" valign="top" width="22.540000000000003%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001411352680_p129003482434"><a name="zh-cn_topic_0000001411352680_p129003482434"></a><a name="zh-cn_topic_0000001411352680_p129003482434"></a>flow_attr.h</p>
</td>
<td class="cellrowborder" valign="top" width="28.870000000000008%" headers="mcps1.1.5.1.3 "><p id="p50161671413"><a name="p50161671413"></a><a name="p50161671413"></a>定义timeBatch和countBatch两种功能实现UDF组batch能力。</p>
</td>
<td class="cellrowborder" valign="top" width="23.520000000000003%" headers="mcps1.1.5.1.4 "><p id="p168651614151"><a name="p168651614151"></a><a name="p168651614151"></a>libflow_graph.so</p>
</td>
</tr>
</tbody>
</table>

>![](public_sys-resources/icon-note.gif) **说明：** 
>头文件中出现的char\_t类型是char类型的别名。

## DataFlow构图接口<a name="section32413319013"></a>

**表 1**  DataFlow构图接口

<a name="table1644114511445"></a>
<table><thead align="left"><tr id="row184417511246"><th class="cellrowborder" valign="top" width="33.47%" id="mcps1.2.3.1.1"><p id="p6441125120410"><a name="p6441125120410"></a><a name="p6441125120410"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.53%" id="mcps1.2.3.1.2"><p id="p1144117511349"><a name="p1144117511349"></a><a name="p1144117511349"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row17441125116414"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1667729113212"><a name="p1667729113212"></a><a name="p1667729113212"></a><a href="FlowOperator类.md">FlowOperator类</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p7671529123218"><a name="p7671529123218"></a><a name="p7671529123218"></a>FlowOperator类是DataFlow Graph的节点基类，继承于GE的Operator。</p>
</td>
</tr>
<tr id="row944214511419"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p16661729143215"><a name="p16661729143215"></a><a name="p16661729143215"></a><a href="FlowData的构造函数和析构函数.md">FlowData的构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001461072793_p157993218525"><a name="zh-cn_topic_0000001461072793_p157993218525"></a><a name="zh-cn_topic_0000001461072793_p157993218525"></a>FlowData构造函数和析构函数，构造函数会返回一个FlowData节点。</p>
</td>
</tr>
<tr id="row12318757162"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p18651829123218"><a name="p18651829123218"></a><a name="p18651829123218"></a><a href="FlowNode构造函数和析构函数.md">FlowNode构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001461312437_p067953535319"><a name="zh-cn_topic_0000001461312437_p067953535319"></a><a name="zh-cn_topic_0000001461312437_p067953535319"></a>FlowNode构造函数和析构函数，构造函数返回一个FlowNode节点。</p>
</td>
</tr>
<tr id="row1231816571467"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p19631329163218"><a name="p19631329163218"></a><a name="p19631329163218"></a><a href="SetInput.md">SetInput</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001411032868_p19118164419546"><a name="zh-cn_topic_0000001411032868_p19118164419546"></a><a name="zh-cn_topic_0000001411032868_p19118164419546"></a>给FlowNode设置输入，表示将src_op的第src_index个输出作为FlowNode的第dst_index个输入，返回设置好输入的FlowNode节点。</p>
</td>
</tr>
<tr id="row931813574614"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p562829163210"><a name="p562829163210"></a><a name="p562829163210"></a><a href="AddPp.md">AddPp</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001411352684_p15227121203"><a name="zh-cn_topic_0000001411352684_p15227121203"></a><a name="zh-cn_topic_0000001411352684_p15227121203"></a>给FlowNode添加映射的ProcessPoint，当前一个FlowNode仅能添加一个ProcessPoint，添加后会默认将FlowNode的输入输出和ProcessPoint的输入输出按顺序进行映射。</p>
</td>
</tr>
<tr id="row469180173913"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p126911003393"><a name="p126911003393"></a><a name="p126911003393"></a><a href="MapInput.md">MapInput</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001461192589_p1339913564556"><a name="zh-cn_topic_0000001461192589_p1339913564556"></a><a name="zh-cn_topic_0000001461192589_p1339913564556"></a>给FlowNode映射输入，表示将FlowNode的第node_input_index个输入给到ProcessPoint的第pp_input_index个输入，并且给ProcessPoint的该输入设置上attrs里的所有属性，返回映射好的FlowNode节点。该函数可选，不被调用时会默认按顺序映射FlowNode和ProcessPoint的输入。</p>
</td>
</tr>
<tr id="row1692402398"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p16692130123916"><a name="p16692130123916"></a><a name="p16692130123916"></a><a href="MapOutput.md">MapOutput</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001461432453_p18295932155813"><a name="zh-cn_topic_0000001461432453_p18295932155813"></a><a name="zh-cn_topic_0000001461432453_p18295932155813"></a>给FlowNode映射输出，表示将ProcessPoint的第pp_output_index个输出给到FlowNode的第node_output_index个输出，返回映射好的FlowNode节点。</p>
</td>
</tr>
<tr id="row857616411814"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p195763411611"><a name="p195763411611"></a><a name="p195763411611"></a><a href="SetBalanceScatter.md">SetBalanceScatter</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p162061576213"><a name="p162061576213"></a><a name="p162061576213"></a>设置节点balance scatter属性，具有balance scatter属性的UDF可以使用balance options设置负载均衡输出。</p>
</td>
</tr>
<tr id="row5692203397"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p196927019398"><a name="p196927019398"></a><a name="p196927019398"></a><a href="SetBalanceGather.md">SetBalanceGather</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1119214395274"><a name="p1119214395274"></a><a name="p1119214395274"></a>设置节点balance gather属性，具有balance gather属性的UDF可以使用balance options设置负载均衡亲和输出。</p>
</td>
</tr>
<tr id="row1669270203910"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p86922015396"><a name="p86922015396"></a><a name="p86922015396"></a><a href="FlowGraph构造函数和析构函数.md">FlowGraph构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001461072797_p1648418251426"><a name="zh-cn_topic_0000001461072797_p1648418251426"></a><a name="zh-cn_topic_0000001461072797_p1648418251426"></a>FlowGraph构造函数和析构函数，构造函数会返回一张空的FlowGraph图。</p>
</td>
</tr>
<tr id="row46921301396"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1869260183916"><a name="p1869260183916"></a><a name="p1869260183916"></a><a href="SetInputs.md">SetInputs</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001410872980_p290817358314"><a name="zh-cn_topic_0000001410872980_p290817358314"></a><a name="zh-cn_topic_0000001410872980_p290817358314"></a>设置FlowGraph的输入节点，会自动根据节点的输出连接关系构建出一张FlowGraph图，并返回该图。</p>
</td>
</tr>
<tr id="row16692150183910"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p569216023917"><a name="p569216023917"></a><a name="p569216023917"></a><a href="SetOutputs.md">SetOutputs</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001461312441_p290817358314"><a name="zh-cn_topic_0000001461312441_p290817358314"></a><a name="zh-cn_topic_0000001461312441_p290817358314"></a>设置FlowGraph的输出节点，并返回该图。</p>
</td>
</tr>
<tr id="row7692140153917"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p86922023917"><a name="p86922023917"></a><a name="p86922023917"></a><a href="SetOutputs（index）.md">SetOutputs（index）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p11193825131520"><a name="p11193825131520"></a><a name="p11193825131520"></a>设置FlowGraph中的FlowNode和FlowNode输出index的关联关系，并返回该图。常用于设置FlowNode部分输出场景，比如FlowNode1有2个输出，但是作为FlowNode2输入的时候只需要FlowNode1的一个输出，这种情况下可以设置FlowNode1的一个输出index。</p>
</td>
</tr>
<tr id="row1169220011394"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p186921503393"><a name="p186921503393"></a><a name="p186921503393"></a><a href="SetContainsNMappingNode.md">SetContainsNMappingNode</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p169214083914"><a name="p169214083914"></a><a name="p169214083914"></a>设置FlowGraph是否包含n_mapping节点。</p>
</td>
</tr>
<tr id="row12692130113916"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p19692120153910"><a name="p19692120153910"></a><a name="p19692120153910"></a><a href="SetInputsAlignAttrs.md">SetInputsAlignAttrs</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p636317219415"><a name="p636317219415"></a><a name="p636317219415"></a>设置FlowGraph中的输入对齐属性。</p>
</td>
</tr>
<tr id="row1669211043917"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p56921207395"><a name="p56921207395"></a><a name="p56921207395"></a><a href="const-ge-Graph-ToGeGraph()-const.md">const ge::Graph &amp;ToGeGraph() const</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p96923043910"><a name="p96923043910"></a><a name="p96923043910"></a>将FlowGraph转换到GE的Graph。</p>
</td>
</tr>
<tr id="row1469311014398"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p469319003914"><a name="p469319003914"></a><a name="p469319003914"></a><a href="SetGraphPpBuilderAsync.md">SetGraphPpBuilderAsync</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1996919196418"><a name="p1996919196418"></a><a name="p1996919196418"></a>设置FlowGraph中的GraphPp的Builder是否异步执行。</p>
</td>
</tr>
<tr id="row1320794075518"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p14207154055519"><a name="p14207154055519"></a><a name="p14207154055519"></a><a href="SetExceptionCatch.md">SetExceptionCatch</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1490417496347"><a name="p1490417496347"></a><a name="p1490417496347"></a>设置用户异常捕获功能是否开启。</p>
</td>
</tr>
<tr id="row1869320073914"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p116931302393"><a name="p116931302393"></a><a name="p116931302393"></a><a href="ProcessPoint析构函数.md">ProcessPoint析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001434536460_zh-cn_topic_0204328224_zh-cn_topic_0182636384_p13843256"><a name="zh-cn_topic_0000001434536460_zh-cn_topic_0204328224_zh-cn_topic_0182636384_p13843256"></a><a name="zh-cn_topic_0000001434536460_zh-cn_topic_0204328224_zh-cn_topic_0182636384_p13843256"></a>ProcessPoint析构函数。</p>
</td>
</tr>
<tr id="row13693170103919"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p206931306394"><a name="p206931306394"></a><a name="p206931306394"></a><a href="GetProcessPointType.md">GetProcessPointType</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001484652277_p192271624587"><a name="zh-cn_topic_0000001484652277_p192271624587"></a><a name="zh-cn_topic_0000001484652277_p192271624587"></a>获取ProcessPoint的类型。</p>
</td>
</tr>
<tr id="row11693404394"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p0693170153912"><a name="p0693170153912"></a><a name="p0693170153912"></a><a href="GetProcessPointName.md">GetProcessPointName</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001435012410_p192271624587"><a name="zh-cn_topic_0000001435012410_p192271624587"></a><a name="zh-cn_topic_0000001435012410_p192271624587"></a>获取ProcessPoint的名称。</p>
</td>
</tr>
<tr id="row183321753183816"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p133331653133817"><a name="p133331653133817"></a><a name="p133331653133817"></a><a href="GetCompileConfig.md">GetCompileConfig</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001484611753_p192271624587"><a name="zh-cn_topic_0000001484611753_p192271624587"></a><a name="zh-cn_topic_0000001484611753_p192271624587"></a>获取ProcessPoint编译配置的文件。</p>
</td>
</tr>
<tr id="row333385333820"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p19333153163812"><a name="p19333153163812"></a><a name="p19333153163812"></a><a href="Serialize（ProcessPoint类）.md">Serialize（ProcessPoint类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001434853134_p16972182814218"><a name="zh-cn_topic_0000001434853134_p16972182814218"></a><a name="zh-cn_topic_0000001434853134_p16972182814218"></a>ProcessPoint的序列化方法。由ProcessPoint的子类去实现该方法的功能。</p>
</td>
</tr>
<tr id="row133355310381"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p43338535386"><a name="p43338535386"></a><a name="p43338535386"></a><a href="FunctionPp构造函数和析构函数.md">FunctionPp构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001461432457_p192271624587"><a name="zh-cn_topic_0000001461432457_p192271624587"></a><a name="zh-cn_topic_0000001461432457_p192271624587"></a>FunctionPp的构造函数和析构函数，构造函数会返回一个FunctionPp对象。</p>
</td>
</tr>
<tr id="row033313531380"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p333375310386"><a name="p333375310386"></a><a name="p333375310386"></a><a href="SetCompileConfig（FunctionPp类）.md">SetCompileConfig（FunctionPp类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001411352688_p20264558531"><a name="zh-cn_topic_0000001411352688_p20264558531"></a><a name="zh-cn_topic_0000001411352688_p20264558531"></a>设置FunctionPp的json配置文件名字和路径，该配置文件用于将FunctionPp和UDF进行映射。</p>
</td>
</tr>
<tr id="row633365343815"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p11333175303814"><a name="p11333175303814"></a><a name="p11333175303814"></a><a href="AddInvokedClosure-(添加调用的GraphPp).md">AddInvokedClosure (添加调用的GraphPp)</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001411192724_p1532416920"><a name="zh-cn_topic_0000001411192724_p1532416920"></a><a name="zh-cn_topic_0000001411192724_p1532416920"></a>添加FunctionPp调用的GraphPp，返回添加好的FunctionPp。</p>
</td>
</tr>
<tr id="row10333253143819"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p19333453153817"><a name="p19333453153817"></a><a name="p19333453153817"></a><a href="AddInvokedClosure-(添加调用的ProcessPoint子类).md">AddInvokedClosure (添加调用的ProcessPoint子类)</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p32183124511"><a name="p32183124511"></a><a name="p32183124511"></a>添加FunctionPp调用的GraphPp，返回添加好的FunctionPp。</p>
</td>
</tr>
<tr id="row385601514916"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1085641510491"><a name="p1085641510491"></a><a name="p1085641510491"></a><a href="AddInvokedClosure-(添加调用的FlowGraphPp).md">AddInvokedClosure (添加调用的FlowGraphPp)</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p089154314494"><a name="p089154314494"></a><a name="p089154314494"></a>添加FunctionPp调用的FlowGraphPp，返回添加好的FunctionPp。</p>
</td>
</tr>
<tr id="row12333175313380"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p933345312382"><a name="p933345312382"></a><a name="p933345312382"></a><a href="SetInitParam.md">SetInitParam</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001461072801_p2793201513"><a name="zh-cn_topic_0000001461072801_p2793201513"></a><a name="zh-cn_topic_0000001461072801_p2793201513"></a>设置FunctionPp的初始化参数，返回设置好的FunctionPp。</p>
</td>
</tr>
<tr id="row15318857869"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p5617295327"><a name="p5617295327"></a><a name="p5617295327"></a><a href="Serialize（FunctionPp类）.md">Serialize（FunctionPp类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001435016924_p16972182814218"><a name="zh-cn_topic_0000001435016924_p16972182814218"></a><a name="zh-cn_topic_0000001435016924_p16972182814218"></a>FunctionPp的序列化方法。</p>
</td>
</tr>
<tr id="row193181257064"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p186020291323"><a name="p186020291323"></a><a name="p186020291323"></a><a href="GetInvokedClosures.md">GetInvokedClosures</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p2972586311"><a name="p2972586311"></a><a name="p2972586311"></a>获取FunctionPp调用的GraphPp。</p>
</td>
</tr>
<tr id="row1555985025510"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p3560105085510"><a name="p3560105085510"></a><a name="p3560105085510"></a><a href="GraphPp构造函数和析构函数.md">GraphPp构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001461312445_p699292615173"><a name="zh-cn_topic_0000001461312445_p699292615173"></a><a name="zh-cn_topic_0000001461312445_p699292615173"></a>GraphPp构造函数和析构函数，构造函数会返回一个GraphPp对象。</p>
</td>
</tr>
<tr id="row75601450155517"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p14560550125511"><a name="p14560550125511"></a><a name="p14560550125511"></a><a href="SetCompileConfig（GraphPp类）.md">SetCompileConfig（GraphPp类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001411032876_p25962301112"><a name="zh-cn_topic_0000001411032876_p25962301112"></a><a name="zh-cn_topic_0000001411032876_p25962301112"></a>设置GraphPp的json配置文件路径和文件名。配置文件用于AscendGraph的描述和编译。</p>
</td>
</tr>
<tr id="row155601550115517"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1456014508556"><a name="p1456014508556"></a><a name="p1456014508556"></a><a href="Serialize（GraphPp类）.md">Serialize（GraphPp类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001484976265_p16972182814218"><a name="zh-cn_topic_0000001484976265_p16972182814218"></a><a name="zh-cn_topic_0000001484976265_p16972182814218"></a>GraphPp的序列化方法。</p>
</td>
</tr>
<tr id="row1256055055510"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p55604500550"><a name="p55604500550"></a><a name="p55604500550"></a><a href="GetGraphBuilder（GraphPp类）.md">GetGraphBuilder（GraphPp类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p178811540518"><a name="p178811540518"></a><a name="p178811540518"></a>获取GraphPp中Graph的创建函数。</p>
</td>
</tr>
<tr id="row11471181935020"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p12472619145019"><a name="p12472619145019"></a><a name="p12472619145019"></a><a href="FlowGraphPp构造函数和析构函数.md">FlowGraphPp构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p156543915111"><a name="p156543915111"></a><a name="p156543915111"></a>FlowGraphPp构造函数和析构函数，构造函数会返回一个FlowGraphPp对象。</p>
</td>
</tr>
<tr id="row1716132316501"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p41611623165010"><a name="p41611623165010"></a><a name="p41611623165010"></a><a href="Serialize（FlowGraphPp类）.md">Serialize（FlowGraphPp类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p17887847205111"><a name="p17887847205111"></a><a name="p17887847205111"></a>FlowGraphPp的序列化方法。</p>
</td>
</tr>
<tr id="row12422102735012"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p842214276500"><a name="p842214276500"></a><a name="p842214276500"></a><a href="GetGraphBuilder（FlowGraphPp类）.md">GetGraphBuilder（FlowGraphPp类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p15763143071418"><a name="p15763143071418"></a><a name="p15763143071418"></a>获取FlowGraphPp中Graph的创建函数。</p>
</td>
</tr>
<tr id="row656010502553"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1656615515717"><a name="p1656615515717"></a><a name="p1656615515717"></a><a href="TimeBatch.md">TimeBatch</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p11256727112615"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p11256727112615"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p11256727112615"></a>TimeBatch功能是基于UDF为前提的。</p>
<p id="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p74172912515"><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p74172912515"></a><a name="zh-cn_topic_0000001417515936_zh-cn_topic_0000001409649741_p74172912515"></a>正常模型每次处理一个数据，当需要一次处理一批数据时，就需要将这批数据组成一个Batch，最基本的Batch方式是将这批N个数据直接拼接，然后shape前加N，而某些场景需要将某段或者某几段时间数据组成一个batch，并且按特定的维度拼接，则可以通过使用TimeBatch功能来组Batch。</p>
</td>
</tr>
<tr id="row44421251043"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1756545125720"><a name="p1756545125720"></a><a name="p1756545125720"></a><a href="CountBatch.md">CountBatch</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p558229153214"><a name="p558229153214"></a><a name="p558229153214"></a>CountBatch功能是指基于UDF为计算处理点将多个数据按batchSize组成batch。</p>
</td>
</tr>
</tbody>
</table>

## DataFlow运行接口<a name="section144531976011"></a>

**表 2**  DataFlow运行接口

<a name="table10223122443215"></a>
<table><thead align="left"><tr id="row1222312249325"><th class="cellrowborder" valign="top" width="33.47%" id="mcps1.2.3.1.1"><p id="p2224142411323"><a name="p2224142411323"></a><a name="p2224142411323"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.53%" id="mcps1.2.3.1.2"><p id="p15224142443217"><a name="p15224142443217"></a><a name="p15224142443217"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row1722492419321"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p32746329328"><a name="p32746329328"></a><a name="p32746329328"></a><a href="FeedDataFlowGraph（feed所有输入）.md">FeedDataFlowGraph（feed所有输入）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001415994657_p64091283392"><a name="zh-cn_topic_0000001415994657_p64091283392"></a><a name="zh-cn_topic_0000001415994657_p64091283392"></a>将所有数据输入到Graph图。</p>
</td>
</tr>
<tr id="row13900173642415"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p490143614243"><a name="p490143614243"></a><a name="p490143614243"></a><a href="FeedDataFlowGraph（按索引feed输入）.md">FeedDataFlowGraph（按索引feed输入）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p5145198162813"><a name="p5145198162813"></a><a name="p5145198162813"></a>将数据按索引输入到Graph图。</p>
</td>
</tr>
<tr id="row12730143916245"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p973053920247"><a name="p973053920247"></a><a name="p973053920247"></a><a href="FeedDataFlowGraph（feed所有FlowMsg）.md">FeedDataFlowGraph（feed所有FlowMsg）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p873003915249"><a name="p873003915249"></a><a name="p873003915249"></a>将数据输入到Graph图。</p>
</td>
</tr>
<tr id="row11203933102416"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1220433372412"><a name="p1220433372412"></a><a name="p1220433372412"></a><a href="FeedDataFlowGraph（按索引feed-FlowMsg）.md">FeedDataFlowGraph（按索引feed FlowMsg）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p320453315248"><a name="p320453315248"></a><a name="p320453315248"></a>将数据按索引输入到Graph图。</p>
</td>
</tr>
<tr id="row1122472493211"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1627283293216"><a name="p1627283293216"></a><a name="p1627283293216"></a><a href="FeedRawData.md">FeedRawData</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p165794312610"><a name="p165794312610"></a><a name="p165794312610"></a>将原始数据输入到Graph图。</p>
</td>
</tr>
<tr id="row6224122443212"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1127173210321"><a name="p1127173210321"></a><a name="p1127173210321"></a><a href="FetchDataFlowGraph（获取所有输出数据）.md">FetchDataFlowGraph（获取所有输出数据）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001364713352_p64091283392"><a name="zh-cn_topic_0000001364713352_p64091283392"></a><a name="zh-cn_topic_0000001364713352_p64091283392"></a>获取图输出数据。</p>
</td>
</tr>
<tr id="row1222432443211"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1927053220327"><a name="p1927053220327"></a><a name="p1927053220327"></a><a href="FetchDataFlowGraph（按索引获取输出数据）.md">FetchDataFlowGraph（按索引获取输出数据）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001365193284_p64091283392"><a name="zh-cn_topic_0000001365193284_p64091283392"></a><a name="zh-cn_topic_0000001365193284_p64091283392"></a>按索引获取图输出数据。</p>
</td>
</tr>
<tr id="row162337257546"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p10233425195418"><a name="p10233425195418"></a><a name="p10233425195418"></a><a href="FetchDataFlowGraph（获取所有输出FlowMsg）.md">FetchDataFlowGraph（获取所有输出FlowMsg）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p723382525416"><a name="p723382525416"></a><a name="p723382525416"></a>获取图输出数据。</p>
</td>
</tr>
<tr id="row14899102611542"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p19899132625411"><a name="p19899132625411"></a><a name="p19899132625411"></a><a href="FetchDataFlowGraph（按索引获取输出FlowMsg）.md">FetchDataFlowGraph（按索引获取输出FlowMsg）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p289912620544"><a name="p289912620544"></a><a name="p289912620544"></a>按索引获取图输出数据。</p>
</td>
</tr>
<tr id="row922412247322"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p3268332183213"><a name="p3268332183213"></a><a name="p3268332183213"></a><a href="DataFlowInfo数据类型构造函数和析构函数.md">DataFlowInfo数据类型构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p2083913014136"><a name="p2083913014136"></a><a name="p2083913014136"></a>DataFlowInfo构造函数和析构函数。</p>
</td>
</tr>
<tr id="row12224202410326"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p3267123213216"><a name="p3267123213216"></a><a name="p3267123213216"></a><a href="SetUserData（DataFlowInfo数据类型）.md">SetUserData（DataFlowInfo数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p155251965131"><a name="p155251965131"></a><a name="p155251965131"></a>设置用户信息。</p>
</td>
</tr>
<tr id="row8224192493216"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1266113219324"><a name="p1266113219324"></a><a name="p1266113219324"></a><a href="GetUserData（DataFlowInfo数据类型）.md">GetUserData（DataFlowInfo数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p71561713111319"><a name="p71561713111319"></a><a name="p71561713111319"></a>获取用户信息。</p>
</td>
</tr>
<tr id="row1736795715816"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p336755715815"><a name="p336755715815"></a><a name="p336755715815"></a><a href="SetStartTime（DataFlowInfo数据类型）.md">SetStartTime（DataFlowInfo数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001439470042_zh-cn_topic_0000001359548690_p1562613391532"><a name="zh-cn_topic_0000001439470042_zh-cn_topic_0000001359548690_p1562613391532"></a><a name="zh-cn_topic_0000001439470042_zh-cn_topic_0000001359548690_p1562613391532"></a>设置数据的开始时间戳。</p>
</td>
</tr>
<tr id="row1436725795812"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p3367205715816"><a name="p3367205715816"></a><a name="p3367205715816"></a><a href="GetStartTime（DataFlowInfo数据类型）.md">GetStartTime（DataFlowInfo数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p88614206137"><a name="p88614206137"></a><a name="p88614206137"></a>获取数据的开始时间戳。</p>
</td>
</tr>
<tr id="row10367115795810"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p10367857135813"><a name="p10367857135813"></a><a name="p10367857135813"></a><a href="SetEndTime（DataFlowInfo数据类型）.md">SetEndTime（DataFlowInfo数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001439310790_zh-cn_topic_0000001359708658_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001439310790_zh-cn_topic_0000001359708658_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001439310790_zh-cn_topic_0000001359708658_zh-cn_topic_0000001264921066_p36600850"></a>设置数据的结束时间戳。</p>
</td>
</tr>
<tr id="row2367135718587"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1336785715816"><a name="p1336785715816"></a><a name="p1336785715816"></a><a href="GetEndTime（DataFlowInfo数据类型）.md">GetEndTime（DataFlowInfo数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p18311533181320"><a name="p18311533181320"></a><a name="p18311533181320"></a>获取数据的结束时间戳。</p>
</td>
</tr>
<tr id="row143671257165810"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p5367175713586"><a name="p5367175713586"></a><a name="p5367175713586"></a><a href="SetFlowFlags（DataFlowInfo数据类型）.md">SetFlowFlags（DataFlowInfo数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001438993194_zh-cn_topic_0000001409388721_p186651645278"><a name="zh-cn_topic_0000001438993194_zh-cn_topic_0000001409388721_p186651645278"></a><a name="zh-cn_topic_0000001438993194_zh-cn_topic_0000001409388721_p186651645278"></a>设置数据中的flags。</p>
</td>
</tr>
<tr id="row6224182453210"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p626483210327"><a name="p626483210327"></a><a name="p626483210327"></a><a href="GetFlowFlags（DataFlowInfo数据类型）.md">GetFlowFlags（DataFlowInfo数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p9527184451310"><a name="p9527184451310"></a><a name="p9527184451310"></a>获取数据中的flags。</p>
</td>
</tr>
<tr id="row17701491210"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1377018917118"><a name="p1377018917118"></a><a name="p1377018917118"></a><a href="SetTransactionId（DataFlowInfo数据类型）.md">SetTransactionId（DataFlowInfo数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001439153150_zh-cn_topic_0000001408829021_p204566483916"><a name="zh-cn_topic_0000001439153150_zh-cn_topic_0000001408829021_p204566483916"></a><a name="zh-cn_topic_0000001439153150_zh-cn_topic_0000001408829021_p204566483916"></a>设置DataFlow数据传输使用的事务ID。</p>
</td>
</tr>
<tr id="row88011712719"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1880216121016"><a name="p1880216121016"></a><a name="p1880216121016"></a><a href="GetTransactionId（DataFlowInfo数据类型）.md">GetTransactionId（DataFlowInfo数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p108604465916"><a name="p108604465916"></a><a name="p108604465916"></a>获取DataFlow数据传输使用的事务ID。</p>
</td>
</tr>
<tr id="row17505541706"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p195051041803"><a name="p195051041803"></a><a name="p195051041803"></a><a href="FlowMsg数据类型构造函数和析构函数.md">FlowMsg数据类型构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p72888431999"><a name="p72888431999"></a><a name="p72888431999"></a>FlowMsg构造函数和析构函数。</p>
</td>
</tr>
<tr id="row1505141302"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p10505154116019"><a name="p10505154116019"></a><a name="p10505154116019"></a><a href="GetMsgType（FlowMsg数据类型）.md">GetMsgType（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001409268917_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001409268917_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001409268917_zh-cn_topic_0000001264921066_p36600850"></a>获取FlowMsg的消息类型。</p>
</td>
</tr>
<tr id="row1950554118010"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p185053411904"><a name="p185053411904"></a><a name="p185053411904"></a><a href="SetMsgType（FlowMsg数据类型）.md">SetMsgType（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001409388721_p186651645278"><a name="zh-cn_topic_0000001409388721_p186651645278"></a><a name="zh-cn_topic_0000001409388721_p186651645278"></a>设置FlowMsg的消息类型。</p>
</td>
</tr>
<tr id="row750515411014"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p175053419013"><a name="p175053419013"></a><a name="p175053419013"></a><a href="GetTensor（FlowMsg数据类型）.md">GetTensor（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001409388717_p148801903014"><a name="zh-cn_topic_0000001409388717_p148801903014"></a><a name="zh-cn_topic_0000001409388717_p148801903014"></a><span id="ph56860123117"><a name="ph56860123117"></a><a name="ph56860123117"></a>获取FlowMsg中的Tensor指针</span>。</p>
</td>
</tr>
<tr id="row95058412019"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p050514416011"><a name="p050514416011"></a><a name="p050514416011"></a><a href="GetRetCode（FlowMsg数据类型）.md">GetRetCode（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001408829017_p254635918010"><a name="zh-cn_topic_0000001408829017_p254635918010"></a><a name="zh-cn_topic_0000001408829017_p254635918010"></a>获取输入FlowMsg中的错误码。</p>
</td>
</tr>
<tr id="row14505114118010"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1550610417014"><a name="p1550610417014"></a><a name="p1550610417014"></a><a href="SetRetCode（FlowMsg数据类型）.md">SetRetCode（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001359389150_p489119523119"><a name="zh-cn_topic_0000001359389150_p489119523119"></a><a name="zh-cn_topic_0000001359389150_p489119523119"></a>设置FlowMsg中的错误码。</p>
</td>
</tr>
<tr id="row145065417011"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p125061941701"><a name="p125061941701"></a><a name="p125061941701"></a><a href="SetStartTime（FlowMsg数据类型）.md">SetStartTime（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001359548690_p1562613391532"><a name="zh-cn_topic_0000001359548690_p1562613391532"></a><a name="zh-cn_topic_0000001359548690_p1562613391532"></a>设置FlowMsg消息头中的开始时间戳。</p>
</td>
</tr>
<tr id="row250618411204"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p145063414014"><a name="p145063414014"></a><a name="p145063414014"></a><a href="GetStartTime（FlowMsg数据类型）.md">GetStartTime（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001409228617_p198905321146"><a name="zh-cn_topic_0000001409228617_p198905321146"></a><a name="zh-cn_topic_0000001409228617_p198905321146"></a>获取FlowMsg消息中的开始时间戳。</p>
</td>
</tr>
<tr id="row94897281406"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1048918285013"><a name="p1048918285013"></a><a name="p1048918285013"></a><a href="SetEndTime（FlowMsg数据类型）.md">SetEndTime（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001359708658_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001359708658_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001359708658_zh-cn_topic_0000001264921066_p36600850"></a>设置FlowMsg消息头中的结束时间戳。</p>
</td>
</tr>
<tr id="row162041089115"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1620448316"><a name="p1620448316"></a><a name="p1620448316"></a><a href="GetEndTime（FlowMsg数据类型）.md">GetEndTime（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001409268921_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001409268921_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001409268921_zh-cn_topic_0000001264921066_p36600850"></a>获取FlowMsg消息中的结束时间戳。</p>
</td>
</tr>
<tr id="row1520428018"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p22051987113"><a name="p22051987113"></a><a name="p22051987113"></a><a href="SetFlowFlags（FlowMsg数据类型）.md">SetFlowFlags（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p9285121218262"><a name="p9285121218262"></a><a name="p9285121218262"></a>设置FlowMsg消息头中的flags。</p>
</td>
</tr>
<tr id="row112057811120"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p182051786116"><a name="p182051786116"></a><a name="p182051786116"></a><a href="GetFlowFlags（FlowMsg数据类型）.md">GetFlowFlags（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001408829021_p204566483916"><a name="zh-cn_topic_0000001408829021_p204566483916"></a><a name="zh-cn_topic_0000001408829021_p204566483916"></a>获取FlowMsg消息头中的flags。</p>
</td>
</tr>
<tr id="row11205198314"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p120514816110"><a name="p120514816110"></a><a name="p120514816110"></a><a href="GetTransactionId（FlowMsg数据类型）.md">GetTransactionId（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p42051583118"><a name="p42051583118"></a><a name="p42051583118"></a>获取FlowMsg消息中的事务ID。</p>
</td>
</tr>
<tr id="row20205684118"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p02051081717"><a name="p02051081717"></a><a name="p02051081717"></a><a href="SetTransactionId（FlowMsg数据类型）.md">SetTransactionId（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p16544736112616"><a name="p16544736112616"></a><a name="p16544736112616"></a>设置FlowMsg消息中的事务ID。</p>
</td>
</tr>
<tr id="row13205284117"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p7205682111"><a name="p7205682111"></a><a name="p7205682111"></a><a href="SetUserData（FlowMsg数据类型）.md">SetUserData（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p02672049152613"><a name="p02672049152613"></a><a name="p02672049152613"></a>设置用户信息。</p>
</td>
</tr>
<tr id="row42053818119"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p9205081716"><a name="p9205081716"></a><a name="p9205081716"></a><a href="GetUserData（FlowMsg数据类型）.md">GetUserData（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p24594570262"><a name="p24594570262"></a><a name="p24594570262"></a>获取用户信息。</p>
</td>
</tr>
<tr id="row16447826705"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p844842613020"><a name="p844842613020"></a><a name="p844842613020"></a><a href="GetRawData（FlowMsg数据类型）.md">GetRawData（FlowMsg数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p0245192172713"><a name="p0245192172713"></a><a name="p0245192172713"></a>获取RawData类型的数据对应的数据指针和数据大小。</p>
</td>
</tr>
<tr id="row6378151514011"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p173791515607"><a name="p173791515607"></a><a name="p173791515607"></a><a href="AllocTensor（FlowBufferFactory数据类型）.md">AllocTensor（FlowBufferFactory数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002013837145_p3702153414173"><a name="zh-cn_topic_0000002013837145_p3702153414173"></a><a name="zh-cn_topic_0000002013837145_p3702153414173"></a>根据shape、data type和对齐大小申请Tensor。</p>
</td>
</tr>
<tr id="row13213238012"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p82132311014"><a name="p82132311014"></a><a name="p82132311014"></a><a href="AllocTensorMsg（FlowBufferFactory数据类型）.md">AllocTensorMsg（FlowBufferFactory数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p557553142819"><a name="p557553142819"></a><a name="p557553142819"></a>根据shape、data type和对齐大小申请FlowMsg。</p>
</td>
</tr>
<tr id="row261715241705"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1261710244010"><a name="p1261710244010"></a><a name="p1261710244010"></a><a href="AllocRawDataMsg（FlowBufferFactory数据类型）.md">AllocRawDataMsg（FlowBufferFactory数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001913682208_p3702153414173"><a name="zh-cn_topic_0000001913682208_p3702153414173"></a><a name="zh-cn_topic_0000001913682208_p3702153414173"></a>根据输入的size申请一块连续内存，用于承载raw data类型的数据。</p>
</td>
</tr>
<tr id="row3841319706"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p084319401"><a name="p084319401"></a><a name="p084319401"></a><a href="AllocEmptyDataMsg（FlowBufferFactory数据类型）.md">AllocEmptyDataMsg（FlowBufferFactory数据类型）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1281314307502"><a name="p1281314307502"></a><a name="p1281314307502"></a>申请空数据的MsgType类型的message。</p>
</td>
</tr>
<tr id="row119032115013"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p119115211606"><a name="p119115211606"></a><a name="p119115211606"></a><a href="ToFlowMsg（tensor）.md">ToFlowMsg（tensor）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p184621956132817"><a name="p184621956132817"></a><a name="p184621956132817"></a>根据输入的Tensor转换成用于承载Tensor的FlowMsg。</p>
</td>
</tr>
<tr id="row172792175019"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p727916171002"><a name="p727916171002"></a><a name="p727916171002"></a><a href="ToFlowMsg（raw-data）.md">ToFlowMsg（raw data）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1817264152915"><a name="p1817264152915"></a><a name="p1817264152915"></a>根据输入的raw data转换成用于承载raw data的FlowMsg。</p>
</td>
</tr>
</tbody>
</table>

