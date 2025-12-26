# DataFlow运行接口简介<a name="ZH-CN_TOPIC_0000001977312242"></a>

本文档主要描述模型执行接口，您可以在“$\{INSTALL\_DIR\}/include/ge”路径下查看对应接口的头文件。

$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。若安装的Ascend-cann-toolkit软件包，以root安装举例，则安装后文件存储路径为：/usr/local/Ascend/ascend-toolkit/latest。

<a name="zh-cn_topic_0000001416390457_table19650738184916"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001416390457_row5651338124915"><th class="cellrowborder" valign="top" width="21.09%" id="mcps1.1.5.1.1"><p id="zh-cn_topic_0000001416390457_p19651193864919"><a name="zh-cn_topic_0000001416390457_p19651193864919"></a><a name="zh-cn_topic_0000001416390457_p19651193864919"></a>接口分类</p>
</th>
<th class="cellrowborder" valign="top" width="25.990000000000002%" id="mcps1.1.5.1.2"><p id="zh-cn_topic_0000001416390457_p344352618433"><a name="zh-cn_topic_0000001416390457_p344352618433"></a><a name="zh-cn_topic_0000001416390457_p344352618433"></a>头文件路径</p>
</th>
<th class="cellrowborder" valign="top" width="32.29%" id="mcps1.1.5.1.3"><p id="p79991615201413"><a name="p79991615201413"></a><a name="p79991615201413"></a>用途</p>
</th>
<th class="cellrowborder" valign="top" width="20.630000000000003%" id="mcps1.1.5.1.4"><p id="p16862164158"><a name="p16862164158"></a><a name="p16862164158"></a>对应的库文件</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001416390457_row12674164819"><td class="cellrowborder" valign="top" width="21.09%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001416390457_p5268191611811"><a name="zh-cn_topic_0000001416390457_p5268191611811"></a><a name="zh-cn_topic_0000001416390457_p5268191611811"></a>Graph运行接口</p>
</td>
<td class="cellrowborder" valign="top" width="25.990000000000002%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001416390457_p122912517277"><a name="zh-cn_topic_0000001416390457_p122912517277"></a><a name="zh-cn_topic_0000001416390457_p122912517277"></a>ge_api.h</p>
</td>
<td class="cellrowborder" valign="top" width="32.29%" headers="mcps1.1.5.1.3 "><p id="p10655195118327"><a name="p10655195118327"></a><a name="p10655195118327"></a>用于将数据输入<span id="ph883110196517"><a name="ph883110196517"></a><a name="ph883110196517"></a>到</span>DataFlow图和获取DataFlow模型执行结果。</p>
</td>
<td class="cellrowborder" valign="top" width="20.630000000000003%" headers="mcps1.1.5.1.4 "><p id="p1465565118320"><a name="p1465565118320"></a><a name="p1465565118320"></a>libge_runner.so   libdavinci_executor.so    libgraph_base.so</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001416390457_row1565113383499"><td class="cellrowborder" valign="top" width="21.09%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001416390457_p1889858135018"><a name="zh-cn_topic_0000001416390457_p1889858135018"></a><a name="zh-cn_topic_0000001416390457_p1889858135018"></a>数据类型</p>
</td>
<td class="cellrowborder" valign="top" width="25.990000000000002%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001416390457_p080381265912"><a name="zh-cn_topic_0000001416390457_p080381265912"></a><a name="zh-cn_topic_0000001416390457_p080381265912"></a>ge_data_flow_api.h</p>
</td>
<td class="cellrowborder" valign="top" width="32.29%" headers="mcps1.1.5.1.3 "><p id="p16363357123213"><a name="p16363357123213"></a><a name="p16363357123213"></a>支持用户设置和获取DataFlowInfo中的成员变量。</p>
<div class="note" id="note127335482021"><a name="note127335482021"></a><a name="note127335482021"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1733134813218"><a name="p1733134813218"></a><a name="p1733134813218"></a>如果单点编译DataFlowInfo数据类型，建议编译选项增加-Wl,--no-as-needed，确保依赖的so符号在编译时被完整加载。</p>
</div></div>
</td>
<td class="cellrowborder" valign="top" width="20.630000000000003%" headers="mcps1.1.5.1.4 "><p id="p336365711327"><a name="p336365711327"></a><a name="p336365711327"></a>libdavinci_executor.so libge_runner.so</p>
</td>
</tr>
</tbody>
</table>

