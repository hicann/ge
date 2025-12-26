# UDF接口列表<a name="ZH-CN_TOPIC_0000001977316702"></a>

本文档主要描述UDF（User Defined Function）模块对外提供的接口，用户可以调用这些接口进行自定义处理函数的开发，然后通过DataFlow构图在CPU上执行该处理函数。

您可以在“$\{INSTALL\_DIR\}/include/flow\_func”查看对应接口的头文件。

$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。若安装的Ascend-cann-toolkit软件包，以root安装举例，则安装后文件存储路径为：/usr/local/Ascend/ascend-toolkit/latest。

## 接口分类及对应头文件<a name="section143701455184613"></a>

**表 1**  接口分类及对应头文件

<a name="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_table6861546172820"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_row986112467281"><th class="cellrowborder" valign="top" width="26.87%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_p3861046142817"><a name="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_p3861046142817"></a><a name="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_p3861046142817"></a>接口分类</p>
</th>
<th class="cellrowborder" valign="top" width="28.439999999999998%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_p17861134612813"><a name="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_p17861134612813"></a><a name="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_p17861134612813"></a>头文件路径</p>
</th>
<th class="cellrowborder" valign="top" width="44.690000000000005%" id="mcps1.2.4.1.3"><p id="p477523164014"><a name="p477523164014"></a><a name="p477523164014"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_row17163247142218"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_p1471615566212"><a name="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_p1471615566212"></a><a name="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_p1471615566212"></a>AttrValue类</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_p1415202210223"><a name="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_p1415202210223"></a><a name="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_p1415202210223"></a>attr_value.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001304615476_p8060118"><a name="zh-cn_topic_0000001304615476_p8060118"></a><a name="zh-cn_topic_0000001304615476_p8060118"></a>用于获取用户设置的属性值。</p>
</td>
</tr>
<tr id="row2812839105"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001304225444_p1526901915111"><a name="zh-cn_topic_0000001304225444_p1526901915111"></a><a name="zh-cn_topic_0000001304225444_p1526901915111"></a>AscendString类</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001304225444_p182688194115"><a name="zh-cn_topic_0000001304225444_p182688194115"></a><a name="zh-cn_topic_0000001304225444_p182688194115"></a>ascend_string.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="p2086584934016"><a name="p2086584934016"></a><a name="p2086584934016"></a>对String类型的封装。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_row208611046142817"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001304225444_p1045232035512"><a name="zh-cn_topic_0000001304225444_p1045232035512"></a><a name="zh-cn_topic_0000001304225444_p1045232035512"></a>MetaContext类</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001304225444_p1045282065520"><a name="zh-cn_topic_0000001304225444_p1045282065520"></a><a name="zh-cn_topic_0000001304225444_p1045282065520"></a>meta_context.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001304789308_p8060118"><a name="zh-cn_topic_0000001304789308_p8060118"></a><a name="zh-cn_topic_0000001304789308_p8060118"></a>用于UDF上下文信息相关处理，如申请tensor和获取设置的属性等操作。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001304225444_row205662013717"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001304225444_p2097214432505"><a name="zh-cn_topic_0000001304225444_p2097214432505"></a><a name="zh-cn_topic_0000001304225444_p2097214432505"></a>FlowMsg类</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001304225444_p897234375015"><a name="zh-cn_topic_0000001304225444_p897234375015"></a><a name="zh-cn_topic_0000001304225444_p897234375015"></a>flow_msg.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001409228613_p8060118"><a name="zh-cn_topic_0000001409228613_p8060118"></a><a name="zh-cn_topic_0000001409228613_p8060118"></a>用于处理flow func输入输出的相关操作。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_row11861146142810"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001304225444_p745132014556"><a name="zh-cn_topic_0000001304225444_p745132014556"></a><a name="zh-cn_topic_0000001304225444_p745132014556"></a>Tensor类</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="p1771722161414"><a name="p1771722161414"></a><a name="p1771722161414"></a>flow_msg.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001304629368_p8060118"><a name="zh-cn_topic_0000001304629368_p8060118"></a><a name="zh-cn_topic_0000001304629368_p8060118"></a>用于执行Tensor的相关操作。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001304225444_zh-cn_topic_0000001312720969_row2903258114615"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001304225444_p1145018206557"><a name="zh-cn_topic_0000001304225444_p1145018206557"></a><a name="zh-cn_topic_0000001304225444_p1145018206557"></a>MetaFlowFunc类</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001304225444_p745020202555"><a name="zh-cn_topic_0000001304225444_p745020202555"></a><a name="zh-cn_topic_0000001304225444_p745020202555"></a>meta_flow_func.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001357588757_p86542563915"><a name="zh-cn_topic_0000001357588757_p86542563915"></a><a name="zh-cn_topic_0000001357588757_p86542563915"></a>该类在meta_flow_func.h中定义，用户继承该类进行自定义的单func处理函数的编写。</p>
</td>
</tr>
<tr id="row1063218406396"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="p142301647113920"><a name="p142301647113920"></a><a name="p142301647113920"></a>MetaMultiFunc类</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="p11230547113913"><a name="p11230547113913"></a><a name="p11230547113913"></a>meta_multi_func.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="p149863263712"><a name="p149863263712"></a><a name="p149863263712"></a>该类在meta_multi_func.h中定义，用户继承该类进行自定义的多func处理函数的编写。</p>
</td>
</tr>
<tr id="row19210021311"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="p1221062318"><a name="p1221062318"></a><a name="p1221062318"></a>FlowFuncRegistrar类</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="p162108219117"><a name="p162108219117"></a><a name="p162108219117"></a>meta_multi_func.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="p8060118"><a name="p8060118"></a><a name="p8060118"></a>该类在meta_multi_func.h中定义，是注册MetaMultiFunc的辅助模板类。</p>
</td>
</tr>
<tr id="row1989484373919"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="p5230147153919"><a name="p5230147153919"></a><a name="p5230147153919"></a>MetaParams类</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="p16230154783917"><a name="p16230154783917"></a><a name="p16230154783917"></a>meta_params.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="p66772019423"><a name="p66772019423"></a><a name="p66772019423"></a>该类在meta_params.h中定义，在FlowFunc的多func处理函数中使用该类获取共享的变量信息。</p>
</td>
</tr>
<tr id="row136711429399"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="p623018472398"><a name="p623018472398"></a><a name="p623018472398"></a>MetaRunContext类</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="p162311247103918"><a name="p162311247103918"></a><a name="p162311247103918"></a>meta_run_context.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="p250917143425"><a name="p250917143425"></a><a name="p250917143425"></a>用于执行FlowFunc的多func处理函数的上下文信息相关处理，如申请Tensor、设置输出、运行FlowModel等操作。</p>
</td>
</tr>
<tr id="row73991415217"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="p4399941132118"><a name="p4399941132118"></a><a name="p4399941132118"></a>OutOptions类</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="p939924142112"><a name="p939924142112"></a><a name="p939924142112"></a>out_options.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="p1262585185114"><a name="p1262585185114"></a><a name="p1262585185114"></a>业务发布数据时，为了携带相关的option，提供了输出options的类。</p>
</td>
</tr>
<tr id="row1436213394215"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="p113636392215"><a name="p113636392215"></a><a name="p113636392215"></a>BalanceConfig类</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="p113631039172111"><a name="p113631039172111"></a><a name="p113631039172111"></a>balance_config.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="p1444912399421"><a name="p1444912399421"></a><a name="p1444912399421"></a>当需要均衡分发时，需要设置输出数据标识和权重矩阵相关配置信息，根据配置调度模块可以完成多实例之间的均衡分发。</p>
</td>
</tr>
<tr id="row555912112467"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="p156018113462"><a name="p156018113462"></a><a name="p156018113462"></a>FlowBufferFactory类</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="p145601115460"><a name="p145601115460"></a><a name="p145601115460"></a>flow_msg.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="p37751732408"><a name="p37751732408"></a><a name="p37751732408"></a>-</p>
</td>
</tr>
<tr id="row2082133417326"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="p1783103443212"><a name="p1783103443212"></a><a name="p1783103443212"></a>FlowMsgQueue类</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="p1883334133213"><a name="p1883334133213"></a><a name="p1883334133213"></a>flow_msg_queue.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="p47171254142914"><a name="p47171254142914"></a><a name="p47171254142914"></a>流式输入场景下（即flow func函数入参为队列时），用于flow func的输入队列，队列中的数据对象为<a href="FlowMsg类.md">FlowMsg类</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001304225444_row89721943115018"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001304225444_p136013752614"><a name="zh-cn_topic_0000001304225444_p136013752614"></a><a name="zh-cn_topic_0000001304225444_p136013752614"></a>注册宏</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001304225444_p11601107122610"><a name="zh-cn_topic_0000001304225444_p11601107122610"></a><a name="zh-cn_topic_0000001304225444_p11601107122610"></a>meta_flow_func.h</p>
<p id="p516010451628"><a name="p516010451628"></a><a name="p516010451628"></a>meta_multi_func.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="p1477543194012"><a name="p1477543194012"></a><a name="p1477543194012"></a>-</p>
</td>
</tr>
<tr id="row1712110111431"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="p1295471111313"><a name="p1295471111313"></a><a name="p1295471111313"></a>UDF日志接口</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="p2095418114313"><a name="p2095418114313"></a><a name="p2095418114313"></a>flow_func_log.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="p777513314019"><a name="p777513314019"></a><a name="p777513314019"></a>flow_func_log.h提供了日志接口，方便flowfunc开发中进行日志记录。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001304225444_row1343714782920"><td class="cellrowborder" valign="top" width="26.87%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001304225444_p1343812792917"><a name="zh-cn_topic_0000001304225444_p1343812792917"></a><a name="zh-cn_topic_0000001304225444_p1343812792917"></a>错误码</p>
</td>
<td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001304225444_p14438117142919"><a name="zh-cn_topic_0000001304225444_p14438117142919"></a><a name="zh-cn_topic_0000001304225444_p14438117142919"></a>flow_func_defines.h</p>
</td>
<td class="cellrowborder" valign="top" width="44.690000000000005%" headers="mcps1.2.4.1.3 "><p id="p077512319407"><a name="p077512319407"></a><a name="p077512319407"></a>-</p>
</td>
</tr>
</tbody>
</table>

## AttrValue类<a name="section5948174816916"></a>

**表 2**  AttrValue类接口

<a name="table1644114511445"></a>
<table><thead align="left"><tr id="row184417511246"><th class="cellrowborder" valign="top" width="33.47%" id="mcps1.2.3.1.1"><p id="p6441125120410"><a name="p6441125120410"></a><a name="p6441125120410"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.53%" id="mcps1.2.3.1.2"><p id="p1144117511349"><a name="p1144117511349"></a><a name="p1144117511349"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row17441125116414"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p04411451247"><a name="p04411451247"></a><a name="p04411451247"></a><a href="AttrValue构造函数和析构函数.md">AttrValue构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1944117519416"><a name="p1944117519416"></a><a name="p1944117519416"></a>AttrValue构造函数和析构函数。</p>
</td>
</tr>
<tr id="row944214511419"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p164420511141"><a name="p164420511141"></a><a name="p164420511141"></a><a href="GetVal(AscendString-value).md">GetVal(AscendString &amp;value)</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p12442551347"><a name="p12442551347"></a><a name="p12442551347"></a>获取string类型的属性值。</p>
</td>
</tr>
<tr id="row12318757162"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p183188575618"><a name="p183188575618"></a><a name="p183188575618"></a><a href="GetVal(std-vector-AscendString-value).md">GetVal(std::vector&lt;AscendString&gt; &amp;value)</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1931813575610"><a name="p1931813575610"></a><a name="p1931813575610"></a>获取list string类型的属性值。</p>
</td>
</tr>
<tr id="row1231816571467"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p143180574616"><a name="p143180574616"></a><a name="p143180574616"></a><a href="GetVal(int64_t-value).md">GetVal(int64_t &amp;value)</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001357579337_zh-cn_topic_0000001312399913_p7929917163115"><a name="zh-cn_topic_0000001357579337_zh-cn_topic_0000001312399913_p7929917163115"></a><a name="zh-cn_topic_0000001357579337_zh-cn_topic_0000001312399913_p7929917163115"></a>获取int类型的属性值。</p>
</td>
</tr>
<tr id="row931813574614"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1431815712612"><a name="p1431815712612"></a><a name="p1431815712612"></a><a href="GetVal(std-vector-int64_t-value).md">GetVal(std::vector&lt;int64_t&gt; &amp;value)</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001304779872_zh-cn_topic_0000001312399913_p7929917163115"><a name="zh-cn_topic_0000001304779872_zh-cn_topic_0000001312399913_p7929917163115"></a><a name="zh-cn_topic_0000001304779872_zh-cn_topic_0000001312399913_p7929917163115"></a>获取list int类型的属性值。</p>
</td>
</tr>
<tr id="row15318857869"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p143187576610"><a name="p143187576610"></a><a name="p143187576610"></a><a href="GetVal(std-vector-std-vector-int64_t-value).md">GetVal(std::vector&lt;std::vector&lt;int64_t &gt;&gt; &amp;value)</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001304619932_zh-cn_topic_0000001312399913_p7929917163115"><a name="zh-cn_topic_0000001304619932_zh-cn_topic_0000001312399913_p7929917163115"></a><a name="zh-cn_topic_0000001304619932_zh-cn_topic_0000001312399913_p7929917163115"></a>获取list list int类型的属性值。</p>
</td>
</tr>
<tr id="row193181257064"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1931814577618"><a name="p1931814577618"></a><a name="p1931814577618"></a><a href="GetVal(float-value).md">GetVal(float &amp;value)</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001357779633_zh-cn_topic_0000001312399913_p7929917163115"><a name="zh-cn_topic_0000001357779633_zh-cn_topic_0000001312399913_p7929917163115"></a><a name="zh-cn_topic_0000001357779633_zh-cn_topic_0000001312399913_p7929917163115"></a>获取float类型的属性值。</p>
</td>
</tr>
<tr id="row44421251043"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1442195119411"><a name="p1442195119411"></a><a name="p1442195119411"></a><a href="GetVal(std-vector-float-value).md">GetVal(std::vector&lt;float&gt; &amp;value)</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001304939728_zh-cn_topic_0000001312399913_p7929917163115"><a name="zh-cn_topic_0000001304939728_zh-cn_topic_0000001312399913_p7929917163115"></a><a name="zh-cn_topic_0000001304939728_zh-cn_topic_0000001312399913_p7929917163115"></a>获取list float类型的属性值。</p>
</td>
</tr>
<tr id="row111781534560"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1117823416615"><a name="p1117823416615"></a><a name="p1117823416615"></a><a href="GetVal(bool-value).md">GetVal(bool &amp;value)</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001409378513_zh-cn_topic_0000001312399913_p7929917163115"><a name="zh-cn_topic_0000001409378513_zh-cn_topic_0000001312399913_p7929917163115"></a><a name="zh-cn_topic_0000001409378513_zh-cn_topic_0000001312399913_p7929917163115"></a>获取bool类型的属性值。</p>
</td>
</tr>
<tr id="row74651839467"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p174651939869"><a name="p174651939869"></a><a name="p174651939869"></a><a href="GetVal(std-vector-bool-value).md">GetVal(std::vector&lt;bool&gt; &amp;value)</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001409258725_zh-cn_topic_0000001312399913_p7929917163115"><a name="zh-cn_topic_0000001409258725_zh-cn_topic_0000001312399913_p7929917163115"></a><a name="zh-cn_topic_0000001409258725_zh-cn_topic_0000001312399913_p7929917163115"></a>获取list bool类型的属性值。</p>
</td>
</tr>
<tr id="row39686441364"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p15968144561"><a name="p15968144561"></a><a name="p15968144561"></a><a href="GetVal(TensorDataType-value).md">GetVal(TensorDataType &amp;value)</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001359538474_zh-cn_topic_0000001312399913_p7929917163115"><a name="zh-cn_topic_0000001359538474_zh-cn_topic_0000001312399913_p7929917163115"></a><a name="zh-cn_topic_0000001359538474_zh-cn_topic_0000001312399913_p7929917163115"></a>获取TensorDataType类型的属性值。</p>
</td>
</tr>
<tr id="row478510316611"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p137856316618"><a name="p137856316618"></a><a name="p137856316618"></a><a href="GetVal(std-vector-TensorDataType-value).md">GetVal(std::vector&lt;TensorDataType&gt; &amp;value)</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001408818817_zh-cn_topic_0000001312399913_p7929917163115"><a name="zh-cn_topic_0000001408818817_zh-cn_topic_0000001312399913_p7929917163115"></a><a name="zh-cn_topic_0000001408818817_zh-cn_topic_0000001312399913_p7929917163115"></a>获取list TensorDataType类型的属性值。</p>
</td>
</tr>
</tbody>
</table>

## AscendString类<a name="section189210557116"></a>

**表 3**  AscendString类接口

<a name="table0455151019101"></a>
<table><thead align="left"><tr id="row124553101108"><th class="cellrowborder" valign="top" width="33.47%" id="mcps1.2.3.1.1"><p id="p945521041014"><a name="p945521041014"></a><a name="p945521041014"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.53%" id="mcps1.2.3.1.2"><p id="p104557108102"><a name="p104557108102"></a><a name="p104557108102"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row1845561041018"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p215113154106"><a name="p215113154106"></a><a name="p215113154106"></a><a href="AscendString构造函数和析构函数.md">AscendString构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p9480111518458"><a name="p9480111518458"></a><a name="p9480111518458"></a>AscendString构造函数和析构函数。</p>
</td>
</tr>
<tr id="row1845521061012"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p6150015111019"><a name="p6150015111019"></a><a name="p6150015111019"></a><a href="GetString.md">GetString</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1015121615475"><a name="p1015121615475"></a><a name="p1015121615475"></a>获取字符串地址。</p>
</td>
</tr>
<tr id="row16455410161016"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p31491015141015"><a name="p31491015141015"></a><a name="p31491015141015"></a><a href="关系符重载.md">关系符重载</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001357698889_zh-cn_topic_0000001312399913_p7929917163115"><a name="zh-cn_topic_0000001357698889_zh-cn_topic_0000001312399913_p7929917163115"></a><a name="zh-cn_topic_0000001357698889_zh-cn_topic_0000001312399913_p7929917163115"></a>对于AscendString对象大小比较的使用场景（例如map数据结构的key进行排序），通过重载以下关系符实现。</p>
</td>
</tr>
<tr id="row1245521018102"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p161471215121017"><a name="p161471215121017"></a><a name="p161471215121017"></a><a href="GetLength.md">GetLength</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p124416451112"><a name="p124416451112"></a><a name="p124416451112"></a>获取字符串的长度。</p>
</td>
</tr>
</tbody>
</table>

## MetaContext类<a name="section1468565181616"></a>

**表 4**  MetaContext类接口

<a name="table145995144128"></a>
<table><thead align="left"><tr id="row13599131471214"><th class="cellrowborder" valign="top" width="33.47%" id="mcps1.2.3.1.1"><p id="p13599171410122"><a name="p13599171410122"></a><a name="p13599171410122"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.53%" id="mcps1.2.3.1.2"><p id="p1159951419121"><a name="p1159951419121"></a><a name="p1159951419121"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row75991214161219"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p332102316122"><a name="p332102316122"></a><a name="p332102316122"></a><a href="MetaContext构造函数和析构函数.md">MetaContext构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001304385292_zh-cn_topic_0000001265080934_zh-cn_topic_0204328120_zh-cn_topic_0182636384_p13843256"><a name="zh-cn_topic_0000001304385292_zh-cn_topic_0000001265080934_zh-cn_topic_0204328120_zh-cn_topic_0182636384_p13843256"></a><a name="zh-cn_topic_0000001304385292_zh-cn_topic_0000001265080934_zh-cn_topic_0204328120_zh-cn_topic_0182636384_p13843256"></a>MetaContext构造函数和析构函数。</p>
</td>
</tr>
<tr id="row7599814201216"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p16319172301220"><a name="p16319172301220"></a><a name="p16319172301220"></a><a href="AllocTensorMsg（MetaContext类）.md">AllocTensorMsg（MetaContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001304225452_p1476471614810"><a name="zh-cn_topic_0000001304225452_p1476471614810"></a><a name="zh-cn_topic_0000001304225452_p1476471614810"></a>根据shape和data type申请<span id="ph1300183920914"><a name="ph1300183920914"></a><a name="ph1300183920914"></a>T</span>ensor类型的msg。该函数供<a href="Proc.md">Proc</a>调用。</p>
</td>
</tr>
<tr id="row05997148127"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p17318162320125"><a name="p17318162320125"></a><a name="p17318162320125"></a><a href="AllocEmptyDataMsg（MetaContext类）.md">AllocEmptyDataMsg（MetaContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1281314307502"><a name="p1281314307502"></a><a name="p1281314307502"></a>申请空数据的MsgType类型的message。该函数供<a href="Proc.md">Proc</a>调用。</p>
</td>
</tr>
<tr id="row1759971419125"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1531752313129"><a name="p1531752313129"></a><a name="p1531752313129"></a><a href="SetOutput（MetaContext类-tensor）.md">SetOutput（MetaContext类,tensor）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001304065500_zh-cn_topic_0000001312640889_p4089365"><a name="zh-cn_topic_0000001304065500_zh-cn_topic_0000001312640889_p4089365"></a><a name="zh-cn_topic_0000001304065500_zh-cn_topic_0000001312640889_p4089365"></a>设置指定index的output的tensor。该函数供<a href="Proc.md">Proc</a>调用。</p>
</td>
</tr>
<tr id="row1559911451210"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p8316172317129"><a name="p8316172317129"></a><a name="p8316172317129"></a><a href="GetAttr（MetaContext类-获取指针）.md">GetAttr（MetaContext类，获取指针）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"><a name="zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a><a name="zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a>根据属性名获取AttrValue类型的指针。该函数供<a href="Init（MetaFlowFunc类）.md">Init（MetaFlowFunc类）</a>调用。</p>
</td>
</tr>
<tr id="row16600191416125"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p20314102316123"><a name="p20314102316123"></a><a name="p20314102316123"></a><a href="GetAttr（MetaContext类-获取属性值）.md">GetAttr（MetaContext类，获取属性值）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001357345209_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001357345209_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001357345209_zh-cn_topic_0000001264921066_p36600850"></a>根据属性名获取对应的属性值。该函数供<a href="Init（MetaFlowFunc类）.md">Init（MetaFlowFunc类）</a>调用。</p>
</td>
</tr>
<tr id="row4600131413124"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p18313023201216"><a name="p18313023201216"></a><a name="p18313023201216"></a><a href="RunFlowModel（MetaContext类）.md">RunFlowModel（MetaContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a>同步执行指定的模型。该函数供<a href="Proc.md">Proc</a>调用。</p>
</td>
</tr>
<tr id="row19600414111215"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p9312152321219"><a name="p9312152321219"></a><a name="p9312152321219"></a><a href="GetInputNum（MetaContext类）.md">GetInputNum（MetaContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p193794194157"><a name="p193794194157"></a><a name="p193794194157"></a>获取Flowfunc的输入个数。该函数供<a href="Init（MetaFlowFunc类）.md">Init（MetaFlowFunc类）</a>调用。</p>
</td>
</tr>
<tr id="row76001514111218"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p8690325121214"><a name="p8690325121214"></a><a name="p8690325121214"></a><a href="GetOutputNum（MetaContext类）.md">GetOutputNum（MetaContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p93923277152"><a name="p93923277152"></a><a name="p93923277152"></a>获取Flowfunc的输出个数。该函数供<a href="Init（MetaFlowFunc类）.md">Init（MetaFlowFunc类）</a>调用。</p>
</td>
</tr>
<tr id="row86001114151212"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p206891255127"><a name="p206891255127"></a><a name="p206891255127"></a><a href="GetWorkPath（MetaContext类）.md">GetWorkPath（MetaContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p980763381510"><a name="p980763381510"></a><a name="p980763381510"></a>获取Flowfunc的工作路径。该函数供<a href="Init（MetaFlowFunc类）.md">Init（MetaFlowFunc类）</a>调用。</p>
</td>
</tr>
<tr id="row1960041420125"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p16881425121212"><a name="p16881425121212"></a><a name="p16881425121212"></a><a href="GetRunningDeviceId（MetaContext类）.md">GetRunningDeviceId（MetaContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p45155415157"><a name="p45155415157"></a><a name="p45155415157"></a>获取正在运行的设备ID。该函数供<a href="Init（MetaFlowFunc类）.md">Init（MetaFlowFunc类）</a>调用。</p>
</td>
</tr>
<tr id="row46003145123"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p16687425201212"><a name="p16687425201212"></a><a name="p16687425201212"></a><a href="GetUserData（MetaContext类）.md">GetUserData（MetaContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001409388721_p186651645278"><a name="zh-cn_topic_0000001409388721_p186651645278"></a><a name="zh-cn_topic_0000001409388721_p186651645278"></a>获取用户数据。该函数供<a href="Proc.md">Proc</a>调用。</p>
</td>
</tr>
<tr id="row9921205415130"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p16921115431310"><a name="p16921115431310"></a><a name="p16921115431310"></a><a href="AllocTensorMsgWithAlign（MetaContext类）.md">AllocTensorMsgWithAlign（MetaContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p3702153414173"><a name="p3702153414173"></a><a name="p3702153414173"></a>根据shape、data type<span id="ph62001091889"><a name="ph62001091889"></a><a name="ph62001091889"></a>和对齐大小申请Tensor</span>类型的FlowMsg，与<a href="AllocTensorMsg（MetaContext类）.md">AllocTensorMsg</a>函数区别是AllocTensorMsg默认申请以64字节对齐，此函数可以指定对齐大小，方便性能调优。</p>
</td>
</tr>
<tr id="row84631156506"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p114631554508"><a name="p114631554508"></a><a name="p114631554508"></a><a href="RaiseException（MetaContext类）.md">RaiseException（MetaContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p194736444506"><a name="p194736444506"></a><a name="p194736444506"></a>UDF主动上报异常，该异常可以被同作用域内的其他UDF捕获。</p>
</td>
</tr>
<tr id="row331235508"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p143203115013"><a name="p143203115013"></a><a name="p143203115013"></a><a href="GetException（MetaContext类）.md">GetException（MetaContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p9960195665010"><a name="p9960195665010"></a><a name="p9960195665010"></a>UDF获取异常，如果开启了异常捕获功能，需要在UDF中Proc函数开始位置尝试捕获异常。</p>
</td>
</tr>
</tbody>
</table>

## FlowMsg类<a name="section7170195113211"></a>

**表 5**  FlowMsg类接口

<a name="table1354843712162"></a>
<table><thead align="left"><tr id="row1154919370163"><th class="cellrowborder" valign="top" width="33.47%" id="mcps1.2.3.1.1"><p id="p1354963712161"><a name="p1354963712161"></a><a name="p1354963712161"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.53%" id="mcps1.2.3.1.2"><p id="p354917378162"><a name="p354917378162"></a><a name="p354917378162"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row1454912374161"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1242994611612"><a name="p1242994611612"></a><a name="p1242994611612"></a><a href="FlowMsg构造函数和析构函数.md">FlowMsg构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001359708654_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001359708654_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001359708654_zh-cn_topic_0000001264921066_p36600850"></a>FlowMsg构造函数和析构函数。</p>
</td>
</tr>
<tr id="row135493379161"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p44281746191615"><a name="p44281746191615"></a><a name="p44281746191615"></a><a href="GetMsgType（FlowMsg类）.md">GetMsgType（FlowMsg类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001409268917_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001409268917_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001409268917_zh-cn_topic_0000001264921066_p36600850"></a>获取FlowMsg的消息类型。</p>
</td>
</tr>
<tr id="row185491737161616"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p19427846151611"><a name="p19427846151611"></a><a name="p19427846151611"></a><a href="GetTensor（FlowMsg类）.md">GetTensor（FlowMsg类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001409388717_p148801903014"><a name="zh-cn_topic_0000001409388717_p148801903014"></a><a name="zh-cn_topic_0000001409388717_p148801903014"></a><span id="ph19722181511114"><a name="ph19722181511114"></a><a name="ph19722181511114"></a>获取FlowMsg中的Tensor指针</span>。</p>
</td>
</tr>
<tr id="row15549637201617"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p3426184661610"><a name="p3426184661610"></a><a name="p3426184661610"></a><a href="SetRetCode（FlowMsg类）.md">SetRetCode（FlowMsg类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001359389150_p489119523119"><a name="zh-cn_topic_0000001359389150_p489119523119"></a><a name="zh-cn_topic_0000001359389150_p489119523119"></a>设置FlowMsg消息中的错误码。</p>
</td>
</tr>
<tr id="row5549143710169"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p184241246131610"><a name="p184241246131610"></a><a name="p184241246131610"></a><a href="GetRetCode（FlowMsg类）.md">GetRetCode（FlowMsg类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001408829017_p254635918010"><a name="zh-cn_topic_0000001408829017_p254635918010"></a><a name="zh-cn_topic_0000001408829017_p254635918010"></a>获取输入FlowMsg消息中的错误码。</p>
</td>
</tr>
<tr id="row0550113771615"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p104235469168"><a name="p104235469168"></a><a name="p104235469168"></a><a href="SetStartTime（FlowMsg类）.md">SetStartTime（FlowMsg类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001359548690_p1562613391532"><a name="zh-cn_topic_0000001359548690_p1562613391532"></a><a name="zh-cn_topic_0000001359548690_p1562613391532"></a>设置FlowMsg消息头中的开始时间戳。</p>
</td>
</tr>
<tr id="row15550193712163"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1542217469163"><a name="p1542217469163"></a><a name="p1542217469163"></a><a href="GetStartTime（FlowMsg类）.md">GetStartTime（FlowMsg类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001409228617_p198905321146"><a name="zh-cn_topic_0000001409228617_p198905321146"></a><a name="zh-cn_topic_0000001409228617_p198905321146"></a>获取FlowMsg消息中的开始时间戳。</p>
</td>
</tr>
<tr id="row2055023701613"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p8421124691618"><a name="p8421124691618"></a><a name="p8421124691618"></a><a href="SetEndTime（FlowMsg类）.md">SetEndTime（FlowMsg类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001359708658_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001359708658_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001359708658_zh-cn_topic_0000001264921066_p36600850"></a>设置FlowMsg消息头中的结束时间戳。</p>
</td>
</tr>
<tr id="row2055043761616"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p2419246141610"><a name="p2419246141610"></a><a name="p2419246141610"></a><a href="GetEndTime（FlowMsg类）.md">GetEndTime（FlowMsg类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001409268921_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001409268921_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001409268921_zh-cn_topic_0000001264921066_p36600850"></a>获取FlowMsg消息中的结束时间戳。</p>
</td>
</tr>
<tr id="row10550123731611"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p15731449111615"><a name="p15731449111615"></a><a name="p15731449111615"></a><a href="SetFlowFlags（FlowMsg类）.md">SetFlowFlags（FlowMsg类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p9012137315"><a name="p9012137315"></a><a name="p9012137315"></a>设置FlowMsg消息头中的flags。</p>
</td>
</tr>
<tr id="row19550173791615"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p772114911613"><a name="p772114911613"></a><a name="p772114911613"></a><a href="GetFlowFlags（FlowMsg类）.md">GetFlowFlags（FlowMsg类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001408829021_p204566483916"><a name="zh-cn_topic_0000001408829021_p204566483916"></a><a name="zh-cn_topic_0000001408829021_p204566483916"></a>获取FlowMsg消息头中的flags。</p>
</td>
</tr>
<tr id="row455083719162"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p8701749141615"><a name="p8701749141615"></a><a name="p8701749141615"></a><a href="SetRouteLabel.md">SetRouteLabel</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1532272883118"><a name="p1532272883118"></a><a name="p1532272883118"></a>设置路由的标签<strong id="b151154742613"><a name="b151154742613"></a><a name="b151154742613"></a>。</strong></p>
</td>
</tr>
<tr id="row5551113710166"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p11695493161"><a name="p11695493161"></a><a name="p11695493161"></a><a href="GetTransactionId（FlowMsg类）.md">GetTransactionId（FlowMsg类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p54541038123110"><a name="p54541038123110"></a><a name="p54541038123110"></a>获取FlowMsg消息中的事务ID，事务ID从1开始计数，每feed一批数据，事务ID会加一，可用于识别哪一批数据<strong id="b244643412426"><a name="b244643412426"></a><a name="b244643412426"></a>。</strong></p>
</td>
</tr>
<tr id="row1760114565264"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p14601356192619"><a name="p14601356192619"></a><a name="p14601356192619"></a><a href="GetTensorList.md">GetTensorList</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1574144514314"><a name="p1574144514314"></a><a name="p1574144514314"></a>返回FlowMsg中所有的Tensor指针列表。</p>
</td>
</tr>
<tr id="row528919582262"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p6289358152617"><a name="p6289358152617"></a><a name="p6289358152617"></a><a href="GetRawData（FlowMsg类）.md">GetRawData（FlowMsg类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1950735413315"><a name="p1950735413315"></a><a name="p1950735413315"></a>获取RawData类型的数据对应的数据指针和数据大小。</p>
</td>
</tr>
<tr id="row101371248424"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1413716474219"><a name="p1413716474219"></a><a name="p1413716474219"></a><a href="SetMsgType（FlowMsg类）.md">SetMsgType（FlowMsg类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p154161840194212"><a name="p154161840194212"></a><a name="p154161840194212"></a>设置FlowMsg的消息类型。</p>
</td>
</tr>
<tr id="row2529106184215"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p175297611423"><a name="p175297611423"></a><a name="p175297611423"></a><a href="SetTransactionId（FlowMsg类）.md">SetTransactionId（FlowMsg类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p125297613427"><a name="p125297613427"></a><a name="p125297613427"></a>设置FlowMsg消息中的事务ID。</p>
</td>
</tr>
</tbody>
</table>

## Tensor类<a name="section1040402383518"></a>

**表 6**  Tensor类接口

<a name="table728210919335"></a>
<table><thead align="left"><tr id="row6283169113319"><th class="cellrowborder" valign="top" width="33.47%" id="mcps1.2.3.1.1"><p id="p62837912338"><a name="p62837912338"></a><a name="p62837912338"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.53%" id="mcps1.2.3.1.2"><p id="p4283149133319"><a name="p4283149133319"></a><a name="p4283149133319"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row628316920333"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p13853121518332"><a name="p13853121518332"></a><a name="p13853121518332"></a><a href="Tensor构造函数和析构函数.md">Tensor构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001357504881_zh-cn_topic_0000001264921070_zh-cn_topic_0204328222_zh-cn_topic_0182636384_p13843256"><a name="zh-cn_topic_0000001357504881_zh-cn_topic_0000001264921070_zh-cn_topic_0204328222_zh-cn_topic_0182636384_p13843256"></a><a name="zh-cn_topic_0000001357504881_zh-cn_topic_0000001264921070_zh-cn_topic_0204328222_zh-cn_topic_0182636384_p13843256"></a>Tensor构造函数和析构函数。</p>
</td>
</tr>
<tr id="row928317913314"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p885251593314"><a name="p885251593314"></a><a name="p885251593314"></a><a href="GetShape.md">GetShape</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001304385288_zh-cn_topic_0000001265240890_p12001237"><a name="zh-cn_topic_0000001304385288_zh-cn_topic_0000001265240890_p12001237"></a><a name="zh-cn_topic_0000001304385288_zh-cn_topic_0000001265240890_p12001237"></a>获取Tensor的Shape。</p>
</td>
</tr>
<tr id="row1128349123312"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p13851161533312"><a name="p13851161533312"></a><a name="p13851161533312"></a><a href="GetDataType.md">GetDataType</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001304225448_zh-cn_topic_0000001265240878_p1589323202015"><a name="zh-cn_topic_0000001304225448_zh-cn_topic_0000001265240878_p1589323202015"></a><a name="zh-cn_topic_0000001304225448_zh-cn_topic_0000001265240878_p1589323202015"></a>获取Tensor中的数据类型。</p>
</td>
</tr>
<tr id="row728339183319"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p188504154339"><a name="p188504154339"></a><a name="p188504154339"></a><a href="GetData.md">GetData</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001304065496_zh-cn_topic_0000001265399790_p149855"><a name="zh-cn_topic_0000001304065496_zh-cn_topic_0000001265399790_p149855"></a><a name="zh-cn_topic_0000001304065496_zh-cn_topic_0000001265399790_p149855"></a>获取Tensor中的数据。</p>
</td>
</tr>
<tr id="row12838993315"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p10848115163311"><a name="p10848115163311"></a><a name="p10848115163311"></a><a href="GetDataSize.md">GetDataSize</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001357384993_zh-cn_topic_0000001265399802_p30971651"><a name="zh-cn_topic_0000001357384993_zh-cn_topic_0000001265399802_p30971651"></a><a name="zh-cn_topic_0000001357384993_zh-cn_topic_0000001265399802_p30971651"></a>获取Tensor中的数据大小。</p>
</td>
</tr>
<tr id="row42831899338"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p158471615193310"><a name="p158471615193310"></a><a name="p158471615193310"></a><a href="GetElementCnt.md">GetElementCnt</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001357345205_p1390215711126"><a name="zh-cn_topic_0000001357345205_p1390215711126"></a><a name="zh-cn_topic_0000001357345205_p1390215711126"></a>获取Tensor中的元素的个数。</p>
</td>
</tr>
<tr id="row02831995337"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p6846171533317"><a name="p6846171533317"></a><a name="p6846171533317"></a><a href="GetDataBufferSize.md">GetDataBufferSize</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p6896131293516"><a name="p6896131293516"></a><a name="p6896131293516"></a>获取Tensor中的对齐后的数据大小。</p>
</td>
</tr>
<tr id="row142841294333"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p784531517337"><a name="p784531517337"></a><a name="p784531517337"></a><a href="Reshape.md">Reshape</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p138081794612"><a name="p138081794612"></a><a name="p138081794612"></a>对Tensor进行Reshape操作，不改变Tensor的内容。</p>
</td>
</tr>
</tbody>
</table>

## MetaFlowFunc类<a name="section13122357374"></a>

**表 7**  MetaFlowFunc类接口

<a name="table27481540355"></a>
<table><thead align="left"><tr id="row5748205453514"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p1474818543359"><a name="p1474818543359"></a><a name="p1474818543359"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p1674855453511"><a name="p1674855453511"></a><a name="p1674855453511"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row147481541359"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p72990015367"><a name="p72990015367"></a><a name="p72990015367"></a><a href="MetaFlowFunc构造函数和析构函数.md">MetaFlowFunc构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_p13843256"><a name="zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_p13843256"></a><a name="zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_p13843256"></a>用户继承该类进行自定义的单func处理函数的编写。在析构函数中，执行释放相关资源操作。</p>
</td>
</tr>
<tr id="row147489544352"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p152981404364"><a name="p152981404364"></a><a name="p152981404364"></a><a href="SetContext.md">SetContext</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001304385296_p735218494917"><a name="zh-cn_topic_0000001304385296_p735218494917"></a><a name="zh-cn_topic_0000001304385296_p735218494917"></a>设置flow func的上下文信息。</p>
</td>
</tr>
<tr id="row15748165411357"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p9297901364"><a name="p9297901364"></a><a name="p9297901364"></a><a href="Init（MetaFlowFunc类）.md">Init（MetaFlowFunc类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001304225456_p179521451181018"><a name="zh-cn_topic_0000001304225456_p179521451181018"></a><a name="zh-cn_topic_0000001304225456_p179521451181018"></a>用户自定义flow func的初始化函数。</p>
</td>
</tr>
<tr id="row15748125463517"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p14296140163616"><a name="p14296140163616"></a><a name="p14296140163616"></a><a href="Proc.md">Proc</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001357479629_p179521451181018"><a name="zh-cn_topic_0000001357479629_p179521451181018"></a><a name="zh-cn_topic_0000001357479629_p179521451181018"></a>用户自定义flow func的处理函数。</p>
</td>
</tr>
<tr id="row574835412354"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p182955017366"><a name="p182955017366"></a><a name="p182955017366"></a><a href="RegisterFlowFunc.md">RegisterFlowFunc</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p1967710143379"><a name="p1967710143379"></a><a name="p1967710143379"></a>注册flow func。</p>
<p id="p1410314317275"><a name="p1410314317275"></a><a name="p1410314317275"></a>不建议直接使用该函数，建议使用<a href="MetaFlowFunc注册函数宏.md">MetaFlowFunc注册函数宏</a>来注册flow func。</p>
</td>
</tr>
<tr id="row1063619451530"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p463617454310"><a name="p463617454310"></a><a name="p463617454310"></a><a href="ResetFlowFuncState（MetaFlowFunc类）.md">ResetFlowFuncState（MetaFlowFunc类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p35821780416"><a name="p35821780416"></a><a name="p35821780416"></a>在故障恢复场景下，快速重置FlowFunc为初始化状态。</p>
</td>
</tr>
<tr id="row5749165415358"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p17294006364"><a name="p17294006364"></a><a name="p17294006364"></a><a href="其他.md">其他</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p229330113611"><a name="p229330113611"></a><a name="p229330113611"></a>REGISTER_FLOW_FUNC_INNER(name, ctr, clazz)和REGISTER_FLOW_FUNC_IMPL(name, ctr, clazz)是<a href="MetaFlowFunc注册函数宏.md">MetaFlowFunc注册函数宏</a>的实现，不建议用户直接调用。</p>
</td>
</tr>
</tbody>
</table>

## MetaMultiFunc类<a name="section14924013103919"></a>

**表 8**  MetaMultiFunc类接口

<a name="table1347510117383"></a>
<table><thead align="left"><tr id="row847511103815"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p184752173812"><a name="p184752173812"></a><a name="p184752173812"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p10475151163810"><a name="p10475151163810"></a><a name="p10475151163810"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row9475151113816"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p122511478387"><a name="p122511478387"></a><a name="p122511478387"></a><a href="MetaMultiFunc构造函数和析构函数.md">MetaMultiFunc构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p768264473819"><a name="p768264473819"></a><a name="p768264473819"></a>用户继承该类进行自定义的多func处理函数的编写。在析构函数中，执行释放相关资源操作。</p>
</td>
</tr>
<tr id="row047614119383"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1824918715386"><a name="p1824918715386"></a><a name="p1824918715386"></a><a href="Init（MetaMultiFunc类）.md">Init（MetaMultiFunc类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p538335114384"><a name="p538335114384"></a><a name="p538335114384"></a>用户自定义flow func的初始化函数。</p>
</td>
</tr>
<tr id="row2047613163812"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1324818720383"><a name="p1324818720383"></a><a name="p1324818720383"></a><a href="多func处理函数.md">多func处理函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p51921657113817"><a name="p51921657113817"></a><a name="p51921657113817"></a>用户自定义多flow func的处理函数。</p>
</td>
</tr>
<tr id="row647691163816"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p16247187163811"><a name="p16247187163811"></a><a name="p16247187163811"></a><a href="RegisterMultiFunc.md">RegisterMultiFunc</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p09461684397"><a name="p09461684397"></a><a name="p09461684397"></a>注册多flow func。</p>
<p id="p29721659193613"><a name="p29721659193613"></a><a name="p29721659193613"></a>不建议直接使用该函数，建议使用<a href="MetaMultiFunc注册函数宏.md">MetaMultiFunc注册函数宏</a>来注册flow func。</p>
</td>
</tr>
<tr id="row67296291243"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p272913293410"><a name="p272913293410"></a><a name="p272913293410"></a><a href="ResetFlowFuncState（MetaMultiFunc类）.md">ResetFlowFuncState（MetaMultiFunc类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p191856451947"><a name="p191856451947"></a><a name="p191856451947"></a>在故障恢复场景下，快速重置FlowFunc为初始化状态。</p>
</td>
</tr>
</tbody>
</table>

## FlowFuncRegistrar类<a name="section147441311405"></a>

**表 9**  FlowFuncRegistrar类接口

<a name="table2033923963919"></a>
<table><thead align="left"><tr id="row634043916397"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p7340639183913"><a name="p7340639183913"></a><a name="p7340639183913"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p17340739103911"><a name="p17340739103911"></a><a name="p17340739103911"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row1834018391397"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p14457164215399"><a name="p14457164215399"></a><a name="p14457164215399"></a><a href="RegProcFunc.md">RegProcFunc</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p16448103174012"><a name="p16448103174012"></a><a name="p16448103174012"></a>注册多flow func处理函数，结合<a href="MetaMultiFunc注册函数宏.md">MetaMultiFunc注册函数宏</a>来注册flow func。</p>
</td>
</tr>
<tr id="row1334019393393"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1045564211395"><a name="p1045564211395"></a><a name="p1045564211395"></a><a href="CreateMultiFunc.md">CreateMultiFunc</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p211911864013"><a name="p211911864013"></a><a name="p211911864013"></a>创建多func处理对象和处理函数，框架内部使用，用户不直接使用。</p>
</td>
</tr>
</tbody>
</table>

## MetaParams类<a name="section20904915437"></a>

**表 10**  MetaParams类接口

<a name="table16541353174018"></a>
<table><thead align="left"><tr id="row1754111532406"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p254245319403"><a name="p254245319403"></a><a name="p254245319403"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p15421253194012"><a name="p15421253194012"></a><a name="p15421253194012"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row19542253164018"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p91881022418"><a name="p91881022418"></a><a name="p91881022418"></a><a href="MetaParams构造函数和析构函数.md">MetaParams构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p146821285420"><a name="p146821285420"></a><a name="p146821285420"></a>MetaParams构造函数和析构函数。</p>
</td>
</tr>
<tr id="row8542553174015"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p418613224119"><a name="p418613224119"></a><a name="p418613224119"></a><a href="GetName.md">GetName</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p5432171417427"><a name="p5432171417427"></a><a name="p5432171417427"></a>获取Flowfunc的实例名。</p>
</td>
</tr>
<tr id="row65421853174019"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p218514217418"><a name="p218514217418"></a><a name="p218514217418"></a><a href="GetAttr（MetaParams类-获取指针）.md">GetAttr（MetaParams类，获取指针）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002013837129_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"><a name="zh-cn_topic_0000002013837129_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a><a name="zh-cn_topic_0000002013837129_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a>根据属性名获取AttrValue类型的指针。该函数供<a href="Init（MetaFlowFunc类）.md">Init（MetaFlowFunc类）</a>调用。</p>
</td>
</tr>
<tr id="row1954265316403"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p81841274116"><a name="p81841274116"></a><a name="p81841274116"></a><a href="GetAttr（MetaParams类-获取属性值）.md">GetAttr（MetaParams类，获取属性值）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001977316766_zh-cn_topic_0000001357345209_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001977316766_zh-cn_topic_0000001357345209_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001977316766_zh-cn_topic_0000001357345209_zh-cn_topic_0000001264921066_p36600850"></a>根据属性名获取对应的属性值。该函数供<a href="Init（MetaFlowFunc类）.md">Init（MetaFlowFunc类）</a>调用。</p>
</td>
</tr>
<tr id="row6542195344016"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p181822274110"><a name="p181822274110"></a><a name="p181822274110"></a><a href="GetInputNum（MetaParams类）.md">GetInputNum（MetaParams类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002013796625_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000002013796625_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000002013796625_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a>获取Flowfunc的输入个数。该函数供<a href="Init（MetaFlowFunc类）.md">Init（MetaFlowFunc类）</a>调用。</p>
</td>
</tr>
<tr id="row1854212531406"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p21801925412"><a name="p21801925412"></a><a name="p21801925412"></a><a href="GetOutputNum（MetaParams类）.md">GetOutputNum（MetaParams类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002013837137_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000002013837137_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000002013837137_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a>获取Flowfunc的输出个数。该函数供<a href="Init（MetaFlowFunc类）.md">Init（MetaFlowFunc类）</a>调用。</p>
</td>
</tr>
<tr id="row6903174214120"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p890314224114"><a name="p890314224114"></a><a name="p890314224114"></a><a href="GetWorkPath（MetaParams类）.md">GetWorkPath（MetaParams类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001977316774_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001977316774_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001977316774_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a>获取Flowfunc的工作路径。该函数供<a href="Init（MetaFlowFunc类）.md">Init（MetaFlowFunc类）</a>调用。</p>
</td>
</tr>
<tr id="row79374110414"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p13931941194113"><a name="p13931941194113"></a><a name="p13931941194113"></a><a href="GetRunningDeviceId（MetaParams类）.md">GetRunningDeviceId（MetaParams类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p8321185613427"><a name="p8321185613427"></a><a name="p8321185613427"></a>获取正在运行的设备ID。</p>
</td>
</tr>
</tbody>
</table>

## MetaRunContext类<a name="section154853131118"></a>

**表 11**  MetaRunContext类接口

<a name="table126251824204317"></a>
<table><thead align="left"><tr id="row362582464314"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p86251124134316"><a name="p86251124134316"></a><a name="p86251124134316"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p4626202434316"><a name="p4626202434316"></a><a name="p4626202434316"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row17626152444310"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p16623730104316"><a name="p16623730104316"></a><a name="p16623730104316"></a><a href="MetaRunContext构造函数和析构函数.md">MetaRunContext构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p111111320104"><a name="p111111320104"></a><a name="p111111320104"></a>MetaRunContext构造函数和析构函数。</p>
</td>
</tr>
<tr id="row186260242438"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p66220303436"><a name="p66220303436"></a><a name="p66220303436"></a><a href="AllocTensorMsg（MetaRunContext类）.md">AllocTensorMsg（MetaRunContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001977316758_zh-cn_topic_0000001304225452_p1476471614810"><a name="zh-cn_topic_0000001977316758_zh-cn_topic_0000001304225452_p1476471614810"></a><a name="zh-cn_topic_0000001977316758_zh-cn_topic_0000001304225452_p1476471614810"></a>根据shape和data type申请<span id="ph1883164710915"><a name="ph1883164710915"></a><a name="ph1883164710915"></a>T</span>ensor类型的msg。该函数供<a href="Proc.md">Proc</a>调用。</p>
</td>
</tr>
<tr id="row1062682454317"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p362018309434"><a name="p362018309434"></a><a name="p362018309434"></a><a href="SetOutput（MetaRunContext类-tensor）.md">SetOutput（MetaRunContext类,tensor）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002013796617_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312640889_p4089365"><a name="zh-cn_topic_0000002013796617_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312640889_p4089365"></a><a name="zh-cn_topic_0000002013796617_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312640889_p4089365"></a>设置指定index的output的tensor。该函数供<a href="Proc.md">Proc</a>调用。</p>
</td>
</tr>
<tr id="row186271424104316"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1861963014311"><a name="p1861963014311"></a><a name="p1861963014311"></a><a href="RunFlowModel（MetaRunContext类）.md">RunFlowModel（MetaRunContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001977157062_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"><a name="zh-cn_topic_0000001977157062_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a><a name="zh-cn_topic_0000001977157062_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_p36600850"></a>同步执行指定的模型。该函数供<a href="Proc.md">Proc</a>调用。</p>
</td>
</tr>
<tr id="row206281724164310"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1061817307435"><a name="p1061817307435"></a><a name="p1061817307435"></a><a href="AllocEmptyDataMsg（MetaRunContext类）.md">AllocEmptyDataMsg（MetaRunContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p182498891120"><a name="p182498891120"></a><a name="p182498891120"></a>申请空数据的MsgType类型的message。</p>
</td>
</tr>
<tr id="row9628192414317"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p3617133084317"><a name="p3617133084317"></a><a name="p3617133084317"></a><a href="GetUserData（MetaRunContext类）.md">GetUserData（MetaRunContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002013796629_zh-cn_topic_0000001409388721_p186651645278"><a name="zh-cn_topic_0000002013796629_zh-cn_topic_0000001409388721_p186651645278"></a><a name="zh-cn_topic_0000002013796629_zh-cn_topic_0000001409388721_p186651645278"></a>获取用户数据。该函数供<a href="Proc.md">Proc</a>调用。</p>
</td>
</tr>
<tr id="row462932418439"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p8616230164313"><a name="p8616230164313"></a><a name="p8616230164313"></a><a href="SetOutput（MetaRunContext类-输出）.md">SetOutput（MetaRunContext类,输出）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p2507182111110"><a name="p2507182111110"></a><a name="p2507182111110"></a>设置指定index和options的输出，该函数供func函数调用。</p>
</td>
</tr>
<tr id="row6629524154315"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p17614103012435"><a name="p17614103012435"></a><a name="p17614103012435"></a><a href="SetMultiOutputs.md">SetMultiOutputs</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p2090011308111"><a name="p2090011308111"></a><a name="p2090011308111"></a>批量设置指定index和options的输出，该函数供func函数调用。</p>
</td>
</tr>
<tr id="row94093381013"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p340923812016"><a name="p340923812016"></a><a name="p340923812016"></a><a href="AllocTensorMsgWithAlign（MetaRunContext类）.md">AllocTensorMsgWithAlign（MetaRunContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002013837145_p3702153414173"><a name="zh-cn_topic_0000002013837145_p3702153414173"></a><a name="zh-cn_topic_0000002013837145_p3702153414173"></a>根据shape、data type<span id="ph14976129286"><a name="ph14976129286"></a><a name="ph14976129286"></a>和对齐大小申请Tensor</span>类型的FlowMsg，与<a href="AllocTensorMsg（MetaContext类）.md">AllocTensorMsg</a>函数区别是AllocTensorMsg默认申请以64字节对齐，此函数可以指定对齐大小，方便性能调优。</p>
</td>
</tr>
<tr id="row158221351006"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1282210351301"><a name="p1282210351301"></a><a name="p1282210351301"></a><a href="AllocTensorListMsg.md">AllocTensorListMsg</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001913682208_p3702153414173"><a name="zh-cn_topic_0000001913682208_p3702153414173"></a><a name="zh-cn_topic_0000001913682208_p3702153414173"></a>根据输入的dtype shapes数组分配一块连续内存，<span id="ph845412141864"><a name="ph845412141864"></a><a name="ph845412141864"></a>用于承载Tensor数组。</span></p>
</td>
</tr>
<tr id="row174903885118"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p9345114195116"><a name="p9345114195116"></a><a name="p9345114195116"></a><a href="RaiseException（MetaRunContext类）.md">RaiseException（MetaRunContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p14586141195111"><a name="p14586141195111"></a><a name="p14586141195111"></a>UDF主动上报异常，该异常可以被同作用域内的其他UDF捕获。</p>
</td>
</tr>
<tr id="row128329105517"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p8344181495115"><a name="p8344181495115"></a><a name="p8344181495115"></a><a href="GetException（MetaRunContext类）.md">GetException（MetaRunContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p185876115518"><a name="p185876115518"></a><a name="p185876115518"></a>UDF获取异常，如果开启了异常捕获功能，需要在UDF中Proc函数开始位置尝试捕获异常。</p>
</td>
</tr>
<tr id="row11928256174816"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p392815624819"><a name="p392815624819"></a><a name="p392815624819"></a><a href="AllocRawDataMsg（MetaRunContext类）.md">AllocRawDataMsg（MetaRunContext类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p4981102612497"><a name="p4981102612497"></a><a name="p4981102612497"></a>根据输入的size申请一块连续内存，用于承载RawData类型的数据。</p>
</td>
</tr>
<tr id="row122846594486"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p2284195934819"><a name="p2284195934819"></a><a name="p2284195934819"></a><a href="ToFlowMsg.md">ToFlowMsg</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p923353604913"><a name="p923353604913"></a><a name="p923353604913"></a>根据输入的Tensor转换成用于承载Tensor的FlowMsg。</p>
</td>
</tr>
</tbody>
</table>

## OutOptions类<a name="section151011117141516"></a>

**表 12**  OutOptions类接口

<a name="table17894172174410"></a>
<table><thead align="left"><tr id="row20894325447"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p38948284411"><a name="p38948284411"></a><a name="p38948284411"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p1189411224414"><a name="p1189411224414"></a><a name="p1189411224414"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row5894172174411"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p78942024447"><a name="p78942024447"></a><a name="p78942024447"></a><a href="OutOptions构造函数和析构函数.md">OutOptions构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p126641757131412"><a name="p126641757131412"></a><a name="p126641757131412"></a>OutOptions的构造和析构函数。</p>
</td>
</tr>
<tr id="row3895724443"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p38951214412"><a name="p38951214412"></a><a name="p38951214412"></a><a href="MutableBalanceConfig.md">MutableBalanceConfig</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p17115181163113"><a name="p17115181163113"></a><a name="p17115181163113"></a>获取或创建BalanceConfig。</p>
</td>
</tr>
<tr id="row1989516214419"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p0895162134415"><a name="p0895162134415"></a><a name="p0895162134415"></a><a href="GetBalanceConfig.md">GetBalanceConfig</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p10161199181516"><a name="p10161199181516"></a><a name="p10161199181516"></a>获取BalanceConfig。</p>
</td>
</tr>
</tbody>
</table>

## BalanceConfig类<a name="section9368163171611"></a>

**表 13**  BalanceConfig类接口

<a name="table92314512441"></a>
<table><thead align="left"><tr id="row192312517445"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p6238594413"><a name="p6238594413"></a><a name="p6238594413"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p62310511442"><a name="p62310511442"></a><a name="p62310511442"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row19241656443"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p42416554414"><a name="p42416554414"></a><a name="p42416554414"></a><a href="BalanceConfig构造函数和析构函数.md">BalanceConfig构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p11211046131517"><a name="p11211046131517"></a><a name="p11211046131517"></a>BalanceConfig的构造和析构函数。</p>
</td>
</tr>
<tr id="row122416554413"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p82418510444"><a name="p82418510444"></a><a name="p82418510444"></a><a href="SetAffinityPolicy.md">SetAffinityPolicy</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p129411552111519"><a name="p129411552111519"></a><a name="p129411552111519"></a>设置均衡分发亲和性。</p>
</td>
</tr>
<tr id="row1424758443"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p17246564418"><a name="p17246564418"></a><a name="p17246564418"></a><a href="GetAffinityPolicy.md">GetAffinityPolicy</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p22066281612"><a name="p22066281612"></a><a name="p22066281612"></a>获取亲和性。</p>
</td>
</tr>
<tr id="row1924157447"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p62425104417"><a name="p62425104417"></a><a name="p62425104417"></a><a href="SetBalanceWeight.md">SetBalanceWeight</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p1028113711169"><a name="p1028113711169"></a><a name="p1028113711169"></a>设置均衡分发权重信息。</p>
</td>
</tr>
<tr id="row172411534410"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p424115134414"><a name="p424115134414"></a><a name="p424115134414"></a><a href="GetBalanceWeight.md">GetBalanceWeight</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p1520741381615"><a name="p1520741381615"></a><a name="p1520741381615"></a>获取均衡分发权重信息。</p>
</td>
</tr>
<tr id="row14244534411"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p424750448"><a name="p424750448"></a><a name="p424750448"></a><a href="SetDataPos.md">SetDataPos</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001307722046_p179521451181018"><a name="zh-cn_topic_0000001307722046_p179521451181018"></a><a name="zh-cn_topic_0000001307722046_p179521451181018"></a>设置输出数据对应权重矩阵中的位置。</p>
</td>
</tr>
<tr id="row122419511445"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1924195144415"><a name="p1924195144415"></a><a name="p1924195144415"></a><a href="GetDataPos.md">GetDataPos</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p196132025151612"><a name="p196132025151612"></a><a name="p196132025151612"></a>获取输出数据对应权重矩阵中的位置。</p>
</td>
</tr>
</tbody>
</table>

## FlowBufferFactory类<a name="section12321631175012"></a>

**表 14**  FlowBufferFactory类接口

<a name="table1023353111503"></a>
<table><thead align="left"><tr id="row123393116508"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p22331031125020"><a name="p22331031125020"></a><a name="p22331031125020"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p323393119505"><a name="p323393119505"></a><a name="p323393119505"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row112331331175012"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p4311248175014"><a name="p4311248175014"></a><a name="p4311248175014"></a><a href="AllocTensor（FlowBufferFactory类）.md">AllocTensor（FlowBufferFactory类）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p15310348205010"><a name="p15310348205010"></a><a name="p15310348205010"></a>根据shape、data type<span id="ph19612710088"><a name="ph19612710088"></a><a name="ph19612710088"></a>和对齐大小申请Tensor</span>，默认申请以64字节对齐，可以指定对齐大小，方便性能调优。</p>
</td>
</tr>
</tbody>
</table>

## FlowMsgQueue类<a name="section17983582079"></a>

**表 15**  FlowMsgQueue类接口

<a name="table398165813719"></a>
<table><thead align="left"><tr id="row4984585715"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p209810585716"><a name="p209810585716"></a><a name="p209810585716"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p18984581179"><a name="p18984581179"></a><a name="p18984581179"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row10988584713"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1162812191186"><a name="p1162812191186"></a><a name="p1162812191186"></a><a href="FlowMsgQueue构造函数和析构函数.md">FlowMsgQueue构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p058441791015"><a name="p058441791015"></a><a name="p058441791015"></a>FlowMsgQueue的构造和析构函数。</p>
</td>
</tr>
<tr id="row2061810225913"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p16181222898"><a name="p16181222898"></a><a name="p16181222898"></a><a href="Dequeue.md">Dequeue</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p1476732419109"><a name="p1476732419109"></a><a name="p1476732419109"></a>设置均衡分发亲和性。</p>
</td>
</tr>
<tr id="row1210572515916"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1410512515912"><a name="p1410512515912"></a><a name="p1410512515912"></a><a href="Depth.md">Depth</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p3280153301020"><a name="p3280153301020"></a><a name="p3280153301020"></a>获取队列的深度，即获取队列可容纳元素的最大个数。</p>
</td>
</tr>
<tr id="row139606264917"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p096062618916"><a name="p096062618916"></a><a name="p096062618916"></a><a href="Size.md">Size</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p3911640161017"><a name="p3911640161017"></a><a name="p3911640161017"></a>获取队列中当前元素的个数。</p>
</td>
</tr>
</tbody>
</table>

## 注册宏<a name="section13640141081710"></a>

**表 16**  注册宏

<a name="table108741860449"></a>
<table><thead align="left"><tr id="row108749618441"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p118745674411"><a name="p118745674411"></a><a name="p118745674411"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p1887414613444"><a name="p1887414613444"></a><a name="p1887414613444"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row12875166124416"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p108751165440"><a name="p108751165440"></a><a name="p108751165440"></a><a href="MetaFlowFunc注册函数宏.md">MetaFlowFunc注册函数宏</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p17688163114811"><a name="p17688163114811"></a><a name="p17688163114811"></a>注册MetaFlowFunc的实现类。</p>
</td>
</tr>
<tr id="row1387516614418"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p15875116144420"><a name="p15875116144420"></a><a name="p15875116144420"></a><a href="MetaMultiFunc注册函数宏.md">MetaMultiFunc注册函数宏</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p104181612691"><a name="p104181612691"></a><a name="p104181612691"></a>注册MetaMultiFunc的实现类。</p>
</td>
</tr>
</tbody>
</table>

## UDF日志接口<a name="section663155413194"></a>

**表 17**  UDF日志接口

<a name="table675418912447"></a>
<table><thead align="left"><tr id="row167548918448"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p17541190448"><a name="p17541190448"></a><a name="p17541190448"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p27541799441"><a name="p27541799441"></a><a name="p27541799441"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row17541493448"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p13754891443"><a name="p13754891443"></a><a name="p13754891443"></a><a href="FlowFuncLogger构造函数和析构函数.md">FlowFuncLogger构造函数和析构函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p109221308179"><a name="p109221308179"></a><a name="p109221308179"></a>FlowFuncLogger构造函数和析构函数。</p>
</td>
</tr>
<tr id="row37541493449"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p117541996444"><a name="p117541996444"></a><a name="p117541996444"></a><a href="GetLogger.md">GetLogger</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p369210361170"><a name="p369210361170"></a><a name="p369210361170"></a>获取日志实现类。</p>
</td>
</tr>
<tr id="row57541290441"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1175449154412"><a name="p1175449154412"></a><a name="p1175449154412"></a><a href="GetLogExtHeader.md">GetLogExtHeader</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p59145430171"><a name="p59145430171"></a><a name="p59145430171"></a>获取日志扩展头信息。</p>
</td>
</tr>
<tr id="row6337191318513"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p20337213757"><a name="p20337213757"></a><a name="p20337213757"></a><a href="IsLogEnable.md">IsLogEnable</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p1782943418189"><a name="p1782943418189"></a><a name="p1782943418189"></a>查询对应级别和类型的日志是否开启。</p>
</td>
</tr>
<tr id="row10337213355"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p18297193412515"><a name="p18297193412515"></a><a name="p18297193412515"></a><a href="Error.md">Error</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p128151641111810"><a name="p128151641111810"></a><a name="p128151641111810"></a>记录ERROR级别日志。</p>
</td>
</tr>
<tr id="row1233712131556"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p5337141310515"><a name="p5337141310515"></a><a name="p5337141310515"></a><a href="Warn.md">Warn</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p10521204741810"><a name="p10521204741810"></a><a name="p10521204741810"></a>记录Warn级别日志。</p>
</td>
</tr>
<tr id="row13337413350"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p13371413259"><a name="p13371413259"></a><a name="p13371413259"></a><a href="Info.md">Info</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p66415315188"><a name="p66415315188"></a><a name="p66415315188"></a>记录Info级别日志。</p>
</td>
</tr>
<tr id="row1333715134519"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p233881311516"><a name="p233881311516"></a><a name="p233881311516"></a><a href="Debug.md">Debug</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p879295891816"><a name="p879295891816"></a><a name="p879295891816"></a>记录Debug级别日志。</p>
</td>
</tr>
<tr id="row87552964420"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p27559994413"><a name="p27559994413"></a><a name="p27559994413"></a><a href="运行日志Error级别日志宏.md">运行日志Error级别日志宏</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p35457491919"><a name="p35457491919"></a><a name="p35457491919"></a>运行日志Error级别日志宏。</p>
</td>
</tr>
<tr id="row275518911448"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p167551991449"><a name="p167551991449"></a><a name="p167551991449"></a><a href="运行日志Info级别日志宏.md">运行日志Info级别日志宏</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p94568112192"><a name="p94568112192"></a><a name="p94568112192"></a>运行日志Info级别日志宏。</p>
</td>
</tr>
<tr id="row17755149164412"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p11755159164416"><a name="p11755159164416"></a><a name="p11755159164416"></a><a href="调试日志Error级别日志宏.md">调试日志Error级别日志宏</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p178061118101913"><a name="p178061118101913"></a><a name="p178061118101913"></a>调试日志Error级别日志宏。</p>
</td>
</tr>
<tr id="row107556934415"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p875512914412"><a name="p875512914412"></a><a name="p875512914412"></a><a href="调试日志Warn级别日志宏.md">调试日志Warn级别日志宏</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p28172611913"><a name="p28172611913"></a><a name="p28172611913"></a>调试日志Warn级别日志宏。</p>
</td>
</tr>
<tr id="row14755698447"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p16755391446"><a name="p16755391446"></a><a name="p16755391446"></a><a href="调试日志Info级别日志宏.md">调试日志Info级别日志宏</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p984163271911"><a name="p984163271911"></a><a name="p984163271911"></a>调试日志Info级别日志宏。</p>
</td>
</tr>
<tr id="row10952129714"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p595812979"><a name="p595812979"></a><a name="p595812979"></a><a href="调试日志Debug级别日志宏.md">调试日志Debug级别日志宏</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p1679911371191"><a name="p1679911371191"></a><a name="p1679911371191"></a>调试日志Debug级别日志宏。</p>
</td>
</tr>
</tbody>
</table>

## 错误码<a name="section9889354201"></a>

**表 18**  错误码

<a name="table9675420718"></a>
<table><thead align="left"><tr id="row13716541376"><th class="cellrowborder" valign="top" width="33.44%" id="mcps1.2.3.1.1"><p id="p2719544718"><a name="p2719544718"></a><a name="p2719544718"></a>错误码模块</p>
</th>
<th class="cellrowborder" valign="top" width="66.56%" id="mcps1.2.3.1.2"><p id="p3717545715"><a name="p3717545715"></a><a name="p3717545715"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row117654179"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p11569023497"><a name="p11569023497"></a><a name="p11569023497"></a><a href="UDF错误码.md#section1390959132616">flowfunc</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p12710541177"><a name="p12710541177"></a><a name="p12710541177"></a>提供了flowfunc的错误码供用户使用，主要用于对异常逻辑的判断处理。</p>
</td>
</tr>
<tr id="row1675547715"><td class="cellrowborder" valign="top" width="33.44%" headers="mcps1.2.3.1.1 "><p id="p1556814234916"><a name="p1556814234916"></a><a name="p1556814234916"></a><a href="UDF错误码.md#section119131377263">AICPU</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.56%" headers="mcps1.2.3.1.2 "><p id="p20785416711"><a name="p20785416711"></a><a name="p20785416711"></a>AICPU在执行模型的过程中，有可能向用户上报的错误码。</p>
</td>
</tr>
</tbody>
</table>

