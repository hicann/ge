# GraphProcessPoint<a name="ZH-CN_TOPIC_0000002181418389"></a>

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

GraphProcessPoint构造函数，返回一个GraphProcessPoint对象。

## 函数原型<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section84161445741"></a>

```
GraphProcessPoint(framework, graph_file, load_params={}, compile_config_path="", name=None)
```

## 参数说明<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_section63604780"></a>

<a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_table2051894852017"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row4558174815206"><th class="cellrowborder" valign="top" width="22.220000000000002%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"></a><strong id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="15.06%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="62.72%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"></a><strong id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"></a>取值说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row35581048202018"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p076384473910"><a name="p076384473910"></a><a name="p076384473910"></a>framework</p>
</td>
<td class="cellrowborder" valign="top" width="15.06%" headers="mcps1.1.4.1.2 "><p id="p77639449393"><a name="p77639449393"></a><a name="p77639449393"></a>Framework</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="p11762164443917"><a name="p11762164443917"></a><a name="p11762164443917"></a>IR文件的框架类型，详见<a href="dataflow-Framework.md">dataflow.Framework</a>。</p>
</td>
</tr>
<tr id="row1432294943917"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p63239497394"><a name="p63239497394"></a><a name="p63239497394"></a>graph_file</p>
</td>
<td class="cellrowborder" valign="top" width="15.06%" headers="mcps1.1.4.1.2 "><p id="p17323124963915"><a name="p17323124963915"></a><a name="p17323124963915"></a>str</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="p173231449193917"><a name="p173231449193917"></a><a name="p173231449193917"></a>IR文件路径。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row6558548162013"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p135696184217"><a name="p135696184217"></a><a name="p135696184217"></a>load_params</p>
</td>
<td class="cellrowborder" valign="top" width="15.06%" headers="mcps1.1.4.1.2 "><p id="p20561061428"><a name="p20561061428"></a><a name="p20561061428"></a>Dict[str, str]</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="p83431614162713"><a name="p83431614162713"></a><a name="p83431614162713"></a>配置参数map映射表，key为参数类型，value为参数值，均为String格式，用于描述原始模型解析参数。</p>
</td>
</tr>
<tr id="row772313478391"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p5723174723920"><a name="p5723174723920"></a><a name="p5723174723920"></a>compile_config_path</p>
</td>
<td class="cellrowborder" valign="top" width="15.06%" headers="mcps1.1.4.1.2 "><p id="p15848144216429"><a name="p15848144216429"></a><a name="p15848144216429"></a>str</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="p187246478395"><a name="p187246478395"></a><a name="p187246478395"></a>编译graph时的配置文件路径。</p>
<p id="p5878906477"><a name="p5878906477"></a><a name="p5878906477"></a>配置文件实例如下：</p>
<pre class="screen" id="zh-cn_topic_0000001411032876_screen198482475239"><a name="zh-cn_topic_0000001411032876_screen198482475239"></a><a name="zh-cn_topic_0000001411032876_screen198482475239"></a>{"build_option":{},"inputs_tensor_desc":[{"data_type":"DT_UINT32","shape":[3]},{"data_type":"DT_UINT32","shape":[3]}]}</pre>
</td>
</tr>
<tr id="row09381350143913"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p1093985013393"><a name="p1093985013393"></a><a name="p1093985013393"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="15.06%" headers="mcps1.1.4.1.2 "><p id="p1852642104211"><a name="p1852642104211"></a><a name="p1852642104211"></a>str</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="p18939185011391"><a name="p18939185011391"></a><a name="p18939185011391"></a>处理点名称，框架会自动保证名称唯一，不设置时会自动生成GraphProcessPoint, GraphProcessPoint_1, GraphProcessPoint_2,...的名称。</p>
</td>
</tr>
</tbody>
</table>

**表 1**  GraphProcessPoint的json配置文件

<a name="zh-cn_topic_0000001411032876_table1179952915232"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001411032876_row779992992311"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001411032876_p2799429182313"><a name="zh-cn_topic_0000001411032876_p2799429182313"></a><a name="zh-cn_topic_0000001411032876_p2799429182313"></a>配置项</p>
</th>
<th class="cellrowborder" valign="top" width="27.87%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001411032876_p197991229102314"><a name="zh-cn_topic_0000001411032876_p197991229102314"></a><a name="zh-cn_topic_0000001411032876_p197991229102314"></a>可选/必选</p>
</th>
<th class="cellrowborder" valign="top" width="44.5%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001411032876_p680013295233"><a name="zh-cn_topic_0000001411032876_p680013295233"></a><a name="zh-cn_topic_0000001411032876_p680013295233"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001411032876_row7800029102314"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411032876_p78211813182511"><a name="zh-cn_topic_0000001411032876_p78211813182511"></a><a name="zh-cn_topic_0000001411032876_p78211813182511"></a>build_options</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411032876_p19821213112516"><a name="zh-cn_topic_0000001411032876_p19821213112516"></a><a name="zh-cn_topic_0000001411032876_p19821213112516"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411032876_p19821213152513"><a name="zh-cn_topic_0000001411032876_p19821213152513"></a><a name="zh-cn_topic_0000001411032876_p19821213152513"></a>值为map&lt;string, string&gt;, 有需要设置时参考Ascend Graph中的build_options。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411032876_row1728411013253"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411032876_p14821513132519"><a name="zh-cn_topic_0000001411032876_p14821513132519"></a><a name="zh-cn_topic_0000001411032876_p14821513132519"></a>inputs_tensor_desc</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411032876_p1482181310256"><a name="zh-cn_topic_0000001411032876_p1482181310256"></a><a name="zh-cn_topic_0000001411032876_p1482181310256"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411032876_p1182151310258"><a name="zh-cn_topic_0000001411032876_p1182151310258"></a><a name="zh-cn_topic_0000001411032876_p1182151310258"></a>值为list，Graph的输入节点，list元素为tensor的描述。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411032876_row1413118317257"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411032876_p682119137252"><a name="zh-cn_topic_0000001411032876_p682119137252"></a><a name="zh-cn_topic_0000001411032876_p682119137252"></a>inputs_tensor_desc.data_type</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411032876_p128211613182515"><a name="zh-cn_topic_0000001411032876_p128211613182515"></a><a name="zh-cn_topic_0000001411032876_p128211613182515"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411032876_p16949428681"><a name="zh-cn_topic_0000001411032876_p16949428681"></a><a name="zh-cn_topic_0000001411032876_p16949428681"></a>字符串类型。</p>
<p id="zh-cn_topic_0000001411032876_p282191320252"><a name="zh-cn_topic_0000001411032876_p282191320252"></a><a name="zh-cn_topic_0000001411032876_p282191320252"></a>取值为Graph中的data_type对应字符串。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411032876_row360312532518"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411032876_p11821101342518"><a name="zh-cn_topic_0000001411032876_p11821101342518"></a><a name="zh-cn_topic_0000001411032876_p11821101342518"></a>inputs_tensor_desc.shape</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411032876_p19822141318253"><a name="zh-cn_topic_0000001411032876_p19822141318253"></a><a name="zh-cn_topic_0000001411032876_p19822141318253"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411032876_p78227136254"><a name="zh-cn_topic_0000001411032876_p78227136254"></a><a name="zh-cn_topic_0000001411032876_p78227136254"></a>值为整数类型的列表。</p>
<p id="zh-cn_topic_0000001411032876_p5873471289"><a name="zh-cn_topic_0000001411032876_p5873471289"></a><a name="zh-cn_topic_0000001411032876_p5873471289"></a>取值为Graph中的shape。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section413535858"></a>

正常场景下返回None。

异常情况下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例<a name="section17821439839"></a>

```
import dataflow as df
pp1 = df.GraphProcessPoint(...)
```

## 约束说明<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section2021419196520"></a>

无

