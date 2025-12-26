# SetCompileConfig（GraphPp类）<a name="ZH-CN_TOPIC_0000001977312234"></a>

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

## 函数功能<a name="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_section3729174918713"></a>

设置GraphPp的json配置文件路径和文件名。配置文件用于AscendGraph的描述和编译。

返回设置好的GraphPp。

## 函数原型<a name="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_section84161445741"></a>

```
GraphPp &SetCompileConfig(const char_t *json_file_path)
```

## 参数说明<a name="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_section63604780"></a>

<a name="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_table47561922"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_row29169897"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p13951479"><a name="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p13951479"></a><a name="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p13951479"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="27.900000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p56327989"><a name="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p56327989"></a><a name="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p56327989"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="44.47%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p66531170"><a name="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p66531170"></a><a name="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p66531170"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_row20315681"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001411032876_p1629646151812"><a name="zh-cn_topic_0000001411032876_p1629646151812"></a><a name="zh-cn_topic_0000001411032876_p1629646151812"></a>json_file_path</p>
</td>
<td class="cellrowborder" valign="top" width="27.900000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001411032876_p42924661814"><a name="zh-cn_topic_0000001411032876_p42924661814"></a><a name="zh-cn_topic_0000001411032876_p42924661814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="44.47%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001411032876_p1028346101817"><a name="zh-cn_topic_0000001411032876_p1028346101817"></a><a name="zh-cn_topic_0000001411032876_p1028346101817"></a>GraphPp的json配置文件路径和文件名。</p>
<p id="zh-cn_topic_0000001411032876_p123038151362"><a name="zh-cn_topic_0000001411032876_p123038151362"></a><a name="zh-cn_topic_0000001411032876_p123038151362"></a>GraphPp的json配置文件用于AscendGraph的描述和编译。</p>
<p id="zh-cn_topic_0000001411032876_p20553175719237"><a name="zh-cn_topic_0000001411032876_p20553175719237"></a><a name="zh-cn_topic_0000001411032876_p20553175719237"></a>示例如下，参数解释请参考<a href="#zh-cn_topic_0000001411032876_table1179952915232">表1</a>。</p>
<pre class="screen" id="zh-cn_topic_0000001411032876_screen198482475239"><a name="zh-cn_topic_0000001411032876_screen198482475239"></a><a name="zh-cn_topic_0000001411032876_screen198482475239"></a>{"build_option":{},"inputs_tensor_desc":[{"data_type":"DT_UINT32","shape":[3]},{"data_type":"DT_UINT32","shape":[3]}]}</pre>
</td>
</tr>
</tbody>
</table>

**表 1**  GraphPp的json配置文件

<a name="zh-cn_topic_0000001411032876_table1179952915232"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001411032876_row779992992311"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001411032876_p2799429182313"><a name="zh-cn_topic_0000001411032876_p2799429182313"></a><a name="zh-cn_topic_0000001411032876_p2799429182313"></a>配置项</p>
</th>
<th class="cellrowborder" valign="top" width="27.87%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001411032876_p197991229102314"><a name="zh-cn_topic_0000001411032876_p197991229102314"></a><a name="zh-cn_topic_0000001411032876_p197991229102314"></a>可选/必选</p>
</th>
<th class="cellrowborder" valign="top" width="44.5%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001411032876_p680013295233"><a name="zh-cn_topic_0000001411032876_p680013295233"></a><a name="zh-cn_topic_0000001411032876_p680013295233"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001411032876_row7800029102314"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411032876_p78211813182511"><a name="zh-cn_topic_0000001411032876_p78211813182511"></a><a name="zh-cn_topic_0000001411032876_p78211813182511"></a>build_option</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411032876_p19821213112516"><a name="zh-cn_topic_0000001411032876_p19821213112516"></a><a name="zh-cn_topic_0000001411032876_p19821213112516"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411032876_p19821213152513"><a name="zh-cn_topic_0000001411032876_p19821213152513"></a><a name="zh-cn_topic_0000001411032876_p19821213152513"></a>值为map&lt;string, string&gt;, 有需要设置时参考Ascend Graph中的build_option。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411032876_row1728411013253"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411032876_p14821513132519"><a name="zh-cn_topic_0000001411032876_p14821513132519"></a><a name="zh-cn_topic_0000001411032876_p14821513132519"></a>inputs_tensor_desc</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411032876_p1482181310256"><a name="zh-cn_topic_0000001411032876_p1482181310256"></a><a name="zh-cn_topic_0000001411032876_p1482181310256"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411032876_p1182151310258"><a name="zh-cn_topic_0000001411032876_p1182151310258"></a><a name="zh-cn_topic_0000001411032876_p1182151310258"></a>值为list，Ascend Graph的输入节点，list元素为Tensor的描述。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411032876_row1413118317257"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411032876_p682119137252"><a name="zh-cn_topic_0000001411032876_p682119137252"></a><a name="zh-cn_topic_0000001411032876_p682119137252"></a>inputs_tensor_desc.data_type</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411032876_p128211613182515"><a name="zh-cn_topic_0000001411032876_p128211613182515"></a><a name="zh-cn_topic_0000001411032876_p128211613182515"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411032876_p16949428681"><a name="zh-cn_topic_0000001411032876_p16949428681"></a><a name="zh-cn_topic_0000001411032876_p16949428681"></a>字符串类型。</p>
<p id="zh-cn_topic_0000001411032876_p282191320252"><a name="zh-cn_topic_0000001411032876_p282191320252"></a><a name="zh-cn_topic_0000001411032876_p282191320252"></a>取值为Ascend Graph中的data_type对应字符串。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411032876_row360312532518"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411032876_p11821101342518"><a name="zh-cn_topic_0000001411032876_p11821101342518"></a><a name="zh-cn_topic_0000001411032876_p11821101342518"></a>inputs_tensor_desc.shape</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411032876_p19822141318253"><a name="zh-cn_topic_0000001411032876_p19822141318253"></a><a name="zh-cn_topic_0000001411032876_p19822141318253"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411032876_p78227136254"><a name="zh-cn_topic_0000001411032876_p78227136254"></a><a name="zh-cn_topic_0000001411032876_p78227136254"></a>值为整数类型的列表</p>
<p id="zh-cn_topic_0000001411032876_p5873471289"><a name="zh-cn_topic_0000001411032876_p5873471289"></a><a name="zh-cn_topic_0000001411032876_p5873471289"></a>取值为Ascend Graph中的shape。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_section413535858"></a>

返回设置好的GraphPp。

## 异常处理<a name="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_section1548781517515"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001411032876_zh-cn_topic_0000001265240866_section2021419196520"></a>

无。

