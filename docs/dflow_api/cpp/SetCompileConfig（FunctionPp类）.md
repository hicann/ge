# SetCompileConfig（FunctionPp类）<a name="ZH-CN_TOPIC_0000002013792089"></a>

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

## 函数功能<a name="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_section3729174918713"></a>

设置FunctionPp的json配置文件名字和路径，该配置文件用于将FunctionPp和UDF进行映射。

返回设置好的FunctionPp。

## 函数原型<a name="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_section84161445741"></a>

```
FunctionPp &SetCompileConfig(const char_t *json_file_path)
```

## 参数说明<a name="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_section63604780"></a>

<a name="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_table47561922"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_row29169897"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p13951479"><a name="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p13951479"></a><a name="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p13951479"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="27.900000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p56327989"><a name="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p56327989"></a><a name="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p56327989"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="44.47%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p66531170"><a name="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p66531170"></a><a name="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p66531170"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_row20315681"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001411352688_p8247225151011"><a name="zh-cn_topic_0000001411352688_p8247225151011"></a><a name="zh-cn_topic_0000001411352688_p8247225151011"></a>json_file_path</p>
</td>
<td class="cellrowborder" valign="top" width="27.900000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001411352688_p1824752521017"><a name="zh-cn_topic_0000001411352688_p1824752521017"></a><a name="zh-cn_topic_0000001411352688_p1824752521017"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="44.47%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001411352688_p172470254103"><a name="zh-cn_topic_0000001411352688_p172470254103"></a><a name="zh-cn_topic_0000001411352688_p172470254103"></a>FunctionPp的json配置文件路径和名字。</p>
<p id="zh-cn_topic_0000001411352688_p1230115853017"><a name="zh-cn_topic_0000001411352688_p1230115853017"></a><a name="zh-cn_topic_0000001411352688_p1230115853017"></a>FunctionPp的json配置文件用于UDF的描述和编译。</p>
<p id="zh-cn_topic_0000001411352688_p20553175719237"><a name="zh-cn_topic_0000001411352688_p20553175719237"></a><a name="zh-cn_topic_0000001411352688_p20553175719237"></a>配置文件示例请参考<a href="#li1641451223110">FunctionPp的json配置文件</a>，参数解释请参考<a href="#zh-cn_topic_0000001411352688_table1179952915232">表1</a>。</p>
</td>
</tr>
</tbody>
</table>

-   <a name="li1641451223110"></a>FunctionPp的json配置文件示例如下。

    ```
    {"func_list":[{"func_name":"Add", "inputs_index":[1,0], "outputs_index":[0]}],"input_num":2,"output_num":1,"target_bin":"libadd.so","workspace":"./","cmakelist_path":"CMakeLists.txt","compiler": "./cpu_compile.json","running_resources_info":[{"type":"cpu","num":2},{"type":"memory","num":100}],"heavy_load":false}
    ```

**表 1**  FunctionPp的json配置文件

<a name="zh-cn_topic_0000001411352688_table1179952915232"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001411352688_row779992992311"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001411352688_p2799429182313"><a name="zh-cn_topic_0000001411352688_p2799429182313"></a><a name="zh-cn_topic_0000001411352688_p2799429182313"></a>配置项</p>
</th>
<th class="cellrowborder" valign="top" width="27.87%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001411352688_p197991229102314"><a name="zh-cn_topic_0000001411352688_p197991229102314"></a><a name="zh-cn_topic_0000001411352688_p197991229102314"></a>可选/必选</p>
</th>
<th class="cellrowborder" valign="top" width="44.5%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001411352688_p680013295233"><a name="zh-cn_topic_0000001411352688_p680013295233"></a><a name="zh-cn_topic_0000001411352688_p680013295233"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001411352688_row7800029102314"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p529316932717"><a name="zh-cn_topic_0000001411352688_p529316932717"></a><a name="zh-cn_topic_0000001411352688_p529316932717"></a>workspace</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p122932992711"><a name="zh-cn_topic_0000001411352688_p122932992711"></a><a name="zh-cn_topic_0000001411352688_p122932992711"></a>必选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p18293099277"><a name="zh-cn_topic_0000001411352688_p18293099277"></a><a name="zh-cn_topic_0000001411352688_p18293099277"></a>值为字符串，UDF的工作空间路径。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352688_row942485910269"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p52938910272"><a name="zh-cn_topic_0000001411352688_p52938910272"></a><a name="zh-cn_topic_0000001411352688_p52938910272"></a>target_bin</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p12293189132712"><a name="zh-cn_topic_0000001411352688_p12293189132712"></a><a name="zh-cn_topic_0000001411352688_p12293189132712"></a>必选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p32931693279"><a name="zh-cn_topic_0000001411352688_p32931693279"></a><a name="zh-cn_topic_0000001411352688_p32931693279"></a>值为字符串，UDF工程编译出来的so名字，为防止被非法篡改，该字符串需要以lib***.so来命名，合法的字符包含大小写字母、数字、下划线和中划线。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352688_row11708155792615"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p1929410922713"><a name="zh-cn_topic_0000001411352688_p1929410922713"></a><a name="zh-cn_topic_0000001411352688_p1929410922713"></a>input_num</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p329410952718"><a name="zh-cn_topic_0000001411352688_p329410952718"></a><a name="zh-cn_topic_0000001411352688_p329410952718"></a>必选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p18294179192719"><a name="zh-cn_topic_0000001411352688_p18294179192719"></a><a name="zh-cn_topic_0000001411352688_p18294179192719"></a>值为数字，表示UDF的输入个数，即FunctionPp的输入个数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352688_row1728411013253"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p15294091276"><a name="zh-cn_topic_0000001411352688_p15294091276"></a><a name="zh-cn_topic_0000001411352688_p15294091276"></a>output_num</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p122940918272"><a name="zh-cn_topic_0000001411352688_p122940918272"></a><a name="zh-cn_topic_0000001411352688_p122940918272"></a>必选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p17294129132717"><a name="zh-cn_topic_0000001411352688_p17294129132717"></a><a name="zh-cn_topic_0000001411352688_p17294129132717"></a>值为数字，表示UDF的输出个数。即FunctionPp的输出个数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352688_row11389156152616"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p629479192713"><a name="zh-cn_topic_0000001411352688_p629479192713"></a><a name="zh-cn_topic_0000001411352688_p629479192713"></a>func_list</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p1629410916273"><a name="zh-cn_topic_0000001411352688_p1629410916273"></a><a name="zh-cn_topic_0000001411352688_p1629410916273"></a>必选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p929459132710"><a name="zh-cn_topic_0000001411352688_p929459132710"></a><a name="zh-cn_topic_0000001411352688_p929459132710"></a>值为list，list的元素为单个function的描述，当前只支持一个function。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352688_row14780105415268"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p629479112717"><a name="zh-cn_topic_0000001411352688_p629479112717"></a><a name="zh-cn_topic_0000001411352688_p629479112717"></a>func_list.func_name</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p029419922716"><a name="zh-cn_topic_0000001411352688_p029419922716"></a><a name="zh-cn_topic_0000001411352688_p029419922716"></a>必选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p72949942720"><a name="zh-cn_topic_0000001411352688_p72949942720"></a><a name="zh-cn_topic_0000001411352688_p72949942720"></a>值为字符串，函数名称，要和UDF里定义的function名称一致。多function场景下，func_name不允许重复。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352688_row1413118317257"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p18294149142718"><a name="zh-cn_topic_0000001411352688_p18294149142718"></a><a name="zh-cn_topic_0000001411352688_p18294149142718"></a>func_list.inputs_index</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p1294791273"><a name="zh-cn_topic_0000001411352688_p1294791273"></a><a name="zh-cn_topic_0000001411352688_p1294791273"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p112946919277"><a name="zh-cn_topic_0000001411352688_p112946919277"></a><a name="zh-cn_topic_0000001411352688_p112946919277"></a>值为list，list元素为数字，表示该function取FunctionPp的哪些输入，单function情况下当前无效。多function情况下该字段必选。且多个处理函数input index不共享，不能重复。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352688_row360312532518"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p172941922711"><a name="zh-cn_topic_0000001411352688_p172941922711"></a><a name="zh-cn_topic_0000001411352688_p172941922711"></a>func_list.outputs_index</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p1929417982718"><a name="zh-cn_topic_0000001411352688_p1929417982718"></a><a name="zh-cn_topic_0000001411352688_p1929417982718"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p629499132714"><a name="zh-cn_topic_0000001411352688_p629499132714"></a><a name="zh-cn_topic_0000001411352688_p629499132714"></a>值为list，list元素为数字，表示该function对应FunctionPp的哪些输出，单function情况下当前无效。多function情况下output index可共享。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352688_row3433954193117"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p1912518122324"><a name="zh-cn_topic_0000001411352688_p1912518122324"></a><a name="zh-cn_topic_0000001411352688_p1912518122324"></a>cmakelist_path</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p9125171215322"><a name="zh-cn_topic_0000001411352688_p9125171215322"></a><a name="zh-cn_topic_0000001411352688_p9125171215322"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p91256123328"><a name="zh-cn_topic_0000001411352688_p91256123328"></a><a name="zh-cn_topic_0000001411352688_p91256123328"></a>值为字符串，源码编译的CMakeLists文件相对于workspace的路径，如果未指定，则取workspace下面的默认CMakeLists文件。</p>
<p id="zh-cn_topic_0000001411352688_p720574265918"><a name="zh-cn_topic_0000001411352688_p720574265918"></a><a name="zh-cn_topic_0000001411352688_p720574265918"></a>CMakeLists文件的详细信息请参考<a href="#zh-cn_topic_0000001411352688_li0604184475614">CMakeLists文件</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352688_row18249558123113"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p21258126329"><a name="zh-cn_topic_0000001411352688_p21258126329"></a><a name="zh-cn_topic_0000001411352688_p21258126329"></a>compiler</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p912515126329"><a name="zh-cn_topic_0000001411352688_p912515126329"></a><a name="zh-cn_topic_0000001411352688_p912515126329"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p1312513124324"><a name="zh-cn_topic_0000001411352688_p1312513124324"></a><a name="zh-cn_topic_0000001411352688_p1312513124324"></a>值为字符串，异构环境下编译源码的交叉编译工具路径配置文件，如果未指定，则取资源类型默认的编译工具。</p>
<p id="zh-cn_topic_0000001411352688_p1787714383554"><a name="zh-cn_topic_0000001411352688_p1787714383554"></a><a name="zh-cn_topic_0000001411352688_p1787714383554"></a>compiler的json配置文件内容示例和各字段含义请参考<a href="#zh-cn_topic_0000001411352688_li178313415720">•compiler的json配置文件</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352688_row12204056173113"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p181259127327"><a name="zh-cn_topic_0000001411352688_p181259127327"></a><a name="zh-cn_topic_0000001411352688_p181259127327"></a>running_resources_info</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p21251912153212"><a name="zh-cn_topic_0000001411352688_p21251912153212"></a><a name="zh-cn_topic_0000001411352688_p21251912153212"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p11251412173212"><a name="zh-cn_topic_0000001411352688_p11251412173212"></a><a name="zh-cn_topic_0000001411352688_p11251412173212"></a>值为list，运行当前so需要的资源信息，list的元素为单个资源信息的描述。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352688_row114281352133117"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p61251312193212"><a name="zh-cn_topic_0000001411352688_p61251312193212"></a><a name="zh-cn_topic_0000001411352688_p61251312193212"></a>running_resources_info.type</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p171251012173215"><a name="zh-cn_topic_0000001411352688_p171251012173215"></a><a name="zh-cn_topic_0000001411352688_p171251012173215"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p1939610961717"><a name="zh-cn_topic_0000001411352688_p1939610961717"></a><a name="zh-cn_topic_0000001411352688_p1939610961717"></a>当配置了running_resources_info时，该字段必选。</p>
<p id="zh-cn_topic_0000001411352688_p20126181293214"><a name="zh-cn_topic_0000001411352688_p20126181293214"></a><a name="zh-cn_topic_0000001411352688_p20126181293214"></a>值为字符串，运行当前so需要的资源信息的类型，可选类型是"cpu"和"memory"。当资源类型是"memory"时，单位是M。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001411352688_row152608488319"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p12126111243219"><a name="zh-cn_topic_0000001411352688_p12126111243219"></a><a name="zh-cn_topic_0000001411352688_p12126111243219"></a>running_resources_info.num</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p7126101293215"><a name="zh-cn_topic_0000001411352688_p7126101293215"></a><a name="zh-cn_topic_0000001411352688_p7126101293215"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p156011942181716"><a name="zh-cn_topic_0000001411352688_p156011942181716"></a><a name="zh-cn_topic_0000001411352688_p156011942181716"></a>当配置了running_resources_info时，该字段必选。</p>
<p id="zh-cn_topic_0000001411352688_p612691216326"><a name="zh-cn_topic_0000001411352688_p612691216326"></a><a name="zh-cn_topic_0000001411352688_p612691216326"></a>值为数字，运行当前so需要的资源信息的数量。</p>
</td>
</tr>
<tr id="row1742311318408"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="p1042313104019"><a name="p1042313104019"></a><a name="p1042313104019"></a>heavy_load</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="p1042493154020"><a name="p1042493154020"></a><a name="p1042493154020"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="p7856145594213"><a name="p7856145594213"></a><a name="p7856145594213"></a>表示节点对算力的诉求。</p>
<a name="ul690551920448"></a><a name="ul690551920448"></a><ul id="ul690551920448"><li>true：重载，表示对算力的诉求大。</li><li>false：轻载，表示对算力的诉求小。</li></ul>
<p id="p107281926164115"><a name="p107281926164115"></a><a name="p107281926164115"></a>默认值为false。</p>
<p id="p1856113011415"><a name="p1856113011415"></a><a name="p1856113011415"></a>当该参数取值为"true"时会影响UDF的部署位置。</p>
</td>
</tr>
<tr id="row132571544013"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="p02617151405"><a name="p02617151405"></a><a name="p02617151405"></a>buf_cfg</p>
</td>
<td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="p13261215164013"><a name="p13261215164013"></a><a name="p13261215164013"></a>可选</p>
</td>
<td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="p1826101510406"><a name="p1826101510406"></a><a name="p1826101510406"></a>用户可以自定义配置内存池档位，通过自定义档位可以提升内存申请效率及减少内存碎片。如未设置该参数，将使用默认的档位配置初始化内存模块，默认档位配置及该参数样例请参考<a href="#li112771103413">•buf_cfg的json配置</a>，该配置最多支持64个档位，超过64编译报错。</p>
</td>
</tr>
</tbody>
</table>

-   <a name="zh-cn_topic_0000001411352688_li178313415720"></a>compiler的json配置

    内容示例如下，各字段解释如所示。

    ```
    {"compiler":[{"resource_type":"X86","toolchain":"/usr/bin/g++"},{"resource_type":"Aarch","toolchain":"/usr/bin/g++"},{"resource_type":"Ascend","toolchain":"/usr/local/Ascend/hcc"}]}
    ```

    **表 2**  compiler的json配置说明

    <a name="zh-cn_topic_0000001411352688_table4711165145313"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001411352688_row171285145318"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001411352688_p371217512531"><a name="zh-cn_topic_0000001411352688_p371217512531"></a><a name="zh-cn_topic_0000001411352688_p371217512531"></a>配置项</p>
    </th>
    <th class="cellrowborder" valign="top" width="27.87%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001411352688_p77126516536"><a name="zh-cn_topic_0000001411352688_p77126516536"></a><a name="zh-cn_topic_0000001411352688_p77126516536"></a>可选/必选</p>
    </th>
    <th class="cellrowborder" valign="top" width="44.5%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001411352688_p17712155125316"><a name="zh-cn_topic_0000001411352688_p17712155125316"></a><a name="zh-cn_topic_0000001411352688_p17712155125316"></a>描述</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001411352688_row20712175145319"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p18111351559"><a name="zh-cn_topic_0000001411352688_p18111351559"></a><a name="zh-cn_topic_0000001411352688_p18111351559"></a>compiler</p>
    </td>
    <td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p1581115545512"><a name="zh-cn_topic_0000001411352688_p1581115545512"></a><a name="zh-cn_topic_0000001411352688_p1581115545512"></a>必选</p>
    </td>
    <td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p381120575511"><a name="zh-cn_topic_0000001411352688_p381120575511"></a><a name="zh-cn_topic_0000001411352688_p381120575511"></a>值为list，list的元素为单个资源类型的编译工具的描述。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001411352688_row1471215525314"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p148111514556"><a name="zh-cn_topic_0000001411352688_p148111514556"></a><a name="zh-cn_topic_0000001411352688_p148111514556"></a>compiler.resource_type</p>
    </td>
    <td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p08110515513"><a name="zh-cn_topic_0000001411352688_p08110515513"></a><a name="zh-cn_topic_0000001411352688_p08110515513"></a>必选</p>
    </td>
    <td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p1781115115518"><a name="zh-cn_topic_0000001411352688_p1781115115518"></a><a name="zh-cn_topic_0000001411352688_p1781115115518"></a>值为字符串，设备支持的资源类型。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001411352688_row11712125145318"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001411352688_p88118513556"><a name="zh-cn_topic_0000001411352688_p88118513556"></a><a name="zh-cn_topic_0000001411352688_p88118513556"></a>compiler.toolchain</p>
    </td>
    <td class="cellrowborder" valign="top" width="27.87%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001411352688_p17811155135518"><a name="zh-cn_topic_0000001411352688_p17811155135518"></a><a name="zh-cn_topic_0000001411352688_p17811155135518"></a>必选</p>
    </td>
    <td class="cellrowborder" valign="top" width="44.5%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001411352688_p981120505510"><a name="zh-cn_topic_0000001411352688_p981120505510"></a><a name="zh-cn_topic_0000001411352688_p981120505510"></a>值为字符串，该资源类型对应的编译工具路径。</p>
    </td>
    </tr>
    </tbody>
    </table>

-   <a name="li112771103413"></a>buf\_cfg的json配置

    内容示例如下，各字段解释如[表3](#table16292457124212)所示。

    ```
    "buf_cfg":[{"total_size":2097152,"blk_size":256,"max_buf_size":8192,"page_type":"normal"},        // 1.total:2M  max:8K
               {"total_size":10485760,"blk_size":4096,"max_buf_size":8388608,"page_type":"normal"},   // 2.total:10M  max:8M
               {"total_size":2097152,"blk_size":256,"max_buf_size":8192,"page_type":"huge"},          // 3.total:2M  max:8K     
               {"total_size":10485760,"blk_size":8192,"max_buf_size":8388608,"page_type":"huge"},     // 4.total:10M  max:8M
               {"total_size":69206016,"blk_size":8192,"max_buf_size":67108864,"page_type":"huge"}]    // 5.total:66M  max:64M
    ```

    >![](public_sys-resources/icon-note.gif) **说明：** 
    >如上样例共配置了5个内存档位，前两条针对普通内存，后三条针对大页内存。
    >使用该配置初始化内存管理模块后，如果进程申请8M大页内存，驱动会根据第4条配置项，生成并管理一个10M内存池，从其中申请8M内存。
    >如本进程需要再次申请1M大页内存，由于第三条配置项中一次最大只能申请8K，因此仍然会落到第4条配置项对应的内存池中，此时上一次申请10M只使用了8M，剩余的内存仍大于1M，因此会在上一次生成的10M内存池中申请1M内存供本次使用。

    默认档位如下：

    <a name="table18569837988"></a>
    <table><thead align="left"><tr id="row115691637585"><th class="cellrowborder" valign="top" width="6.959999999999999%" id="mcps1.1.6.1.1"><p id="p463313418915"><a name="p463313418915"></a><a name="p463313418915"></a>ID</p>
    </th>
    <th class="cellrowborder" valign="top" width="23.630000000000003%" id="mcps1.1.6.1.2"><p id="p205696375812"><a name="p205696375812"></a><a name="p205696375812"></a>total_size</p>
    </th>
    <th class="cellrowborder" valign="top" width="23.830000000000002%" id="mcps1.1.6.1.3"><p id="p756913374817"><a name="p756913374817"></a><a name="p756913374817"></a>blk_size</p>
    </th>
    <th class="cellrowborder" valign="top" width="28.439999999999998%" id="mcps1.1.6.1.4"><p id="p097517135918"><a name="p097517135918"></a><a name="p097517135918"></a>max_buf_size</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.14%" id="mcps1.1.6.1.5"><p id="p12619724390"><a name="p12619724390"></a><a name="p12619724390"></a>page_type</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row256913375817"><td class="cellrowborder" valign="top" width="6.959999999999999%" headers="mcps1.1.6.1.1 "><p id="p1263364990"><a name="p1263364990"></a><a name="p1263364990"></a>0</p>
    </td>
    <td class="cellrowborder" valign="top" width="23.630000000000003%" headers="mcps1.1.6.1.2 "><p id="p1556920376810"><a name="p1556920376810"></a><a name="p1556920376810"></a>2M</p>
    </td>
    <td class="cellrowborder" valign="top" width="23.830000000000002%" headers="mcps1.1.6.1.3 "><p id="p15875143711917"><a name="p15875143711917"></a><a name="p15875143711917"></a>256B</p>
    </td>
    <td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.1.6.1.4 "><p id="p456917371582"><a name="p456917371582"></a><a name="p456917371582"></a>8K</p>
    <p id="p155691376814"><a name="p155691376814"></a><a name="p155691376814"></a></p>
    </td>
    <td class="cellrowborder" valign="top" width="17.14%" headers="mcps1.1.6.1.5 "><p id="p13619152412915"><a name="p13619152412915"></a><a name="p13619152412915"></a>normal</p>
    </td>
    </tr>
    <tr id="row056914371789"><td class="cellrowborder" valign="top" width="6.959999999999999%" headers="mcps1.1.6.1.1 "><p id="p66331244916"><a name="p66331244916"></a><a name="p66331244916"></a>1</p>
    </td>
    <td class="cellrowborder" valign="top" width="23.630000000000003%" headers="mcps1.1.6.1.2 "><p id="p20569113718814"><a name="p20569113718814"></a><a name="p20569113718814"></a>32M</p>
    </td>
    <td class="cellrowborder" valign="top" width="23.830000000000002%" headers="mcps1.1.6.1.3 "><p id="p1537815598913"><a name="p1537815598913"></a><a name="p1537815598913"></a>8K</p>
    </td>
    <td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.1.6.1.4 "><p id="p155691837987"><a name="p155691837987"></a><a name="p155691837987"></a>8M</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.14%" headers="mcps1.1.6.1.5 "><p id="p166191241199"><a name="p166191241199"></a><a name="p166191241199"></a>normal</p>
    </td>
    </tr>
    <tr id="row6933141615117"><td class="cellrowborder" valign="top" width="6.959999999999999%" headers="mcps1.1.6.1.1 "><p id="p1693361621115"><a name="p1693361621115"></a><a name="p1693361621115"></a>3</p>
    </td>
    <td class="cellrowborder" valign="top" width="23.630000000000003%" headers="mcps1.1.6.1.2 "><p id="p1393315162117"><a name="p1393315162117"></a><a name="p1393315162117"></a>2M</p>
    </td>
    <td class="cellrowborder" valign="top" width="23.830000000000002%" headers="mcps1.1.6.1.3 "><p id="p5933316131118"><a name="p5933316131118"></a><a name="p5933316131118"></a>256B</p>
    </td>
    <td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.1.6.1.4 "><p id="p119337168116"><a name="p119337168116"></a><a name="p119337168116"></a>8K</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.14%" headers="mcps1.1.6.1.5 "><p id="p109341716111116"><a name="p109341716111116"></a><a name="p109341716111116"></a>huge</p>
    </td>
    </tr>
    <tr id="row0780203417114"><td class="cellrowborder" valign="top" width="6.959999999999999%" headers="mcps1.1.6.1.1 "><p id="p9780143416115"><a name="p9780143416115"></a><a name="p9780143416115"></a>4</p>
    </td>
    <td class="cellrowborder" valign="top" width="23.630000000000003%" headers="mcps1.1.6.1.2 "><p id="p197801346112"><a name="p197801346112"></a><a name="p197801346112"></a>66M</p>
    </td>
    <td class="cellrowborder" valign="top" width="23.830000000000002%" headers="mcps1.1.6.1.3 "><p id="p17780153471113"><a name="p17780153471113"></a><a name="p17780153471113"></a>8K</p>
    </td>
    <td class="cellrowborder" valign="top" width="28.439999999999998%" headers="mcps1.1.6.1.4 "><p id="p178110348114"><a name="p178110348114"></a><a name="p178110348114"></a>64M</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.14%" headers="mcps1.1.6.1.5 "><p id="p1078133491113"><a name="p1078133491113"></a><a name="p1078133491113"></a>huge</p>
    </td>
    </tr>
    </tbody>
    </table>

    **表 3**  buf\_cfg的json配置说明

    <a name="table16292457124212"></a>
    <table><thead align="left"><tr id="row13292057114211"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.2.4.1.1"><p id="p16292205764216"><a name="p16292205764216"></a><a name="p16292205764216"></a>配置项</p>
    </th>
    <th class="cellrowborder" valign="top" width="27.839999999999996%" id="mcps1.2.4.1.2"><p id="p429210575426"><a name="p429210575426"></a><a name="p429210575426"></a>可选/必选</p>
    </th>
    <th class="cellrowborder" valign="top" width="44.529999999999994%" id="mcps1.2.4.1.3"><p id="p17292185719427"><a name="p17292185719427"></a><a name="p17292185719427"></a>描述</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row429295718421"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="p029205794213"><a name="p029205794213"></a><a name="p029205794213"></a>total_size</p>
    </td>
    <td class="cellrowborder" valign="top" width="27.839999999999996%" headers="mcps1.2.4.1.2 "><p id="p42925575423"><a name="p42925575423"></a><a name="p42925575423"></a>必选</p>
    </td>
    <td class="cellrowborder" valign="top" width="44.529999999999994%" headers="mcps1.2.4.1.3 "><p id="p6292125719420"><a name="p6292125719420"></a><a name="p6292125719420"></a>当前档位内存池的大小，单位Byte。</p>
    <p id="p147311559725"><a name="p147311559725"></a><a name="p147311559725"></a>约束：普通内存total_size是4K的倍数，大页内存total_size是2M的倍数，且total_size是blk_size的倍数。</p>
    </td>
    </tr>
    <tr id="row529265719425"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="p182921557164215"><a name="p182921557164215"></a><a name="p182921557164215"></a>blk_size</p>
    </td>
    <td class="cellrowborder" valign="top" width="27.839999999999996%" headers="mcps1.2.4.1.2 "><p id="p3292105717428"><a name="p3292105717428"></a><a name="p3292105717428"></a>必选</p>
    </td>
    <td class="cellrowborder" valign="top" width="44.529999999999994%" headers="mcps1.2.4.1.3 "><p id="p5292155712427"><a name="p5292155712427"></a><a name="p5292155712427"></a>当前档位一次可以申请的最小内存值，单位Byte。</p>
    <p id="p137781817315"><a name="p137781817315"></a><a name="p137781817315"></a>约束：blk_size要求满足2^n，且在(0,2M]之间，小于max_buf_size。</p>
    </td>
    </tr>
    <tr id="row14292105744219"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="p162925573429"><a name="p162925573429"></a><a name="p162925573429"></a>max_buf_size</p>
    </td>
    <td class="cellrowborder" valign="top" width="27.839999999999996%" headers="mcps1.2.4.1.2 "><p id="p2292115712421"><a name="p2292115712421"></a><a name="p2292115712421"></a>必选</p>
    </td>
    <td class="cellrowborder" valign="top" width="44.529999999999994%" headers="mcps1.2.4.1.3 "><p id="p152921857164214"><a name="p152921857164214"></a><a name="p152921857164214"></a>当前档位一次可以申请的最大内存值，单位Byte。</p>
    <p id="p10966710233"><a name="p10966710233"></a><a name="p10966710233"></a>约束：max_buf_size小于total_size，max_buf_size同一中page_type下必须保持严格递增。</p>
    </td>
    </tr>
    <tr id="row671414894318"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.1 "><p id="p1371554813434"><a name="p1371554813434"></a><a name="p1371554813434"></a>page_type</p>
    </td>
    <td class="cellrowborder" valign="top" width="27.839999999999996%" headers="mcps1.2.4.1.2 "><p id="p186711554432"><a name="p186711554432"></a><a name="p186711554432"></a>必选</p>
    </td>
    <td class="cellrowborder" valign="top" width="44.529999999999994%" headers="mcps1.2.4.1.3 "><p id="p771718131938"><a name="p771718131938"></a><a name="p771718131938"></a>当前档位对应的内存类型。取值如下。</p>
    <a name="ul106291347162117"></a><a name="ul106291347162117"></a><ul id="ul106291347162117"><li>huge：大页内存</li><li>normal：普通内存</li></ul>
    </td>
    </tr>
    </tbody>
    </table>

-   <a name="zh-cn_topic_0000001411352688_li0604184475614"></a>CMakeLists文件

    DataFlow UDF编译模块解析异构环境的resource.json资源配置和cpu\_compiler配置，根据resource.json资源配置类型匹配选择cpu\_compiler中指定的交叉编译工具，如果用户未指定cpu\_compiler.json配置文件或者cpu\_compiler.json未配置该类型的编译工具，则取环境上默认的编译工具进行编译。不同资源类型的编译工具名称和路径如下。$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。若安装的Ascend-cann-toolkit软件包，以root安装举例，则安装后文件存储路径为：/usr/local/Ascend/ascend-toolkit/latest。

    -   X86和Aarch场景下：g++
    -   Ascend场景下：$\{INSTALL\_DIR\}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++

    用户的源代码工程要遵从如下规则：

    -   用户提供的源码工程目录下要包括所有执行代码和依赖库源码。
    -   用户要配置好FunctionPp执行代码和依赖库的编译脚本。
    -   编译脚本要使用RELEASE\_DIR变量作为最终输出目录，如果有依赖的so文件，用户需要把依赖的so文件拷贝到该路径下。
    -   编译脚本要使用RESOURCE\_TYPE变量判断资源类型，如果当前UDF不支持某一个资源类型，需要将对应的注释放开。
    -   CMakeLists sample如下：

        ```
        cmake_minimum_required(VERSION 3.5)
        PROJECT(UDF)
        if ("x${RESOURCE_TYPE}" STREQUAL "xAscend")
          message(STATUS "ascend compiler enter")
          # if unsupport current resource type, please uncomment the next line.
          # message(FATAL_ERROR "Unsupport compile Ascend target!")
        elseif("x${RESOURCE_TYPE}" STREQUAL "xAarch")
          message(STATUS "aarch64 compiler enter")
          # if unsupport current resource type, please uncomment the next line.
          # message(FATAL_ERROR "Unsupport compile Aarch64 target!")
        else()
          message(STATUS "x86 compiler enter")
          # if unsupport current resource type, please uncomment the next line.
          # message(FATAL_ERROR "Unsupport compile X86 target!")
        endif()
        
        if(DEFINED ENV{ASCEND_INSTALL_PATH})
          set(ASCEND_INSTALL_PATH $ENV{ASCEND_INSTALL_PATH})
          message(STATUS "Read ASCEND_INSTALL_PATH from ENV: ${ASCEND_INSTALL_PATH}")
        else()
          set(ASCEND_INSTALL_PATH /usr/local/Ascend)
          message(STATUS "Default ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}, you can export ASCEND_INSTALL_PATH to set this environment")
        endif()
        
        # set dynamic library output path
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${RELEASE_DIR})
        # set static library output path
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${RELEASE_DIR})
        
        message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY= ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
        
        set(INC_DIR "${ASCEND_INSTALL_PATH}/latest/x86_64-linux/include/flow_func")
        file(GLOB SRC_LIST "*.cpp")
        
        # Specify cross compiler
        add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
        
        # set c++ compiler
        set(CMAKE_CXX_COMPILER ${TOOLCHAIN})
        
        # =========================UDF so compile============================
        # check if SRC_LIST is exist
        if("x${SRC_LIST}" STREQUAL "x")
            message(UDF "=========no source file=============")
            add_custom_target(${UDF_TARGET_LIB}
                COMMAND echo "no source to make lib${UDF_TARGET_LIB}.so")
            return(0)
        endif()
        
        add_library(${UDF_TARGET_LIB} SHARED
          ${SRC_LIST}
        )
        
        target_include_directories(${UDF_TARGET_LIB} PRIVATE
          ${INC_DIR}
        )
        
        target_compile_options(${UDF_TARGET_LIB} PRIVATE
          -O2
          -std=c++11
          -ftrapv  
          -fstack-protector-all
          -fPIC
        )
        
        if ("x${RESOURCE_TYPE}" STREQUAL "xAscend")
          target_link_libraries(${UDF_TARGET_LIB} PRIVATE 
            -Wl,--whole-archive
            ${ASCEND_INSTALL_PATH}/latest/x86_64-linux/lib64/stub/aarch64/libflow_func.so
            -Wl,--no-whole-archive
          )
          # If there have any dependent so, please release the following comments and copy dependent so to ${PROJECT_BINARY_DIR}/${RELEASE_DIR}
          # [[execute_process(
            COMMAND cp ${ASCEND_INSTALL_PATH}/latest/x86_64-linux/lib64/stub/aarch64/libflow_func.so ${PROJECT_BINARY_DIR}/${RELEASE_DIR}
          )]]
        elseif("x${RESOURCE_TYPE}" STREQUAL "xAarch")
          target_link_libraries(${UDF_TARGET_LIB} PRIVATE 
            -Wl,--whole-archive
            ${ASCEND_INSTALL_PATH}/latest/x86_64-linux/lib64/stub/aarch64/libflow_func.so
            -Wl,--no-whole-archive
          )
          # If there have any dependent so, please release the following comments and copy dependent so to ${PROJECT_BINARY_DIR}/${RELEASE_DIR}
          # [[execute_process(
            COMMAND cp ${ASCEND_INSTALL_PATH}/latest/x86_64-linux/lib64/stub/aarch64/libflow_func.so ${PROJECT_BINARY_DIR}/${RELEASE_DIR}
          )]]
        else()
          target_link_libraries(${UDF_TARGET_LIB} PRIVATE 
            -Wl,--whole-archive
            ${ASCEND_INSTALL_PATH}/latest/x86_64-linux/lib64/stub/x86_64/libflow_func.so
            -Wl,--no-whole-archive
          )
          # If there have any dependent so, please release the following comments and copy dependent so to ${PROJECT_BINARY_DIR}/${RELEASE_DIR}
          # [[execute_process(
            COMMAND cp ${ASCEND_INSTALL_PATH}/latest/x86_64-linux/lib64/stub/x86_64/libflow_func.so ${PROJECT_BINARY_DIR}/${RELEASE_DIR}
          )]]
        endif()
        
        ```

## 返回值<a name="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_section413535858"></a>

返回设置好的FunctionPp。

## 异常处理<a name="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_section1548781517515"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001411352688_zh-cn_topic_0000001265240866_section2021419196520"></a>

无。

