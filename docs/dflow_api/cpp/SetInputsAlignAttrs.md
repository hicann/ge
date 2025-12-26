# SetInputsAlignAttrs<a name="ZH-CN_TOPIC_0000001977152502"></a>

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

## 函数功能<a name="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_section3729174918713"></a>

设置FlowGraph中的输入对齐属性。

## 函数原型<a name="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_section84161445741"></a>

```
FlowGraph &SetInputsAlignAttrs(uint32_t align_max_cache_num, int32_t align_timeout, bool dropout_when_not_align = false)
```

## 参数说明<a name="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_section63604780"></a>

<a name="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_table47561922"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_row29169897"><th class="cellrowborder" valign="top" width="27.6%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p13951479"><a name="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p13951479"></a><a name="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p13951479"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="27.93%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p56327989"><a name="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p56327989"></a><a name="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p56327989"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="44.47%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p66531170"><a name="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p66531170"></a><a name="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p66531170"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_row20315681"><td class="cellrowborder" valign="top" width="27.6%" headers="mcps1.1.4.1.1 "><p id="p4695132143117"><a name="p4695132143117"></a><a name="p4695132143117"></a>align_max_cache_num</p>
</td>
<td class="cellrowborder" valign="top" width="27.93%" headers="mcps1.1.4.1.2 "><p id="p1969511324317"><a name="p1969511324317"></a><a name="p1969511324317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="44.47%" headers="mcps1.1.4.1.3 "><p id="p1820820471428"><a name="p1820820471428"></a><a name="p1820820471428"></a>数据对齐最大缓存数量，默认为0，表示不开启数据对齐功能，取值&gt;0表示开启，最大值为1024。</p>
<p id="p10694332163116"><a name="p10694332163116"></a><a name="p10694332163116"></a>每个缓存表示一组输入。</p>
</td>
</tr>
<tr id="row10923143073116"><td class="cellrowborder" valign="top" width="27.6%" headers="mcps1.1.4.1.1 "><p id="p8923630103119"><a name="p8923630103119"></a><a name="p8923630103119"></a>align_timeout</p>
</td>
<td class="cellrowborder" valign="top" width="27.93%" headers="mcps1.1.4.1.2 "><p id="p139232030163118"><a name="p139232030163118"></a><a name="p139232030163118"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="44.47%" headers="mcps1.1.4.1.3 "><p id="p3923193073116"><a name="p3923193073116"></a><a name="p3923193073116"></a>每组数据对齐等待超时时间，单位ms。</p>
<p id="p14692613833"><a name="p14692613833"></a><a name="p14692613833"></a>-1表示永不超时，配置需要大于0并不超过600 * 1000ms(10分钟)。</p>
</td>
</tr>
<tr id="row1720512913311"><td class="cellrowborder" valign="top" width="27.6%" headers="mcps1.1.4.1.1 "><p id="p1520582916316"><a name="p1520582916316"></a><a name="p1520582916316"></a>dropout_when_not_align</p>
</td>
<td class="cellrowborder" valign="top" width="27.93%" headers="mcps1.1.4.1.2 "><p id="p720502933117"><a name="p720502933117"></a><a name="p720502933117"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="44.47%" headers="mcps1.1.4.1.3 "><p id="p195031547441"><a name="p195031547441"></a><a name="p195031547441"></a>超时或超过缓存最大数之后没有对齐的数据是否要丢弃。</p>
<a name="ul163104610510"></a><a name="ul163104610510"></a><ul id="ul163104610510"><li>true：是</li><li>false：否</li></ul>
<p id="p320515296315"><a name="p320515296315"></a><a name="p320515296315"></a>默认为false。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_section413535858"></a>

返回设置了对齐属性的FlowGraph图。

## 异常处理<a name="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_section1548781517515"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001461312441_zh-cn_topic_0000001265240866_section2021419196520"></a>

无。

