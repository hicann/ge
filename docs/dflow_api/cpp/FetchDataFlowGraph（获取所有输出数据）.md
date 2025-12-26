# FetchDataFlowGraph（获取所有输出数据）<a name="ZH-CN_TOPIC_0000001977312246"></a>

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

## 函数功能<a name="zh-cn_topic_0000001364713352_section44282627"></a>

获取图输出数据。

## 函数原型<a name="zh-cn_topic_0000001364713352_section1831611148519"></a>

```
Status FetchDataFlowGraph(uint32_t graph_id, std::vector<Tensor> &outputs, DataFlowInfo &info, int32_t timeout)
```

## 参数说明<a name="zh-cn_topic_0000001364713352_section62999330"></a>

<a name="zh-cn_topic_0000001364713352_table62794401190"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001364713352_row7279184010914"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001364713352_p142791040498"><a name="zh-cn_topic_0000001364713352_p142791040498"></a><a name="zh-cn_topic_0000001364713352_p142791040498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.469999999999999%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001364713352_p82796401992"><a name="zh-cn_topic_0000001364713352_p82796401992"></a><a name="zh-cn_topic_0000001364713352_p82796401992"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.9%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001364713352_p1727911401893"><a name="zh-cn_topic_0000001364713352_p1727911401893"></a><a name="zh-cn_topic_0000001364713352_p1727911401893"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001364713352_row16279740593"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001364713352_p152793401491"><a name="zh-cn_topic_0000001364713352_p152793401491"></a><a name="zh-cn_topic_0000001364713352_p152793401491"></a>graph_id</p>
</td>
<td class="cellrowborder" valign="top" width="14.469999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001364713352_p162794401912"><a name="zh-cn_topic_0000001364713352_p162794401912"></a><a name="zh-cn_topic_0000001364713352_p162794401912"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.9%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001364713352_p172796409911"><a name="zh-cn_topic_0000001364713352_p172796409911"></a><a name="zh-cn_topic_0000001364713352_p172796409911"></a>要执行图对应的ID。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001364713352_row12791640693"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001364713352_p52792402920"><a name="zh-cn_topic_0000001364713352_p52792402920"></a><a name="zh-cn_topic_0000001364713352_p52792402920"></a>outputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.469999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001364713352_p22794401596"><a name="zh-cn_topic_0000001364713352_p22794401596"></a><a name="zh-cn_topic_0000001364713352_p22794401596"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="57.9%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001364713352_p127913401494"><a name="zh-cn_topic_0000001364713352_p127913401494"></a><a name="zh-cn_topic_0000001364713352_p127913401494"></a>计算图输出Tensor，用户无需分配内存空间，执行完成后GE会分配内存并赋值。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001364713352_row112791540794"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001364713352_p18280440999"><a name="zh-cn_topic_0000001364713352_p18280440999"></a><a name="zh-cn_topic_0000001364713352_p18280440999"></a>info</p>
</td>
<td class="cellrowborder" valign="top" width="14.469999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001364713352_p1128019409915"><a name="zh-cn_topic_0000001364713352_p1128019409915"></a><a name="zh-cn_topic_0000001364713352_p1128019409915"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="57.9%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001364713352_p10280144019911"><a name="zh-cn_topic_0000001364713352_p10280144019911"></a><a name="zh-cn_topic_0000001364713352_p10280144019911"></a>输出数据流标志（flow flag）。具体请参考<a href="DataFlowInfo数据类型.md">DataFlowInfo数据类型</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001364713352_row82801401917"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001364713352_p15280104015915"><a name="zh-cn_topic_0000001364713352_p15280104015915"></a><a name="zh-cn_topic_0000001364713352_p15280104015915"></a>timeout</p>
</td>
<td class="cellrowborder" valign="top" width="14.469999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001364713352_p728024018911"><a name="zh-cn_topic_0000001364713352_p728024018911"></a><a name="zh-cn_topic_0000001364713352_p728024018911"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.9%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001364713352_p128014401297"><a name="zh-cn_topic_0000001364713352_p128014401297"></a><a name="zh-cn_topic_0000001364713352_p128014401297"></a>数据获取超时时间，单位：ms，取值为-1时表示从不超时。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001364713352_section30123063"></a>

函数状态结果

<a name="zh-cn_topic_0000001364713352_table2601186"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001364713352_row1832323"><th class="cellrowborder" valign="top" width="32.65%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001364713352_p14200498"><a name="zh-cn_topic_0000001364713352_p14200498"></a><a name="zh-cn_topic_0000001364713352_p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="24.36%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001364713352_p9389685"><a name="zh-cn_topic_0000001364713352_p9389685"></a><a name="zh-cn_topic_0000001364713352_p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="42.99%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001364713352_p22367029"><a name="zh-cn_topic_0000001364713352_p22367029"></a><a name="zh-cn_topic_0000001364713352_p22367029"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001364713352_row66898905"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001364713352_p50102218"><a name="zh-cn_topic_0000001364713352_p50102218"></a><a name="zh-cn_topic_0000001364713352_p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="24.36%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001364713352_p31747823"><a name="zh-cn_topic_0000001364713352_p31747823"></a><a name="zh-cn_topic_0000001364713352_p31747823"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="42.99%" headers="mcps1.1.4.1.3 "><a name="ul93492385537"></a><a name="ul93492385537"></a><ul id="ul93492385537"><li>SUCCESS：数据获取成功。</li><li>FAILED：数据获取失败。</li><li>其他错误码请参考<a href="UDF错误码.md">UDF错误码</a>。</li></ul>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001364713352_section24049039"></a>

无

