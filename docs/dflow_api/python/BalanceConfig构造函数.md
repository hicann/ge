# BalanceConfig构造函数<a name="ZH-CN_TOPIC_0000001976840882"></a>

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

## 函数功能<a name="zh-cn_topic_0000001481724054_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section36893359"></a>

BalanceConfig构造函数。

## 函数原型<a name="zh-cn_topic_0000001481724054_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section136951948195410"></a>

```
__init__(self, row_num: int, col_num: int, affinity_policy: AffinityPolicy = AffinityPolicy.NO_AFFINITY) -> None
```

## 参数说明<a name="zh-cn_topic_0000001481724054_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section63604780"></a>

<a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_table66993202"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_row41236172"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p51795644"><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p51795644"></a><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p51795644"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="25.6%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p34697616"><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p34697616"></a><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p34697616"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="46.77%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p17794566"><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p17794566"></a><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p17794566"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_row32073719"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p19619162613113"><a name="p19619162613113"></a><a name="p19619162613113"></a>row_num</p>
</td>
<td class="cellrowborder" valign="top" width="25.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001600884160_p996815225551"><a name="zh-cn_topic_0000001600884160_p996815225551"></a><a name="zh-cn_topic_0000001600884160_p996815225551"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p29612923"><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p29612923"></a><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p29612923"></a>权重矩阵行数</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_row10981106134517"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p861912619310"><a name="p861912619310"></a><a name="p861912619310"></a>col_num</p>
</td>
<td class="cellrowborder" valign="top" width="25.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001600884160_p497162295513"><a name="zh-cn_topic_0000001600884160_p497162295513"></a><a name="zh-cn_topic_0000001600884160_p497162295513"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.77%" headers="mcps1.1.4.1.3 "><p id="p1345963433418"><a name="p1345963433418"></a><a name="p1345963433418"></a>权重矩阵列数</p>
</td>
</tr>
<tr id="row18987112193414"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p598716127344"><a name="p598716127344"></a><a name="p598716127344"></a>affinity_policy</p>
</td>
<td class="cellrowborder" valign="top" width="25.6%" headers="mcps1.1.4.1.2 "><p id="p13987512113415"><a name="p13987512113415"></a><a name="p13987512113415"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.77%" headers="mcps1.1.4.1.3 "><p id="p19988812123419"><a name="p19988812123419"></a><a name="p19988812123419"></a>亲和策略，默认非亲和</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001481724054_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section35572112"></a>

返回BalanceConfig类型的对象。

## 异常处理<a name="zh-cn_topic_0000001481724054_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section51713556"></a>

无

## 约束说明<a name="zh-cn_topic_0000001481724054_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section62768825"></a>

无

