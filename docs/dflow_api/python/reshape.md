# reshape<a name="ZH-CN_TOPIC_0000001976840850"></a>

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

## 函数功能<a name="zh-cn_topic_0000001439153154_zh-cn_topic_0000001357345205_zh-cn_topic_0000001265240874_section46831934202919"></a>

对tensor进行Reshape操作，不改变tensor的内容。

## 函数原型<a name="zh-cn_topic_0000001439153154_zh-cn_topic_0000001357345205_zh-cn_topic_0000001265240874_section186399319293"></a>

```
def reshape(self, shape: Union[List[int], Tuple[int]])
```

## 参数说明<a name="zh-cn_topic_0000001439153154_zh-cn_topic_0000001357345205_zh-cn_topic_0000001265240874_section15860103919294"></a>

<a name="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_table66993202"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_row41236172"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p51795644"><a name="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p51795644"></a><a name="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p51795644"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="25.6%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p34697616"><a name="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p34697616"></a><a name="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p34697616"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="46.77%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p17794566"><a name="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p17794566"></a><a name="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p17794566"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_row32073719"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p47834478"><a name="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p47834478"></a><a name="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p47834478"></a>shape</p>
</td>
<td class="cellrowborder" valign="top" width="25.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p49387472"><a name="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p49387472"></a><a name="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p49387472"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p29612923"><a name="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p29612923"></a><a name="zh-cn_topic_0000001359389150_zh-cn_topic_0000001312720989_p29612923"></a>要改变的目标shape。</p>
<p id="p9541124716473"><a name="p9541124716473"></a><a name="p9541124716473"></a>要求shape元素个数必须和原来shape的个数一致。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001439153154_zh-cn_topic_0000001357345205_zh-cn_topic_0000001265240874_section1490114312914"></a>

-   0：SUCCESS
-   other：FAILED，具体请参考[UDF错误码](UDF错误码.md)。

## 异常处理<a name="zh-cn_topic_0000001439153154_zh-cn_topic_0000001357345205_zh-cn_topic_0000001265240874_section16219204732913"></a>

无

## 约束说明<a name="zh-cn_topic_0000001439153154_zh-cn_topic_0000001357345205_zh-cn_topic_0000001265240874_section6477115015296"></a>

如果对输入进行reshape动作，可能会影响其他使用本输入的节点正常执行。

