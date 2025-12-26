# get\_user\_data（UDF）<a name="ZH-CN_TOPIC_0000001976840878"></a>

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

## 函数功能<a name="zh-cn_topic_0000001481728758_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section51668594"></a>

获取用户定义数据。

## 函数原型<a name="zh-cn_topic_0000001481728758_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section45209275152"></a>

```
get_user_data(self, size: int , offset: int = 0) -> bytearray
```

## 参数说明<a name="zh-cn_topic_0000001481728758_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section62364163"></a>

<a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_table66993202"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_row41236172"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p51795644"><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p51795644"></a><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p51795644"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="25.6%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p34697616"><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p34697616"></a><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p34697616"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="46.77%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p17794566"><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p17794566"></a><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p17794566"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_row32073719"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p19619162613113"><a name="p19619162613113"></a><a name="p19619162613113"></a>size</p>
</td>
<td class="cellrowborder" valign="top" width="25.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001600884160_p996815225551"><a name="zh-cn_topic_0000001600884160_p996815225551"></a><a name="zh-cn_topic_0000001600884160_p996815225551"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p29612923"><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p29612923"></a><a name="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_zh-cn_topic_0000001312720989_p29612923"></a>用户数据长度。取值范围[0, 64]。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001481244598_zh-cn_topic_0000001439153142_zh-cn_topic_0000001304065500_row10981106134517"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p861912619310"><a name="p861912619310"></a><a name="p861912619310"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="25.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001600884160_p497162295513"><a name="zh-cn_topic_0000001600884160_p497162295513"></a><a name="zh-cn_topic_0000001600884160_p497162295513"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001600884160_p1127232683810"><a name="zh-cn_topic_0000001600884160_p1127232683810"></a><a name="zh-cn_topic_0000001600884160_p1127232683810"></a>用户数据的偏移值，需要遵循如下约束。</p>
<p id="zh-cn_topic_0000001600884160_p1960362925219"><a name="zh-cn_topic_0000001600884160_p1960362925219"></a><a name="zh-cn_topic_0000001600884160_p1960362925219"></a>[0, 64), size+offset&lt;=64</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001481728758_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section24406563"></a>

返回用户自定义的数据，类型是bytearray。

## 异常处理<a name="zh-cn_topic_0000001481728758_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section18332482"></a>

无

## 约束说明<a name="zh-cn_topic_0000001481728758_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section30774618"></a>

offset不传时默认值=0。返回的类型是bytearray，用户需要根据定义的结构反向解析。

-   string类型用byte\_array.decode\("utf-8"\)解析。
-   int类型可以用int.from\_bytes\(byte\_array, byteorder='big'\)解析，byteorder根据环境设置。
-   float类型可以用struct.unpack\('f', byte\_array\)\[0\]解析。

