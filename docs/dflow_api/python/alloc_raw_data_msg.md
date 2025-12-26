# alloc\_raw\_data\_msg<a name="ZH-CN_TOPIC_0000002094794960"></a>

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

根据输入的size申请一块连续内存，用于承载raw data类型的FlowMsg。

## 函数原型<a name="zh-cn_topic_0000001481728758_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section45209275152"></a>

```
alloc_raw_data_msg(self, size, align:Optional[int] = 64) -> FlowMsg
```

## 参数说明<a name="zh-cn_topic_0000001481728758_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section62364163"></a>

<a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_table66993202"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_row41236172"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p51795644"><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p51795644"></a><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p51795644"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="25.629999999999995%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p34697616"><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p34697616"></a><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p34697616"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="46.739999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p17794566"><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p17794566"></a><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p17794566"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_row32073719"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p47834478"><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p47834478"></a><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p47834478"></a>size</p>
</td>
<td class="cellrowborder" valign="top" width="25.629999999999995%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p49387472"><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p49387472"></a><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p49387472"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.739999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p29612923"><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p29612923"></a><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p29612923"></a>申请内存大小，单位字节。</p>
</td>
</tr>
<tr id="row1741163752015"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p1874113752014"><a name="p1874113752014"></a><a name="p1874113752014"></a>align</p>
</td>
<td class="cellrowborder" valign="top" width="25.629999999999995%" headers="mcps1.1.4.1.2 "><p id="p1974116378207"><a name="p1974116378207"></a><a name="p1974116378207"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.739999999999995%" headers="mcps1.1.4.1.3 "><p id="p16742037152013"><a name="p16742037152013"></a><a name="p16742037152013"></a>申请内存地址对齐大小，取值范围【32、64、128、256、512、1024】，默认值为64。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001481728758_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section24406563"></a>

正常返回FlowMsg的实例。申请失败返回None。

## 异常处理<a name="zh-cn_topic_0000001481728758_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section18332482"></a>

无

## 约束说明<a name="zh-cn_topic_0000001481728758_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section30774618"></a>

无

