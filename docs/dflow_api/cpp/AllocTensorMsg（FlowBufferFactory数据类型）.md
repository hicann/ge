# AllocTensorMsg（FlowBufferFactory数据类型）<a name="ZH-CN_TOPIC_0000002094275286"></a>

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

## 函数功能<a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001408829021_zh-cn_topic_0000001264921066_section51668594"></a>

根据shape、data type和对齐大小申请FlowMsg。

## 函数原型<a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001408829021_zh-cn_topic_0000001264921066_section45209275152"></a>

```
FlowMsgPtr AllocTensorMsg(const std::vector<int64_t> &shape, DataType data_type, uint32_t align = 512U)
```

## 参数说明<a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001408829021_zh-cn_topic_0000001264921066_section62364163"></a>

<a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001409388721_zh-cn_topic_0000001312720989_table66993202"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002013837145_zh-cn_topic_0000001409388721_zh-cn_topic_0000001312720989_row41236172"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002013837145_zh-cn_topic_0000001409388721_zh-cn_topic_0000001312720989_p51795644"><a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001409388721_zh-cn_topic_0000001312720989_p51795644"></a><a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001409388721_zh-cn_topic_0000001312720989_p51795644"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="25.6%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002013837145_zh-cn_topic_0000001409388721_zh-cn_topic_0000001312720989_p34697616"><a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001409388721_zh-cn_topic_0000001312720989_p34697616"></a><a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001409388721_zh-cn_topic_0000001312720989_p34697616"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="46.77%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002013837145_zh-cn_topic_0000001409388721_zh-cn_topic_0000001312720989_p17794566"><a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001409388721_zh-cn_topic_0000001312720989_p17794566"></a><a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001409388721_zh-cn_topic_0000001312720989_p17794566"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002013837145_zh-cn_topic_0000001409388721_zh-cn_topic_0000001312720989_row32073719"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002013837145_p219515316174"><a name="zh-cn_topic_0000002013837145_p219515316174"></a><a name="zh-cn_topic_0000002013837145_p219515316174"></a>shape</p>
</td>
<td class="cellrowborder" valign="top" width="25.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002013837145_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p49387472"><a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p49387472"></a><a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p49387472"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002013837145_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p29612923"><a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p29612923"></a><a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p29612923"></a>Tensor FlowMsg的shape。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002013837145_row10638191716386"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002013837145_p111941553161710"><a name="zh-cn_topic_0000002013837145_p111941553161710"></a><a name="zh-cn_topic_0000002013837145_p111941553161710"></a>data_type</p>
</td>
<td class="cellrowborder" valign="top" width="25.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002013837145_zh-cn_topic_0000001304225452_p119816634515"><a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001304225452_p119816634515"></a><a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001304225452_p119816634515"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002013837145_zh-cn_topic_0000001304225452_p998166134517"><a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001304225452_p998166134517"></a><a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001304225452_p998166134517"></a>Tensor FlowMsg的DataType 。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002013837145_row1327216269381"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002013837145_p119385341714"><a name="zh-cn_topic_0000002013837145_p119385341714"></a><a name="zh-cn_topic_0000002013837145_p119385341714"></a>align</p>
</td>
<td class="cellrowborder" valign="top" width="25.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002013837145_p121929532171"><a name="zh-cn_topic_0000002013837145_p121929532171"></a><a name="zh-cn_topic_0000002013837145_p121929532171"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002013837145_p10192353181716"><a name="zh-cn_topic_0000002013837145_p10192353181716"></a><a name="zh-cn_topic_0000002013837145_p10192353181716"></a>申请内存地址对齐大小，取值范围 【32、64、128、256、512、1024】，当前为预留参数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001408829021_zh-cn_topic_0000001264921066_section24406563"></a>

申请的FlowMsg指针。

## 异常处理<a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001408829021_zh-cn_topic_0000001264921066_section18332482"></a>

申请不到FlowMsg指针则返回NULL。

## 约束说明<a name="zh-cn_topic_0000002013837145_zh-cn_topic_0000001408829021_zh-cn_topic_0000001264921066_section30774618"></a>

无。

