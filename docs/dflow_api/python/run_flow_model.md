# run\_flow\_model<a name="ZH-CN_TOPIC_0000001977000606"></a>

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

## 函数功能<a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section51668594"></a>

同步执行指定的模型。

## 函数原型<a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section45209275152"></a>

```
run_flow_model(self, model_key:str, input_msgs: List[FlowMsg], timeout: int) -> Tuple[int, FlowMsg]
```

## 参数说明<a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section62364163"></a>

<a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_table66993202"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_row41236172"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p51795644"><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p51795644"></a><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p51795644"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="25.6%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p34697616"><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p34697616"></a><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p34697616"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="46.77%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p17794566"><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p17794566"></a><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p17794566"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_row32073719"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p47834478"><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p47834478"></a><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p47834478"></a>model_key</p>
</td>
<td class="cellrowborder" valign="top" width="25.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p49387472"><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p49387472"></a><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p49387472"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p29612923"><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p29612923"></a><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001312720989_p29612923"></a>指定的模型key。当使用C++开发DataFlow时，与AddInvokedClosure中指定的name一致。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_row1697801575511"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p10979121517555"><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p10979121517555"></a><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p10979121517555"></a>input_msgs</p>
</td>
<td class="cellrowborder" valign="top" width="25.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p4979191525518"><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p4979191525518"></a><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p4979191525518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p1397916155554"><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p1397916155554"></a><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p1397916155554"></a>提供给模型的输入。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_row14692135283017"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p1669225223018"><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p1669225223018"></a><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p1669225223018"></a>timeout</p>
</td>
<td class="cellrowborder" valign="top" width="25.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p106927523301"><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p106927523301"></a><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p106927523301"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p1369225215308"><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p1369225215308"></a><a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_p1369225215308"></a>等待模型执行超时时间，单位ms，-1表示永不超时。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section24406563"></a>

-   0：SUCCESS
-   other：FAILED，具体请参考[UDF错误码](UDF错误码.md)。

## 异常处理<a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section18332482"></a>

无

## 约束说明<a name="zh-cn_topic_0000001481564194_zh-cn_topic_0000001439470038_zh-cn_topic_0000001468175517_zh-cn_topic_0000001264921066_section30774618"></a>

无

