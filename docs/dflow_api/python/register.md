# register<a name="ZH-CN_TOPIC_0000002094618864"></a>

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

## 函数功能<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section3729174918713"></a>

注册自定义类型对应的序列化、反序列化、计算size的函数，可结合feed，fetch接口使用，用于feed/fetch任意Python类型。

## 函数原型<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section84161445741"></a>

```
register(msg_type, clz, serialize_func, deserialize_func, size_func=None)
```

## 参数说明<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_section63604780"></a>

<a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_table2051894852017"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row4558174815206"><th class="cellrowborder" valign="top" width="22.16%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"></a><strong id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="15.120000000000001%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="62.72%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"></a><strong id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"></a>取值说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row6558548162013"><td class="cellrowborder" valign="top" width="22.16%" headers="mcps1.1.4.1.1 "><p id="p16010348455"><a name="p16010348455"></a><a name="p16010348455"></a>msg_type</p>
</td>
<td class="cellrowborder" valign="top" width="15.120000000000001%" headers="mcps1.1.4.1.2 "><p id="p13542175918453"><a name="p13542175918453"></a><a name="p13542175918453"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="p859183434516"><a name="p859183434516"></a><a name="p859183434516"></a>注册的类型ID。</p>
</td>
</tr>
<tr id="row7167114171519"><td class="cellrowborder" valign="top" width="22.16%" headers="mcps1.1.4.1.1 "><p id="p161671041141513"><a name="p161671041141513"></a><a name="p161671041141513"></a>clz</p>
</td>
<td class="cellrowborder" valign="top" width="15.120000000000001%" headers="mcps1.1.4.1.2 "><p id="p1167141171513"><a name="p1167141171513"></a><a name="p1167141171513"></a>类型定义</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="p0167134151519"><a name="p0167134151519"></a><a name="p0167134151519"></a>类型定义，比如int，str，或者自定义的class。</p>
</td>
</tr>
<tr id="row1150917518134"><td class="cellrowborder" valign="top" width="22.16%" headers="mcps1.1.4.1.1 "><p id="p135091521319"><a name="p135091521319"></a><a name="p135091521319"></a>serialize_func</p>
</td>
<td class="cellrowborder" valign="top" width="15.120000000000001%" headers="mcps1.1.4.1.2 "><p id="p1650913513131"><a name="p1650913513131"></a><a name="p1650913513131"></a>function</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="p45091512133"><a name="p45091512133"></a><a name="p45091512133"></a>序列化函数，输入是任意的Python对象，输出bytes<span>类型的数据，即对象被序列化后的字节流</span>。</p>
</td>
</tr>
<tr id="row1341612216137"><td class="cellrowborder" valign="top" width="22.16%" headers="mcps1.1.4.1.1 "><p id="p12416122161312"><a name="p12416122161312"></a><a name="p12416122161312"></a>deserialize_func</p>
</td>
<td class="cellrowborder" valign="top" width="15.120000000000001%" headers="mcps1.1.4.1.2 "><p id="p204169222139"><a name="p204169222139"></a><a name="p204169222139"></a>function</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="p19416222201313"><a name="p19416222201313"></a><a name="p19416222201313"></a>反序列化函数，<span>输入类型为</span>bytes<span>，表示要反序列化的字节流</span>，输出为<span>被反序列化的对象。可以是任何Python对象类型</span>。</p>
</td>
</tr>
<tr id="row1336629141314"><td class="cellrowborder" valign="top" width="22.16%" headers="mcps1.1.4.1.1 "><p id="p123612981315"><a name="p123612981315"></a><a name="p123612981315"></a>size_func</p>
</td>
<td class="cellrowborder" valign="top" width="15.120000000000001%" headers="mcps1.1.4.1.2 "><p id="p183662981318"><a name="p183662981318"></a><a name="p183662981318"></a>function</p>
</td>
<td class="cellrowborder" valign="top" width="62.72%" headers="mcps1.1.4.1.3 "><p id="p12361829181317"><a name="p12361829181317"></a><a name="p12361829181317"></a>计算序列化后内存大小的函数，单位字节，预留字段。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section413535858"></a>

正常场景下返回None。

异常情况下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例<a name="section17821439839"></a>

```
import cloudpickle
import dataflow as df

class TestClass():
    def __init__(self, name, val):
        self.name = name
        self.val = val

df.msg_type_register.register(1026, TestClass, lambda obj: cloudpickle.dumps(obj), lambda buffer: cloudpickle.loads(buffer))
```

## 约束说明<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section2021419196520"></a>

无

