# npu\_model<a name="ZH-CN_TOPIC_0000002192518470"></a>

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

如果UDF部署在host侧，执行时数据需要从device拷贝到本地进行运算。对于PyTorch场景，如果计算全在device侧，输入输出也是在device侧，执行时数据需要从device拷贝到host，执行后PyTorch再将数据搬到device侧，影响执行性能，使用npu\_model可以优化为不搬移数据（即直接下沉到device执行）的方式触发执行。

## 函数原型<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section84161445741"></a>

```
装饰器@npu_model
```

## 参数说明<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_section63604780"></a>

<a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_table2051894852017"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row4558174815206"><th class="cellrowborder" valign="top" width="22.220000000000002%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"></a><strong id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="15.040000000000001%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="62.739999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"></a><strong id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"></a>取值说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row1879239200"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p5564108182019"><a name="p5564108182019"></a><a name="p5564108182019"></a>optimize_level</p>
</td>
<td class="cellrowborder" valign="top" width="15.040000000000001%" headers="mcps1.1.4.1.2 "><p id="p2879631209"><a name="p2879631209"></a><a name="p2879631209"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="62.739999999999995%" headers="mcps1.1.4.1.3 "><a name="ul49001236125"></a><a name="ul49001236125"></a><ul id="ul49001236125"><li>1：PyTorch场景下，通过UDF nn引擎完成输入输出数据下沉到device执行，默认值为1。</li><li>2：把PyTorch模型编译成图，直接作为nn模型导出，优化为npu模型加载执行，需要配合input_descs使用。<div class="note" id="note117021914243"><a name="note117021914243"></a><a name="note117021914243"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p10701319112413"><a name="p10701319112413"></a><a name="p10701319112413"></a>该配置项在修饰类的时候起作用，修饰函数不能配置。</p>
</div></div>
</li></ul>
</td>
</tr>
<tr id="row914714596192"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p8642644218"><a name="p8642644218"></a><a name="p8642644218"></a>input_descs</p>
</td>
<td class="cellrowborder" valign="top" width="15.040000000000001%" headers="mcps1.1.4.1.2 "><p id="p5148125913193"><a name="p5148125913193"></a><a name="p5148125913193"></a>[<a href="dataflow-TensorDesc.md">TensorDesc</a>]</p>
</td>
<td class="cellrowborder" valign="top" width="62.739999999999995%" headers="mcps1.1.4.1.3 "><p id="p13910261322"><a name="p13910261322"></a><a name="p13910261322"></a>当optimize_level=2时，用于表达torch导出成图的输入tensor描述，示例如下：</p>
<a name="screen245012519174"></a><a name="screen245012519174"></a><pre class="screen" codetype="Python" id="screen245012519174">input_descs=[TensorDesc(dtype = df.DT_INT64, shape = [2,1,4]),TensorDesc(dtype =
df.DT_FLOAT, shape = [2,1,4])],</pre>
<p id="p14910166183217"><a name="p14910166183217"></a><a name="p14910166183217"></a>当shape中某一维度为负值，表示输入是动态的，通过npu_model最终会导出成动态图。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row6558548162013"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p8565162815120"><a name="p8565162815120"></a><a name="p8565162815120"></a>num_returns</p>
</td>
<td class="cellrowborder" valign="top" width="15.040000000000001%" headers="mcps1.1.4.1.2 "><p id="p13542175918453"><a name="p13542175918453"></a><a name="p13542175918453"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="62.739999999999995%" headers="mcps1.1.4.1.3 "><p id="p159541233496"><a name="p159541233496"></a><a name="p159541233496"></a>装饰器装饰函数时，用于表示函数的输出个数，不设置该参数时默认函数返回一个返回值。该参数与使用<span>type annotations</span>方式标识函数返回个数与类型的方式选择其一即可。</p>
</td>
</tr>
<tr id="row1732183172618"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p134821218181318"><a name="p134821218181318"></a><a name="p134821218181318"></a>resources</p>
</td>
<td class="cellrowborder" valign="top" width="15.040000000000001%" headers="mcps1.1.4.1.2 "><p id="p1167141171513"><a name="p1167141171513"></a><a name="p1167141171513"></a>dict</p>
</td>
<td class="cellrowborder" valign="top" width="62.739999999999995%" headers="mcps1.1.4.1.3 "><p id="p16125185619483"><a name="p16125185619483"></a><a name="p16125185619483"></a>用于标识当前func需要的资源信息，支持memory、num_cpus和num_npus。memory单位为M; num_npus表示需要使用npu资源数量，为预留参数，当前仅支持1。例如：{"memory": 100, "num_cpus": 1, "num_npus": 1}</p>
</td>
</tr>
<tr id="row2377193362615"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p1620472171312"><a name="p1620472171312"></a><a name="p1620472171312"></a>env_hook_func</p>
</td>
<td class="cellrowborder" valign="top" width="15.040000000000001%" headers="mcps1.1.4.1.2 "><p id="p10204121171320"><a name="p10204121171320"></a><a name="p10204121171320"></a>function</p>
</td>
<td class="cellrowborder" valign="top" width="62.739999999999995%" headers="mcps1.1.4.1.3 "><p id="p152048219130"><a name="p152048219130"></a><a name="p152048219130"></a>此钩子函数用于给用户自行扩展在Python UDF初始化之前必要的Python环境准备或import操作。</p>
</td>
</tr>
<tr id="row18377161111217"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p537721112216"><a name="p537721112216"></a><a name="p537721112216"></a>visible_device_enable</p>
</td>
<td class="cellrowborder" valign="top" width="15.040000000000001%" headers="mcps1.1.4.1.2 "><p id="p6377161112212"><a name="p6377161112212"></a><a name="p6377161112212"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="62.739999999999995%" headers="mcps1.1.4.1.3 "><p id="p16377111192116"><a name="p16377111192116"></a><a name="p16377111192116"></a>开启后，UDF进程会根据用户配置num_npus资源自动设置ASCEND_RT_VISIBLE_DEVICES，调用get_running_device_id接口获取对应的逻辑ID，当前num_npus仅支持1，因此该场景下get_running_device_id结果为0。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section413535858"></a>

正常场景下返回被装饰的函数。

异常情况下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例<a name="section17821439839"></a>

```
@df.npu_model(optimize_level=1)
class FakeModel1(nn.Module):
    def __init__(self):
        super().__init__()

    # 模拟模型推理
    @df.method()
    def forward(self, input_image):
        return F.interpolate(input_image, size=(256, 256), mode='bilinear')

@df.npu_model(optimize_level=1, input_descs=[df.TensorDesc(dtype=df.DT_FLOAT, shape=[1, 3, 768, 768])])
class FakeModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = 0.5
        self.std = 0.5

    # 模拟模型推理
    @df.method()
    def forward(self, input_image):
        return (input_image - self.mean) / self.std

@df.npu_model()
def preprocess(input_image):
    # 模拟图片裁切
    transform = transforms.Compose([transforms.CenterCrop(512)])
    return transform(input_image)

@df.npu_model()
def postprocess(input_image):
    mean = 0.5
    std = 0.5
    img = input_image * std + mean
    return F.interpolate(img, size=(512, 512), mode='bilinear')
```

## 约束说明<a name="zh-cn_topic_0000001411032868_zh-cn_topic_0000001265240866_section2021419196520"></a>

-   需安装对应Python版本的torch\_npu包。
-   输入输出必须为npu tensor。
-   一组输入对应一组输出，不支持流式输入输出。

