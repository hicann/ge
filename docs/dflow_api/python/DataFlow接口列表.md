# DataFlow接口列表<a name="ZH-CN_TOPIC_0000002013394133"></a>

使用DataFlow Python接口构造DataFlow图进行推理。支持定义图处理点，UDF处理点，描述处理点之间的数据流关系；支持导入TensorFlow, ONNX, MindSpore的IR文件作为图处理点计算逻辑定义。

## DataFlow构图接口<a name="section32413319013"></a>

**表 1**  DataFlow构图接口

<a name="table1644114511445"></a>
<table><thead align="left"><tr id="row184417511246"><th class="cellrowborder" valign="top" width="66.12%" id="mcps1.2.3.1.1"><p id="p6441125120410"><a name="p6441125120410"></a><a name="p6441125120410"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="33.879999999999995%" id="mcps1.2.3.1.2"><p id="p1144117511349"><a name="p1144117511349"></a><a name="p1144117511349"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row17441125116414"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p1745133310426"><a name="p1745133310426"></a><a name="p1745133310426"></a><a href="dataflow-FlowData.md">dataflow.FlowData</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p1472802714307"><a name="p1472802714307"></a><a name="p1472802714307"></a>DataFlow Graph中的数据节点，每个FlowData对应一个输入。</p>
</td>
</tr>
<tr id="row944214511419"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p844993304214"><a name="p844993304214"></a><a name="p844993304214"></a><a href="FlowNode.md">FlowNode</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p6585114913377"><a name="p6585114913377"></a><a name="p6585114913377"></a>DataFlow Graph中的计算节点。</p>
</td>
</tr>
<tr id="row12318757162"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p8133103034210"><a name="p8133103034210"></a><a name="p8133103034210"></a><a href="add_process_point.md">add_process_point</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001411352684_p15227121203"><a name="zh-cn_topic_0000001411352684_p15227121203"></a><a name="zh-cn_topic_0000001411352684_p15227121203"></a>给FlowNode添加映射的pp，当前一个FlowNode仅能添加一个pp，添加后会默认将FlowNode的输入输出和pp的输入输出按顺序进行映射。</p>
</td>
</tr>
<tr id="row1231816571467"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p151316301426"><a name="p151316301426"></a><a name="p151316301426"></a><a href="map_input.md">map_input</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001461192589_p1339913564556"><a name="zh-cn_topic_0000001461192589_p1339913564556"></a><a name="zh-cn_topic_0000001461192589_p1339913564556"></a>给FlowNode映射输入，表示将FlowNode的第node_input_index个输入给到ProcessPoint的第pp_input_index个输入，并且给ProcessPoint的该输入设置上attr里的所有属性，返回映射好的FlowNode节点。该函数可选，不被调用时会默认按顺序去映射FlowNode和ProcessPoint的输入。</p>
</td>
</tr>
<tr id="row931813574614"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p313003044211"><a name="p313003044211"></a><a name="p313003044211"></a><a href="map_output.md">map_output</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001461432453_p18295932155813"><a name="zh-cn_topic_0000001461432453_p18295932155813"></a><a name="zh-cn_topic_0000001461432453_p18295932155813"></a>给FlowNode映射输出，表示将pp的第pp_output_index个输出给到FlowNode的第node_output_index个输出，返回映射好的FlowNode节点。</p>
</td>
</tr>
<tr id="row469180173913"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p912819300425"><a name="p912819300425"></a><a name="p912819300425"></a><a href="set_attr.md">set_attr</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001461432453_p073229135814"><a name="zh-cn_topic_0000001461432453_p073229135814"></a><a name="zh-cn_topic_0000001461432453_p073229135814"></a>设置FlowNode的属性。</p>
</td>
</tr>
<tr id="row1692402398"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p412793015428"><a name="p412793015428"></a><a name="p412793015428"></a><a href="__call__.md">__call__</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p645884245013"><a name="p645884245013"></a><a name="p645884245013"></a>调用FlowNode进行计算。</p>
</td>
</tr>
<tr id="row857616411814"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p10126183054220"><a name="p10126183054220"></a><a name="p10126183054220"></a><a href="set_balance_scatter.md">set_balance_scatter</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p74583478503"><a name="p74583478503"></a><a name="p74583478503"></a>设置节点balance scatter属性，具有balance scatter属性的UDF可以使用balance options设置负载均衡输出。</p>
</td>
</tr>
<tr id="row5692203397"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p1112443074211"><a name="p1112443074211"></a><a name="p1112443074211"></a><a href="set_balance_gather.md">set_balance_gather</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p112751352135012"><a name="p112751352135012"></a><a name="p112751352135012"></a>设置节点balance gather属性，具有balance gather属性的UDF可以使用balance options设置负载均衡亲和输出。</p>
</td>
</tr>
<tr id="row184191488528"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p2419287521"><a name="p2419287521"></a><a name="p2419287521"></a><a href="set_alias.md">set_alias</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p15804163215812"><a name="p15804163215812"></a><a name="p15804163215812"></a>设置节点别名，使用option:ge.experiment.data_flow_deploy_info_path指定节点部署位置时，flow_node_list字段可使用别名进行指定。</p>
</td>
</tr>
<tr id="row1669270203910"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p1712318307426"><a name="p1712318307426"></a><a name="p1712318307426"></a><a href="dataflow-FlowFlag.md">dataflow.FlowFlag</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p131669015114"><a name="p131669015114"></a><a name="p131669015114"></a>设置FlowMsg消息头中的flags。</p>
</td>
</tr>
<tr id="row1169220011394"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p1458262674216"><a name="p1458262674216"></a><a name="p1458262674216"></a><a href="FlowGraph.md">FlowGraph</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p151822212517"><a name="p151822212517"></a><a name="p151822212517"></a>DataFlow的graph，由输入节点FlowData和计算节点FlowNode构成。</p>
</td>
</tr>
<tr id="row12692130113916"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p135806265428"><a name="p135806265428"></a><a name="p135806265428"></a><a href="set_contains_n_mapping_node.md">set_contains_n_mapping_node</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p15801264429"><a name="p15801264429"></a><a name="p15801264429"></a>设置FlowGraph是否包含n_mapping节点。</p>
</td>
</tr>
<tr id="row1669211043917"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p1257992611422"><a name="p1257992611422"></a><a name="p1257992611422"></a><a href="set_inputs_align_attrs.md">set_inputs_align_attrs</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p613315351418"><a name="p613315351418"></a><a name="p613315351418"></a>设置FlowGraph中的输入对齐属性。</p>
</td>
</tr>
<tr id="row177523134554"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p177521013115515"><a name="p177521013115515"></a><a name="p177521013115515"></a><a href="set_exception_catch.md">set_exception_catch</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p1037104143720"><a name="p1037104143720"></a><a name="p1037104143720"></a>设置用户异常捕获功能是否开启。</p>
</td>
</tr>
<tr id="row1469311014398"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p757822654220"><a name="p757822654220"></a><a name="p757822654220"></a><a href="dataflow-FlowOutput.md">dataflow.FlowOutput</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001461312441_p290817358314"><a name="zh-cn_topic_0000001461312441_p290817358314"></a><a name="zh-cn_topic_0000001461312441_p290817358314"></a>描述FlowNode的输出。</p>
</td>
</tr>
<tr id="row1869320073914"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p1576326174220"><a name="p1576326174220"></a><a name="p1576326174220"></a><a href="dataflow-Framework.md">dataflow.Framework</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p164171949105112"><a name="p164171949105112"></a><a name="p164171949105112"></a>设置原始网络模型的框架类型。</p>
</td>
</tr>
<tr id="row13693170103919"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p6575132610427"><a name="p6575132610427"></a><a name="p6575132610427"></a><a href="FuncProcessPoint.md">FuncProcessPoint</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001461432457_p192271624587"><a name="zh-cn_topic_0000001461432457_p192271624587"></a><a name="zh-cn_topic_0000001461432457_p192271624587"></a>FuncProcessPoint的构造函数，返回一个FuncProcessPoint对象。</p>
</td>
</tr>
<tr id="row11693404394"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p1057412624210"><a name="p1057412624210"></a><a name="p1057412624210"></a><a href="set_init_param.md">set_init_param</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001461072801_p2793201513"><a name="zh-cn_topic_0000001461072801_p2793201513"></a><a name="zh-cn_topic_0000001461072801_p2793201513"></a>设置FuncProcessPoint的初始化参数。</p>
</td>
</tr>
<tr id="row183321753183816"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p9572172654218"><a name="p9572172654218"></a><a name="p9572172654218"></a><a href="add_invoked_closure.md">add_invoked_closure</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001411192724_p1532416920"><a name="zh-cn_topic_0000001411192724_p1532416920"></a><a name="zh-cn_topic_0000001411192724_p1532416920"></a>添加FuncProcessPoint调用的GraphProcessPoint或者FlowGraphProcessPoint，返回添加好的FuncProcessPoint。</p>
</td>
</tr>
<tr id="row7621157175215"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p762115575524"><a name="p762115575524"></a><a name="p762115575524"></a><a href="GraphProcessPoint.md">GraphProcessPoint</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p77041016533"><a name="p77041016533"></a><a name="p77041016533"></a>GraphProcessPoint构造函数，返回一个GraphProcessPoint对象。</p>
</td>
</tr>
<tr id="row1385449121113"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p2852049181116"><a name="p2852049181116"></a><a name="p2852049181116"></a><a href="fnode.md">fnode</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p338918404127"><a name="p338918404127"></a><a name="p338918404127"></a>根据当前的GraphProcessPoint生成一个FlowNode，返回一个FlowNode对象。</p>
</td>
</tr>
<tr id="row13264184720537"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p1557182619426"><a name="p1557182619426"></a><a name="p1557182619426"></a><a href="dataflow-FlowGraphProcessPoint.md">dataflow.FlowGraphProcessPoint</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p1550211462368"><a name="p1550211462368"></a><a name="p1550211462368"></a>GraphProcessPoint构造函数，返回一个GraphProcessPoint对象。</p>
</td>
</tr>
<tr id="row133355310381"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p7570102644210"><a name="p7570102644210"></a><a name="p7570102644210"></a><a href="Tensor.md">Tensor</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p17697172711302"><a name="p17697172711302"></a><a name="p17697172711302"></a>Tensor的构造函数。</p>
</td>
</tr>
<tr id="row033313531380"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p19915522194214"><a name="p19915522194214"></a><a name="p19915522194214"></a><a href="numpy.md">numpy</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p19469758103019"><a name="p19469758103019"></a><a name="p19469758103019"></a>将Tensor转换到numpy的ndarray。</p>
</td>
</tr>
<tr id="row633365343815"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p9914202213428"><a name="p9914202213428"></a><a name="p9914202213428"></a><a href="dataflow-TensorDesc.md">dataflow.TensorDesc</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p19694559132715"><a name="p19694559132715"></a><a name="p19694559132715"></a>Tensor的描述函数。</p>
</td>
</tr>
<tr id="row13726247134"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p8721424161316"><a name="p8721424161316"></a><a name="p8721424161316"></a><a href="dataflow-alloc_tensor.md">dataflow.alloc_tensor</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_p1476471614810"><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_p1476471614810"></a><a name="zh-cn_topic_0000001532163945_zh-cn_topic_0000001438993186_zh-cn_topic_0000001304225452_p1476471614810"></a>根据shape、data type以及对齐大小申请dataflow tensor。</p>
</td>
</tr>
<tr id="row13133182712135"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p5133727191318"><a name="p5133727191318"></a><a name="p5133727191318"></a><a href="dataflow-utils-generate_deploy_template.md">dataflow.utils.generate_deploy_template</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p7133132701311"><a name="p7133132701311"></a><a name="p7133132701311"></a>根据FlowGraph生成指定部署位置的option:"ge.experiment.data_flow_deploy_info_path"所需要的文件的模板。</p>
</td>
</tr>
<tr id="row18606112561319"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p15606725161316"><a name="p15606725161316"></a><a name="p15606725161316"></a><a href="register.md">register</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p587215031718"><a name="p587215031718"></a><a name="p587215031718"></a>注册自定义类型对应的序列化、反序列化、计算size的函数，可结合feed，fetch接口使用，用于feed/fetch任意Python类型。</p>
</td>
</tr>
<tr id="row169161171415"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p991611118148"><a name="p991611118148"></a><a name="p991611118148"></a><a href="registered.md">registered</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p156221215131818"><a name="p156221215131818"></a><a name="p156221215131818"></a>判断消息类型ID是否被注册过。</p>
</td>
</tr>
<tr id="row1661111521418"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p761151591417"><a name="p761151591417"></a><a name="p761151591417"></a><a href="get_msg_type（dataflow）.md">get_msg_type（dataflow）</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p272152811810"><a name="p272152811810"></a><a name="p272152811810"></a>根据类型定义获取注册的消息类型ID。</p>
</td>
</tr>
<tr id="row35121513181412"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p35121613191416"><a name="p35121613191416"></a><a name="p35121613191416"></a><a href="get_serialize_func.md">get_serialize_func</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p133351033171814"><a name="p133351033171814"></a><a name="p133351033171814"></a>根据消息类型ID获取注册的序列化函数。</p>
</td>
</tr>
<tr id="row1465831001410"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p196584106140"><a name="p196584106140"></a><a name="p196584106140"></a><a href="get_deserialize_func.md">get_deserialize_func</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p3147114011185"><a name="p3147114011185"></a><a name="p3147114011185"></a>根据消息类型ID获取注册的反序列化函数。</p>
</td>
</tr>
<tr id="row27071570144"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p3707374148"><a name="p3707374148"></a><a name="p3707374148"></a><a href="get_size_func.md">get_size_func</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p142915483189"><a name="p142915483189"></a><a name="p142915483189"></a>根据消息类型ID获取注册的计算序列化内存大小的函数。</p>
</td>
</tr>
<tr id="row9959194171413"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p149591649146"><a name="p149591649146"></a><a name="p149591649146"></a><a href="deserialize_from_file.md">deserialize_from_file</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p1421161818199"><a name="p1421161818199"></a><a name="p1421161818199"></a>从序列化的pickle文件进行反序列化恢复Python对象。</p>
</td>
</tr>
<tr id="row2963108171417"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p6964788141"><a name="p6964788141"></a><a name="p6964788141"></a><a href="pyflow.md">pyflow</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p122461332509"><a name="p122461332509"></a><a name="p122461332509"></a>支持将函数作为pipeline任务在本地或者远端运行。</p>
</td>
</tr>
<tr id="row104805321415"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p1048016351419"><a name="p1048016351419"></a><a name="p1048016351419"></a><a href="method.md">method</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p24801038141"><a name="p24801038141"></a><a name="p24801038141"></a>对于复杂场景支持将类作为pipeline任务在本地或者远端运行。</p>
</td>
</tr>
<tr id="row1849320112549"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p1849351115413"><a name="p1849351115413"></a><a name="p1849351115413"></a><a href="npu_model.md">npu_model</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p1118616408575"><a name="p1118616408575"></a><a name="p1118616408575"></a>如果UDF部署在host侧，执行时数据需要从device拷贝到本地进行运算。对于PyTorch场景，如果计算全在device侧，输入输出也是在device侧，执行时数据需要从device拷贝到host，执行后PyTorch再将数据搬到device侧，影响执行性能，使用npu_model可以优化为不搬移数据（即直接下沉到device执行）的方式触发执行。</p>
</td>
</tr>
<tr id="row10333253143819"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p159125224428"><a name="p159125224428"></a><a name="p159125224428"></a><a href="dataflow-CountBatch.md">dataflow.CountBatch</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p3829121785614"><a name="p3829121785614"></a><a name="p3829121785614"></a>CountBatch功能是指基于UDF为计算处理点将多个数据按batchSize组成batch。该功能应用于dataflow异步场景。</p>
</td>
</tr>
<tr id="row12333175313380"><td class="cellrowborder" valign="top" width="66.12%" headers="mcps1.2.3.1.1 "><p id="p69111822154216"><a name="p69111822154216"></a><a name="p69111822154216"></a><a href="dataflow-TimeBatch.md">dataflow.TimeBatch</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.879999999999995%" headers="mcps1.2.3.1.2 "><p id="p44256404482"><a name="p44256404482"></a><a name="p44256404482"></a>TimeBatch功能是基于UDF为前提的。</p>
<p id="p3425204074817"><a name="p3425204074817"></a><a name="p3425204074817"></a>正常模型每次处理一个数据，当需要一次处理一批数据时，就需要将这批数据组成一个Batch。最基本的Batch方式是将这批N个数据直接拼接，然后shape前加N，而某些场景需要将某段或者某几段时间数据组成一个batch，并且按特定的维度拼接，则可以通过使用TimeBatch功能来组Batch。</p>
</td>
</tr>
</tbody>
</table>

## DataFlow运行接口<a name="section144531976011"></a>

**表 2**  DataFlow运行接口

<a name="table10223122443215"></a>
<table><thead align="left"><tr id="row1222312249325"><th class="cellrowborder" valign="top" width="33.47%" id="mcps1.2.3.1.1"><p id="p2224142411323"><a name="p2224142411323"></a><a name="p2224142411323"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="66.53%" id="mcps1.2.3.1.2"><p id="p15224142443217"><a name="p15224142443217"></a><a name="p15224142443217"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row172212917225"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p20226717194210"><a name="p20226717194210"></a><a name="p20226717194210"></a><a href="dataflow-init.md">dataflow.init</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p11733748165312"><a name="p11733748165312"></a><a name="p11733748165312"></a>初始化dataflow时的options。</p>
</td>
</tr>
<tr id="row107851320151015"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1875302319105"><a name="p1875302319105"></a><a name="p1875302319105"></a><a href="FlowInfo.md">FlowInfo</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p9753112310107"><a name="p9753112310107"></a><a name="p9753112310107"></a>DataFlow的flow信息。</p>
</td>
</tr>
<tr id="row14714022191011"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1075322317107"><a name="p1075322317107"></a><a name="p1075322317107"></a><a href="set_user_data.md">set_user_data</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p18753823171014"><a name="p18753823171014"></a><a name="p18753823171014"></a>设置用户信息。</p>
</td>
</tr>
<tr id="row3972101841019"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1675322331013"><a name="p1675322331013"></a><a name="p1675322331013"></a><a href="get_user_data（dataflow）.md">get_user_data（dataflow）</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1875392391017"><a name="p1875392391017"></a><a name="p1875392391017"></a>获取用户信息。</p>
</td>
</tr>
<tr id="row82752026101012"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p17275142601014"><a name="p17275142601014"></a><a name="p17275142601014"></a><a href="user_data.md">user_data</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p296216321395"><a name="p296216321395"></a><a name="p296216321395"></a>获取用户信息。</p>
</td>
</tr>
<tr id="row1182674412116"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p782618441119"><a name="p782618441119"></a><a name="p782618441119"></a><a href="data_size.md">data_size</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p01419578157"><a name="p01419578157"></a><a name="p01419578157"></a>获取user_data的长度。</p>
</td>
</tr>
<tr id="row113318475114"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p73344711113"><a name="p73344711113"></a><a name="p73344711113"></a><a href="start_time.md">start_time</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p206713214164"><a name="p206713214164"></a><a name="p206713214164"></a>以属性方式读取和设置FlowInfo的开始时间。</p>
</td>
</tr>
<tr id="row467510489116"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p1067574813116"><a name="p1067574813116"></a><a name="p1067574813116"></a><a href="end_time.md">end_time</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p15205101019163"><a name="p15205101019163"></a><a name="p15205101019163"></a>以属性方法读取和设置FlowInfo的结束时间。</p>
</td>
</tr>
<tr id="row13116543181116"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p161161643121118"><a name="p161161643121118"></a><a name="p161161643121118"></a><a href="flow_flags.md">flow_flags</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1533561612169"><a name="p1533561612169"></a><a name="p1533561612169"></a>以属性方法读取和设置FlowInfo的flow_flags。</p>
</td>
</tr>
<tr id="row167402030121012"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p874043061019"><a name="p874043061019"></a><a name="p874043061019"></a><a href="transaction_id.md">transaction_id</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p13725021121614"><a name="p13725021121614"></a><a name="p13725021121614"></a>以属性方式读写事务ID。</p>
</td>
</tr>
<tr id="row1122472493211"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p0225161714217"><a name="p0225161714217"></a><a name="p0225161714217"></a><a href="feed_data.md">feed_data</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1342635416533"><a name="p1342635416533"></a><a name="p1342635416533"></a>将数据输入到Graph。</p>
</td>
</tr>
<tr id="row592071212237"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p09207122238"><a name="p09207122238"></a><a name="p09207122238"></a><a href="feed.md">feed</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p7920412122315"><a name="p7920412122315"></a><a name="p7920412122315"></a>将数据输入到Graph，支持可序列化的任意的输入。</p>
</td>
</tr>
<tr id="row6224122443212"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p142231317144211"><a name="p142231317144211"></a><a name="p142231317144211"></a><a href="fetch_data.md">fetch_data</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p2113161115419"><a name="p2113161115419"></a><a name="p2113161115419"></a>获取Graph输出数据。</p>
</td>
</tr>
<tr id="row841917157234"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p194191415172318"><a name="p194191415172318"></a><a name="p194191415172318"></a><a href="fetch.md">fetch</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p17640159152415"><a name="p17640159152415"></a><a name="p17640159152415"></a>获取Graph输出数据。支持可序列化的任意的输出。</p>
</td>
</tr>
<tr id="row1222432443211"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p6222191717422"><a name="p6222191717422"></a><a name="p6222191717422"></a><a href="dataflow-finalize.md">dataflow.finalize</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p2019578547"><a name="p2019578547"></a><a name="p2019578547"></a>释放dataflow初始化的资源。</p>
</td>
</tr>
<tr id="row9749517122320"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p12749111742319"><a name="p12749111742319"></a><a name="p12749111742319"></a><a href="dataflow-get_running_device_id.md">dataflow.get_running_device_id</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p1656597172517"><a name="p1656597172517"></a><a name="p1656597172517"></a>UDF执行时获取当前UDF的运行device_id, 信息来源和UDF部署位置的配置。</p>
</td>
</tr>
<tr id="row861212754620"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p76124710463"><a name="p76124710463"></a><a name="p76124710463"></a><a href="dataflow-get_running_instance_id.md">dataflow.get_running_instance_id</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p4452202720101"><a name="p4452202720101"></a><a name="p4452202720101"></a>UDF执行时获取当前UDF的运行实例ID，该信息来源于data_flow_deploy_info.json中的logic_device_list配置。</p>
</td>
</tr>
<tr id="row1046512444616"><td class="cellrowborder" valign="top" width="33.47%" headers="mcps1.2.3.1.1 "><p id="p646514444617"><a name="p646514444617"></a><a name="p646514444617"></a><a href="dataflow-get_running_instance_num.md">dataflow.get_running_instance_num</a></p>
</td>
<td class="cellrowborder" valign="top" width="66.53%" headers="mcps1.2.3.1.2 "><p id="p9019587132"><a name="p9019587132"></a><a name="p9019587132"></a>UDF执行时获取当前UDF的运行实例个数，该信息来源于data_flow_deploy_info.json中的logic_device_list配置。</p>
</td>
</tr>
</tbody>
</table>

## 模块<a name="section192596163391"></a>

dataflow module：公共接口的命名空间

## 类<a name="section2906163819395"></a>

-   class CountBatch：CountBatch属性的类
-   class FlowData：输入节点类
-   class FlowFlag：数据标记类
-   class FlowGraph：dataflow的图类
-   class FlowInfo：指定输入输出数据携带的信息类
-   class FlowNode：计算节点类
-   class FlowOutput：计算节点的输出类
-   class Framework：IR文件的框架类型的枚举类
-   class FuncProcessPoint：UDF处理点类
-   class GraphProcessPoint：图处理点类
-   class Tensor：张量数据类
-   class TensorDesc：张量的描述类
-   class TimeBatch：TimeBatch属性的类

## 函数<a name="section530695619390"></a>

-   init\(...\)：dataflow的资源初始化方法
-   finalize\(...\)：dataflow的资源释放方法

## 其他成员<a name="section157818198408"></a>

**表 3**  其他成员

<a name="table41909719419"></a>
<table><thead align="left"><tr id="row13190147104111"><th class="cellrowborder" valign="top" width="28.77%" id="mcps1.2.3.1.1"><p id="p152181119591"><a name="p152181119591"></a><a name="p152181119591"></a>名称</p>
</th>
<th class="cellrowborder" valign="top" width="71.23%" id="mcps1.2.3.1.2"><p id="p2019011714115"><a name="p2019011714115"></a><a name="p2019011714115"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row639963715312"><td class="cellrowborder" valign="top" width="28.77%" headers="mcps1.2.3.1.1 "><p id="p639912371315"><a name="p639912371315"></a><a name="p639912371315"></a>DT_FLOAT</p>
</td>
<td class="cellrowborder" valign="top" width="71.23%" headers="mcps1.2.3.1.2 "><p id="p144801671184"><a name="p144801671184"></a><a name="p144801671184"></a>df.data_type.DType的对象</p>
<p id="p0480571683"><a name="p0480571683"></a><a name="p0480571683"></a>32位单精度浮点数</p>
</td>
</tr>
<tr id="row13886512163"><td class="cellrowborder" valign="top" width="28.77%" headers="mcps1.2.3.1.1 "><p id="p148861712961"><a name="p148861712961"></a><a name="p148861712961"></a>DT_FLOAT16</p>
</td>
<td class="cellrowborder" valign="top" width="71.23%" headers="mcps1.2.3.1.2 "><p id="p422814172081"><a name="p422814172081"></a><a name="p422814172081"></a>df.data_type.DType的对象</p>
<p id="p1622815171587"><a name="p1622815171587"></a><a name="p1622815171587"></a>16位半精度浮点数</p>
</td>
</tr>
<tr id="row1575720364"><td class="cellrowborder" valign="top" width="28.77%" headers="mcps1.2.3.1.1 "><p id="p117502020616"><a name="p117502020616"></a><a name="p117502020616"></a>DT_INT8</p>
</td>
<td class="cellrowborder" valign="top" width="71.23%" headers="mcps1.2.3.1.2 "><p id="p13180162712819"><a name="p13180162712819"></a><a name="p13180162712819"></a>df.data_type.DType的对象</p>
<p id="p2180927584"><a name="p2180927584"></a><a name="p2180927584"></a>有符号8位整数</p>
</td>
</tr>
<tr id="row1849317323612"><td class="cellrowborder" valign="top" width="28.77%" headers="mcps1.2.3.1.1 "><p id="p44932032162"><a name="p44932032162"></a><a name="p44932032162"></a>DT_INT16</p>
</td>
<td class="cellrowborder" valign="top" width="71.23%" headers="mcps1.2.3.1.2 "><p id="p1191617376810"><a name="p1191617376810"></a><a name="p1191617376810"></a>df.data_type.DType的对象</p>
<p id="p15916437785"><a name="p15916437785"></a><a name="p15916437785"></a>有符号16位整数</p>
</td>
</tr>
<tr id="row1378514271169"><td class="cellrowborder" valign="top" width="28.77%" headers="mcps1.2.3.1.1 "><p id="p27851127866"><a name="p27851127866"></a><a name="p27851127866"></a>DT_UINT16</p>
</td>
<td class="cellrowborder" valign="top" width="71.23%" headers="mcps1.2.3.1.2 "><p id="p194454489816"><a name="p194454489816"></a><a name="p194454489816"></a>df.data_type.DType的对象</p>
<p id="p2044513489814"><a name="p2044513489814"></a><a name="p2044513489814"></a>无符号16位整数</p>
</td>
</tr>
<tr id="row2160182620616"><td class="cellrowborder" valign="top" width="28.77%" headers="mcps1.2.3.1.1 "><p id="p171608261567"><a name="p171608261567"></a><a name="p171608261567"></a>DT_UINT8</p>
</td>
<td class="cellrowborder" valign="top" width="71.23%" headers="mcps1.2.3.1.2 "><p id="p399158281"><a name="p399158281"></a><a name="p399158281"></a>df.data_type.DType的对象</p>
<p id="p159911588816"><a name="p159911588816"></a><a name="p159911588816"></a>无符号8位整数</p>
</td>
</tr>
<tr id="row353010241861"><td class="cellrowborder" valign="top" width="28.77%" headers="mcps1.2.3.1.1 "><p id="p9530142413610"><a name="p9530142413610"></a><a name="p9530142413610"></a>DT_INT32</p>
</td>
<td class="cellrowborder" valign="top" width="71.23%" headers="mcps1.2.3.1.2 "><p id="p853312101199"><a name="p853312101199"></a><a name="p853312101199"></a>df.data_type.DType的对象</p>
<p id="p253311102919"><a name="p253311102919"></a><a name="p253311102919"></a>有符号32位整数</p>
</td>
</tr>
<tr id="row1668322168"><td class="cellrowborder" valign="top" width="28.77%" headers="mcps1.2.3.1.1 "><p id="p1266862215611"><a name="p1266862215611"></a><a name="p1266862215611"></a>DT_INT64</p>
</td>
<td class="cellrowborder" valign="top" width="71.23%" headers="mcps1.2.3.1.2 "><p id="p266141916916"><a name="p266141916916"></a><a name="p266141916916"></a>df.data_type.DType的对象</p>
<p id="p566110191912"><a name="p566110191912"></a><a name="p566110191912"></a>有符号64位整数</p>
</td>
</tr>
<tr id="row450841418612"><td class="cellrowborder" valign="top" width="28.77%" headers="mcps1.2.3.1.1 "><p id="p10508314061"><a name="p10508314061"></a><a name="p10508314061"></a>DT_UINT32</p>
</td>
<td class="cellrowborder" valign="top" width="71.23%" headers="mcps1.2.3.1.2 "><p id="p1649118311599"><a name="p1649118311599"></a><a name="p1649118311599"></a>df.data_type.DType的对象</p>
<p id="p74911311996"><a name="p74911311996"></a><a name="p74911311996"></a>无符号32位整数</p>
</td>
</tr>
<tr id="row7913315160"><td class="cellrowborder" valign="top" width="28.77%" headers="mcps1.2.3.1.1 "><p id="p89131154619"><a name="p89131154619"></a><a name="p89131154619"></a>DT_UINT64</p>
</td>
<td class="cellrowborder" valign="top" width="71.23%" headers="mcps1.2.3.1.2 "><p id="p0609164013914"><a name="p0609164013914"></a><a name="p0609164013914"></a>df.data_type.DType的对象</p>
<p id="p8609174017914"><a name="p8609174017914"></a><a name="p8609174017914"></a>无符号64位整数</p>
</td>
</tr>
<tr id="row15297518862"><td class="cellrowborder" valign="top" width="28.77%" headers="mcps1.2.3.1.1 "><p id="p92971718667"><a name="p92971718667"></a><a name="p92971718667"></a>DT_BOOL</p>
</td>
<td class="cellrowborder" valign="top" width="71.23%" headers="mcps1.2.3.1.2 "><p id="p957019491294"><a name="p957019491294"></a><a name="p957019491294"></a>df.data_type.DType的对象</p>
<p id="p8570134911919"><a name="p8570134911919"></a><a name="p8570134911919"></a>布尔类型</p>
</td>
</tr>
<tr id="row155451512420"><td class="cellrowborder" valign="top" width="28.77%" headers="mcps1.2.3.1.1 "><p id="p125466513417"><a name="p125466513417"></a><a name="p125466513417"></a>DT_DOUBLE</p>
</td>
<td class="cellrowborder" valign="top" width="71.23%" headers="mcps1.2.3.1.2 "><p id="p1864315581393"><a name="p1864315581393"></a><a name="p1864315581393"></a>df.data_type.DType的对象</p>
<p id="p96431258690"><a name="p96431258690"></a><a name="p96431258690"></a>64位双精度浮点数</p>
</td>
</tr>
<tr id="row134011461082"><td class="cellrowborder" valign="top" width="28.77%" headers="mcps1.2.3.1.1 "><p id="p1480871016812"><a name="p1480871016812"></a><a name="p1480871016812"></a>DT_STRING</p>
</td>
<td class="cellrowborder" valign="top" width="71.23%" headers="mcps1.2.3.1.2 "><p id="p134778237813"><a name="p134778237813"></a><a name="p134778237813"></a>df.data_type.DType的对象</p>
<p id="p9304122810811"><a name="p9304122810811"></a><a name="p9304122810811"></a>字符串类型</p>
</td>
</tr>
</tbody>
</table>

