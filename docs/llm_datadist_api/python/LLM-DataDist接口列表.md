# LLM-DataDist接口列表<a name="ZH-CN_TOPIC_0000002408010605"></a>

LLM-DataDist：大模型分布式集群和数据加速组件，提供了集群KV数据管理能力，以支持全量图和增量图分离部署。

-   支持的产品形态如下：
    -   Atlas A2 推理系列产品
    -   Atlas A3 训练系列产品/Atlas A3 推理系列产品

-   当前仅支持Python3.9与Python3.11。安装方法请参考Python官网[https://www.python.org/](https://www.python.org/)。
-   最大注册50GB的Device内存。注册内存越大，占用的OS内存越多。

LLM-DataDist接口列表如下。

## LLM-DataDist<a name="section1983152244318"></a>

**表 1**  LLM-DataDist接口

<a name="table52841713164813"></a>
<table><thead align="left"><tr id="row1728513134484"><th class="cellrowborder" valign="top" width="37.519999999999996%" id="mcps1.2.3.1.1"><p id="p228561310481"><a name="p228561310481"></a><a name="p228561310481"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="62.480000000000004%" id="mcps1.2.3.1.2"><p id="p1428514137484"><a name="p1428514137484"></a><a name="p1428514137484"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row142855132487"><td class="cellrowborder" valign="top" width="37.519999999999996%" headers="mcps1.2.3.1.1 "><p id="p16149739131210"><a name="p16149739131210"></a><a name="p16149739131210"></a><a href="LLMDataDist构造函数.md">LLMDataDist构造函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.480000000000004%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"><a name="zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a><a name="zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a>构造LLMDataDist。</p>
</td>
</tr>
<tr id="row15244124619485"><td class="cellrowborder" valign="top" width="37.519999999999996%" headers="mcps1.2.3.1.1 "><p id="p16148123971210"><a name="p16148123971210"></a><a name="p16148123971210"></a><a href="init.md">init</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.480000000000004%" headers="mcps1.2.3.1.2 "><p id="p14609010148"><a name="p14609010148"></a><a name="p14609010148"></a>初始化LLMDataDist。</p>
</td>
</tr>
<tr id="row552416440482"><td class="cellrowborder" valign="top" width="37.519999999999996%" headers="mcps1.2.3.1.1 "><p id="p8708193711219"><a name="p8708193711219"></a><a name="p8708193711219"></a><a href="finalize.md">finalize</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.480000000000004%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="p638310518145"><a name="p638310518145"></a><a name="p638310518145"></a>释放LLMDataDist。</p>
</td>
</tr>
<tr id="row189281911134910"><td class="cellrowborder" valign="top" width="37.519999999999996%" headers="mcps1.2.3.1.1 "><p id="p470612376125"><a name="p470612376125"></a><a name="p470612376125"></a><a href="link_clusters.md">link_clusters</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.480000000000004%" headers="mcps1.2.3.1.2 "><p id="p72641725163214"><a name="p72641725163214"></a><a name="p72641725163214"></a>建链。</p>
</td>
</tr>
<tr id="row175770137493"><td class="cellrowborder" valign="top" width="37.519999999999996%" headers="mcps1.2.3.1.1 "><p id="p870510375121"><a name="p870510375121"></a><a name="p870510375121"></a><a href="unlink_clusters.md">unlink_clusters</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.480000000000004%" headers="mcps1.2.3.1.2 "><p id="p1444103433213"><a name="p1444103433213"></a><a name="p1444103433213"></a>断链。</p>
</td>
</tr>
<tr id="row4210134064817"><td class="cellrowborder" valign="top" width="37.519999999999996%" headers="mcps1.2.3.1.1 "><p id="p188221714627"><a name="p188221714627"></a><a name="p188221714627"></a><a href="check_link_status.md">check_link_status</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.480000000000004%" headers="mcps1.2.3.1.2 "><p id="p113115013324"><a name="p113115013324"></a><a name="p113115013324"></a>调用此接口可快速检测链路状态是否正常。</p>
</td>
</tr>
<tr id="row149891915492"><td class="cellrowborder" valign="top" width="37.519999999999996%" headers="mcps1.2.3.1.1 "><p id="p1970215378129"><a name="p1970215378129"></a><a name="p1970215378129"></a><a href="kv_cache_manager.md">kv_cache_manager</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.480000000000004%" headers="mcps1.2.3.1.2 "><p id="p125316415338"><a name="p125316415338"></a><a name="p125316415338"></a>获取KvCacheManager实例。</p>
</td>
</tr>
<tr id="row1168518252918"><td class="cellrowborder" valign="top" width="37.519999999999996%" headers="mcps1.2.3.1.1 "><p id="p18686825192"><a name="p18686825192"></a><a name="p18686825192"></a><a href="switch_role.md">switch_role</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.480000000000004%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="p56863255916"><a name="p56863255916"></a><a name="p56863255916"></a>切换当前LLMDataDist的角色，建议仅在使用PagedAttention的场景使用。</p>
</td>
</tr>
</tbody>
</table>

## LLMConfig<a name="section15510113219494"></a>

**表 2**  LLMConfig接口

<a name="table3510133218492"></a>
<table><thead align="left"><tr id="row7511183218498"><th class="cellrowborder" valign="top" width="37.56%" id="mcps1.2.3.1.1"><p id="p11511123214497"><a name="p11511123214497"></a><a name="p11511123214497"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="62.44%" id="mcps1.2.3.1.2"><p id="p55111032104911"><a name="p55111032104911"></a><a name="p55111032104911"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row12511113213492"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p131119437493"><a name="p131119437493"></a><a name="p131119437493"></a><a href="LLMConfig构造函数.md">LLMConfig构造函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p6117437495"><a name="p6117437495"></a><a name="p6117437495"></a>构造LLMConfig。</p>
</td>
</tr>
<tr id="row165111332194912"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p7101443124910"><a name="p7101443124910"></a><a name="p7101443124910"></a><a href="generate_options.md">generate_options</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p8398131155119"><a name="p8398131155119"></a><a name="p8398131155119"></a>生成配置项字典。</p>
</td>
</tr>
<tr id="row1371712198222"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p179134318494"><a name="p179134318494"></a><a name="p179134318494"></a><a href="device_id.md">device_id</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p1456101135113"><a name="p1456101135113"></a><a name="p1456101135113"></a>设置当前进程Device ID，对应底层ge.exec.deviceId配置项。</p>
</td>
</tr>
<tr id="row5808121192220"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p781643154914"><a name="p781643154914"></a><a name="p781643154914"></a><a href="sync_kv_timeout.md">sync_kv_timeout</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p131692172510"><a name="p131692172510"></a><a name="p131692172510"></a>配置拉取kv等接口超时时间，对应底层llm.SyncKvCacheWaitTime配置项。</p>
</td>
</tr>
<tr id="row15511132154916"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p1019442962214"><a name="p1019442962214"></a><a name="p1019442962214"></a><a href="enable_switch_role.md">enable_switch_role</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p185751117239"><a name="p185751117239"></a><a name="p185751117239"></a>配置是否支持角色平滑切换，对应底层llm.EnableSwitchRole配置项。</p>
</td>
</tr>
<tr id="row7126161417224"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p17711436494"><a name="p17711436494"></a><a name="p17711436494"></a><a href="ge_options.md">ge_options</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p11889721165117"><a name="p11889721165117"></a><a name="p11889721165117"></a>配置额外的GE配置项。</p>
</td>
</tr>
<tr id="row1951118326492"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p141034334910"><a name="p141034334910"></a><a name="p141034334910"></a><a href="listen_ip_info.md">listen_ip_info</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p162501367510"><a name="p162501367510"></a><a name="p162501367510"></a>PROMPT侧设置集群侦听信息，对应底层llm.listenIpInfo配置项。</p>
</td>
</tr>
<tr id="row1046742101210"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p124614220122"><a name="p124614220122"></a><a name="p124614220122"></a><a href="mem_utilization.md">mem_utilization</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p10461422125"><a name="p10461422125"></a><a name="p10461422125"></a>配置ge.flowGraphMemMaxSize内存的利用率。默认值0.95。</p>
</td>
</tr>
<tr id="row2979171019232"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p2979181032316"><a name="p2979181032316"></a><a name="p2979181032316"></a><a href="buf_pool_cfg.md">buf_pool_cfg</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p1068364719231"><a name="p1068364719231"></a><a name="p1068364719231"></a>用户指定内存档位配置，提高内存申请性能和使用率。</p>
</td>
</tr>
</tbody>
</table>

## KvCacheManager<a name="section5109124383518"></a>

**表 3**  KvCacheManager接口

<a name="table8109943183513"></a>
<table><thead align="left"><tr id="row711084333518"><th class="cellrowborder" valign="top" width="37.56%" id="mcps1.2.3.1.1"><p id="p15110144343510"><a name="p15110144343510"></a><a name="p15110144343510"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="62.44%" id="mcps1.2.3.1.2"><p id="p20110184319356"><a name="p20110184319356"></a><a name="p20110184319356"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row393682525419"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p1593620256543"><a name="p1593620256543"></a><a name="p1593620256543"></a><a href="KvCacheManager构造函数.md">KvCacheManager构造函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p1493617253547"><a name="p1493617253547"></a><a name="p1493617253547"></a>介绍KvCacheManager构造函数。</p>
</td>
</tr>
<tr id="row13110204319353"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p13531557143515"><a name="p13531557143515"></a><a name="p13531557143515"></a><a href="is_initialized.md">is_initialized</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p1613773412560"><a name="p1613773412560"></a><a name="p1613773412560"></a>查询KvCacheManager实例是否已初始化。</p>
</td>
</tr>
<tr id="row1011014316355"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p452135713351"><a name="p452135713351"></a><a name="p452135713351"></a><a href="allocate_cache.md">allocate_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p384834285619"><a name="p384834285619"></a><a name="p384834285619"></a>分配Cache，Cache分配成功后，会同时被cache_id与cache_keys引用，只有当这些引用都解除后，cache所占用的资源才会实际释放。</p>
</td>
</tr>
<tr id="row1611010438357"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p1351657133510"><a name="p1351657133510"></a><a name="p1351657133510"></a><a href="deallocate_cache.md">deallocate_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p8497217813"><a name="p8497217813"></a><a name="p8497217813"></a>释放Cache。</p>
</td>
</tr>
<tr id="row11101343123510"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p05011575352"><a name="p05011575352"></a><a name="p05011575352"></a><a href="remove_cache_key.md">remove_cache_key</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p19393845315"><a name="p19393845315"></a><a name="p19393845315"></a>移除CacheKey，仅当LLMRole为PROMPT时可调用。</p>
</td>
</tr>
<tr id="row15110343113519"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p44935720353"><a name="p44935720353"></a><a name="p44935720353"></a><a href="pull_cache.md">pull_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p13937451412"><a name="p13937451412"></a><a name="p13937451412"></a>根据CacheKey，从对应的Prompt节点拉取KV到本地KV Cache，仅当LLMRole为DECODER时可调用。</p>
</td>
</tr>
<tr id="row911144323519"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p16484577353"><a name="p16484577353"></a><a name="p16484577353"></a><a href="copy_cache.md">copy_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p117201917413"><a name="p117201917413"></a><a name="p117201917413"></a>拷贝KV。</p>
</td>
</tr>
<tr id="row1144310113611"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p154352003619"><a name="p154352003619"></a><a name="p154352003619"></a><a href="get_cache_tensors.md">get_cache_tensors</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p17546127194118"><a name="p17546127194118"></a><a name="p17546127194118"></a>获取cache tensor。</p>
</td>
</tr>
<tr id="row9443160113615"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p343511063612"><a name="p343511063612"></a><a name="p343511063612"></a><a href="allocate_blocks_cache.md">allocate_blocks_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p1152415153419"><a name="p1152415153419"></a><a name="p1152415153419"></a>PagedAttention场景下，分配多个blocks的Cache。</p>
</td>
</tr>
<tr id="row0443806363"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p14436404362"><a name="p14436404362"></a><a name="p14436404362"></a><a href="pull_blocks.md">pull_blocks</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p119061837174119"><a name="p119061837174119"></a><a name="p119061837174119"></a>PagedAttention场景下，根据BlocksCacheKey，通过block列表的方式从对应的Prompt节点拉取KV到本地KV Cache，仅当LLMRole为DECODER时可调用。</p>
</td>
</tr>
<tr id="row1544318017362"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p743615063612"><a name="p743615063612"></a><a name="p743615063612"></a><a href="copy_blocks.md">copy_blocks</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p1817014412418"><a name="p1817014412418"></a><a name="p1817014412418"></a>PagedAttention场景下，拷贝KV。</p>
</td>
</tr>
<tr id="row166915311459"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p369163124517"><a name="p369163124517"></a><a name="p369163124517"></a><a href="swap_blocks.md">swap_blocks</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p9468154720564"><a name="p9468154720564"></a><a name="p9468154720564"></a>对cpu_cache和npu_cache进行换入换出。</p>
</td>
</tr>
<tr id="row891212184513"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p1491218118458"><a name="p1491218118458"></a><a name="p1491218118458"></a><a href="transfer_cache_async.md">transfer_cache_async</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p1681105425618"><a name="p1681105425618"></a><a name="p1681105425618"></a>异步分层传输KV Cache。</p>
</td>
</tr>
</tbody>
</table>

## KvCache<a name="section432835451913"></a>

**表 4**  KVCache接口

<a name="table103285547191"></a>
<table><thead align="left"><tr id="row143286541190"><th class="cellrowborder" valign="top" width="37.56%" id="mcps1.2.3.1.1"><p id="p19328205461916"><a name="p19328205461916"></a><a name="p19328205461916"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="62.44%" id="mcps1.2.3.1.2"><p id="p43281754121910"><a name="p43281754121910"></a><a name="p43281754121910"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row13328554141914"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p198201623152016"><a name="p198201623152016"></a><a name="p198201623152016"></a><a href="KvCache构造函数.md">KvCache构造函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p6547121182110"><a name="p6547121182110"></a><a name="p6547121182110"></a>构造KVCache。</p>
</td>
</tr>
<tr id="row93293542198"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p58191323142015"><a name="p58191323142015"></a><a name="p58191323142015"></a><a href="cache_id.md">cache_id</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p16784297591"><a name="p16784297591"></a><a name="p16784297591"></a>获取KvCache的id。</p>
</td>
</tr>
<tr id="row1235694633510"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p10356846193512"><a name="p10356846193512"></a><a name="p10356846193512"></a><a href="cache_desc.md">cache_desc</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p19282257183617"><a name="p19282257183617"></a><a name="p19282257183617"></a>获取KvCache描述。</p>
</td>
</tr>
<tr id="row05035263514"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p750952113517"><a name="p750952113517"></a><a name="p750952113517"></a><a href="per_device_tensor_addrs.md">per_device_tensor_addrs</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p63464718374"><a name="p63464718374"></a><a name="p63464718374"></a>获取KvCache的地址。</p>
</td>
</tr>
<tr id="row8522550113519"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p17523850193515"><a name="p17523850193515"></a><a name="p17523850193515"></a><a href="create_cpu_cache.md">create_cpu_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p7345201411379"><a name="p7345201411379"></a><a name="p7345201411379"></a>创建cpu cache。</p>
</td>
</tr>
</tbody>
</table>

## LLMClusterInfo<a name="section112159586521"></a>

**表 5**  LLMClusterInfo接口

<a name="table132152058155219"></a>
<table><thead align="left"><tr id="row1021525815520"><th class="cellrowborder" valign="top" width="37.56%" id="mcps1.2.3.1.1"><p id="p122151558155216"><a name="p122151558155216"></a><a name="p122151558155216"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="62.44%" id="mcps1.2.3.1.2"><p id="p172151658145219"><a name="p172151658145219"></a><a name="p172151658145219"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row1721525865218"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p7275121210531"><a name="p7275121210531"></a><a name="p7275121210531"></a><a href="LLMClusterInfo构造函数.md">LLMClusterInfo构造函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p295595545314"><a name="p295595545314"></a><a name="p295595545314"></a>构造LLMClusterInfo。</p>
</td>
</tr>
<tr id="row14216135815218"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p11274012105318"><a name="p11274012105318"></a><a name="p11274012105318"></a><a href="remote_cluster_id.md">remote_cluster_id</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p183511110542"><a name="p183511110542"></a><a name="p183511110542"></a>设置对端集群ID。</p>
</td>
</tr>
<tr id="row102166588529"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p4273161218530"><a name="p4273161218530"></a><a name="p4273161218530"></a><a href="append_local_ip_info.md">append_local_ip_info</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p91221778548"><a name="p91221778548"></a><a name="p91221778548"></a>添加本地集群IP信息。</p>
</td>
</tr>
<tr id="row32164588527"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p192727126533"><a name="p192727126533"></a><a name="p192727126533"></a><a href="append_remote_ip_info.md">append_remote_ip_info</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p1392251175420"><a name="p1392251175420"></a><a name="p1392251175420"></a>添加远端集群IP信息。</p>
</td>
</tr>
</tbody>
</table>

## CacheTask<a name="section417392252418"></a>

**表 6**  CacheTask

<a name="table1602121318331"></a>
<table><thead align="left"><tr id="row2602141310336"><th class="cellrowborder" valign="top" width="37.56%" id="mcps1.2.3.1.1"><p id="p146029134339"><a name="p146029134339"></a><a name="p146029134339"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="62.44%" id="mcps1.2.3.1.2"><p id="p160211319337"><a name="p160211319337"></a><a name="p160211319337"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row1604191318338"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p1560481316335"><a name="p1560481316335"></a><a name="p1560481316335"></a><a href="CacheTask构造函数.md">CacheTask构造函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p16043134331"><a name="p16043134331"></a><a name="p16043134331"></a>构造CacheTask。</p>
</td>
</tr>
<tr id="row26042137339"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p1604171393310"><a name="p1604171393310"></a><a name="p1604171393310"></a><a href="synchronize.md">synchronize</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p56042133330"><a name="p56042133330"></a><a name="p56042133330"></a>等待所有层传输完成，并获取整体执行结果。</p>
</td>
</tr>
<tr id="row196042135334"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p17604171314337"><a name="p17604171314337"></a><a name="p17604171314337"></a><a href="get_results.md">get_results</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p4604111319338"><a name="p4604111319338"></a><a name="p4604111319338"></a>等待所有层传输完成，并获取每个TransferConfig对应执行结果。</p>
</td>
</tr>
</tbody>
</table>

## 其他<a name="section75364163542"></a>

**表 7**  其他

<a name="table0536101618549"></a>
<table><thead align="left"><tr id="row155361716195412"><th class="cellrowborder" valign="top" width="37.56%" id="mcps1.2.3.1.1"><p id="p13536141675418"><a name="p13536141675418"></a><a name="p13536141675418"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="62.44%" id="mcps1.2.3.1.2"><p id="p16537201616541"><a name="p16537201616541"></a><a name="p16537201616541"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row6155326133814"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p14123162743820"><a name="p14123162743820"></a><a name="p14123162743820"></a><a href="LLMRole.md">LLMRole</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p91234274380"><a name="p91234274380"></a><a name="p91234274380"></a>LLMRole的枚举值。</p>
</td>
</tr>
<tr id="row114683443817"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p5391131082513"><a name="p5391131082513"></a><a name="p5391131082513"></a><a href="Placement.md">Placement</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p1049717237256"><a name="p1049717237256"></a><a name="p1049717237256"></a>CacheDesc的字段，表示cache所在的设备类型。</p>
</td>
</tr>
<tr id="row4537216135414"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p1697012416548"><a name="p1697012416548"></a><a name="p1697012416548"></a><a href="CacheDesc.md">CacheDesc</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p56031913185513"><a name="p56031913185513"></a><a name="p56031913185513"></a>构造CacheDesc。</p>
</td>
</tr>
<tr id="row953712164547"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p189693246542"><a name="p189693246542"></a><a name="p189693246542"></a><a href="CacheKey.md">CacheKey</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p3351142095517"><a name="p3351142095517"></a><a name="p3351142095517"></a>构造CacheKey。</p>
</td>
</tr>
<tr id="row199547341279"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p19955133417276"><a name="p19955133417276"></a><a name="p19955133417276"></a><a href="CacheKeyByIdAndIndex.md">CacheKeyByIdAndIndex</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="p11989636101712"><a name="p11989636101712"></a><a name="p11989636101712"></a>构造CacheKeyByIdAndIndex，通常在<a href="pull_cache.md">pull_cache</a>接口中作为参数类型使用。</p>
</td>
</tr>
<tr id="row16537171695410"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p15968182465415"><a name="p15968182465415"></a><a name="p15968182465415"></a><a href="BlocksCacheKey.md">BlocksCacheKey</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p628413251557"><a name="p628413251557"></a><a name="p628413251557"></a>PagedAttention场景下，构造BlocksCacheKey。</p>
</td>
</tr>
<tr id="row1128012419269"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p10112101875716"><a name="p10112101875716"></a><a name="p10112101875716"></a><a href="LayerSynchronizer.md">LayerSynchronizer</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p68621658593"><a name="p68621658593"></a><a name="p68621658593"></a>等待模型指定层执行完成，用户需要继承LayerSynchronizer并实现该接口。</p>
<p id="p20862205155919"><a name="p20862205155919"></a><a name="p20862205155919"></a>该接口会在执行KvCacheManager.transfer_cache_async时被调用，当该接口返回成功，则开始当前层cache的传输。</p>
</td>
</tr>
<tr id="row19832854399"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p2373521205718"><a name="p2373521205718"></a><a name="p2373521205718"></a><a href="TransferConfig.md">TransferConfig</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p740441385918"><a name="p740441385918"></a><a name="p740441385918"></a>构造TransferConfig。</p>
</td>
</tr>
<tr id="row117001932114119"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p107001632174113"><a name="p107001632174113"></a><a name="p107001632174113"></a><a href="TransferWithCacheKeyConfig.md">TransferWithCacheKeyConfig</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p17376205519434"><a name="p17376205519434"></a><a name="p17376205519434"></a>构造TransferWithCacheKeyConfig。</p>
</td>
</tr>
<tr id="row479991033917"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p52621214143917"><a name="p52621214143917"></a><a name="p52621214143917"></a><a href="LLMException.md">LLMException</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p1626317140398"><a name="p1626317140398"></a><a name="p1626317140398"></a>获取异常的错误码。错误码列表详见<a href="LLMStatusCode.md">LLMStatusCode</a>。</p>
</td>
</tr>
<tr id="row12637230102613"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p1563713011261"><a name="p1563713011261"></a><a name="p1563713011261"></a><a href="LLMStatusCode.md">LLMStatusCode</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p963743013268"><a name="p963743013268"></a><a name="p963743013268"></a>LLMStatusCode的枚举值。</p>
</td>
</tr>
<tr id="row181251356185720"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p191251656105713"><a name="p191251656105713"></a><a name="p191251656105713"></a><a href="DataType.md">DataType</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p9125205635715"><a name="p9125205635715"></a><a name="p9125205635715"></a>DataType的枚举类。</p>
</td>
</tr>
</tbody>
</table>

