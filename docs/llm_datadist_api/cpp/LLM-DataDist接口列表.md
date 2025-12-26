# LLM-DataDist接口列表<a name="ZH-CN_TOPIC_0000002373943566"></a>

LLM-DataDist：大模型分布式集群和数据加速组件，提供了集群KV数据管理能力，支持全量图和增量图分离部署。

支持的产品形态如下：

-   Atlas A2 推理系列产品
-   Atlas A3 训练系列产品/Atlas A3 推理系列产品

相关接口存放在："$\{INSTALL\_DIR\}/include/llm\_datadist/llm\_datadist.h"。$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。若安装的Ascend-cann-toolkit软件包，以root安装举例，则安装后文件存储路径为：/usr/local/Ascend/ascend-toolkit/latest。

接口对应的库文件是：libllm\_engine.so。

LLM-DataDist接口列表如下。

**表 1**  LLM-DataDist接口\_V1

<a name="table52841713164813"></a>
<table><thead align="left"><tr id="row1728513134484"><th class="cellrowborder" valign="top" width="37.56%" id="mcps1.2.3.1.1"><p id="p228561310481"><a name="p228561310481"></a><a name="p228561310481"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="62.44%" id="mcps1.2.3.1.2"><p id="p1428514137484"><a name="p1428514137484"></a><a name="p1428514137484"></a>简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row142855132487"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p16149739131210"><a name="p16149739131210"></a><a name="p16149739131210"></a><a href="LlmDataDist构造函数.md">LlmDataDist构造函数</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"><a name="zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a><a name="zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a>构造LLM-DataDist。</p>
</td>
</tr>
<tr id="row1893115411539"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p17931241115316"><a name="p17931241115316"></a><a name="p17931241115316"></a><a href="LlmDataDist().md">~LlmDataDist()</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p189315416536"><a name="p189315416536"></a><a name="p189315416536"></a>LLM-DataDist对象析构函数。</p>
</td>
</tr>
<tr id="row15244124619485"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p16148123971210"><a name="p16148123971210"></a><a name="p16148123971210"></a><a href="Initialize.md">Initialize</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p14609010148"><a name="p14609010148"></a><a name="p14609010148"></a>初始化LLM-DataDist。</p>
</td>
</tr>
<tr id="row552416440482"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p8708193711219"><a name="p8708193711219"></a><a name="p8708193711219"></a><a href="Finalize.md">Finalize</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p638310518145"><a name="p638310518145"></a><a name="p638310518145"></a>释放LLM-DataDist。</p>
</td>
</tr>
<tr id="row189281911134910"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p870510375121"><a name="p870510375121"></a><a name="p870510375121"></a><a href="SetRole.md">SetRole</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001893731858_zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"><a name="zh-cn_topic_0000001893731858_zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a><a name="zh-cn_topic_0000001893731858_zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a>设置当前LLM-DataDist的角色。</p>
</td>
</tr>
<tr id="row175770137493"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p87031837151211"><a name="p87031837151211"></a><a name="p87031837151211"></a><a href="LinkLlmClusters.md">LinkLlmClusters</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001893731798_zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"><a name="zh-cn_topic_0000001893731798_zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a><a name="zh-cn_topic_0000001893731798_zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a>建链。</p>
</td>
</tr>
<tr id="row4210134064817"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p1970215378129"><a name="p1970215378129"></a><a name="p1970215378129"></a><a href="UnlinkLlmClusters.md">UnlinkLlmClusters</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001935851421_zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"><a name="zh-cn_topic_0000001935851421_zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a><a name="zh-cn_topic_0000001935851421_zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_p45666040"></a>断链。</p>
</td>
</tr>
<tr id="row01701344112715"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p18170844112715"><a name="p18170844112715"></a><a name="p18170844112715"></a><a href="PullKvCache.md">PullKvCache</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p392581814428"><a name="p392581814428"></a><a name="p392581814428"></a>以连续内存方式拉取KV Cache。</p>
</td>
</tr>
<tr id="row92684814274"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p32614818278"><a name="p32614818278"></a><a name="p32614818278"></a><a href="PullKvBlocks.md">PullKvBlocks</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p592481874214"><a name="p592481874214"></a><a name="p592481874214"></a>以block列表的方式拉取KV Cache。</p>
</td>
</tr>
<tr id="row584335052715"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p1184316503278"><a name="p1184316503278"></a><a name="p1184316503278"></a><a href="CopyKvCache.md">CopyKvCache</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p837095111437"><a name="p837095111437"></a><a name="p837095111437"></a>以连续内存方式拷贝KV Cache。</p>
</td>
</tr>
<tr id="row8247946192718"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p11247114612277"><a name="p11247114612277"></a><a name="p11247114612277"></a><a href="CopyKvBlocks.md">CopyKvBlocks</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p83705512437"><a name="p83705512437"></a><a name="p83705512437"></a>以block列表的方式拷贝KV Cache。</p>
</td>
</tr>
<tr id="row16272105314207"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p02729536206"><a name="p02729536206"></a><a name="p02729536206"></a><a href="PushKvCache.md">PushKvCache</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p13368186151011"><a name="p13368186151011"></a><a name="p13368186151011"></a>推送Cache到远端节点，仅当角色为Prompt时可调用。</p>
</td>
</tr>
<tr id="row11011553203"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p7101155582011"><a name="p7101155582011"></a><a name="p7101155582011"></a><a href="PushKvBlocks.md">PushKvBlocks</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p1930718302408"><a name="p1930718302408"></a><a name="p1930718302408"></a>通过block列表的方式，推送Cache到远端节点，仅当角色为Prompt时可调用。</p>
</td>
</tr>
<tr id="row456143203313"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p75621732133311"><a name="p75621732133311"></a><a name="p75621732133311"></a><a href="AllocateCache.md">AllocateCache</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p1056233263318"><a name="p1056233263318"></a><a name="p1056233263318"></a>分配Cache。</p>
</td>
</tr>
<tr id="row1570718301334"><td class="cellrowborder" valign="top" width="37.56%" headers="mcps1.2.3.1.1 "><p id="p7708203023314"><a name="p7708203023314"></a><a name="p7708203023314"></a><a href="DeallocateCache.md">DeallocateCache</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.44%" headers="mcps1.2.3.1.2 "><p id="p870818304338"><a name="p870818304338"></a><a name="p870818304338"></a>释放Cache。</p>
</td>
</tr>
</tbody>
</table>

