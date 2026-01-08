# LLMRAGOnMedicaKG
self-implement of disease centered Medical graph from zero to full and sever as question answering base. 从无到有搭建一个以疾病为中心的一定规模医药领域知识图谱，并以该知识图谱，结合LLM完成自动问答与分析服务。

**作者**：医疗知识图谱问答系统——何阳  
**日期**：2025-12-20

# 一、项目介绍

目前知识图谱在各个领域全面开花，如教育、医疗、司法、金融等。本项目立足医药领域，以垂直型医药网站为数据来源，以疾病为核心，构建起一个包含7类规模为4.4万的知识实体，11类规模约30万实体关系的知识图谱。
本项目将包括以下两部分的内容：
1) 基于垂直网站数据的医药知识图谱构建    
2) 基于医药知识图谱的自动问答，基于LLM的方式

实际上，我们在之前的项目 (https://github.com/liuhuanyong/QABasedOnMedicalKnowledgeGraph) 中已经开源过基于朴素KG实现方式的问答，其中涉及到知识图谱构建部分，用到的代码、用到的数据，可以从该项目中继承。

# 二、项目运行方式

1、配置要求：要求配置neo4j数据库及相应的python依赖包。neo4j数据库用户名密码记住，并修改相应文件。    
2、知识图谱数据导入：python build_medicalgraph.py，导入的数据较多，估计需要几个小时。     
3、该项目依赖Qwen-1.8B-Chat作为底层LLM模型，可以执行 `python qwen7b_server.py` 搭建服务（默认端口3001）   
4、配置服务地址：在 `chat_with_llm.py` 中修改 `MODEL_URL="http://127.0.0.1:3001/generate"`（默认本机）     
5、开始执行问答：有两种方式
   - **方式1：Web可视化界面（推荐）**：运行 `python qwen7b_server.py` 后，在浏览器访问 `http://127.0.0.1:3001/` 即可使用美观的Web界面进行问答
   - **方式2：命令行界面**：先运行 `python qwen7b_server.py` 启动服务，然后在新终端运行 `python chat_with_llm.py` 开始问答
  
# 三、医疗知识图谱构建
# 3.1 业务驱动的知识图谱构建框架
![image](https://github.com/liuhuanyong/QABasedOnMedicalKnowledgeGraph/blob/master/img/kg_route.png)

# 3.2 脚本目录
prepare_data/datasoider.py：网络资讯采集脚本  
prepare_data/datasoider.py：网络资讯采集脚本  
prepare_data/max_cut.py：基于词典的最大向前/向后切分脚本  
build_medicalgraph.py：知识图谱入库脚本    　　

# 3.3 医药领域知识图谱规模
1.3.1 neo4j图数据库存储规模
![image](https://github.com/liuhuanyong/QABasedOnMedicalKnowledgeGraph/blob/master/img/graph_summary.png)

3.3.2 知识图谱实体类型

| 实体类型 | 中文含义 | 实体数量 |举例 |
| :--- | :---: | :---: | :--- |
| Check | 诊断检查项目 | 3,353| 支气管造影;关节镜检查|
| Department | 医疗科目 | 54 |  整形美容科;烧伤科|
| Disease | 疾病 | 8,807 |  血栓闭塞性脉管炎;胸降主动脉动脉瘤|
| Drug | 药品 | 3,828 |  京万红痔疮膏;布林佐胺滴眼液|
| Food | 食物 | 4,870 |  番茄冲菜牛肉丸汤;竹笋炖羊肉|
| Producer | 在售药品 | 17,201 |  通药制药青霉素V钾片;青阳醋酸地塞米松片|
| Symptom | 疾病症状 | 5,998 |  乳腺组织肥厚;脑实质深部出血|
| Total | 总计 | 44,111 | 约4.4万实体量级|


3.3.3 知识图谱实体关系类型

| 实体关系类型 | 中文含义 | 关系数量 | 举例|
| :--- | :---: | :---: | :--- |
| belongs_to | 属于 | 8,844| <妇科,属于,妇产科>|
| common_drug | 疾病常用药品 | 14,649 | <阳强,常用,甲磺酸酚妥拉明分散片>|
| do_eat |疾病宜吃食物 | 22,238| <胸椎骨折,宜吃,黑鱼>|
| drugs_of |  药品在售药品 | 17,315| <青霉素V钾片,在售,通药制药青霉素V钾片>|
| need_check | 疾病所需检查 | 39,422| <单侧肺气肿,所需检查,支气管造影>|
| no_eat | 疾病忌吃食物 | 22,247| <唇病,忌吃,杏仁>|
| recommand_drug | 疾病推荐药品 | 59,467 | <混合痔,推荐用药,京万红痔疮膏>|
| recommand_eat | 疾病推荐食谱 | 40,221 | <鞘膜积液,推荐食谱,番茄冲菜牛肉丸汤>|
| has_symptom | 疾病症状 | 5,998 |  <早期乳腺癌,疾病症状,乳腺组织肥厚>|
| acompany_with | 疾病并发疾病 | 12,029 | <下肢交通静脉瓣膜关闭不全,并发疾病,血栓闭塞性脉管炎>|
| Total | 总计 | 294,149 | 约30万关系量级|


# 四、基于医疗知识图谱的RAG问答

基本思想（RAG流程）：

**step1: linking entity** - 针对问题进行实体识别，本项目采用基于AC自动机通过加载图谱词表进行匹配获得；

**step2: recall kg facts** - 通过上一步得到的多个实体，检索知识图谱中相关的三元组。系统会优先选择与目标实体直接相关的三元组，并通过优先级排序筛选出最相关的知识（最多50条）；

**step3: generate answer** - 通过召回好的三元组，拼接prompt，使用LLM（Qwen-1.8B-Chat）完成问答生成；

## 系统特点

- **知识增强**：结合知识图谱的结构化知识和大语言模型的生成能力
- **可解释性**：通过三元组检索过程，用户可以看到答案的来源依据
- **显存优化**：针对6GB GPU进行了显存优化，支持长文本输入和生成
- **智能检索**：优先选择直接相关的三元组，避免不相关信息干扰
- **Web可视化界面**：提供美观的Web界面，无需命令行操作，直接在浏览器中使用
- **代码规范**：所有核心代码文件都添加了详细的中文注释，包括文件说明、函数文档、参数说明等，便于学习和二次开发


    def chat(self, query):
        print("step1: linking entity.....")
        entity_dict = self.entity_linking(query)
        depth = 1
        facts = list()
        answer = ""
        default = "抱歉，我在知识库中没有找到对应的实体，无法回答。"
        if not entity_dict:
            print("no entity founded...finished...")
            return default
        print("step2：recall kg facts....")
        for entity_name, types in entity_dict.items():
            for entity_type in types:
                rels = self.link_entity_rel(query, entity_name, entity_type)
                entity_triples = self.recall_facts(rels, entity_type, entity_name, depth)
                facts += entity_triples
        # 限制总三元组数量，避免prompt过长
        max_total_triples = 50
        if len(facts) > max_total_triples:
            facts = facts[:max_total_triples]
        fact_prompt = self.format_prompt(query, facts)
        print("step3：generate answer...")
        answer, _ = model.chat(query=fact_prompt, history=[])
        return answer

# 五、实验结果

系统在医疗知识图谱问答任务上取得了良好的效果。实验表明：

- **实体识别准确**：能够准确识别疾病、症状等医疗实体
- **知识检索有效**：从知识图谱中检索到相关三元组，并通过优先级排序保证相关性
- **答案生成专业**：生成的答案结构清晰、专业准确，包含治疗方式、常用药品、就诊科室、预防措施等信息
- **问题类型多样**：能够处理"怎么办"（治疗类）和"怎么预防"（预防类）等多种问题类型

## 实验案例

**案例1：治疗类问题**
- 用户：我头痛怎么办
- 系统回答：包含治疗方式、常用药品、就诊科室、预防措施等结构化信息

**案例2：预防类问题**  
- 用户：那头痛怎么预防
- 系统回答：包含合理饮食、睡眠充足、戒烟限酒、控制压力、定期体检等5条预防建议

# 总结

1、本文完成了引入LLM-KG的方式进行医疗领域RAG的开源方案；    
2、核心思路在于实体识别、知识图谱检索、LLM生成，结合了知识图谱的结构化知识和大语言模型的生成能力；    
3、系统具有良好的可解释性，用户可以通过调试信息了解答案来源；    
4、提供Web可视化界面和命令行两种使用方式，提升用户体验；    
5、代码结构清晰，注释详细，便于学习和二次开发；    
6、开源的意义是思路指引，而不是一味搬运、索取、坐享其成，大家一同建设好生态；

---

**项目维护者**：何阳  
**最后更新**：2025-12-20





