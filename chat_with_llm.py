# -*- coding: utf-8 -*-
"""
基于知识图谱的RAG问答系统（命令行版本）

功能：
1. 实体识别：从用户问题中识别医疗实体（疾病、症状等）
2. 知识检索：从Neo4j知识图谱中检索相关三元组
3. 答案生成：基于检索到的知识，使用LLM生成答案

工作流程：
Step1: 实体识别与链接（linking entity）
Step2: 知识图谱检索（recall kg facts）
Step3: 答案生成（generate answer）

作者：医疗知识图谱问答系统
日期：2025
"""

import os
from question_classifier import QuestionClassifier
from llm_server import ModelAPI
from build_medicalgraph import MedicalGraph

# ==================== 初始化组件 ====================
# 实体识别器：用于从问题中识别医疗实体
entity_parser = QuestionClassifier()

# 知识图谱连接：连接到Neo4j数据库
kg = MedicalGraph()

# LLM API客户端：默认指向本机Qwen服务
# 如需远程模型，请改成对应IP/域名与端口，例如："http://192.168.1.100:3001/generate"
model = ModelAPI(MODEL_URL="http://127.0.0.1:3001/generate")

class KGRAG():
    """
    知识图谱检索增强生成（KGRAG）类
    
    功能：
    1. 实体识别与链接：识别问题中的医疗实体并确定相关属性字段
    2. 知识图谱检索：从Neo4j中检索与实体相关的知识三元组
    3. Prompt构建：将检索到的三元组和问题组合成prompt
    4. 答案生成：调用LLM生成最终答案
    """
    def __init__(self):
        """
        初始化KGRAG类
        
        设置：
        - cn_dict: 中文字段名映射字典（英文字段名 -> 中文显示名）
        - entity_rel_dict: 各实体类型对应的关系字段列表
        """
        self.cn_dict = {
                "name":"名称",
                "desc":"疾病简介",
                "cause":"疾病病因",
                "prevent":"预防措施",
                "cure_department":"治疗科室",
                "cure_lasttime":"治疗周期",
                "cure_way":"治疗方式",
                "cured_prob":"治愈概率",
                "easy_get":"易感人群",
                "belongs_to":"所属科室",
                "common_drug":"常用药品",
                "do_eat":"宜吃",
                "drugs_of":"生产药品",
                "need_check":"诊断检查",
                "no_eat":"忌吃",
                "recommand_drug":"好评药品",
                "recommand_eat":"推荐食谱",
                "has_symptom":"症状",
                "acompany_with":"并发症",
                "Check":"诊断检查项目",
                "Department":"医疗科目",
                "Disease":"疾病",
                "Drug":"药品",
                "Food":"食物",
                "Producer":"在售药品",
                "Symptom":"疾病症状"
        }
        self.entity_rel_dict = {
            "check":["name", 'need_check'],
            "department":["name", 'belongs_to'],
            "disease":["prevent", "cure_way", "name", "cure_lasttime", "cured_prob", "cause", "cure_department", "desc", "easy_get", 'recommand_eat', 'no_eat', 'do_eat', "common_drug", 'drugs_of', 'recommand_drug', 'need_check', 'has_symptom', 'acompany_with', 'belongs_to'],
            "drug":["name", "common_drug", 'drugs_of', 'recommand_drug'],
            "food":["name"],
            "producer":["name"],
            "symptom":["name", 'has_symptom'],
        }
        return

    def _truncate_val(self, val, max_len=120):
        """
        截断过长文本值
        
        参数:
            val: 需要截断的值（可以是任何类型）
            max_len (int): 最大长度，默认120字符
        
        返回:
            str: 截断后的字符串，如果超过长度则添加"..."
        
        用途:
            防止知识图谱中的长文本（如疾病病因描述）导致prompt过长
        """
        s = str(val)
        return s if len(s) <= max_len else s[:max_len] + "..."

    def entity_linking(self, query):
        """
        实体识别与链接
        
        参数:
            query (str): 用户输入的问题
        
        返回:
            dict: 实体字典，格式为 {实体名: [实体类型列表]}
                 例如：{"头痛": ["disease", "symptom"]}
        """
        return entity_parser.check_medical(query)

    def link_entity_rel(self, query, entity, entity_type):
        """
        获取实体相关的属性字段列表
        
        参数:
            query (str): 用户问题
            entity (str): 实体名称
            entity_type (str): 实体类型（disease, symptom等）
        
        返回:
            set: 该实体类型的所有候选字段集合
        
        注意:
            - 直接返回该实体类型的所有候选字段，避免调用LLM造成延迟
            - 这样可以保证检索到足够的信息，避免遗漏关键知识
        """
        cate = [self.cn_dict.get(i) for i in self.entity_rel_dict.get(entity_type, [])]
        cls_rel = set(cate)
        # 简化日志输出，只显示关键信息
        print(f"[link_entity_rel] entity={entity}, type={entity_type}, used_fields={len(cls_rel)}个字段")
        return cls_rel

    def recall_facts(self, cls_rel, entity_type, entity_name, depth=1):
        """
        从知识图谱中检索相关三元组
        
        参数:
            cls_rel (set): 需要检索的关系字段集合
            entity_type (str): 实体类型（disease, symptom等）
            entity_name (str): 实体名称
            depth (int): 检索深度，默认1（只检索直接关系）
        
        返回:
            list: 知识三元组列表，格式为 ["<实体,关系,值>", ...]
        
        注意:
            - 优先选择直接关于目标实体的三元组（高优先级）
            - 限制三元组数量（最多30条），避免prompt过长
            - 自动截断过长的文本值，防止prompt被大段描述淹没
        """
        entity_dict = {
            "check":"Check",
            "department":"Department",
            "disease":"Disease",
            "drug":"Drug",
            "food":"Food",
            "producer":"Producer",
            "symptom":"Symptom"
        }
        # "MATCH p=(m:Disease)-[r*..2]-(n) where m.name = '耳聋' return p "
        sql = "MATCH p=(m:{entity_type})-[r*..{depth}]-(n) where m.name = '{entity_name}' return p".format(depth=depth, entity_type=entity_dict.get(entity_type), entity_name=entity_name)
        print(sql)
        ress = kg.g.run(sql).data()
        # 分为两类：直接关于目标实体的三元组（高优先级）和其他相关三元组（低优先级）
        direct_triples = []  # 直接关于目标实体的属性
        related_triples = []  # 通过关系连接的其他实体
        
        for res in ress:
            p_data = res["p"]
            nodes = p_data.nodes
            rels = p_data.relationships
            
            # 处理节点属性（直接关于目标实体的）
            for node in nodes:
                node_name = node["name"]
                # 优先选择目标实体本身的属性
                if node_name == entity_name:
                    for k, v in node.items():
                        if v == node_name or not v:
                            continue
                        if self.cn_dict.get(k) not in cls_rel:
                            continue
                        triple = "<" + ','.join([str(node_name), str(self.cn_dict.get(k)), self._truncate_val(v)]) + ">"
                        direct_triples.append(triple)
            
            # 处理关系（只保留与目标实体直接相关的关系）
            for rel in rels:
                if rel.start_node["name"] == rel.end_node["name"]:
                    continue
                if rel["name"] not in cls_rel:
                    continue
                # 优先选择以目标实体为起点的关系
                if rel.start_node["name"] == entity_name or rel.end_node["name"] == entity_name:
                    triple = "<" + ','.join([str(rel.start_node["name"]), str(rel["name"]), str(rel.end_node["name"])]) + ">"
                    if rel.start_node["name"] == entity_name:
                        direct_triples.append(triple)
                    else:
                        related_triples.append(triple)
        
        # 去重并合并：优先使用直接相关的三元组
        direct_triples = list(set(direct_triples))
        related_triples = list(set(related_triples))
        
        # 优先选择直接相关的，然后补充相关的
        all_triples = direct_triples + related_triples
        
        # 为了控制 prompt 长度，避免显存溢出
        max_triples = 30
        if len(all_triples) > max_triples:
            # 优先保留直接相关的，如果还不够再补充相关的
            if len(direct_triples) >= max_triples:
                triples = direct_triples[:max_triples]
            else:
                triples = direct_triples + related_triples[:max_triples - len(direct_triples)]
            print(f"[recall_facts] entity={entity_name}, type={entity_type}, total_triples={len(all_triples)}, direct={len(direct_triples)}, truncated_to={len(triples)}")
        else:
            triples = all_triples
            print(f"[recall_facts] entity={entity_name}, type={entity_type}, total_triples={len(triples)}, direct={len(direct_triples)}")
        
        return triples


    def format_prompt(self, query, context):
        """
        构建用于LLM生成的prompt
        
        参数:
            query (str): 用户问题
            context (list): 检索到的知识三元组列表
        
        返回:
            str: 格式化后的prompt字符串
        
        注意:
            - 将三元组格式化为编号列表，便于LLM阅读
            - 提供明确的使用指南，告诉LLM如何根据问题类型选择相关信息
            - 要求LLM不要回答"不知道"，即使信息有限也要给出建议
        """
        # 将三元组列表格式化为更易读的字符串
        if not context:
            context_str = "无相关三元组"
        else:
            context_str = "\n".join([f"  {i+1}. {triple}" for i, triple in enumerate(context[:50])])
        
        prompt = f"""你是一个医疗知识问答助手。请根据以下知识三元组回答问题。

知识三元组（格式：<实体, 关系, 值>）：
{context_str}

用户问题：{query}

请仔细阅读上述知识三元组，找出与问题直接相关的信息，并用简洁、专业的中文回答问题。回答时要：
1. 优先使用与问题中提到的实体直接相关的三元组
2. 如果问题问"怎么办"或"如何治疗"，重点关注"治疗方式"、"常用药品"、"治疗科室"等关系
3. 如果问题问"原因"或"病因"，重点关注"疾病病因"关系
4. 如果问题问"症状"，重点关注"症状"关系
5. 不要回答不知道或抱歉；即使信息有限，也请结合三元组给出保守的建议（如常见治疗方式、就诊科室、常见药物、预防措施）

回答："""
        return prompt

    def chat(self, query):
        "{'耳聋': ['disease', 'symptom']}"
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
        # 进一步按关系优先级过滤，减少离题内容
        priority_rels = {"治疗方式", "常用药品", "好评药品", "治疗科室", "诊断检查", "治疗周期", "治愈概率", "预防措施", "疾病病因", "症状", "易感人群", "推荐食谱", "宜吃", "忌吃"}
        priority_facts = [t for t in facts if any(rel in t for rel in priority_rels)]
        if priority_facts:
            facts = priority_facts
        # 限制总三元组数量，避免 prompt 过长导致显存溢出
        max_total_triples = 50
        if len(facts) > max_total_triples:
            print(f"[WARNING] 总三元组数 {len(facts)} 超过限制，截断到 {max_total_triples} 条")
            facts = facts[:max_total_triples]
        fact_prompt = self.format_prompt(query, facts)
        # 调试：打印前200个字符的 prompt，方便排查
        print(f"step3：generate answer... (prompt长度: {len(fact_prompt)}, 三元组数: {len(facts)})")
        if len(facts) > 0:
            print(f"[DEBUG] 前3个三元组示例: {facts[:3]}")
        answer, _ = model.chat(query=fact_prompt, history=[])
        return answer

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    """
    命令行交互式问答程序
    
    使用方法：
    1. 运行此脚本：python chat_with_llm.py
    2. 输入问题，按回车提交
    3. 系统会自动识别实体、检索知识图谱、生成答案
    4. 输入"退出"或按Ctrl+C退出程序
    
    注意：
    - 需要先启动qwen7b_server.py服务
    - 需要确保Neo4j数据库已启动并导入数据
    """
    print("[INFO] 初始化RAG问答系统...")
    chatbot = KGRAG()
    print("[INFO] 系统初始化完成，可以开始提问了！")
    print("[INFO] 输入'退出'或按Ctrl+C退出程序\n")
    
    while True:
        try:
            query = input("USER:").strip()
            if not query:
                continue
            if query.lower() in ['退出', 'exit', 'quit']:
                print("[INFO] 再见！")
                break
            answer = chatbot.chat(query)
            print("KGRAG_BOT:", answer)
            print()  # 空行分隔
        except KeyboardInterrupt:
            print("\n[INFO] 程序已退出")
            break
        except Exception as e:
            print(f"[ERROR] 发生错误: {e}")
            import traceback
            traceback.print_exc()
