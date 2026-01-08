# -*- coding: utf-8 -*-
"""
Qwen LLM服务端 + Web可视化界面

功能：
1. 提供Flask API服务，封装Qwen-1.8B-Chat模型
2. 提供Web可视化界面，用户可在浏览器中直接使用
3. 可选：集成RAG功能，支持知识图谱检索增强生成

主要接口：
- GET  / : Web可视化界面
- POST /generate : 基础LLM生成接口
- POST /rag : RAG完整问答接口（可选）

作者：医疗知识图谱问答系统——何阳
日期：2025-12-20
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import json
from flask import Flask, request, jsonify

# 导入RAG相关模块（可选，如果不需要RAG功能可以注释掉）
try:
    from question_classifier import QuestionClassifier
    from build_medicalgraph import MedicalGraph
    from llm_server import ModelAPI
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("[WARNING] RAG模块未导入，完整问答功能不可用")


## 注意
# 这里改为使用 HuggingFace 上的小模型 Qwen-1.8B-Chat（自动下载到本地缓存）
MODEL_NAME = "Qwen/Qwen-1_8B-Chat"

# 设备配置：优先使用GPU，没有GPU时自动切换到CPU
use_gpu = torch.cuda.is_available()
device = torch.device("cuda") if use_gpu else torch.device("cpu")

# 数据类型配置：GPU使用fp16（加速并节省显存），CPU使用fp32（保证精度）
dtype = torch.float16 if use_gpu else torch.float32

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ==================== 模型加载 ====================
if use_gpu:
    # GPU模式：直接加载到 cuda:0，使用fp16精度
    print(f"[INFO] 使用GPU模式，设备: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": 0},  # 单卡GPU，映射到设备0
    )
else:
    # CPU模式：开启 low_cpu_mem_usage 以降低峰值内存占用
    print(f"[INFO] 使用CPU模式")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,  # 降低内存峰值
    ).to(device)

# 加载生成配置
model.generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print("[INFO] 模型加载完成")

# ==================== 模型推理函数 ====================
def predict_model(data):
    """
    使用Qwen模型进行文本生成
    
    参数:
        data (dict): 包含以下字段的字典
            - message: [{"content": "用户输入的问题"}]
            - max_tokens: 最大生成长度（可选，默认64，最大256）
    
    返回:
        str: 模型生成的回答文本
    
    注意:
        - 使用 model.chat 接口（官方推荐），自动处理prompt模板
        - 自动处理输入长度限制和显存溢出
        - 支持GPU/CPU自动切换
    """
    text = data["message"][0]["content"]
    
    # 限制输入长度，避免显存溢出（6GB GPU 建议不超过 2000 tokens）
    max_input_tokens = 2000
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_input_tokens:
        # 截断到最大长度，保留前面的内容
        truncated_tokens = tokens[:max_input_tokens]
        text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        print(f"[WARNING] 输入过长，已截断到 {max_input_tokens} tokens")

    # 默认最大生成长度适中，兼顾时延和完整性
    max_new_tokens = data.get("max_tokens", 64)
    # 限制最大生成长度，避免显存溢出
    max_new_tokens = min(max_new_tokens, 256)

    # Qwen chat 接口本身不直接支持 max_new_tokens 形参，
    # 通过临时修改 generation_config 来控制
    old_max_new_tokens = model.generation_config.max_new_tokens
    model.generation_config.max_new_tokens = max_new_tokens
    
    # 清理显存缓存（如果使用 GPU）
    if use_gpu:
        torch.cuda.empty_cache()
    
    try:
        response, _ = model.chat(tokenizer, query=text, history=[])
    except TypeError:
        response, _ = model.chat(tokenizer, text, history=[])
    except torch.cuda.OutOfMemoryError as oom_err:
        # 显存不足时清理缓存并重试一次
        if use_gpu:
            torch.cuda.empty_cache()
            print("[WARNING] CUDA OOM，已清理缓存，尝试缩短输入重试...")
            # 进一步缩短输入
            shorter_tokens = tokens[:1000] if len(tokens) > 1000 else tokens
            text = tokenizer.decode(shorter_tokens, skip_special_tokens=True)
            model.generation_config.max_new_tokens = 32
            try:
                response, _ = model.chat(tokenizer, query=text, history=[])
            except:
                raise Exception("显存不足，即使缩短输入后仍无法处理。请减少输入长度或重启服务释放显存。")
        else:
            raise oom_err
    finally:
        # 还原原始配置，避免影响后续调用
        model.generation_config.max_new_tokens = old_max_new_tokens
        # 再次清理显存
        if use_gpu:
            torch.cuda.empty_cache()

    return response

# ==================== Flask应用初始化 ====================
app = Flask(import_name=__name__)

# ==================== API路由 ====================
@app.route("/generate", methods=["POST", "GET"])
def generate():
    """
    基础LLM生成接口
    
    支持的调用方式：
    1. POST JSON格式：
       {
         "message": [{"content": "用户问题"}],
         "max_tokens": 256  // 可选，默认64，最大256
       }
    
    2. GET 查询参数：
       /generate?q=用户问题
    
    返回格式：
    {
        "output": ["生成的答案"],
        "status": "success" | "error",
        "history": []
    }
    """
    try:
        # 优先尝试解析 JSON 体
        data = request.get_json(silent=True)
        if not data:
            # 兼容 GET 参数，或表单/空 body 的 POST
            q = request.args.get("q", "").strip()
            if not q:
                return jsonify({"output": [""], "status": "error", "history": [], "msg": "缺少输入内容"}), 400
            data = {"message": [{"content": q}], "max_tokens": 64}
    except Exception as parse_err:
        return jsonify({"output": [""], "status": "error", "history": [], "msg": f"请求解析失败: {parse_err}"}), 400

    print("request payload:", data)

    try:
        res = predict_model(data)
        label = "success"
    except Exception as e:
        import traceback
        traceback.print_exc()  # 打印完整堆栈便于排查
        res = ""
        label = "error"
        print(e)
    # 返回 history 字段以兼容客户端解析（即便为空）
    return jsonify({"output":[res], "status":label, "history":[]})

# ==================== Web可视化界面 ====================
@app.route("/", methods=["GET"])
def index():
    """
    Web可视化界面路由
    
    返回一个美观的HTML页面，用户可以在浏览器中直接使用问答功能
    界面特点：
    - 美观的渐变背景和卡片式设计
    - 输入框直接输入问题
    - 提供示例问题，点击即可快速填入
    - 实时显示答案，界面友好
    - 支持Ctrl+Enter快捷键提交
    """
    html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>医疗知识图谱问答系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 900px;
            padding: 40px;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .input-group {
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            min-height: 100px;
            font-family: inherit;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .button-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        button {
            flex: 1;
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        button:active {
            transform: translateY(0);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .answer-box {
            background: #f5f5f5;
            border-radius: 10px;
            padding: 20px;
            min-height: 150px;
            border: 2px solid #e0e0e0;
        }
        .answer-box h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 18px;
        }
        .answer-content {
            color: #555;
            line-height: 1.8;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .loading {
            text-align: center;
            color: #667eea;
            padding: 20px;
        }
        .error {
            color: #e74c3c;
            background: #ffeaea;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .example-questions {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
        }
        .example-questions h4 {
            color: #666;
            margin-bottom: 10px;
            font-size: 14px;
        }
        .example-btn {
            display: inline-block;
            padding: 8px 15px;
            margin: 5px;
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 5px;
            cursor: pointer;
            font-size: 13px;
            color: #333;
            transition: all 0.2s;
        }
        .example-btn:hover {
            background: #e0e0e0;
            border-color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 医疗知识图谱问答系统</h1>
        <p class="subtitle">基于知识图谱检索增强生成（RAG）的智能医疗问答</p>
        
        <div class="input-group">
            <textarea id="questionInput" placeholder="请输入您的医疗问题，例如：我头痛怎么办？"></textarea>
        </div>
        
        <div class="button-group">
            <button id="submitBtn" onclick="askQuestion()">提问</button>
            <button onclick="clearAnswer()">清空</button>
        </div>
        
        <div class="answer-box" id="answerBox" style="display: none;">
            <h3>💡 回答：</h3>
            <div class="answer-content" id="answerContent"></div>
        </div>
        
        <div class="example-questions">
            <h4>💬 示例问题：</h4>
            <span class="example-btn" onclick="fillQuestion('我头痛怎么办')">我头痛怎么办</span>
            <span class="example-btn" onclick="fillQuestion('那头痛怎么预防')">那头痛怎么预防</span>
            <span class="example-btn" onclick="fillQuestion('乳腺癌的症状有哪些')">乳腺癌的症状有哪些</span>
            <span class="example-btn" onclick="fillQuestion('失眠怎么治疗')">失眠怎么治疗</span>
            <span class="example-btn" onclick="fillQuestion('肝病要吃啥药')">肝病要吃啥药</span>
        </div>
    </div>

    <script>
        function fillQuestion(question) {
            document.getElementById('questionInput').value = question;
        }
        
        function clearAnswer() {
            document.getElementById('answerBox').style.display = 'none';
            document.getElementById('answerContent').innerHTML = '';
            document.getElementById('questionInput').value = '';
        }
        
        async function askQuestion() {
            const question = document.getElementById('questionInput').value.trim();
            if (!question) {
                alert('请输入问题！');
                return;
            }
            
            const submitBtn = document.getElementById('submitBtn');
            const answerBox = document.getElementById('answerBox');
            const answerContent = document.getElementById('answerContent');
            
            // 禁用按钮，显示加载状态
            submitBtn.disabled = true;
            submitBtn.textContent = '思考中...';
            answerBox.style.display = 'block';
            answerContent.innerHTML = '<div class="loading">🤔 正在思考中，请稍候...</div>';
            
            try {
                // 优先使用RAG接口（如果可用），否则使用基础LLM接口
                const useRAG = true; // 设置为true使用RAG，false使用基础LLM
                const endpoint = useRAG ? '/rag' : '/generate';
                const body = useRAG 
                    ? JSON.stringify({question: question})
                    : JSON.stringify({message: [{content: question}], max_tokens: 256});
                
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: body
                });
                
                const data = await response.json();
                
                if (data.status === 'success' && data.output && data.output[0]) {
                    answerContent.innerHTML = '<div class="answer-content">' + data.output[0] + '</div>';
                } else {
                    answerContent.innerHTML = '<div class="error">❌ 抱歉，生成答案时出现错误。请稍后重试。</div>';
                }
            } catch (error) {
                answerContent.innerHTML = '<div class="error">❌ 网络错误：' + error.message + '</div>';
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = '提问';
            }
        }
        
        // 支持回车键提交（Ctrl+Enter）
        document.getElementById('questionInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                askQuestion();
            }
        });
    </script>
</body>
</html>
    """
    return html_content

# ==================== RAG完整问答接口（可选） ====================
# 如果RAG模块可用，则初始化RAG相关组件并提供完整问答接口
if RAG_AVAILABLE:
    entity_parser = QuestionClassifier()
    kg = MedicalGraph()
    rag_model = ModelAPI(MODEL_URL="http://127.0.0.1:3001/generate")
    
    # RAG问答类：整合知识图谱检索和大语言模型生成
    class KGRAG:
        """
        知识图谱检索增强生成（KGRAG）类
        
        功能：
        1. 实体识别：从用户问题中识别医疗实体（疾病、症状等）
        2. 知识检索：从Neo4j知识图谱中检索相关三元组
        3. 答案生成：基于检索到的知识，使用LLM生成答案
        """
        def __init__(self):
            """
            初始化KGRAG类
            
            设置：
            - cn_dict: 中文字段名映射字典（英文->中文）
            - entity_rel_dict: 实体类型对应的关系字段列表
            """
            self.cn_dict = {
                "name":"名称", "desc":"疾病简介", "cause":"疾病病因", "prevent":"预防措施",
                "cure_department":"治疗科室", "cure_lasttime":"治疗周期", "cure_way":"治疗方式",
                "cured_prob":"治愈概率", "easy_get":"易感人群", "belongs_to":"所属科室",
                "common_drug":"常用药品", "do_eat":"宜吃", "drugs_of":"生产药品",
                "need_check":"诊断检查", "no_eat":"忌吃", "recommand_drug":"好评药品",
                "recommand_eat":"推荐食谱", "has_symptom":"症状", "acompany_with":"并发症"
            }
            self.entity_rel_dict = {
                "disease":["prevent", "cure_way", "name", "cure_lasttime", "cured_prob", "cause", 
                          "cure_department", "desc", "easy_get", "recommand_eat", "no_eat", "do_eat", 
                          "common_drug", "drugs_of", "recommand_drug", "need_check", "has_symptom", 
                          "acompany_with", "belongs_to"],
                "symptom":["name", "has_symptom"],
            }
        
        def entity_linking(self, query):
            return entity_parser.check_medical(query)
        
        def link_entity_rel(self, query, entity, entity_type):
            cate = [self.cn_dict.get(i) for i in self.entity_rel_dict.get(entity_type, [])]
            return set(cate)
        
        def recall_facts(self, cls_rel, entity_type, entity_name, depth=1):
            entity_dict = {"disease":"Disease", "symptom":"Symptom"}
            sql = f"MATCH p=(m:{entity_dict.get(entity_type)})-[r*..{depth}]-(n) where m.name = '{entity_name}' return p"
            ress = kg.g.run(sql).data()
            direct_triples = []
            for res in ress:
                p_data = res["p"]
                nodes = p_data.nodes
                for node in nodes:
                    if node["name"] == entity_name:
                        for k, v in node.items():
                            if v != entity_name and v and self.cn_dict.get(k) in cls_rel:
                                v_str = str(v)[:200]  # 截断长文本
                                triple = f"<{node['name']},{self.cn_dict.get(k)},{v_str}>"
                                direct_triples.append(triple)
            return list(set(direct_triples))[:30]
        
        def chat(self, query):
            entity_dict = self.entity_linking(query)
            if not entity_dict:
                return "抱歉，我在知识库中没有找到对应的实体，无法回答。"
            facts = []
            for entity_name, types in entity_dict.items():
                for entity_type in types:
                    rels = self.link_entity_rel(query, entity_name, entity_type)
                    entity_triples = self.recall_facts(rels, entity_type, entity_name, 1)
                    facts += entity_triples
            facts = facts[:50]
            context_str = "\n".join([f"  {i+1}. {triple}" for i, triple in enumerate(facts)])
            prompt = f"""你是一个医疗知识问答助手。请根据以下知识三元组回答问题。

知识三元组（格式：<实体, 关系, 值>）：
{context_str}

用户问题：{query}

请基于上述知识三元组，用简洁、专业的中文直接回答问题。回答时要：
1. 优先使用与问题中提到的实体直接相关的三元组
2. 如果问题问"怎么办"或"如何治疗"，重点关注"治疗方式"、"常用药品"、"治疗科室"等关系
3. 如果问题问"原因"或"病因"，重点关注"疾病病因"关系
4. 如果问题问"症状"，重点关注"症状"关系
5. 不要回答不知道或抱歉，即使信息有限也要给出建议

回答："""
            answer, _ = rag_model.chat(query=prompt, history=[], max_tokens=256)
            return answer
    
    kgrag = KGRAG()
    
    @app.route("/rag", methods=["POST", "GET"])
    def rag_generate():
        """
        RAG完整问答接口
        
        支持的调用方式：
        1. POST JSON格式：
           {
             "question": "用户问题" 或 "q": "用户问题"
           }
        
        2. GET 查询参数：
           /rag?q=用户问题
        
        返回格式：
        {
            "output": ["生成的答案"],
            "status": "success" | "error",
            "msg": "错误信息（如果有）"
        }
        
        注意：
        - 此接口会先检索知识图谱，再生成答案
        - 比基础/generate接口更准确，但速度稍慢
        """
        try:
            if request.method == "GET":
                q = request.args.get("q", "").strip()
            else:
                data = request.get_json(silent=True) or {}
                q = data.get("question", data.get("q", "")).strip()
            
            if not q:
                return jsonify({"output": [""], "status": "error", "msg": "缺少问题"}), 400
            
            answer = kgrag.chat(q)
            return jsonify({"output": [answer], "status": "success"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"output": [""], "status": "error", "msg": str(e)}), 500

# ==================== 其他路由 ====================
@app.route("/favicon.ico", methods=["GET"])
def favicon():
    """
    处理浏览器自动请求favicon的情况，避免404错误
    """
    return "", 204

# ==================== 主程序入口 ====================
if __name__ == '__main__':
    """
    启动Flask服务
    
    配置说明：
    - port=3001: 服务端口号
    - debug=False: 生产环境建议关闭调试模式
    - host='0.0.0.0': 允许外网访问（如果需要本地访问，可改为'127.0.0.1'）
    """
    print("[INFO] 正在启动Qwen服务...")
    print(f"[INFO] Web界面地址: http://127.0.0.1:3001/")
    print(f"[INFO] API接口地址: http://127.0.0.1:3001/generate")
    if RAG_AVAILABLE:
        print(f"[INFO] RAG接口地址: http://127.0.0.1:3001/rag")
    app.run(port=3001, debug=False, host='0.0.0.0')
