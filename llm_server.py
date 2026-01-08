# -*- coding: utf-8 -*-
"""
LLM API客户端封装

功能：
- 封装对LLM服务的HTTP请求
- 处理超时和错误重试
- 支持历史对话（history）
- 支持自定义最大生成长度

作者：医疗知识图谱问答系统
日期：2025
"""

import requests
import json
import time


class ModelAPI():
    """
    LLM API客户端类
    
    用于与Qwen LLM服务端进行HTTP通信
    """
    def __init__(self, MODEL_URL):
        """
        初始化ModelAPI客户端
        
        参数:
            MODEL_URL (str): LLM服务的URL地址，例如 "http://127.0.0.1:3001/generate"
        """
        self.url = MODEL_URL

    def send_request(self, message, history, max_tokens=256):
        """
        发送HTTP请求到LLM服务
        
        参数:
            message (list): 消息列表，格式为 [{"role": "user", "content": "问题"}]
            history (list): 历史对话记录（可选）
            max_tokens (int): 最大生成长度，默认256
        
        返回:
            tuple: (生成的回答文本, 更新后的历史记录)
        
        注意:
            - 超时时间设置为120秒（考虑CPU推理较慢）
            - 自动处理HTTP错误和JSON解析错误
        """
        data = json.dumps({"message": message, "history": history, "max_tokens": max_tokens})
        headers = {'Content-Type': 'application/json'}
        try:
            # CPU上模型推理较慢，适当放宽超时时间
            res = requests.post(self.url, data=data, headers=headers, timeout=120)
            res.raise_for_status()  # 检查HTTP状态码
            payload = res.json()
            # 提取生成的答案
            predict = payload.get("output", [""])[0] if payload.get("output") else ""
            history = payload.get("history", history)
            return predict, history
        except Exception as e:
            print(f"[ERROR] 请求失败: {e}")
            return "", []

    def chat(self, query, history=None, max_tokens=256):
        """
        对话接口，支持自动重试
        
        参数:
            query (str): 用户输入的问题
            history (list): 历史对话记录（可选，默认None）
            max_tokens (int): 最大生成长度，默认256
        
        返回:
            tuple: (生成的回答文本, 更新后的历史记录)
        
        注意:
            - 最多重试10次，每次失败后等待1秒
            - 如果所有重试都失败，返回空字符串
        """
        # 默认保持调用方传入的历史，不强制重置
        if history is None:
            history = []
        message = [{"role": "user", "content": query}]
        count = 0
        response = ''
        # 最多重试10次
        while count <= 10:
            try:
                count += 1
                response, history = self.send_request(message, history, max_tokens=max_tokens)
                if response:
                    return response, history
            except Exception as e:
                print(f'[ERROR] 第{count}次请求异常: {e}')
                time.sleep(1)  # 等待1秒后重试
        return response, history

# ==================== 测试代码 ====================
if __name__ == '__main__':
    """
    测试代码：演示如何使用ModelAPI
    """
    model = ModelAPI(MODEL_URL="http://127.0.0.1:3001/generate")
    res = model.chat(query="你叫啥", history=[])
    print(res)
