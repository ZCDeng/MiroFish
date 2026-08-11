"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config
from .logger import get_logger

logger = get_logger('mirofish.llm_client')


class TruncatedResponseError(RuntimeError):
    """模型在 token 预算内没能产出内容，加预算重试有救。"""


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content or ""
        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()

        # 推理模型（DeepSeek v4、o 系列等）把思考过程算进 completion_tokens，
        # 但它不出现在 content 里。预算给小了，思考就能把额度吃光，
        # 接口照常 200 返回，content 是空字符串，finish_reason='length'。
        # 实测 deepseek-v4-flash 做本体生成：max_tokens=4096 时 4096 个 token
        # 全花在推理上、content 长度 0；给到 8192 才正常出 2181 字。
        # 这里不吞掉，交给 chat_json 去加预算重试。
        if not content and choice.finish_reason == "length":
            raise TruncatedResponseError(
                f"模型在 max_tokens={max_tokens} 内没产出内容（推理占满了预算）"
            )
        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        # 推理模型可能把整个预算花在思考上，一个字都不输出。一路翻倍到
        # Config.LLM_MAX_TOKENS_CEILING 为止，到顶还不出内容才认输。
        budget = max_tokens
        response = None
        while True:
            try:
                response = self.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=budget,
                    response_format={"type": "json_object"},
                )
                break
            except TruncatedResponseError as e:
                nxt = min(budget * 2, Config.LLM_MAX_TOKENS_CEILING)
                if nxt <= budget:
                    raise ValueError(
                        f"模型在 max_tokens={budget}（已达上限）内仍未产出内容，"
                        f"调高 LLM_MAX_TOKENS_CEILING 或换个非推理模型"
                    ) from e
                logger.warning(f"{e}，把预算从 {budget} 提到 {nxt} 重试")
                budget = nxt

        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            # 别把模型原文整段拼进异常 —— 它会被 api 层连同 traceback 一起
            # 返回给浏览器，里面可能带着上传文档的片段
            preview = cleaned_response[:200]
            raise ValueError(
                f"LLM返回的JSON格式无效（前200字符）: {preview}"
            )

