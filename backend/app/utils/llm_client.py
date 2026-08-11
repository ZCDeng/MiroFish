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
from .openai_chat_compat import create_chat_completion

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
        # GPT-5 系列不收 temperature，且 token 上限的参数名是 max_completion_tokens。
        # 其余模型的请求体一字不改。config.py:43 的默认模型是 gpt-4o-mini，
        # 谁把它换成 gpt-5* 就会踩这个 400。
        response = create_chat_completion(
            self.client,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
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
        #
        # JSON 模式下 finish_reason='length' 一定不可用：要么推理吃光预算 content 为空，
        # 要么对象被从中间切断，两种都得加预算重来。纯文本模式截断了还能读，
        # 所以只在完全没内容时才报 —— report_agent 和 graphiti_tools 有几处
        # 直接调 chat() 生成长文，不能因为超长就抛。
        if choice.finish_reason == "length" and (response_format or not content):
            raise TruncatedResponseError(
                f"模型在 max_tokens={max_tokens} 内没产出完整内容"
                f"（content 长度 {len(content)}）"
            )
        # content_filter 之类的异常终止要说清楚，别让空 content 伪装成 JSON 解析失败
        if choice.finish_reason not in (None, "stop", "length", "tool_calls"):
            raise ValueError(f"模型异常终止：finish_reason={choice.finish_reason}")
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

        # 清理 BOM 和 markdown 代码块标记。
        # BOM（﻿）不算 Python 的 whitespace，strip() 吃不掉，会一路活到
        # json.loads 然后报一个看不懂的错。
        cleaned_response = response.lstrip("﻿").strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.lstrip("﻿").strip()

        # 两个候选依次试：原文，以及把全角引号换成半角的版本。
        # 模型写中文时常把 JSON 的定界引号打成全角：
        #   {"headquarters": "旧金山“, ”year”: “2023”}
        # 全角引号不是合法定界符，运气好直接 JSONDecodeError，运气不好会被后面
        # 某个真引号闭合，解析「成功」但值里卷进了后半个对象（那种由
        # utils/graphiti_attrs.clean_attr_value 在读取侧兜）。
        # 替换只在严格解析失败之后才发生，正常内容碰不到。
        candidates = [cleaned_response]
        repaired = cleaned_response.replace("“", '"').replace("”", '"')
        if repaired != cleaned_response:
            candidates.append(repaired)

        for idx, candidate in enumerate(candidates):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                # 有的 provider 在完整 JSON 之后追一段说明文字（「以上就是本体定义」）。
                # raw_decode 只认从头解出来的那个对象，后面的丢掉。
                # 它不会去「修补」被截断的 JSON —— 截断的前缀本身就解不出完整对象。
                try:
                    parsed, _ = json.JSONDecoder().raw_decode(candidate)
                except json.JSONDecodeError:
                    continue
                logger.warning("模型在 JSON 之后追加了额外文本，已丢弃尾部")

            if not isinstance(parsed, dict):
                # 调用方（如 ontology_generator._validate_and_process）一上来就
                # result["entity_types"] = ...，拿到 list 会 TypeError，
                # 报一个跟真正原因无关的错。这里直接拒绝。
                raise ValueError(
                    f"LLM返回的顶层不是 JSON 对象，而是 {type(parsed).__name__}"
                )
            if idx > 0:
                logger.warning("模型返回的 JSON 用了全角引号，已修正后解析成功")
            return parsed

        # 别把模型原文整段拼进异常 —— 它会被 api 层连同 traceback 一起
        # 返回给浏览器，里面可能带着上传文档的片段
        preview = cleaned_response[:200]
        raise ValueError(f"LLM返回的JSON格式无效（前200字符）: {preview}")

