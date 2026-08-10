"""
Graphiti 客户端构造

四个地方要 Graphiti 客户端：graph_builder 写图、graphiti_tools 检索、
graphiti_entity_reader 读节点、graphiti_memory_updater 写模拟期活动。
以前只有 graph_builder 配齐了 llm_client / embedder / cross_encoder，另外三处
把参数留空，graphiti-core 就地 new 一个 OpenAIClient / OpenAIEmbedder /
OpenAIRerankerClient 打 api.openai.com —— 绕开这里配的 SiliconFlow，而且写图用的
向量和检索用的向量落在两个不同空间，检索结果没有意义。

现在四处都走 build_graphiti_client()。配置缺失直接抛，不再静默降级。
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.graphiti_client")


class FallbackCrossEncoder(CrossEncoderClient):
    async def rank(self, query: str, passages: List[str]) -> List[tuple[str, float]]:
        total = max(len(passages), 1)
        return [
            (p, float(total - idx) / float(total)) for idx, p in enumerate(passages)
        ]


class RobustOpenAIGenericClient(OpenAIGenericClient):
    MAX_RETRIES = 3

    @staticmethod
    def _parse_json_content(content: str) -> Optional[Dict[str, Any]]:
        if not content:
            return None

        # Strip markdown code fences that GLM often wraps output in
        text = content.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence) :].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return RobustOpenAIGenericClient._normalize_glm_response(parsed)
            return None
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            return None

        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return RobustOpenAIGenericClient._normalize_glm_response(parsed)
            return None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize_glm_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        GLM 系列模型有时将 'name' 字段返回为 'entity_name' 或其他变体。
        此方法对 Graphiti 已知的响应结构进行字段名正规化，确保后续 Pydantic 验证通过。
        """
        # Fix extracted_entities: entity_name → name
        if "extracted_entities" in parsed and isinstance(parsed["extracted_entities"], list):
            for entity in parsed["extracted_entities"]:
                if isinstance(entity, dict) and "name" not in entity:
                    for alt in ("entity_name", "entityName", "entity", "label", "node_name"):
                        if alt in entity:
                            entity["name"] = entity.pop(alt)
                            break
        # Fix missed_entities (reflexion): items should be strings
        if "missed_entities" in parsed and isinstance(parsed["missed_entities"], list):
            fixed = []
            for item in parsed["missed_entities"]:
                if isinstance(item, dict):
                    fixed.append(item.get("name") or item.get("entity_name") or str(item))
                else:
                    fixed.append(item)
            parsed["missed_entities"] = fixed
        return parsed

    @staticmethod
    def _schema_to_example(schema_str: str) -> str:
        """
        将 JSON Schema 字符串转换为示例格式。
        GLM 系列模型无法正确理解 {"type":"object","properties":{...}} 格式，
        会把 schema 结构本身嵌入返回值而非填充数据。
        转换为示例格式后模型能正确理解并填充真实数据。
        """
        try:
            schema = json.loads(schema_str)
        except (json.JSONDecodeError, TypeError):
            return schema_str

        defs = schema.get("$defs", {})

        def resolve_ref(ref: str) -> dict:
            # ref is like "#/$defs/SomeName"
            parts = ref.lstrip("#/").split("/")
            node = schema
            for p in parts:
                node = node.get(p, {})
            return node if isinstance(node, dict) else {}

        def gen_example(s):
            if not isinstance(s, dict):
                return s
            # Resolve $ref first
            if "$ref" in s:
                s = resolve_ref(s["$ref"])
            t = s.get("type")
            if t == "object":
                return {k: gen_example(v) for k, v in s.get("properties", {}).items()}
            elif t == "array":
                return [gen_example(s.get("items", {}))]
            elif t == "string":
                desc = s.get("description") or s.get("title", "")
                return f"<{desc}>" if desc else "value"
            elif t == "integer":
                return 0
            elif t == "number":
                return 0.0
            elif t == "boolean":
                return True
            elif t == "null":
                return None
            return None

        try:
            example = gen_example(schema)
            return json.dumps(example, ensure_ascii=False)
        except Exception:
            return schema_str

    async def _generate_response(
        self,
        messages,
        response_model=None,
        max_tokens=8192,
        model_size=None,
    ):
        openai_messages = []
        model_name = self.model or Config.GRAPHITI_MODEL_NAME or "gpt-4o-mini"
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == "user":
                openai_messages.append({"role": "user", "content": m.content})
            elif m.role == "system":
                openai_messages.append({"role": "system", "content": m.content})

        # 非 OpenAI 模型（GLM / Qwen 等）不能正确理解 JSON Schema 格式，
        # 需将最后一条 user message 中的 schema 描述转换为示例格式
        graphiti_base = Config.GRAPHITI_BASE_URL or ""
        # 反着判：只有确认打的是 OpenAI 官方端点才跳过转换。
        # 原来是正着列举 bigmodel/zhipu/dashscope/glm/qwen，换供应商就漏 ——
        # 现在 GRAPHITI_MODEL_NAME 是 DeepSeek-V3 @ siliconflow，两个条件都不命中，
        # 主模型走 graphiti 原生的 JSON Schema，而小模型 Qwen2.5-7B 含 qwen 会命中，
        # 同一次建图里两个模型收到的 prompt 结构不一样。
        _openai_official = "api.openai.com" in graphiti_base or (
            not graphiti_base and not Config.GRAPHITI_BASE_URL
        )
        _non_openai = not _openai_official
        if _non_openai:
            if openai_messages and openai_messages[-1]["role"] == "user":
                content = openai_messages[-1]["content"]
                # Graphiti 注入的 schema 标记
                marker = "Respond with a JSON object in the following format:\n\n"
                if marker in content:
                    prefix, schema_part = content.split(marker, 1)
                    example = self._schema_to_example(schema_part.strip())
                    openai_messages[-1]["content"] = (
                        prefix
                        + "Respond with a JSON object matching this example structure (fill in real data):\n\n"
                        + example
                    )

        api_params = {
            "model": model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if model_name.startswith(("gpt-5", "o1", "o3", "o4")):
            api_params["max_completion_tokens"] = self.max_tokens
        else:
            api_params["max_tokens"] = self.max_tokens
        last_error = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(**api_params)
                content = response.choices[0].message.content or ""
                parsed = self._parse_json_content(content)
                if parsed is not None:
                    return parsed
            except openai.RateLimitError as e:
                raise e
            except Exception as e:
                last_error = e

            if attempt < self.MAX_RETRIES:
                await asyncio.sleep(2**attempt)

        raise ValueError(
            f"LLM returned non-JSON content after {self.MAX_RETRIES + 1} attempts. Last error: {last_error}"
        )


# 索引只需要建一次。graphiti-core 的 Neo4jDriver 在构造时会起一批 CREATE INDEX 协程，
# 事件循环先关掉的话它们跑不完 —— 空库上实测只落地 7 个 RANGE/LOOKUP，4 个 FULLTEXT
# 一个都没有，而 NODE_HYBRID_SEARCH_RRF 的 BM25 那一半正好依赖全文索引。
_indices_ready = False
_indices_lock = asyncio.Lock()


async def ensure_indices(client: Graphiti) -> None:
    """在第一次写图之前补一次索引，进程内只跑一次。"""
    global _indices_ready
    if _indices_ready:
        return
    async with _indices_lock:
        if _indices_ready:
            return
        try:
            await client.build_indices_and_constraints()
            _indices_ready = True
            logger.info("Graphiti 索引已确认")
        except Exception as e:
            # 建索引失败不该挡住写入，但要留下痕迹 —— 检索质量会受影响
            logger.warning(f"建索引失败，检索质量可能下降: {e}")


def build_graphiti_client() -> Graphiti:
    """构造配置齐全的 Graphiti 客户端。

    llm_client / embedder / cross_encoder 三件套一次配齐。缺配置就抛，
    以前那两个兜底（sha256 伪向量的 FallbackEmbedder、embedder=None 落到
    graphiti 默认 OpenAIEmbedder）都已删除 —— 它们让配置错误看起来像能跑。
    """
    if not Config.NEO4J_URI:
        raise ValueError("NEO4J_URI 未配置")
    if not (Config.GRAPHITI_API_KEY and Config.GRAPHITI_BASE_URL):
        raise ValueError(
            "GRAPHITI_API_KEY / GRAPHITI_BASE_URL 未配置（默认继承 LLM_API_KEY / LLM_BASE_URL）"
        )
    if not (Config.GRAPHITI_EMBEDDER_API_KEY and Config.GRAPHITI_EMBEDDER_BASE_URL):
        raise ValueError(
            "GRAPHITI_EMBEDDER_API_KEY / GRAPHITI_EMBEDDER_BASE_URL 未配置。"
            "缺了它们向量检索就是废的，不再静默回落到哈希向量。"
        )

    llm_client = RobustOpenAIGenericClient(
        config=LLMConfig(
            api_key=Config.GRAPHITI_API_KEY,
            base_url=Config.GRAPHITI_BASE_URL,
            model=Config.GRAPHITI_MODEL_NAME,
            small_model=Config.GRAPHITI_SMALL_MODEL_NAME,
            max_tokens=Config.GRAPHITI_MAX_TOKENS,
        ),
        client=AsyncOpenAI(
            api_key=Config.GRAPHITI_API_KEY,
            base_url=Config.GRAPHITI_BASE_URL,
            timeout=Config.GRAPHITI_REQUEST_TIMEOUT,
            max_retries=Config.GRAPHITI_REQUEST_RETRIES,
        ),
    )
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=Config.GRAPHITI_EMBEDDER_API_KEY,
            base_url=Config.GRAPHITI_EMBEDDER_BASE_URL,
            embedding_model=Config.GRAPHITI_EMBEDDER_MODEL,
        )
    )

    return Graphiti(
        uri=Config.NEO4J_URI,
        user=Config.NEO4J_USER,
        password=Config.NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=FallbackCrossEncoder(),
    )
