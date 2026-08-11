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
from typing import Any, Dict, List, Optional, Tuple

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
    """按传入顺序线性打分，不做真正的重排。

    graphiti-core 默认的 OpenAIRerankerClient 会打 api.openai.com，这里用它顶住。
    当前的 search 配方走 RRF，不会触发 cross_encoder；真要做语义重排得换成
    SiliconFlow 上的 rerank 模型。
    """

    async def rank(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        total = max(len(passages), 1)
        return [
            (p, float(total - idx) / float(total)) for idx, p in enumerate(passages)
        ]


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


def build_llm_client() -> OpenAIGenericClient:
    """图谱抽取用的 LLM 客户端。

    这里曾经有一个 RobustOpenAIGenericClient，override 了 _generate_response，
    把 graphiti 注入的 JSON Schema 转成示例格式（_schema_to_example），再对返回值
    做字段名正规化。那是 2026-04 为 GLM-4-flash 写的，当时 graphiti-core 0.11.6
    只会把 schema 塞进 prompt，GLM 读不懂。

    0.29.3 之后这些全部由上游做掉了，而且做得更好：structured_output_mode 默认
    json_schema，走 provider 的原生结构化输出；基类用 tenacity 做指数退避重试；
    _strip_code_fences 处理 markdown 围栏；空响应抛 EmptyResponseError。

    留着那个 override 反而有害 —— 它把带 schema 的 prompt 换成有损示例，模型于是
    自由发挥。实测同一段 extract_text 提示词打 DeepSeek-V3：包装版 0/3 通过
    （返回 entity_text / entity_type 之类的自创字段名），原版 3/3 通过。
    """
    return OpenAIGenericClient(
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
        structured_output_mode=resolve_structured_output_mode(),
    )


# base_url 里出现这些字样的 provider 不支持 response_format={"type":"json_schema"}，
# 只能退到 json_object。实测 api.deepseek.com 会返回
# 400 "This response_format type is unavailable now"。
_NO_JSON_SCHEMA_HOSTS = ("api.deepseek.com", "dashscope", "bigmodel", "zhipu")


def resolve_structured_output_mode() -> str:
    """决定用 json_schema 还是 json_object。

    json_schema 由服务端强制结构，能省掉一整类「模型自己发明字段名」的问题。
    json_object 只保证返回合法 JSON，schema 由 graphiti 拼进 prompt
    （openai_generic_client.py:194-200），约束弱一些但兼容性好。

    显式配了 GRAPHITI_STRUCTURED_OUTPUT_MODE 就听它的，否则按 base_url 猜。
    """
    explicit = (Config.GRAPHITI_STRUCTURED_OUTPUT_MODE or "").lower()
    if explicit in ("json_schema", "json_object"):
        return explicit

    base = (Config.GRAPHITI_BASE_URL or "").lower()
    if any(h in base for h in _NO_JSON_SCHEMA_HOSTS):
        logger.info(f"provider 不支持 json_schema，退到 json_object: {base}")
        return "json_object"
    return "json_schema"


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
        llm_client=build_llm_client(),
        embedder=embedder,
        cross_encoder=FallbackCrossEncoder(),
    )
