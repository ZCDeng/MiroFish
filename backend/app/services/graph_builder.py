"""
图谱构建服务
接口2：使用Graphiti API构建Standalone Graph
"""

import uuid
import threading
import asyncio
import hashlib
import json
import re
from typing import Dict, Any, List, Optional, Callable, cast
from dataclasses import dataclass
from datetime import datetime

import openai
from ..config import Config
from openai import AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.errors import GraphitiError, NodeNotFoundError
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType
from pydantic import BaseModel, Field

from ..models.task import TaskManager, TaskStatus
from ..utils.logger import get_logger
from .text_processor import TextProcessor


logger = get_logger("mirofish.graph")


class FallbackCrossEncoder(CrossEncoderClient):
    async def rank(self, query: str, passages: List[str]) -> List[tuple[str, float]]:
        total = max(len(passages), 1)
        return [
            (p, float(total - idx) / float(total)) for idx, p in enumerate(passages)
        ]


class FallbackEmbedder(EmbedderClient):
    async def create(self, input_data) -> List[float]:
        if not isinstance(input_data, str):
            input_data = str(input_data)

        buf = bytearray()
        seed = input_data.encode("utf-8", errors="ignore")
        counter = 0
        while len(buf) < 1024:
            h = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            buf.extend(h)
            counter += 1

        return [(b / 255.0) * 2.0 - 1.0 for b in buf[:1024]]

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        return [await self.create(item) for item in input_data_list]


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
        _non_openai = (
            "bigmodel" in graphiti_base
            or "zhipu" in graphiti_base
            or "dashscope" in graphiti_base
            or "glm" in model_name.lower()
            or "qwen" in model_name.lower()
        )
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
        # Qwen3 系列默认开启思维链(CoT)，关闭以大幅降低延迟
        if "qwen3" in model_name.lower():
            api_params["extra_body"] = {"enable_thinking": False}

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


@dataclass
class GraphInfo:
    """图谱信息"""

    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """
    图谱构建服务
    负责调用Graphiti API构建知识图谱
    """

    def __init__(self):
        self.neo4j_uri = Config.NEO4J_URI
        self.neo4j_user = Config.NEO4J_USER
        self.neo4j_password = Config.NEO4J_PASSWORD

        if not self.neo4j_uri:
            raise ValueError("NEO4J_URI 未配置")

        self.task_manager = TaskManager()

    def _get_client(self) -> Graphiti:
        llm_config = LLMConfig(
            api_key=Config.GRAPHITI_API_KEY,
            base_url=Config.GRAPHITI_BASE_URL,
            model=Config.GRAPHITI_MODEL_NAME,
            small_model=Config.GRAPHITI_SMALL_MODEL_NAME,
            max_tokens=Config.GRAPHITI_MAX_TOKENS,
        )
        llm_client = RobustOpenAIGenericClient(
            config=llm_config,
            client=AsyncOpenAI(
                api_key=Config.GRAPHITI_API_KEY,
                base_url=Config.GRAPHITI_BASE_URL,
                timeout=Config.GRAPHITI_REQUEST_TIMEOUT,
                max_retries=Config.GRAPHITI_REQUEST_RETRIES,
            ),
        )
        embedder = FallbackEmbedder()
        cross_encoder = FallbackCrossEncoder()

        return Graphiti(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3,
    ) -> str:
        """
        异步构建图谱
        """
        # 创建任务
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            },
        )

        # 在后台线程中执行构建
        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(
                task_id,
                text,
                ontology,
                graph_name,
                chunk_size,
                chunk_overlap,
                batch_size,
            ),
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
    ):
        """图谱构建工作线程"""
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message="开始构建图谱...",
            )

            # 1. 创建图谱
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id, progress=10, message=f"图谱已创建: {graph_id}"
            )

            # 2. 设置本体 (Graphiti不需要显式设置本体，我们在添加episode时传入)
            self.task_manager.update_task(task_id, progress=15, message="本体已准备")

            # 3. 文本分块
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id, progress=20, message=f"文本已分割为 {total_chunks} 个块"
            )

            # 4. 分批发送数据
            episode_uuids = self.add_text_batches(
                graph_id,
                chunks,
                ontology,
                batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.4),  # 20-60%
                    message=msg,
                ),
            )

            # 5. 等待处理完成 (Graphiti add_episode_bulk 是同步等待的，所以这里不需要额外等待)
            self.task_manager.update_task(
                task_id, progress=60, message="处理数据完成..."
            )

            self._wait_for_episodes(
                episode_uuids,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=60 + int(prog * 0.3),  # 60-90%
                    message=msg,
                ),
            )

            # 6. 获取图谱信息
            self.task_manager.update_task(
                task_id, progress=90, message="获取图谱信息..."
            )

            graph_info = self._get_graph_info(graph_id)

            # 完成
            self.task_manager.complete_task(
                task_id,
                {
                    "graph_id": graph_id,
                    "graph_info": graph_info.to_dict(),
                    "chunks_processed": total_chunks,
                },
            )

        except Exception as e:
            import traceback

            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    def create_graph(self, name: str) -> str:
        """创建图谱（返回一个唯一的group_id）"""
        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"
        return graph_id

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """Graphiti不需要显式设置本体，保留此方法以兼容API"""
        pass

    def _parse_ontology_to_entity_types(
        self, ontology: Dict[str, Any]
    ) -> Dict[str, BaseModel]:
        """将本体定义转换为Graphiti需要的Pydantic模型"""
        entity_types = {}
        raw_entity_defs = ontology.get("entity_types", [])
        if not isinstance(raw_entity_defs, list):
            raw_entity_defs = []

        max_entity_types = max(Config.GRAPHITI_MAX_ENTITY_TYPES, 0)
        max_entity_attrs = max(Config.GRAPHITI_MAX_ENTITY_ATTRIBUTES, 0)
        selected_entity_defs = (
            raw_entity_defs[:max_entity_types]
            if max_entity_types > 0
            else raw_entity_defs
        )

        for entity_def in selected_entity_defs:
            name = entity_def["name"]
            description = entity_def.get("description", f"A {name} entity.")

            attrs = {"__doc__": description}
            annotations = {}

            raw_attrs = entity_def.get("attributes", [])
            if not isinstance(raw_attrs, list):
                raw_attrs = []
            selected_attrs = (
                raw_attrs[:max_entity_attrs] if max_entity_attrs > 0 else raw_attrs
            )

            for attr_def in selected_attrs:
                attr_name = attr_def["name"]
                annotations[attr_name] = Optional[str]
                attrs[attr_name] = None

            attrs["__annotations__"] = annotations
            entity_class = type(name, (BaseModel,), attrs)
            entity_class.__doc__ = description
            entity_types[name] = cast(BaseModel, entity_class)

        return entity_types

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        ontology: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[str]:
        """分批添加文本到图谱"""
        episode_uuids = []
        failed_chunks = {"count": 0}
        total_chunks = len(chunks)

        entity_types = None
        if ontology:
            entity_types = self._parse_ontology_to_entity_types(ontology)

        async def _add_batches():
            client = self._get_client()
            episode_timeout = min(
                Config.GRAPHITI_EPISODE_TIMEOUT,
                max(Config.GRAPHITI_BUILD_TIMEOUT - 5.0, 5.0),
            )
            try:
                for i in range(0, total_chunks, batch_size):
                    if cancel_event is not None and cancel_event.is_set():
                        raise TimeoutError("add_text_batches cancelled")

                    batch_chunks = chunks[i : i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (total_chunks + batch_size - 1) // batch_size

                    if progress_callback:
                        progress = (i + len(batch_chunks)) / total_chunks
                        progress_callback(
                            f"发送第 {batch_num}/{total_batches} 批数据 ({len(batch_chunks)} 块)...",
                            progress,
                        )

                    for j, chunk in enumerate(batch_chunks):
                        if cancel_event is not None and cancel_event.is_set():
                            raise TimeoutError("add_text_batches cancelled")

                        prepared_chunk = " ".join(chunk.split())
                        if not prepared_chunk:
                            failed_chunks["count"] += 1
                            continue

                        max_chars = max(Config.GRAPHITI_CHUNK_MAX_CHARS, 0)
                        if max_chars > 0 and len(prepared_chunk) > max_chars:
                            prepared_chunk = prepared_chunk[:max_chars]

                        max_attempts = max(Config.GRAPHITI_REQUEST_RETRIES + 1, 2)
                        chunk_done = False
                        for attempt in range(1, max_attempts + 1):
                            if cancel_event is not None and cancel_event.is_set():
                                raise TimeoutError("add_text_batches cancelled")

                            try:
                                result = await asyncio.wait_for(
                                    client.add_episode(
                                        name=f"Chunk {i + j + 1}",
                                        episode_body=prepared_chunk,
                                        source_description="Text chunk",
                                        reference_time=datetime.now(),
                                        source=EpisodeType.text,
                                        group_id=graph_id,
                                        entity_types=entity_types,
                                    ),
                                    timeout=episode_timeout,
                                )
                                episode_uuids.append(result.episode.uuid)
                                chunk_done = True
                                break
                            except (NodeNotFoundError, GraphitiError) as e:
                                failed_chunks["count"] += 1
                                logger.warning(
                                    f"Graphiti node/data error in batch {batch_num}, chunk {j + 1}: {e}. "
                                    "Continuing with remaining chunks."
                                )
                                break
                            except asyncio.TimeoutError:
                                if entity_types is not None:
                                    try:
                                        result = await asyncio.wait_for(
                                            client.add_episode(
                                                name=f"Chunk {i + j + 1}",
                                                episode_body=prepared_chunk,
                                                source_description="Text chunk",
                                                reference_time=datetime.now(),
                                                source=EpisodeType.text,
                                                group_id=graph_id,
                                                entity_types=None,
                                            ),
                                            timeout=episode_timeout,
                                        )
                                        episode_uuids.append(result.episode.uuid)
                                        chunk_done = True
                                        logger.warning(
                                            f"Chunk {i + j + 1} retried without ontology after timeout."
                                        )
                                        break
                                    except Exception:
                                        pass

                                if attempt < max_attempts:
                                    backoff = float(2 ** (attempt - 1))
                                    logger.warning(
                                        f"Episode timeout in batch {batch_num}, chunk {j + 1}, "
                                        f"attempt {attempt}/{max_attempts}. Retrying in {backoff:.0f}s..."
                                    )
                                    await asyncio.sleep(backoff)
                                    continue
                                failed_chunks["count"] += 1
                                logger.warning(
                                    f"Episode timeout in batch {batch_num}, chunk {j + 1} after {max_attempts} attempts. "
                                    "Continuing with remaining chunks."
                                )
                                break
                            except openai.RateLimitError as e:
                                backoff = 60.0
                                if attempt < max_attempts:
                                    logger.warning(
                                        f"Rate limit (429) in batch {batch_num}, chunk {j + 1}, "
                                        f"attempt {attempt}/{max_attempts}. Waiting {backoff:.0f}s..."
                                    )
                                    await asyncio.sleep(backoff)
                                    continue
                                failed_chunks["count"] += 1
                                logger.warning(
                                    f"Rate limit (429) final in batch {batch_num}, chunk {j + 1}. "
                                    "Continuing with remaining chunks."
                                )
                                break
                            except Exception as e:
                                if entity_types is not None and (
                                    "Property values can only be of primitive types"
                                    in str(e)
                                    or "'list' object has no attribute 'get'" in str(e)
                                ):
                                    try:
                                        result = await asyncio.wait_for(
                                            client.add_episode(
                                                name=f"Chunk {i + j + 1}",
                                                episode_body=prepared_chunk,
                                                source_description="Text chunk",
                                                reference_time=datetime.now(),
                                                source=EpisodeType.text,
                                                group_id=graph_id,
                                                entity_types=None,
                                            ),
                                            timeout=episode_timeout,
                                        )
                                        episode_uuids.append(result.episode.uuid)
                                        chunk_done = True
                                        logger.warning(
                                            f"Chunk {i + j + 1} retried without ontology after CypherTypeError."
                                        )
                                        break
                                    except Exception:
                                        pass

                                if attempt < max_attempts:
                                    backoff = float(2 ** (attempt - 1))
                                    logger.warning(
                                        f"Transient error in batch {batch_num}, chunk {j + 1}, attempt {attempt}/{max_attempts}: "
                                        f"{type(e).__name__}: {e}. Retrying in {backoff:.0f}s..."
                                    )
                                    await asyncio.sleep(backoff)
                                    continue
                                failed_chunks["count"] += 1
                                logger.warning(
                                    f"Final error in batch {batch_num}, chunk {j + 1}: {type(e).__name__}: {e}. "
                                    "Continuing with remaining chunks."
                                )
                                break

                        if not chunk_done:
                            continue
            finally:
                await client.close()

        asyncio.run(_add_batches())
        if failed_chunks["count"] == total_chunks and total_chunks > 0:
            raise RuntimeError(
                "All chunks failed to be added to graph. "
                "Check Graphiti/Neo4j state and ontology compatibility."
            )
        return episode_uuids

    def _wait_for_episodes(
        self,
        episode_uuids: List[str],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        timeout: int = 600,
    ):
        """Graphiti add_episode is awaited, so no need to wait here"""
        if progress_callback:
            progress_callback("处理完成", 1.0)

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """获取图谱信息"""

        async def _get_info():
            client = self._get_client()
            try:
                records, _, _ = await client.driver.execute_query(
                    "MATCH (n:Entity) WHERE n.group_id = $group_id RETURN count(n) as node_count",
                    group_id=graph_id,
                )
                node_count = records[0]["node_count"] if records else 0

                records, _, _ = await client.driver.execute_query(
                    "MATCH (n:Entity)-[r]->(m:Entity) WHERE n.group_id = $group_id AND n.uuid <> m.uuid RETURN count(r) as edge_count",
                    group_id=graph_id,
                )
                edge_count = records[0]["edge_count"] if records else 0

                records, _, _ = await client.driver.execute_query(
                    "MATCH (n:Entity) WHERE n.group_id = $group_id UNWIND labels(n) AS label RETURN DISTINCT label",
                    group_id=graph_id,
                )
                entity_types = [
                    r["label"] for r in records if r["label"] not in ["Entity", "Node"]
                ]

                return GraphInfo(
                    graph_id=graph_id,
                    node_count=node_count,
                    edge_count=edge_count,
                    entity_types=entity_types,
                )
            finally:
                await client.close()

        return asyncio.run(_get_info())

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        获取完整图谱数据（包含详细信息）
        """

        async def _get_data():
            client = self._get_client()
            try:
                records, _, _ = await client.driver.execute_query(
                    "MATCH (n:Entity) WHERE n.group_id = $group_id RETURN n",
                    group_id=graph_id,
                )
                nodes_data = []
                node_map = {}
                for record in records:
                    n = record["n"]
                    uuid_ = n.get("uuid", "")
                    name = n.get("name", "")
                    node_map[uuid_] = name
                    nodes_data.append(
                        {
                            "uuid": uuid_,
                            "name": name,
                            "labels": [
                                l
                                for l in (list(n.labels) if n.labels else [])
                                if l != "Entity"
                            ],
                            "summary": n.get("summary", ""),
                            "attributes": {
                                k: v
                                for k, v in dict(n).items()
                                if isinstance(v, (str, int, float, bool, type(None)))
                            },
                            "created_at": str(n.get("created_at", "")),
                        }
                    )

                records, _, _ = await client.driver.execute_query(
                    "MATCH (n:Entity)-[r]->(m:Entity) WHERE n.group_id = $group_id AND n.uuid <> m.uuid RETURN r, type(r) AS rel_type, n.uuid AS source_uuid, m.uuid AS target_uuid",
                    group_id=graph_id,
                )
                edges_data = []
                for record in records:
                    r = record["r"]
                    source_uuid = record["source_uuid"]
                    target_uuid = record["target_uuid"]
                    edges_data.append(
                        {
                            "uuid": r.get("uuid", ""),
                            "name": record.get("rel_type", ""),
                            "fact": r.get("fact", ""),
                            "fact_type": record.get("rel_type", ""),
                            "source_node_uuid": source_uuid,
                            "target_node_uuid": target_uuid,
                            "source_node_name": node_map.get(source_uuid, ""),
                            "target_node_name": node_map.get(target_uuid, ""),
                            "attributes": {
                                k: v
                                for k, v in dict(r).items()
                                if isinstance(v, (str, int, float, bool, type(None)))
                            },
                            "created_at": str(r.get("created_at", "")),
                            "valid_at": str(r.get("valid_at", "")),
                            "invalid_at": str(r.get("invalid_at", "")),
                            "expired_at": str(r.get("expired_at", "")),
                            "episodes": r.get("episodes", []),
                        }
                    )

                return {
                    "graph_id": graph_id,
                    "nodes": nodes_data,
                    "edges": edges_data,
                    "node_count": len(nodes_data),
                    "edge_count": len(edges_data),
                }
            finally:
                await client.close()

        return asyncio.run(_get_data())

    def delete_graph(self, graph_id: str):
        """删除图谱"""

        async def _delete():
            client = self._get_client()
            try:
                await client.driver.execute_query(
                    "MATCH (n) WHERE n.group_id = $group_id DETACH DELETE n",
                    group_id=graph_id,
                )
            finally:
                await client.close()

        asyncio.run(_delete())
