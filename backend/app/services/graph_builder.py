"""
图谱构建服务
接口2：使用Graphiti API构建Standalone Graph
"""

import os
import uuid
import time
import threading
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import EpisodeType
from pydantic import BaseModel, Field

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from ..utils.logger import get_logger
from .text_processor import TextProcessor


logger = get_logger("mirofish.graph")


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
        return Graphiti(
            uri=self.neo4j_uri, user=self.neo4j_user, password=self.neo4j_password
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
    ) -> Dict[str, type[BaseModel]]:
        """将本体定义转换为Graphiti需要的Pydantic模型"""
        entity_types = {}
        for entity_def in ontology.get("entity_types", []):
            name = entity_def["name"]
            description = entity_def.get("description", f"A {name} entity.")

            attrs = {"__doc__": description}
            annotations = {}

            for attr_def in entity_def.get("attributes", []):
                attr_name = attr_def["name"]
                attr_desc = attr_def.get("description", attr_name)
                attrs[attr_name] = Field(description=attr_desc, default=None)
                annotations[attr_name] = Optional[str]

            attrs["__annotations__"] = annotations
            entity_class = type(name, (BaseModel,), attrs)
            entity_class.__doc__ = description
            entity_types[name] = entity_class

        return entity_types

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        ontology: Dict[str, Any] = None,
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """分批添加文本到图谱"""
        episode_uuids = []
        total_chunks = len(chunks)

        entity_types = None
        if ontology:
            entity_types = self._parse_ontology_to_entity_types(ontology)

        async def _add_batches():
            client = self._get_client()
            try:
                for i in range(0, total_chunks, batch_size):
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
                        try:
                            ep_uuid = str(uuid.uuid4())
                            await client.add_episode(
                                name=f"Chunk {i + j + 1}",
                                episode_body=chunk,
                                source_description="Text chunk",
                                reference_time=datetime.now(),
                                source=EpisodeType.TEXT_CHUNK,
                                graph_id=graph_id,
                                entity_types=entity_types,
                            )
                            episode_uuids.append(ep_uuid)
                        except NodeNotFoundError as e:
                            logger.warning(
                                f"NodeNotFoundError in batch {batch_num}, chunk {j + 1}: {e}. "
                                "Continuing with remaining chunks."
                            )
                            continue
            finally:
                await client.close()

        asyncio.run(_add_batches())
        return episode_uuids

    def _wait_for_episodes(
        self,
        episode_uuids: List[str],
        progress_callback: Optional[Callable] = None,
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
                    "MATCH (n:EntityNode) WHERE n.group_id = $group_id RETURN count(n) as node_count",
                    group_id=graph_id,
                )
                node_count = records[0]["node_count"] if records else 0

                records, _, _ = await client.driver.execute_query(
                    "MATCH (n:EntityNode)-[r:EntityEdge]->(m:EntityNode) WHERE n.group_id = $group_id RETURN count(r) as edge_count",
                    group_id=graph_id,
                )
                edge_count = records[0]["edge_count"] if records else 0

                records, _, _ = await client.driver.execute_query(
                    "MATCH (n:EntityNode) WHERE n.group_id = $group_id UNWIND labels(n) AS label RETURN DISTINCT label",
                    group_id=graph_id,
                )
                entity_types = [
                    r["label"]
                    for r in records
                    if r["label"] not in ["EntityNode", "Node"]
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
                    "MATCH (n:EntityNode) WHERE n.group_id = $group_id RETURN n",
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
                            "labels": list(n.labels) if n.labels else [],
                            "summary": n.get("summary", ""),
                            "attributes": dict(n),
                            "created_at": str(n.get("created_at", "")),
                        }
                    )

                records, _, _ = await client.driver.execute_query(
                    "MATCH (n:EntityNode)-[r:EntityEdge]->(m:EntityNode) WHERE n.group_id = $group_id RETURN r, n.uuid AS source_uuid, m.uuid AS target_uuid",
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
                            "name": r.get("name", ""),
                            "fact": r.get("fact", ""),
                            "fact_type": r.get("name", ""),
                            "source_node_uuid": source_uuid,
                            "target_node_uuid": target_uuid,
                            "source_node_name": node_map.get(source_uuid, ""),
                            "target_node_name": node_map.get(target_uuid, ""),
                            "attributes": dict(r),
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
