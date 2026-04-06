"""
Graphiti实体读取与过滤服务
从Graphiti图谱中读取节点，筛选出符合预定义实体类型的节点
"""

import time
import asyncio
from typing import Dict, Any, List, Optional, Set, Callable, TypeVar
from dataclasses import dataclass, field

from graphiti_core import Graphiti

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.graphiti_entity_reader')

# 用于泛型返回类型
T = TypeVar('T')


@dataclass
class EntityNode:
    """实体节点数据结构"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    # 相关的边信息
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    # 相关的其他节点信息
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }
    
    def get_entity_type(self) -> Optional[str]:
        """获取实体类型（排除默认的Entity基础标签）"""
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """过滤后的实体集合"""
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class GraphitiEntityReader:
    """
    Graphiti实体读取与过滤服务
    """
    
    def __init__(self):
        self.neo4j_uri = Config.NEO4J_URI
        self.neo4j_user = Config.NEO4J_USER
        self.neo4j_password = Config.NEO4J_PASSWORD
        
        if not self.neo4j_uri:
            raise ValueError("NEO4J_URI 未配置")
            
    def _get_client(self) -> Graphiti:
        return Graphiti(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password
        )
    
    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """获取图谱的所有节点"""
        logger.info(f"获取图谱 {graph_id} 的所有节点...")
        
        async def _get():
            client = self._get_client()
            try:
                records, _, _ = await client.driver.execute_query(
                    "MATCH (n:Entity) WHERE n.group_id = $group_id RETURN n",
                    group_id=graph_id
                )
                nodes_data = []
                for record in records:
                    n = record["n"]
                    nodes_data.append({
                        "uuid": n.get("uuid", ""),
                        "name": n.get("name", ""),
                        "labels": list(n.labels) if n.labels else [],
                        "summary": n.get("summary", ""),
                        "attributes": dict(n),
                    })
                return nodes_data
            finally:
                await client.close()
                
        nodes_data = asyncio.run(_get())
        logger.info(f"共获取 {len(nodes_data)} 个节点")
        return nodes_data

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """获取图谱的所有边"""
        logger.info(f"获取图谱 {graph_id} 的所有边...")
        
        async def _get():
            client = self._get_client()
            try:
                records, _, _ = await client.driver.execute_query(
                    "MATCH (n:Entity)-[r:RELATES_TO]->(m:Entity) WHERE n.group_id = $group_id RETURN r, n.uuid AS source_uuid, m.uuid AS target_uuid",
                    group_id=graph_id
                )
                edges_data = []
                for record in records:
                    r = record["r"]
                    edges_data.append({
                        "uuid": r.get("uuid", ""),
                        "name": r.get("name", ""),
                        "fact": r.get("fact", ""),
                        "source_node_uuid": record["source_uuid"],
                        "target_node_uuid": record["target_uuid"],
                        "attributes": dict(r),
                    })
                return edges_data
            finally:
                await client.close()
                
        edges_data = asyncio.run(_get())
        logger.info(f"共获取 {len(edges_data)} 条边")
        return edges_data
    
    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """获取指定节点的所有相关边"""
        async def _get():
            client = self._get_client()
            try:
                records, _, _ = await client.driver.execute_query(
                    "MATCH (n:Entity)-[r:RELATES_TO]-(m:Entity) WHERE n.uuid = $uuid RETURN r, startNode(r).uuid AS source_uuid, endNode(r).uuid AS target_uuid",
                    uuid=node_uuid
                )
                edges_data = []
                for record in records:
                    r = record["r"]
                    edges_data.append({
                        "uuid": r.get("uuid", ""),
                        "name": r.get("name", ""),
                        "fact": r.get("fact", ""),
                        "source_node_uuid": record["source_uuid"],
                        "target_node_uuid": record["target_uuid"],
                        "attributes": dict(r),
                    })
                return edges_data
            finally:
                await client.close()
                
        try:
            return asyncio.run(_get())
        except Exception as e:
            logger.warning(f"获取节点 {node_uuid} 的边失败: {str(e)}")
            return []
    
    def filter_defined_entities(
        self, 
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True
    ) -> FilteredEntities:
        """筛选出符合预定义实体类型的节点"""
        logger.info(f"开始筛选图谱 {graph_id} 的实体...")
        
        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)
        
        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []
        node_map = {n["uuid"]: n for n in all_nodes}
        
        filtered_entities = []
        entity_types_found = set()
        
        for node in all_nodes:
            labels = node.get("labels", [])
            custom_labels = [l for l in labels if l not in ["Entity", "Node"]]

            # 没有子类型时，用基础 Entity 类型
            effective_labels = custom_labels if custom_labels else (["Entity"] if "Entity" in labels else [])

            if not effective_labels:
                continue

            if defined_entity_types:
                matching_labels = [l for l in effective_labels if l in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = effective_labels[0]
            
            entity_types_found.add(entity_type)
            
            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node["summary"],
                attributes=node["attributes"],
            )
            
            if enrich_with_edges:
                related_edges = []
                related_node_uuids = set()
                
                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])
                
                entity.related_edges = related_edges
                
                related_nodes = []
                for related_uuid in related_node_uuids:
                    if related_uuid in node_map:
                        related_node = node_map[related_uuid]
                        related_nodes.append({
                            "uuid": related_node["uuid"],
                            "name": related_node["name"],
                            "labels": related_node["labels"],
                            "summary": related_node.get("summary", ""),
                        })
                
                entity.related_nodes = related_nodes
            
            filtered_entities.append(entity)
        
        logger.info(f"筛选完成: 总节点 {total_count}, 符合条件 {len(filtered_entities)}, "
                   f"实体类型: {entity_types_found}")
        
        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )
    
    def get_entity_with_context(
        self, 
        graph_id: str, 
        entity_uuid: str
    ) -> Optional[EntityNode]:
        """获取单个实体及其完整上下文"""
        async def _get():
            client = self._get_client()
            try:
                records, _, _ = await client.driver.execute_query(
                    "MATCH (n:Entity) WHERE n.uuid = $uuid RETURN n",
                    uuid=entity_uuid
                )
                if not records:
                    return None
                n = records[0]["n"]
                return {
                    "uuid": n.get("uuid", ""),
                    "name": n.get("name", ""),
                    "labels": list(n.labels) if n.labels else [],
                    "summary": n.get("summary", ""),
                    "attributes": dict(n),
                }
            finally:
                await client.close()
                
        try:
            node = asyncio.run(_get())
            if not node:
                return None
            
            edges = self.get_node_edges(entity_uuid)
            all_nodes = self.get_all_nodes(graph_id)
            node_map = {n["uuid"]: n for n in all_nodes}
            
            related_edges = []
            related_node_uuids = set()
            
            for edge in edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])
            
            related_nodes = []
            for related_uuid in related_node_uuids:
                if related_uuid in node_map:
                    related_node = node_map[related_uuid]
                    related_nodes.append({
                        "uuid": related_node["uuid"],
                        "name": related_node["name"],
                        "labels": related_node["labels"],
                        "summary": related_node.get("summary", ""),
                    })
            
            return EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=node["labels"],
                summary=node["summary"],
                attributes=node["attributes"],
                related_edges=related_edges,
                related_nodes=related_nodes,
            )
            
        except Exception as e:
            logger.error(f"获取实体 {entity_uuid} 失败: {str(e)}")
            return None
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str,
        enrich_with_edges: bool = True
    ) -> List[EntityNode]:
        """获取指定类型的所有实体"""
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges
        )
        return result.entities
