"""
Graphiti检索工具服务
封装图谱搜索、节点读取、边查询等工具，供Report Agent使用
"""

import time
import json
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from graphiti_core import Graphiti

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from .graphiti_entity_reader import GraphitiEntityReader

logger = get_logger('mirofish.graphiti_tools')


@dataclass
class SearchResult:
    """搜索结果"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }
    
    def to_text(self) -> str:
        """转换为文本格式，供LLM理解"""
        text_parts = [f"搜索查询: {self.query}", f"找到 {self.total_count} 条相关信息"]
        
        if self.facts:
            text_parts.append("\n### 相关事实:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")
        
        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """节点信息"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }
    
    def to_text(self) -> str:
        """转换为文本格式"""
        entity_type = next((l for l in self.labels if l not in ["EntityNode", "Node"]), "未知类型")
        return f"实体: {self.name} (类型: {entity_type})\n摘要: {self.summary}"


@dataclass
class EdgeInfo:
    """边信息"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # 时间信息
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }
    
    def to_text(self, include_temporal: bool = False) -> str:
        """转换为文本格式"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"关系: {source} --[{self.name}]--> {target}\n事实: {self.fact}"
        
        if include_temporal:
            valid_at = self.valid_at or "未知"
            invalid_at = self.invalid_at or "至今"
            base_text += f"\n时效: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (已过期: {self.expired_at})"
        
        return base_text
    
    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        return self.expired_at is not None
    
    @property
    def is_invalid(self) -> bool:
        """是否已失效"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """深度洞察检索结果"""
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }
    
    def to_text(self) -> str:
        text_parts = [
            f"## 未来预测深度分析",
            f"分析问题: {self.query}",
            f"预测场景: {self.simulation_requirement}",
            f"\n### 预测数据统计",
            f"- 相关预测事实: {self.total_facts}条",
            f"- 涉及实体: {self.total_entities}个",
            f"- 关系链: {self.total_relationships}条"
        ]
        if self.sub_queries:
            text_parts.append(f"\n### 分析的子问题")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")
        if self.semantic_facts:
            text_parts.append(f"\n### 【关键事实】")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        if self.entity_insights:
            text_parts.append(f"\n### 【核心实体】")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', '未知')}** ({entity.get('type', '实体')})")
                if entity.get('summary'):
                    text_parts.append(f"  摘要: \"{entity.get('summary')}\"")
        if self.relationship_chains:
            text_parts.append(f"\n### 【关系链】")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")
        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """广度搜索结果"""
    query: str
    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)
    historical_facts: List[str] = field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }
    
    def to_text(self) -> str:
        text_parts = [
            f"## 广度搜索结果",
            f"查询: {self.query}",
            f"\n### 统计信息",
            f"- 总节点数: {self.total_nodes}",
            f"- 总边数: {self.total_edges}",
            f"- 当前有效事实: {self.active_count}条",
            f"- 历史/过期事实: {self.historical_count}条"
        ]
        if self.active_facts:
            text_parts.append(f"\n### 【当前有效事实】")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        if self.historical_facts:
            text_parts.append(f"\n### 【历史/过期事实】")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        if self.all_nodes:
            text_parts.append(f"\n### 【涉及实体】")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["EntityNode", "Node"]), "实体")
                text_parts.append(f"- **{node.name}** ({entity_type})")
        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """单个Agent的采访结果"""
    agent_name: str
    agent_role: str
    agent_bio: str
    question: str
    response: str
    key_quotes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }
    
    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        text += f"_简介: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**关键引言:**\n"
            for quote in self.key_quotes:
                text += f'> "{quote}"\n'
        return text


@dataclass
class InterviewResult:
    """采访结果"""
    interview_topic: str
    interview_questions: List[str]
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    interviews: List[AgentInterview] = field(default_factory=list)
    selection_reasoning: str = ""
    summary: str = ""
    total_agents: int = 0
    interviewed_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }
    
    def to_text(self) -> str:
        text_parts = [
            "## 深度采访报告",
            f"**采访主题:** {self.interview_topic}",
            f"**采访人数:** {self.interviewed_count} / {self.total_agents} 位模拟Agent",
            "\n### 采访对象选择理由",
            self.selection_reasoning or "（自动选择）",
            "\n---",
            "\n### 采访实录",
        ]
        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### 采访 #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("（无采访记录）\n\n---")
        text_parts.append("\n### 采访摘要与核心观点")
        text_parts.append(self.summary or "（无摘要）")
        return "\n".join(text_parts)


class GraphitiToolsService:
    """
    Graphiti检索工具服务
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.neo4j_uri = Config.NEO4J_URI
        self.neo4j_user = Config.NEO4J_USER
        self.neo4j_password = Config.NEO4J_PASSWORD
        
        if not self.neo4j_uri:
            raise ValueError("NEO4J_URI 未配置")
            
        self._llm_client = llm_client
        self.reader = GraphitiEntityReader()
        logger.info("GraphitiToolsService 初始化完成")
        
    def _get_client(self) -> Graphiti:
        return Graphiti(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password
        )
    
    @property
    def llm(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client
    
    def search_graph(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """图谱语义搜索"""
        logger.info(f"图谱搜索: graph_id={graph_id}, query={query[:50]}...")
        
        async def _search():
            client = self._get_client()
            try:
                search_results = await client.search(
                    query=query,
                    group_ids=[graph_id],
                    num_results=limit
                )
                
                facts = []
                edges = []
                nodes = []
                
                for edge in search_results:
                    if getattr(edge, 'fact', None):
                        facts.append(edge.fact)
                    edges.append({
                        "uuid": getattr(edge, 'uuid', ''),
                        "name": getattr(edge, 'name', ''),
                        "fact": getattr(edge, 'fact', ''),
                        "source_node_uuid": getattr(edge, 'source_node_uuid', ''),
                        "target_node_uuid": getattr(edge, 'target_node_uuid', ''),
                    })
                
                return SearchResult(
                    facts=facts,
                    edges=edges,
                    nodes=nodes,
                    query=query,
                    total_count=len(facts)
                )
            finally:
                await client.close()
                
        try:
            return asyncio.run(_search())
        except Exception as e:
            logger.warning(f"Graphiti Search API失败，降级为本地搜索: {str(e)}")
            return self._local_search(graph_id, query, limit, scope)
    
    def _local_search(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """本地关键词匹配搜索"""
        logger.info(f"使用本地搜索: query={query[:30]}...")
        
        facts = []
        edges_result = []
        nodes_result = []
        
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def match_score(text: str) -> int:
            if not text:
                return 0
            text_lower = text.lower()
            if query_lower in text_lower:
                return 100
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score
        
        try:
            if scope in ["edges", "both"]:
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))
                
                scored_edges.sort(key=lambda x: x[0], reverse=True)
                
                for score, edge in scored_edges[:limit]:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append({
                        "uuid": edge.uuid,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                    })
            
            if scope in ["nodes", "both"]:
                all_nodes = self.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.name) + match_score(node.summary)
                    if score > 0:
                        scored_nodes.append((score, node))
                
                scored_nodes.sort(key=lambda x: x[0], reverse=True)
                
                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "labels": node.labels,
                        "summary": node.summary,
                    })
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
        except Exception as e:
            logger.error(f"本地搜索失败: {str(e)}")
        
        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )
    
    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """获取图谱的所有节点"""
        nodes_data = self.reader.get_all_nodes(graph_id)
        return [NodeInfo(**n) for n in nodes_data]

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """获取图谱的所有边"""
        edges_data = self.reader.get_all_edges(graph_id)
        return [EdgeInfo(**e) for e in edges_data]
    
    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """获取单个节点的详细信息"""
        async def _get():
            client = self._get_client()
            try:
                records, _, _ = await client.driver.execute_query(
                    "MATCH (n:EntityNode) WHERE n.uuid = $uuid RETURN n",
                    uuid=node_uuid
                )
                if not records:
                    return None
                n = records[0]["n"]
                return NodeInfo(
                    uuid=n.get("uuid", ""),
                    name=n.get("name", ""),
                    labels=list(n.labels) if n.labels else [],
                    summary=n.get("summary", ""),
                    attributes=dict(n)
                )
            finally:
                await client.close()
                
        try:
            return asyncio.run(_get())
        except Exception as e:
            logger.error(f"获取节点详情失败: {str(e)}")
            return None
    
    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """获取节点相关的所有边"""
        edges_data = self.reader.get_node_edges(node_uuid)
        return [EdgeInfo(**e) for e in edges_data]
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str
    ) -> List[NodeInfo]:
        """按类型获取实体"""
        all_nodes = self.get_all_nodes(graph_id)
        return [n for n in all_nodes if entity_type in n.labels]
    
    def get_entity_summary(
        self, 
        graph_id: str, 
        entity_name: str
    ) -> Dict[str, Any]:
        """获取指定实体的关系摘要"""
        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )
        
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = next((n for n in all_nodes if n.name.lower() == entity_name.lower()), None)
        
        related_edges = []
        if entity_node:
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)
        
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }
    
    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """获取图谱的统计信息"""
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)
        
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["EntityNode", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1
        
        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1
        
        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }
    
    def get_simulation_context(
        self, 
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """获取模拟相关的上下文信息"""
        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )
        
        stats = self.get_graph_statistics(graph_id)
        all_nodes = self.get_all_nodes(graph_id)
        
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["EntityNode", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })
        
        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities)
        }
    
    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """深度洞察检索"""
        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )
        
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        
        all_facts = []
        all_edges = []
        seen_facts = set()
        
        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )
            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            all_edges.extend(search_result.edges)
        
        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)
        
        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)
        
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)
        
        entity_insights = []
        node_map = {}
        
        for uuid in list(entity_uuids):
            if not uuid:
                continue
            try:
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((l for l in node.labels if l not in ["EntityNode", "Node"]), "实体")
                    related_facts = [f for f in all_facts if node.name.lower() in f.lower()]
                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts
                    })
            except Exception as e:
                logger.debug(f"获取节点 {uuid} 失败: {e}")
                continue
        
        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)
        
        relationship_chains = []
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')
                
                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]
                
                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)
        
        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)
        
        return result
    
    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """使用LLM生成子问题"""
        system_prompt = """你是一个专业的问题分析专家。你的任务是将一个复杂问题分解为多个可以在模拟世界中独立观察的子问题。

要求：
1. 每个子问题应该足够具体，可以在模拟世界中找到相关的Agent行为或事件
2. 子问题应该覆盖原问题的不同维度（如：谁、什么、为什么、怎么样、何时、何地）
3. 子问题应该与模拟场景相关
4. 返回JSON格式：{"sub_queries": ["子问题1", "子问题2", ...]}"""

        user_prompt = f"""模拟需求背景：
{simulation_requirement}

{f"报告上下文：{report_context[:500]}" if report_context else ""}

请将以下问题分解为{max_queries}个子问题：
{query}

返回JSON格式的子问题列表。"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            sub_queries = response.get("sub_queries", [])
            return [str(sq) for sq in sub_queries[:max_queries]]
            
        except Exception as e:
            logger.warning(f"生成子问题失败: {str(e)}，使用默认子问题")
            return [
                query,
                f"{query} 的主要参与者",
                f"{query} 的原因和影响",
                f"{query} 的发展过程"
            ][:max_queries]
    
    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        """广度搜索"""
        result = PanoramaResult(query=query)
        
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)
        
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)
        
        active_facts = []
        historical_facts = []
        
        for edge in all_edges:
            if not edge.fact:
                continue
            
            is_historical = edge.is_expired or edge.is_invalid
            
            if is_historical:
                valid_at = edge.valid_at or "未知"
                invalid_at = edge.invalid_at or edge.expired_at or "未知"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                active_facts.append(edge.fact)
        
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score
        
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)
        
        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)
        
        return result
    
    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """简单搜索"""
        return self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )
    
    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        """深度采访"""
        from .simulation_runner import SimulationRunner
        
        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )
        
        profiles = self._load_agent_profiles(simulation_id)
        
        if not profiles:
            result.summary = "未找到可采访的Agent人设文件"
            return result
        
        result.total_agents = len(profiles)
        
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )
        
        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
        
        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])
        
        INTERVIEW_PROMPT_PREFIX = (
            "你正在接受一次采访。请结合你的人设、所有的过往记忆与行动，"
            "以纯文本方式直接回答以下问题。\n"
            "回复要求：\n"
            "1. 直接用自然语言回答，不要调用任何工具\n"
            "2. 不要返回JSON格式或工具调用格式\n"
            "3. 不要使用Markdown标题（如#、##、###）\n"
            "4. 按问题编号逐一回答，每个回答以「问题X：」开头（X为问题编号）\n"
            "5. 每个问题的回答之间用空行分隔\n"
            "6. 回答要有实质内容，每个问题至少回答2-3句话\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"
        
        try:
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt
                })
            
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,
                timeout=180.0
            )
            
            if not api_result.get("success", False):
                error_msg = api_result.get("error", "未知错误")
                result.summary = f"采访API调用失败: {error_msg}"
                return result
            
            api_interviews = api_result.get("interviews", [])
            
            for i, api_interview in enumerate(api_interviews):
                agent_idx = api_interview.get("agent_id")
                response_text = api_interview.get("response", "")
                
                agent_profile = next((p for p in profiles if p.get("id") == agent_idx), None)
                if not agent_profile:
                    continue
                
                key_quotes = self._extract_key_quotes(response_text)
                
                interview = AgentInterview(
                    agent_name=agent_profile.get("name", f"Agent {agent_idx}"),
                    agent_role=agent_profile.get("role", "未知角色"),
                    agent_bio=agent_profile.get("bio", ""),
                    question="综合采访问题",
                    response=response_text,
                    key_quotes=key_quotes
                )
                result.interviews.append(interview)
            
            result.interviewed_count = len(result.interviews)
            
            if result.interviews:
                result.summary = self._summarize_interviews(
                    interview_requirement=interview_requirement,
                    interviews=result.interviews
                )
            else:
                result.summary = "所有Agent均未返回有效回答"
                
        except Exception as e:
            logger.error(f"采访过程发生异常: {str(e)}")
            result.summary = f"采访过程发生异常: {str(e)}"
            
        return result
        
    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """加载模拟的Agent人设文件"""
        import os
        from ..models.project import ProjectManager
        
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
        profiles_file = os.path.join(sim_dir, "agent_profiles.json")
        
        if os.path.exists(profiles_file):
            try:
                with open(profiles_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"读取人设文件失败: {str(e)}")
        
        return []
        
    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int = 5
    ) -> tuple[List[Dict[str, Any]], List[int], str]:
        """使用LLM选择最适合采访的Agent"""
        if len(profiles) <= max_agents:
            return profiles, [p.get("id") for p in profiles], "Agent总数较少，全选"
            
        profiles_summary = []
        for p in profiles:
            profiles_summary.append(
                f"ID: {p.get('id')}, 姓名: {p.get('name')}, 角色: {p.get('role')}, 简介: {p.get('bio')[:100]}..."
            )
            
        system_prompt = """你是一个专业的记者和调研专家。你的任务是从一组模拟人物中，挑选出最适合回答特定采访问题的人选。
要求：
1. 选择能够提供不同视角、有代表性的人物
2. 优先选择与采访主题最相关的人物
3. 返回JSON格式：{"selected_ids": [1, 2, 3], "reasoning": "选择这些人的理由..."}"""

        user_prompt = f"""模拟背景：{simulation_requirement}
采访需求：{interview_requirement}
需要挑选的人数：最多 {max_agents} 人

候选人物列表：
{chr(10).join(profiles_summary)}

请挑选最合适的人物，并说明理由。"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            selected_ids = response.get("selected_ids", [])
            reasoning = response.get("reasoning", "")
            
            selected_profiles = [p for p in profiles if p.get("id") in selected_ids]
            
            if not selected_profiles:
                selected_profiles = profiles[:max_agents]
                selected_ids = [p.get("id") for p in selected_profiles]
                reasoning = "LLM未返回有效ID，默认选择前几个"
                
            return selected_profiles, selected_ids, reasoning
            
        except Exception as e:
            logger.warning(f"选择Agent失败: {str(e)}，使用默认选择")
            selected_profiles = profiles[:max_agents]
            return selected_profiles, [p.get("id") for p in selected_profiles], "选择过程出错，默认选择前几个"
            
    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """生成采访问题"""
        agent_roles = [p.get("role", "未知") for p in selected_agents]
        
        system_prompt = """你是一个资深记者。请根据采访需求，设计3-5个犀利、有深度、能引发思考的采访问题。
要求：
1. 问题要具体，避免假大空
2. 问题要能引导受访者表达真实观点和情感
3. 返回JSON格式：{"questions": ["问题1", "问题2", ...]}"""

        user_prompt = f"""模拟背景：{simulation_requirement}
采访需求：{interview_requirement}
受访者角色：{', '.join(agent_roles)}

请设计3-5个采访问题。"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )
            
            questions = response.get("questions", [])
            if questions:
                return [str(q) for q in questions]
        except Exception as e:
            logger.warning(f"生成采访问题失败: {str(e)}")
            
        return [
            f"关于'{interview_requirement}'，你有什么看法？",
            "在这个事件中，你最关注的是什么？",
            "你认为接下来会如何发展？"
        ]
        
    def _extract_key_quotes(self, text: str) -> List[str]:
        """从回答中提取关键引言"""
        system_prompt = """你是一个编辑。请从受访者的回答中，提取1-2句最精彩、最能代表其核心观点的原话（金句）。
要求：
1. 必须是原文中的句子，不要改写
2. 句子要完整，有冲击力或深刻见解
3. 返回JSON格式：{"quotes": ["金句1", "金句2"]}"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请提取以下回答的金句：\n\n{text}"}
                ],
                temperature=0.3
            )
            
            return response.get("quotes", [])
        except Exception:
            return []
            
    def _summarize_interviews(
        self,
        interview_requirement: str,
        interviews: List[AgentInterview]
    ) -> str:
        """整合所有采访结果生成摘要"""
        interview_texts = []
        for i in interviews:
            interview_texts.append(f"受访者：{i.agent_name} ({i.agent_role})\n回答：{i.response}")
            
        system_prompt = """你是一个资深新闻主编。请根据多位受访者的回答，撰写一份综合的采访摘要。
要求：
1. 总结核心共识和主要分歧
2. 提炼出最有价值的洞察
3. 结构清晰，客观中立
4. 直接输出Markdown格式的文本，不要输出JSON"""

        user_prompt = f"""采访主题：{interview_requirement}

采访记录：
{chr(10).join(interview_texts)}

请撰写综合采访摘要。"""

        try:
            return self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )
        except Exception as e:
            logger.warning(f"生成采访摘要失败: {str(e)}")
            return "无法生成摘要。"
