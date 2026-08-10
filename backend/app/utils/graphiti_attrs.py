"""
Graphiti 节点/边的本体属性读取

graphiti-core 换过一次属性存储布局，这里把两种布局都吃掉：

- 0.11.x：每个属性摊平成独立的 Neo4j 属性（nodes.py 里 entity_data.update(self.attributes)）。
  嵌套 dict 写不进 Neo4j，会抛 CypherTypeError，当时靠 backend/patched_node_operations.py
  在写入前把非原始类型过滤掉，代价是那些属性直接丢了。
- 0.29.x：整个属性表 json.dumps 进单个 attributes 字段（nodes.py:563 / edges.py:359），
  读回时 json.loads。嵌套结构得以保留，补丁也就不需要了。

用同一个函数读，旧图不必重建。
"""

import json
from typing import Any, Dict

from .logger import get_logger

logger = get_logger("mirofish.graphiti_attrs")

# graphiti-core 自己写在节点/边上的字段，不属于本体定义的属性
GRAPHITI_RESERVED_FIELDS = frozenset(
    {
        "uuid",
        "name",
        "group_id",
        "summary",
        "fact",
        "created_at",
        "valid_at",
        "invalid_at",
        "expired_at",
        "episodes",
        "name_embedding",
        "fact_embedding",
        "attributes",
    }
)


def read_attributes(entity: Any) -> Dict[str, Any]:
    """从 Neo4j 节点或边上取出本体属性。

    entity 是 neo4j driver 返回的 Node / Relationship，dict(entity) 得到它的全部属性。
    """
    props = dict(entity)
    raw = props.get("attributes")

    if isinstance(raw, dict):
        return raw

    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning(f"attributes 不是合法 JSON，按旧布局回退: {raw[:80]}")
        else:
            if isinstance(parsed, dict):
                return parsed
            logger.warning(
                f"attributes 解出来不是 dict（{type(parsed).__name__}），按旧布局回退"
            )

    # 旧布局：属性摊平在节点上，剔掉 graphiti 自己的字段
    return {k: v for k, v in props.items() if k not in GRAPHITI_RESERVED_FIELDS}
