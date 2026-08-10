"""
Graphiti 节点/边的本体属性读取

graphiti-core 换过一次属性存储布局，这里把两种布局都吃掉：

Neo4j 上属性是摊平存的：EntityNode.save 把 self.attributes 的每个键写成节点自己的
一个属性（nodes.py:570-572）。0.29.3 里那句 json.dumps(self.attributes) 只在
driver.provider == KUZU 时走，Neo4j 不走，所以从 0.11 到 0.29 这一点没变。

顺带一个后果：嵌套 dict 依然写不进 Neo4j 属性，CypherTypeError 的风险还在。
graph_builder 里捕获该错误后用 entity_types=None 重跑整块的兜底仍然有意义。

读的时候要把 graphiti 自己写的字段剔掉。注意 labels 也在其中 —— 它既是真正的
Neo4j label，又被当成普通属性存了一份。

json 分支留着是为了兼容将来换 driver，以及万一有人手工塞了 attributes 字段。
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
        "entity_edges",
        "name_embedding",
        "fact_embedding",
        "attributes",
        # graphiti 把 labels 既写成真正的 Neo4j label，又存了一份同名属性。
        # 不剔掉的话它会混进业务属性里。
        "labels",
        "episode_type",
        "source",
        "source_description",
        "content",
    }
)


# 出现这些说明 JSON 结构漏进了值里，可以断定这个值被解析错位污染了
_STRUCTURE_LEAKS = ("```", '"}', "”:", "”,", '”}')
# 确认被污染之后，再回溯到最早的全角引号处截断 —— 错位就是从那里开始的。
# 分两段判断是为了不误伤正常内容里带引号的中文值。
_QUOTE_MARKS = ("“", "”")


def clean_attr_value(value: Any) -> Any:
    """砍掉属性值里因 JSON 解析错位卷进来的垃圾。

    模型写中文时会用全角引号：{"headquarters": "旧金山“, ”founding_year”: “2023”}。
    `“` 不是合法的 JSON 字符串定界符，json.loads 不报错，但会一路读到下一个真正的
    双引号，把后面所有内容（包括 markdown 围栏里的 JSON 块）当成同一个字符串值。

    实测存进库的值长这样：
        '旧金山“, ”founding_year”: “2023”}```json\\n{...}```...'

    这里在最早出现的垃圾标记处截断。治标不治本 —— 根子在写入侧，模型返回的
    JSON 需要在 json.loads 之前把全角引号规范化，但那要覆写 graphiti 的 LLM
    客户端，而那层刚因为过度包装被删掉。先在读取侧兜住，别让垃圾流到下游。
    """
    if not isinstance(value, str):
        return value

    leak = min(
        (value.find(m) for m in _STRUCTURE_LEAKS if value.find(m) != -1),
        default=-1,
    )
    if leak == -1:
        return value

    # 错位是从第一个全角引号开始的，比结构标记更靠前
    quote = min(
        (value.find(q) for q in _QUOTE_MARKS if value.find(q) != -1),
        default=-1,
    )
    cut = quote if 0 <= quote < leak else leak

    cleaned = value[:cut].strip().rstrip("“”\"',：:")
    logger.warning(f"属性值疑似 JSON 解析错位，已截断: {value[:60]!r} → {cleaned!r}")
    return cleaned


def read_attributes(entity: Any) -> Dict[str, Any]:
    """从 Neo4j 节点或边上取出本体属性。

    entity 是 neo4j driver 返回的 Node / Relationship，dict(entity) 得到它的全部属性。
    """
    props = dict(entity)
    raw = props.get("attributes")

    if isinstance(raw, dict):
        return {k: clean_attr_value(v) for k, v in raw.items()}

    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning(f"attributes 不是合法 JSON，按旧布局回退: {raw[:80]}")
        else:
            if isinstance(parsed, dict):
                return {k: clean_attr_value(v) for k, v in parsed.items()}
            logger.warning(
                f"attributes 解出来不是 dict（{type(parsed).__name__}），按旧布局回退"
            )

    # 旧布局：属性摊平在节点上，剔掉 graphiti 自己的字段
    return {
        k: clean_attr_value(v)
        for k, v in props.items()
        if k not in GRAPHITI_RESERVED_FIELDS
    }
