"""Helpers for validating LLM-generated ontology structures."""

from typing import Any, Dict, List, Optional


MAX_ONTOLOGY_TYPES = 10
MAX_ONTOLOGY_ATTRIBUTES = 10
MAX_ONTOLOGY_SOURCE_TARGETS = 10
# graphiti-core 自己写在节点和边上的字段。本体属性重名的话，
# EntityNode.save 的 `if k not in entity_data` 会让那条属性被**静默丢弃**
# （nodes.py:570-572），属性等于白定义。
# 上游 d2f9e56 用加前缀改写来解决；名单按 graphiti 的 EntityNode / EntityEdge
# 实际字段重列，Zep 那份跟这边对不上。
RESERVED_ONTOLOGY_ATTRIBUTE_NAMES = frozenset({
    # EntityNode
    "uuid", "name", "group_id", "labels", "summary", "created_at",
    "name_embedding", "attributes",
    # EntityEdge 额外的
    "fact", "fact_embedding", "episodes", "expired_at", "invalid_at",
    "valid_at", "reference_time", "source_node_uuid", "target_node_uuid",
    # graphiti_attrs 里也按保留字剔掉的
    "episode_type", "source", "source_description", "content", "entity_edges",
})
# 撞名时加的前缀
RESERVED_ATTRIBUTE_PREFIX = "attr_"

_FALLBACK_ATTRIBUTE = {
    "name": "details",
    "type": "text",
    "description": "Additional details about this ontology type.",
}


def normalize_ontology_attribute(attribute: Any) -> Optional[Dict[str, Any]]:
    """Return a safe attribute definition, or ``None`` for unusable values."""

    if isinstance(attribute, str):
        if not attribute.strip():
            return None
        return {
            "name": _safe_attribute_name(attribute),
            "type": "text",
            "description": attribute,
        }

    if not isinstance(attribute, dict):
        return None

    name = attribute.get("name")
    if not isinstance(name, str) or not name.strip():
        return None

    normalized = dict(attribute)
    normalized["name"] = _safe_attribute_name(name)
    description = normalized.get("description")
    if not isinstance(description, str) or not description:
        normalized["description"] = name
    return normalized


def _safe_attribute_name(name: str) -> str:
    """撞 graphiti 保留字的属性名加前缀改写。

    上游只定义了 RESERVED_ONTOLOGY_ATTRIBUTE_NAMES 却没有任何地方使用它，
    所以那份名单一直是摆设。这里补上实际的改写。
    """
    if name.strip().lower() in RESERVED_ONTOLOGY_ATTRIBUTE_NAMES:
        return f"{RESERVED_ATTRIBUTE_PREFIX}{name.strip()}"
    return name


def normalize_ontology_attributes(attributes: Any) -> List[Dict[str, Any]]:
    """Return a non-empty Zep-compatible attribute list within service limits."""

    if not isinstance(attributes, list):
        attributes = []

    normalized_attributes: List[Dict[str, Any]] = []
    for attribute in attributes:
        normalized = normalize_ontology_attribute(attribute)
        if normalized is None:
            continue
        normalized_attributes.append(normalized)
        if len(normalized_attributes) == MAX_ONTOLOGY_ATTRIBUTES:
            break

    if not normalized_attributes:
        normalized_attributes.append(dict(_FALLBACK_ATTRIBUTE))

    return normalized_attributes


def normalize_ontology_source_targets(
    source_targets: Any,
    *,
    limit: int | None = MAX_ONTOLOGY_SOURCE_TARGETS,
) -> List[Dict[str, str]]:
    """Return unique, structurally valid source-target pairs within Zep limits."""

    if not isinstance(source_targets, list):
        return []

    normalized_targets: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source_target in source_targets:
        if not isinstance(source_target, dict):
            continue
        source = source_target.get("source")
        target = source_target.get("target")
        if not isinstance(source, str) or not source.strip():
            continue
        if not isinstance(target, str) or not target.strip():
            continue

        pair = (source.strip(), target.strip())
        if pair in seen:
            continue
        seen.add(pair)
        normalized_targets.append({"source": pair[0], "target": pair[1]})
        if limit is not None and len(normalized_targets) == limit:
            break

    return normalized_targets
