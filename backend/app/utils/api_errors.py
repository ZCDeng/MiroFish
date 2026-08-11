"""
API 错误响应

原来 51 个接口的 except 分支长这样：

    return jsonify({
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc(),
    }), 500

traceback 里有服务端绝对路径、目录结构、每一帧的局部变量名，还常常带着
异常消息本身 —— 而异常消息可能包含 LLM 的原始返回或上传文档的片段。
这些整包发给了浏览器。

现在 traceback 只写服务端日志，响应里给一个 error_id，出问题时拿它去日志里
捞对应的那条。异常消息仍然回传（用户需要知道哪里错了），但截断到 200 字符，
避免把整段模型输出漏出去。
"""

import traceback
import uuid
from typing import Any, Dict, Optional, Tuple

from flask import jsonify

from .logger import get_logger

logger = get_logger("mirofish.api")

# 异常消息回传给前端的长度上限
MAX_ERROR_MESSAGE_CHARS = 200


def error_response(
    exc: BaseException,
    message: Optional[str] = None,
    status: int = 500,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, int]:
    """记录完整 traceback，返回不含 traceback 的错误响应。

    Args:
        exc: 捕获到的异常
        message: 给用户看的文案。留空则用截断后的 str(exc)
        status: HTTP 状态码
        extra: additional fields merged into the payload

    Returns:
        (flask response, status) —— 可以直接 `return error_response(e)`
    """
    error_id = uuid.uuid4().hex[:12]
    logger.error(
        "[%s] %s: %s\n%s",
        error_id,
        type(exc).__name__,
        exc,
        traceback.format_exc(),
    )

    text = message if message is not None else str(exc)
    if len(text) > MAX_ERROR_MESSAGE_CHARS:
        text = text[:MAX_ERROR_MESSAGE_CHARS] + "…"

    payload: Dict[str, Any] = {
        "success": False,
        "error": text,
        "error_id": error_id,
    }
    if extra:
        payload.update(extra)
    return jsonify(payload), status
