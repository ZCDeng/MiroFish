"""
配置管理
统一从项目根目录的 .env 文件加载配置
"""

import os
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
# 路径: MiroFish/.env (相对于 backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), "../../.env")

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # 如果根目录没有 .env，尝试加载环境变量（用于生产环境）
    load_dotenv(override=True)

os.environ.setdefault(
    "SEMAPHORE_LIMIT", os.environ.get("GRAPHITI_SEMAPHORE_LIMIT", "5")
)

# graphiti-core 0.29 起自带 PostHog 遥测，默认开启，会把使用事件发到
# us.i.posthog.com。这里默认关掉；要开就在 .env 里显式设成 true。
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")


class Config:
    """Flask配置类"""

    # Flask配置
    SECRET_KEY = os.environ.get("SECRET_KEY", "mirofish-secret-key")
    # 默认关。开着的话 Werkzeug 交互式调试控制台会跟着起来，
    # 配合 run.py 默认绑 0.0.0.0 和 compose 映射 5002，等于对外开一个可执行任意 Python 的口子。
    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

    # JSON配置 - 禁用ASCII转义，让中文直接显示（而不是 \uXXXX 格式）
    JSON_AS_ASCII = False

    # LLM配置（统一使用OpenAI格式）
    LLM_API_KEY = os.environ.get("LLM_API_KEY")
    LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")
    # OASIS 模拟 Agent 专用的一整套 provider 配置。模拟是全链路里调用量最大的一段
    # （轮数 × agent 数），值得单独指一个便宜的 provider。
    # 三项各自回退到主 LLM，所以只改模型名也能用（同 provider 换小模型）。
    # 模拟脚本读的是 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME 三个环境变量
    # （run_parallel_simulation.py:1015-1017），simulation_runner 起子进程时会用
    # 这三项把它们覆盖掉。
    SIMULATION_AGENT_MODEL = os.environ.get("SIMULATION_AGENT_MODEL", LLM_MODEL_NAME)
    SIMULATION_AGENT_API_KEY = os.environ.get("SIMULATION_AGENT_API_KEY", LLM_API_KEY)
    SIMULATION_AGENT_BASE_URL = os.environ.get(
        "SIMULATION_AGENT_BASE_URL", LLM_BASE_URL
    )
    # chat_json 遇到「推理占满预算、没输出内容」时翻倍重试的上限。
    # 推理模型（DeepSeek v4 等）光思考就能吃掉好几千 token。
    LLM_MAX_TOKENS_CEILING = int(os.environ.get("LLM_MAX_TOKENS_CEILING", "16384"))

    GRAPHITI_API_KEY = os.environ.get("GRAPHITI_API_KEY", LLM_API_KEY)
    GRAPHITI_BASE_URL = os.environ.get("GRAPHITI_BASE_URL", LLM_BASE_URL)
    GRAPHITI_MODEL_NAME = os.environ.get("GRAPHITI_MODEL_NAME", LLM_MODEL_NAME)
    GRAPHITI_SMALL_MODEL_NAME = os.environ.get(
        "GRAPHITI_SMALL_MODEL_NAME", GRAPHITI_MODEL_NAME
    )
    # 结构化输出模式：json_schema 让服务端强制 schema，json_object 只保证是合法 JSON、
    # 由 graphiti 把 schema 拼进 prompt。不是所有 provider 都支持 json_schema ——
    # DeepSeek 官方 (api.deepseek.com) 会返回 400 "This response_format type is
    # unavailable now"，SiliconFlow 则支持。留空则按 base_url 自动判断。
    GRAPHITI_STRUCTURED_OUTPUT_MODE = os.environ.get(
        "GRAPHITI_STRUCTURED_OUTPUT_MODE", ""
    ).strip()

    GRAPHITI_REQUEST_TIMEOUT = float(os.environ.get("GRAPHITI_REQUEST_TIMEOUT", "30"))
    GRAPHITI_REQUEST_RETRIES = int(os.environ.get("GRAPHITI_REQUEST_RETRIES", "1"))
    GRAPHITI_EPISODE_TIMEOUT = float(os.environ.get("GRAPHITI_EPISODE_TIMEOUT", "25"))
    GRAPHITI_SEMAPHORE_LIMIT = int(os.environ.get("GRAPHITI_SEMAPHORE_LIMIT", "5"))
    GRAPHITI_BUILD_TIMEOUT = float(os.environ.get("GRAPHITI_BUILD_TIMEOUT", "120"))
    GRAPHITI_BUILD_TIMEOUT_CAP = float(
        os.environ.get("GRAPHITI_BUILD_TIMEOUT_CAP", "1800")
    )
    GRAPHITI_BATCH_SIZE = int(os.environ.get("GRAPHITI_BATCH_SIZE", "1"))
    GRAPHITI_CHUNK_MAX_CHARS = int(os.environ.get("GRAPHITI_CHUNK_MAX_CHARS", "1200"))
    GRAPHITI_MAX_CHUNKS = int(os.environ.get("GRAPHITI_MAX_CHUNKS", "5"))
    GRAPHITI_ONTOLOGY_CHUNK_THRESHOLD = int(
        os.environ.get("GRAPHITI_ONTOLOGY_CHUNK_THRESHOLD", "40")
    )
    GRAPHITI_MAX_ENTITY_TYPES = int(os.environ.get("GRAPHITI_MAX_ENTITY_TYPES", "6"))
    GRAPHITI_MAX_ENTITY_ATTRIBUTES = int(
        os.environ.get("GRAPHITI_MAX_ENTITY_ATTRIBUTES", "3")
    )
    GRAPHITI_MAX_TOKENS = int(os.environ.get("GRAPHITI_MAX_TOKENS", "8192"))
    # 索引没建成时是否拒绝继续写图。默认拒绝 —— 缺全文索引的图谱看着正常，
    # 但混合检索的 BM25 那一半是废的，属于最难发现的一类故障。
    GRAPHITI_REQUIRE_INDICES = os.environ.get(
        "GRAPHITI_REQUIRE_INDICES", "true"
    ).lower() == "true"
    # 读整张图时的行数上限。graphiti_entity_reader 的几条 Cypher 原来没有 LIMIT，
    # 把全部节点和边一次性 materialize 成 Python list，图一大就是内存爆炸。
    # 上游给 Zep 版的 fetch_all_edges 加过同样的保护（e58d4f1）。
    # 命中上限会打 warning，不会静默截断。
    GRAPHITI_MAX_GRAPH_ROWS = int(os.environ.get("GRAPHITI_MAX_GRAPH_ROWS", "5000"))

    # Neo4j / Graphiti配置
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    GRAPHITI_EMBEDDER_API_KEY = os.environ.get("GRAPHITI_EMBEDDER_API_KEY", os.environ.get("OPENAI_API_KEY"))
    GRAPHITI_EMBEDDER_BASE_URL = os.environ.get("GRAPHITI_EMBEDDER_BASE_URL")
    GRAPHITI_EMBEDDER_MODEL = os.environ.get("GRAPHITI_EMBEDDER_MODEL", "text-embedding-3-small")

    # 文件上传配置
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "../uploads")
    ALLOWED_EXTENSIONS = {"pdf", "md", "txt", "markdown"}

    # 文本处理配置
    DEFAULT_CHUNK_SIZE = 500  # 默认切块大小
    DEFAULT_CHUNK_OVERLAP = 50  # 默认重叠大小

    # OASIS模拟配置
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get("OASIS_DEFAULT_MAX_ROUNDS", "10"))
    OASIS_SIMULATION_DATA_DIR = os.path.join(
        os.path.dirname(__file__), "../uploads/simulations"
    )

    # OASIS平台可用动作配置
    OASIS_TWITTER_ACTIONS = [
        "CREATE_POST",
        "LIKE_POST",
        "REPOST",
        "FOLLOW",
        "DO_NOTHING",
        "QUOTE_POST",
    ]
    OASIS_REDDIT_ACTIONS = [
        "LIKE_POST",
        "DISLIKE_POST",
        "CREATE_POST",
        "CREATE_COMMENT",
        "LIKE_COMMENT",
        "DISLIKE_COMMENT",
        "SEARCH_POSTS",
        "SEARCH_USER",
        "TREND",
        "REFRESH",
        "DO_NOTHING",
        "FOLLOW",
        "MUTE",
    ]

    # Report Agent配置
    REPORT_AGENT_MAX_TOOL_CALLS = int(
        os.environ.get("REPORT_AGENT_MAX_TOOL_CALLS", "5")
    )
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(
        os.environ.get("REPORT_AGENT_MAX_REFLECTION_ROUNDS", "2")
    )
    REPORT_AGENT_TEMPERATURE = float(os.environ.get("REPORT_AGENT_TEMPERATURE", "0.5"))

    @classmethod
    def validate(cls) -> list[str]:
        """验证必要配置"""
        errors: list[str] = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY 未配置")
        if not cls.NEO4J_URI:
            errors.append("NEO4J_URI 未配置")
        if not cls.GRAPHITI_API_KEY:
            errors.append("GRAPHITI_API_KEY 未配置")
        # OPENAI_API_KEY 不再校验：以前 Graphiti 客户端不注入 llm_client 时会就地
        # new 一个 OpenAIClient，构造时要读这个 key。现在四处构造都走
        # build_graphiti_client() 注入 SiliconFlow 的客户端，没人再读它。
        if not (cls.GRAPHITI_EMBEDDER_API_KEY and cls.GRAPHITI_EMBEDDER_BASE_URL):
            errors.append(
                "GRAPHITI_EMBEDDER_API_KEY / GRAPHITI_EMBEDDER_BASE_URL 未配置（向量检索依赖）"
            )
        errors.extend(cls._validate_simulation_provider())
        return errors

    # 模拟层 provider 的三项。key 和 base_url 必须同进同退：
    # 只设其中一个，就会拿 A 家的 key 去打 B 家的端点。
    _SIMULATION_PROVIDER_PAIR = (
        "SIMULATION_AGENT_API_KEY",
        "SIMULATION_AGENT_BASE_URL",
    )

    @classmethod
    def _validate_simulation_provider(cls) -> list[str]:
        """检查模拟 provider 配置的一致性。

        模型名可以单独换（同 provider 换个小模型是常见需求），但 key 和
        base_url 必须成对出现，否则要到子进程真正发请求时才以 401/404 暴露，
        而那时错误信息只说认证失败，不会指向配置。
        """
        import os as _os

        present = [k for k in cls._SIMULATION_PROVIDER_PAIR if _os.environ.get(k)]
        if present and len(present) != len(cls._SIMULATION_PROVIDER_PAIR):
            missing = [k for k in cls._SIMULATION_PROVIDER_PAIR if k not in present]
            return [
                f"模拟 provider 配置不完整：已设 {', '.join(present)}，"
                f"缺 {', '.join(missing)}。key 和 base_url 必须同时指向同一家，"
                "否则会拿一家的密钥去打另一家的端点。"
                "要沿用主 LLM 就三项都别设。"
            ]

        # 只换模型名而 key/base_url 继承主 LLM 时，提醒模型名得是主 provider 认识的
        model_overridden = bool(_os.environ.get("SIMULATION_AGENT_MODEL"))
        if model_overridden and not present:
            if cls.SIMULATION_AGENT_BASE_URL != cls.LLM_BASE_URL:
                return [
                    "SIMULATION_AGENT_MODEL 单独覆盖时，base_url 沿用主 LLM，"
                    f"请确认 {cls.LLM_BASE_URL} 提供 {cls.SIMULATION_AGENT_MODEL}"
                ]
        return []
