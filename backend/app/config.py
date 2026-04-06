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


class Config:
    """Flask配置类"""

    # Flask配置
    SECRET_KEY = os.environ.get("SECRET_KEY", "mirofish-secret-key")
    DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() == "true"

    # JSON配置 - 禁用ASCII转义，让中文直接显示（而不是 \uXXXX 格式）
    JSON_AS_ASCII = False

    # LLM配置（统一使用OpenAI格式）
    LLM_API_KEY = os.environ.get("LLM_API_KEY")
    LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")

    GRAPHITI_API_KEY = os.environ.get("GRAPHITI_API_KEY", LLM_API_KEY)
    GRAPHITI_BASE_URL = os.environ.get("GRAPHITI_BASE_URL", LLM_BASE_URL)
    GRAPHITI_MODEL_NAME = os.environ.get("GRAPHITI_MODEL_NAME", LLM_MODEL_NAME)
    GRAPHITI_SMALL_MODEL_NAME = os.environ.get(
        "GRAPHITI_SMALL_MODEL_NAME", GRAPHITI_MODEL_NAME
    )
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
    GRAPHITI_MAX_CHUNKS = int(os.environ.get("GRAPHITI_MAX_CHUNKS", "10"))
    GRAPHITI_ONTOLOGY_CHUNK_THRESHOLD = int(
        os.environ.get("GRAPHITI_ONTOLOGY_CHUNK_THRESHOLD", "40")
    )
    GRAPHITI_MAX_ENTITY_TYPES = int(os.environ.get("GRAPHITI_MAX_ENTITY_TYPES", "6"))
    GRAPHITI_MAX_ENTITY_ATTRIBUTES = int(
        os.environ.get("GRAPHITI_MAX_ENTITY_ATTRIBUTES", "3")
    )
    GRAPHITI_MAX_TOKENS = int(os.environ.get("GRAPHITI_MAX_TOKENS", "1024"))

    # Neo4j / Graphiti配置
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

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
    def validate(cls):
        """验证必要配置"""
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY 未配置")
        if not cls.NEO4J_URI:
            errors.append("NEO4J_URI 未配置")
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY 未配置")
        if not cls.GRAPHITI_API_KEY:
            errors.append("GRAPHITI_API_KEY 未配置")
        return errors
