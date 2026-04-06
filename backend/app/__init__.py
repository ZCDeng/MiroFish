"""
MiroFish Backend - Flask应用工厂
"""

import os
import warnings

# 抑制 multiprocessing resource_tracker 的警告（来自第三方库如 transformers）
# 需要在所有其他导入之前设置
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, request
from flask_cors import CORS

from .config import Config
from .utils.logger import setup_logger, get_logger


def _recover_orphaned_graph_building_projects():
    from .models.project import ProjectManager, ProjectStatus
    from .models.task import TaskManager, TaskStatus

    task_manager = TaskManager()
    projects = ProjectManager.list_projects(limit=1000)
    for project in projects:
        if project.status != ProjectStatus.GRAPH_BUILDING:
            continue

        task_id = project.graph_build_task_id
        task = task_manager.get_task(task_id) if task_id else None
        # 任务不存在，或任务已持久化为失败状态，都需要修正项目状态
        if task is not None and task.status not in (TaskStatus.FAILED,):
            continue

        project.status = ProjectStatus.FAILED
        if not project.error:
            project.error = (
                "服务重启导致构建任务上下文丢失，已自动标记失败，请重新发起构建。"
            )
        ProjectManager.save_project(project)


def create_app(config_class=Config):
    """Flask应用工厂函数"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # 设置JSON编码：确保中文直接显示（而不是 \uXXXX 格式）
    # Flask >= 2.3 使用 app.json.ensure_ascii，旧版本使用 JSON_AS_ASCII 配置
    if hasattr(app, "json") and hasattr(app.json, "ensure_ascii"):
        setattr(app.json, "ensure_ascii", False)

    # 设置日志
    logger = setup_logger("mirofish")

    # 只在 reloader 子进程中打印启动信息（避免 debug 模式下打印两次）
    is_reloader_process = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    debug_mode = app.config.get("DEBUG", False)
    should_log_startup = not debug_mode or is_reloader_process

    if should_log_startup:
        logger.info("=" * 50)
        logger.info("MiroFish Backend 启动中...")
        logger.info("=" * 50)

    # 启用CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # 注册模拟进程清理函数（确保服务器关闭时终止所有模拟进程）
    from .services.simulation_runner import SimulationRunner

    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("已注册模拟进程清理函数")

    # 请求日志中间件
    @app.before_request
    def log_request():
        logger = get_logger("mirofish.request")
        logger.debug(f"请求: {request.method} {request.path}")
        if request.content_type and "json" in request.content_type:
            logger.debug(f"请求体: {request.get_json(silent=True)}")

    @app.after_request
    def log_response(response):
        logger = get_logger("mirofish.request")
        logger.debug(f"响应: {response.status_code}")
        return response

    # 注册蓝图
    from .api import graph_bp, simulation_bp, report_bp

    app.register_blueprint(graph_bp, url_prefix="/api/graph")
    app.register_blueprint(simulation_bp, url_prefix="/api/simulation")
    app.register_blueprint(report_bp, url_prefix="/api/report")

    # 健康检查
    @app.route("/health")
    def health():
        return {"status": "ok", "service": "MiroFish Backend"}

    if should_log_startup:
        logger.info("MiroFish Backend 启动完成")

    _recover_orphaned_graph_building_projects()

    return app
