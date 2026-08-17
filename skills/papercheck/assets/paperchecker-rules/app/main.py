import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.auth_routes import router as auth_router
from auth.dependencies import enforce_auth_for_request
from auth.service import AuthService
from config.config import settings
from contracts.v2_contract import build_analysis_contract, build_contract, build_error
from services.analysis_service import AnalysisService
from services.exceptions import ServiceAuthError, ServiceValidationError
from services.relevance_service import RelevanceService
from services.report_service import run_smoke_suite
from services.workspace_service import WorkspaceService

# 最大上传文件大小
MAX_FILE_SIZE = settings.max_upload_size

app = FastAPI(
    title="PaperChecker API",
    description="学术论文引用合规性检查API - 重构版",
    version="2.0.0",
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 目录基线
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, settings.temp_dir))
os.makedirs(TEMP_DIR, exist_ok=True)

# 账号系统初始化
AuthService.bootstrap()

# 挂载 auth 路由（独立账号系统）
app.include_router(auth_router)


# 挂载静态文件目录到 /frontend 路径（旧前端）
FRONT_WEB_DIR = os.path.join(PROJECT_ROOT, "front", "web")
if os.path.isdir(FRONT_WEB_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONT_WEB_DIR, html=True), name="frontend")

# 他山设计系统前端（guofeng 主题，单文件 HTML + CSS）
TASHAN_UI_DIR = os.path.join(PROJECT_ROOT, "front", "tashan-ui")
if os.path.isdir(TASHAN_UI_DIR):
    app.mount("/ui", StaticFiles(directory=TASHAN_UI_DIR, html=True), name="frontend-tashan")

# 新前端双轨：仅在构建产物存在时挂载
FRONTEND_V2_DIST = os.path.join(PROJECT_ROOT, "frontend", "dist")
if os.path.isdir(FRONTEND_V2_DIST):
    app.mount("/frontend-v2", StaticFiles(directory=FRONTEND_V2_DIST, html=True), name="frontend-v2")


def _get_request_user(request: Optional[Request]):
    if request is None:
        return None
    return getattr(request.state, "current_user", None)


workspace_service = WorkspaceService(
    project_root=PROJECT_ROOT,
    temp_dir=TEMP_DIR,
    auth_enabled=lambda: settings.auth_enabled,
    user_resolver=_get_request_user,
)
analysis_service = AnalysisService(workspace_service, max_file_size=MAX_FILE_SIZE)
relevance_service = RelevanceService(workspace_service)


def _http_from_service_error(exc: ServiceValidationError) -> HTTPException:
    if isinstance(exc, ServiceAuthError):
        return HTTPException(status_code=401, detail=str(exc))

    msg = str(exc)
    if "不存在" in msg or "not found" in msg.lower():
        return HTTPException(status_code=404, detail=msg)

    return HTTPException(status_code=400, detail=msg)


def _get_user_scoped_dir(request: Optional[Request]) -> str:
    """兼容测试入口：返回当前用户可访问目录。"""
    try:
        return str(workspace_service.get_user_scoped_dir(request=request))
    except ServiceValidationError as exc:
        raise _http_from_service_error(exc)


def _resolve_scoped_path(request: Optional[Request], file_path: str) -> str:
    """兼容测试入口：解析并校验用户可访问文件路径。"""
    try:
        return str(workspace_service.resolve_scoped_path(file_path, request=request))
    except ServiceValidationError as exc:
        raise _http_from_service_error(exc)


@app.middleware("http")
async def auth_guard_middleware(request: Request, call_next):
    if settings.auth_enabled:
        path = request.url.path
        is_public = (
            path == "/"
            or path.startswith("/frontend")
            or path.startswith("/docs")
            or path.startswith("/redoc")
            or path.startswith("/openapi.json")
            or path.startswith("/api/health")
            or path.startswith("/api/auth")
        )
        if not is_public:
            try:
                enforce_auth_for_request(request)
            except HTTPException as exc:
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


@app.get("/")
def read_root():
    return {
        "message": "PaperChecker API - 重构版",
        "version": "2.0.0",
        "modules": ["extractor", "checker", "processor", "reports", "auth"],
    }


@app.get("/api/health")
def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "extractor": "available",
            "checker": "available",
            "processor": "available",
            "reports": "available",
        },
    }


# ---------------------------
# 兼容 API（保持 /api/*）
# ---------------------------


@app.post("/api/full-report")
async def get_full_citation_report(
    request: Request,
    file: UploadFile = File(...),
    author_format: str = Form("full"),
    citation_standard: str = Form("legacy"),
):
    """生成完整引文合规报告（兼容旧输出：raw_report）。"""
    try:
        analysis_result = analysis_service.analyze_upload(
            file=file,
            author_format=author_format,
            citation_standard=citation_standard,
            request=request,
        )
        return JSONResponse(content=analysis_result)
    except ServiceValidationError as exc:
        raise _http_from_service_error(exc)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"分析过程中出现错误: {str(exc)}")


@app.post("/api/full-report-from-path")
async def get_full_citation_report_from_path(
    request: Request = None,
    file_path: str = Form(...),
    author_format: str = Form("full"),
    citation_standard: str = Form("legacy"),
):
    """通过文件路径生成引文合规报告（兼容旧输出：raw_report）。"""
    try:
        analysis_result = analysis_service.analyze_path(
            file_path=file_path,
            author_format=author_format,
            citation_standard=citation_standard,
            request=request,
        )
        return JSONResponse(content=analysis_result)
    except ServiceValidationError as exc:
        raise _http_from_service_error(exc)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"分析过程中出现错误: {str(exc)}")


@app.post("/api/extract-citations")
async def extract_citations_form(request: Request, file_path: str = Form(None)):
    """从文档中提取引用文献（表单）。"""
    try:
        citations = relevance_service.extract_citations(file_path=file_path, request=request)
        return JSONResponse(content={"citations": citations})
    except ServiceValidationError as exc:
        raise _http_from_service_error(exc)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"提取引用文献过程中出现错误: {str(exc)}")


@app.post("/api/extract-citations-json")
async def extract_citations_json(request: Request):
    """从文档中提取引用文献（JSON）。"""
    try:
        body = await request.json()
        file_path = body.get("file_path")
        citations = relevance_service.extract_citations(file_path=file_path, request=request)
        return JSONResponse(content={"citations": citations})
    except ServiceValidationError as exc:
        raise _http_from_service_error(exc)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"提取引用文献过程中出现错误: {str(exc)}")


@app.post("/api/relevance-check")
async def perform_relevance_check(
    request: Request,
    file_path: str = Form(...),
    target_content: str = Form(...),
    task_type: str = Form("文章整体"),
    use_full_content: bool = Form(False),
):
    """执行文献相关性检查（兼容输出字段）。"""
    try:
        result = relevance_service.relevance_check(
            file_path=file_path,
            target_content=target_content,
            task_type=task_type,
            use_full_content=use_full_content,
            request=request,
        )

        # 兼容旧字段 + 新字段并存
        payload = {
            "文档": result.get("document"),
            "生成时间": result.get("generated_time"),
            "document": result.get("document"),
            "test_date": datetime.now().isoformat(),
            "task_type": result.get("task_type"),
            "check_method": result.get("check_method"),
            "relevance_score": result.get("relevance_score", 0),
            "is_suitable_for_citation": result.get("is_suitable_for_citation", False),
            "brief_basis": result.get("brief_basis", ""),
            "detailed_reasoning": result.get("detailed_reasoning", ""),
            "raw_result": result.get("raw_result", ""),
        }
        return JSONResponse(content=payload)
    except ServiceValidationError as exc:
        raise _http_from_service_error(exc)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"相关性检查过程中出现错误: {str(exc)}")


@app.post("/api/upload-only")
async def upload_only(request: Request, file: UploadFile = File(...)):
    """仅上传文件，不进行处理。"""
    try:
        workspace_service.validate_upload(file, max_file_size=MAX_FILE_SIZE)
        file_path = workspace_service.save_upload(file, request=request)
        return JSONResponse(
            content={
                "status": "success",
                "message": "文件上传成功",
                "file_path": str(file_path),
                "filename": file.filename,
                "size": file_path.stat().st_size,
            }
        )
    except ServiceValidationError as exc:
        raise _http_from_service_error(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"上传过程中出现错误: {str(exc)}")


@app.get("/api/list-all-files")
async def list_all_files(request: Request):
    """列出所有上传文件。"""
    try:
        files = workspace_service.list_files(request=request)
        return JSONResponse(content={"files": files})
    except ServiceValidationError as exc:
        raise _http_from_service_error(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取文件列表时出现错误: {str(exc)}")


@app.delete("/api/file")
async def delete_file(request: Request, file_path: str = Query(..., alias="file_path")):
    """删除指定路径文件。"""
    try:
        payload = workspace_service.delete_file(file_path, request=request)
        return JSONResponse(content=payload)
    except ServiceValidationError as exc:
        raise _http_from_service_error(exc)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除文件时出现错误: {str(exc)}")


# ---------------------------
# V2 API（统一契约）
# ---------------------------


@app.post("/api/v2/analysis/report")
async def v2_analysis_report(
    request: Request,
    file: UploadFile = File(...),
    author_format: str = Form("full"),
    citation_standard: str = Form("legacy"),
):
    started_at = datetime.now().isoformat()
    try:
        raw_report = analysis_service.analyze_upload(
            file=file,
            author_format=author_format,
            citation_standard=citation_standard,
            request=request,
        )
        payload = build_analysis_contract(
            raw_report=raw_report,
            status="succeeded",
            metadata={
                "author_format": author_format,
                "citation_standard": citation_standard,
                "entrypoint": "api.v2.analysis.report",
            },
        )
        payload["run"]["started_at"] = started_at
        payload["run"]["finished_at"] = datetime.now().isoformat()
        return JSONResponse(content=payload)
    except ServiceValidationError as exc:
        payload = build_contract(
            operation="analysis.report",
            status="failed",
            summary={},
            issues={},
            evidence={},
            artifacts={},
            error=build_error(code="validation_error", message=str(exc)),
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        return JSONResponse(content=payload, status_code=_http_from_service_error(exc).status_code)


@app.post("/api/v2/analysis/report-from-path")
async def v2_analysis_report_from_path(
    request: Request,
    file_path: str = Form(...),
    author_format: str = Form("full"),
    citation_standard: str = Form("legacy"),
):
    started_at = datetime.now().isoformat()
    try:
        raw_report = analysis_service.analyze_path(
            file_path=file_path,
            author_format=author_format,
            citation_standard=citation_standard,
            request=request,
        )
        payload = build_analysis_contract(
            raw_report=raw_report,
            status="succeeded",
            metadata={
                "author_format": author_format,
                "citation_standard": citation_standard,
                "entrypoint": "api.v2.analysis.report-from-path",
                "file_path": file_path,
            },
        )
        payload["run"]["started_at"] = started_at
        payload["run"]["finished_at"] = datetime.now().isoformat()
        return JSONResponse(content=payload)
    except ServiceValidationError as exc:
        payload = build_contract(
            operation="analysis.report-from-path",
            status="failed",
            error=build_error(code="validation_error", message=str(exc)),
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        return JSONResponse(content=payload, status_code=_http_from_service_error(exc).status_code)


@app.post("/api/v2/analysis/extract-citations")
async def v2_extract_citations(request: Request, file_path: str = Form(...)):
    started_at = datetime.now().isoformat()
    try:
        citations = relevance_service.extract_citations(file_path=file_path, request=request)
        payload = build_contract(
            operation="analysis.extract-citations",
            status="succeeded",
            summary={"citation_count": len(citations)},
            evidence={"citations": citations},
            metadata={"file_path": file_path},
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        return JSONResponse(content=payload)
    except ServiceValidationError as exc:
        payload = build_contract(
            operation="analysis.extract-citations",
            status="failed",
            error=build_error(code="validation_error", message=str(exc)),
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        return JSONResponse(content=payload, status_code=_http_from_service_error(exc).status_code)


@app.post("/api/v2/relevance/check")
async def v2_relevance_check(
    request: Request,
    file_path: str = Form(...),
    target_content: str = Form(...),
    task_type: str = Form("文章整体"),
    use_full_content: bool = Form(False),
):
    started_at = datetime.now().isoformat()
    try:
        result = relevance_service.relevance_check(
            file_path=file_path,
            target_content=target_content,
            task_type=task_type,
            use_full_content=use_full_content,
            request=request,
        )
        payload = build_contract(
            operation="relevance.check",
            status="succeeded",
            summary={
                "relevance_score": result.get("relevance_score", 0),
                "is_suitable_for_citation": bool(result.get("is_suitable_for_citation", False)),
            },
            evidence={"result": result},
            metadata={"file_path": file_path, "task_type": task_type, "use_full_content": bool(use_full_content)},
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        return JSONResponse(content=payload)
    except ServiceValidationError as exc:
        payload = build_contract(
            operation="relevance.check",
            status="failed",
            error=build_error(code="validation_error", message=str(exc)),
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        return JSONResponse(content=payload, status_code=_http_from_service_error(exc).status_code)


@app.post("/api/v2/workspace/upload")
async def v2_workspace_upload(request: Request, file: UploadFile = File(...)):
    started_at = datetime.now().isoformat()
    try:
        workspace_service.validate_upload(file, max_file_size=MAX_FILE_SIZE)
        path = workspace_service.save_upload(file, request=request)
        payload = build_contract(
            operation="workspace.upload",
            status="succeeded",
            summary={"uploaded": 1},
            evidence={"file": {"name": file.filename, "path": str(path), "size": path.stat().st_size}},
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        return JSONResponse(content=payload)
    except ServiceValidationError as exc:
        payload = build_contract(
            operation="workspace.upload",
            status="failed",
            error=build_error(code="validation_error", message=str(exc)),
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        return JSONResponse(content=payload, status_code=_http_from_service_error(exc).status_code)


@app.get("/api/v2/workspace/files")
async def v2_workspace_files(request: Request):
    started_at = datetime.now().isoformat()
    try:
        files = workspace_service.list_files(request=request)
        payload = build_contract(
            operation="workspace.list",
            status="succeeded",
            summary={"file_count": len(files)},
            evidence={"files": files},
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        return JSONResponse(content=payload)
    except ServiceValidationError as exc:
        payload = build_contract(
            operation="workspace.list",
            status="failed",
            error=build_error(code="validation_error", message=str(exc)),
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        return JSONResponse(content=payload, status_code=_http_from_service_error(exc).status_code)


@app.delete("/api/v2/workspace/file")
async def v2_workspace_delete(request: Request, file_path: str = Query(..., alias="file_path")):
    started_at = datetime.now().isoformat()
    try:
        result = workspace_service.delete_file(file_path, request=request)
        payload = build_contract(
            operation="workspace.delete",
            status="succeeded",
            summary={"deleted": 1},
            evidence={"result": result},
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        return JSONResponse(content=payload)
    except ServiceValidationError as exc:
        payload = build_contract(
            operation="workspace.delete",
            status="failed",
            error=build_error(code="validation_error", message=str(exc)),
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        return JSONResponse(content=payload, status_code=_http_from_service_error(exc).status_code)


@app.delete("/api/v2/workspace/files")
async def v2_workspace_clear(request: Request):
    started_at = datetime.now().isoformat()
    try:
        result = workspace_service.clear_files(request=request)
        payload = build_contract(
            operation="workspace.clear",
            status="succeeded",
            summary={"deleted_count": result.get("deleted_count", 0)},
            evidence={"result": result},
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        return JSONResponse(content=payload)
    except ServiceValidationError as exc:
        payload = build_contract(
            operation="workspace.clear",
            status="failed",
            error=build_error(code="validation_error", message=str(exc)),
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        return JSONResponse(content=payload, status_code=_http_from_service_error(exc).status_code)


@app.post("/api/v2/smoke/run")
async def v2_smoke_run(
    manifest: str = Form("tests/smoke_cases_manifest.json"),
    output_dir: Optional[str] = Form(None),
    author_format: str = Form("full"),
    citation_standard: str = Form("legacy"),
):
    started_at = datetime.now().isoformat()
    try:
        result = run_smoke_suite(
            manifest=manifest,
            output_dir=output_dir,
            author_format=author_format,
            citation_standard=citation_standard,
        )
        parsed_flags = result.get("parsed_flags") or {}
        all_passed = parsed_flags.get("SMOKE_ALL_PASSED", "False") == "True"
        payload = build_contract(
            operation="smoke.run",
            status="succeeded" if all_passed else "failed",
            summary={
                "all_passed": all_passed,
                "return_code": result.get("return_code", 1),
            },
            evidence={
                "flags": parsed_flags,
                "smoke_summary": result.get("smoke_summary") or {},
            },
            artifacts={
                "smoke_summary_json": parsed_flags.get("SMOKE_SUMMARY_JSON"),
                "smoke_summary_md": parsed_flags.get("SMOKE_SUMMARY_MD"),
            },
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        status_code = 200 if all_passed else 422
        return JSONResponse(content=payload, status_code=status_code)
    except ServiceValidationError as exc:
        payload = build_contract(
            operation="smoke.run",
            status="failed",
            error=build_error(code="validation_error", message=str(exc)),
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
        )
        return JSONResponse(content=payload, status_code=400)
