import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime

from core.checker.citation_checking.citation_analyzer import analyze_document
from utils.file_handler import save_upload_file, cleanup_file

# 创建API路由器
router = APIRouter()

# 最大上传文件大小 (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

@router.post("/full-report")
async def get_full_citation_report(file: UploadFile = File(...)):
    """
    生成完整的引文合规性报告，类似于standalone_mapping_test.py的输出
    """
    # 检查文件大小
    file.file.seek(0, 2)  # 移动到文件末尾
    file_size = file.file.tell()
    file.file.seek(0)  # 移回文件开头
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大，最大支持 {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    file_path = None
    try:
        # 保存上传的文件
        file_path = save_upload_file(file, "temp_uploads")
        
        # 执行分析任务
        analysis_result = analyze_document(file_path)
        
        return JSONResponse(content=analysis_result)
    
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析过程中出现错误: {str(e)}")
    
    finally:
        # 清理上传的文件
        if file_path:
            cleanup_file(file_path)

@router.get("/health")
def health_check():
    """健康检查端点"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.get("/analyze-document/health")  # For the /api/analyze-document/health path
def analyze_health_check():
    """分析文档功能的健康检查端点"""
    return {"status": "healthy", "function": "analyze_document", "timestamp": datetime.now().isoformat()}