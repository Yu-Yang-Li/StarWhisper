import os
import sys
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import File, UploadFile, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from utils.file_handler import save_upload_file, cleanup_file
from core.polish.analyse import review_document

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024

@router.post("/get_reviews") # 对应命令: POST /polish/get_riviews
async def get_reviews(file: UploadFile = File(...)):
    """
    生成论文精批、润色报告
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

    try:
        # 保存上传的文件
        file_path = save_upload_file(file, "temp_uploads")

        # 执行分析任务
        analysis_result = review_document(file_path)
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