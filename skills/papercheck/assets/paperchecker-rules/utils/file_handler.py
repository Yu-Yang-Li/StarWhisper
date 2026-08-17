import os
import uuid
from typing import Optional
from fastapi import UploadFile, HTTPException
import mimetypes

# 支持的文件类型
SUPPORTED_FILE_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "application/pdf": ".pdf",
}

def validate_file_type(file: UploadFile) -> bool:
    """验证上传文件类型是否支持"""
    # 如果无法从文件名推断MIME类型，则尝试从content_type获取
    mime_type = mimetypes.guess_type(file.filename)[0]
    if not mime_type:
        mime_type = file.content_type
    
    return mime_type in SUPPORTED_FILE_TYPES

def save_upload_file(file: UploadFile, upload_dir: str) -> str:
    """保存上传的文件并返回文件路径"""
    # 验证文件类型
    if not validate_file_type(file):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file.content_type}. 支持的类型: .docx, .doc, .pdf"
        )

    file.file.seek(0, 2)
    expected_size = file.file.tell()
    file.file.seek(0)
    if expected_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="上传文件为空（0 字节）。请重新选择可正常打开的 .docx、.doc 或 .pdf 文件。"
        )
    
    # 生成唯一文件名
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    # 确保上传目录存在
    os.makedirs(upload_dir, exist_ok=True)
    
    # 保存文件
    with open(file_path, "wb") as f:
        content = file.file.read()
        f.write(content)

    actual_size = os.path.getsize(file_path)
    if actual_size <= 0 or actual_size != expected_size:
        cleanup_file(file_path)
        raise HTTPException(
            status_code=400,
            detail=(
                "上传文件保存不完整。"
                f"预期 {expected_size} 字节，实际 {actual_size} 字节；请重新上传原始文件。"
            )
        )
    
    return file_path

def cleanup_file(file_path: str):
    """清理文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except:
        pass  # 如果删除失败，不需要抛出异常
