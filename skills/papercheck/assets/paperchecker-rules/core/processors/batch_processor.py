from .base_processor import BaseProcessor
from .citation_processor import CitationProcessor
from typing import Dict, Any, List
import concurrent.futures
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BatchProcessor(BaseProcessor):
    """批量处理器 - 支持并发处理多个文档"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
    
    def process(self, file_paths: List[str], config: Dict = None) -> Dict[str, Any]:
        """批量处理文档 - 并发执行"""
        start_time = datetime.now()
        
        results = []
        failed_files = []
        
        # 使用线程池并发处理文档
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_path = {
                executor.submit(self._process_single_document, file_path, config): file_path 
                for file_path in file_paths
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"成功处理: {file_path}")
                except Exception as e:
                    logger.error(f"处理文档 {file_path} 时发生错误: {str(e)}")
                    failed_files.append({
                        "document": file_path,
                        "error": str(e),
                        "status": "failed"
                    })
        
        end_time = datetime.now()
        
        return {
            "batch_stats": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "total_documents": len(file_paths),
                "successful_documents": len(results),
                "failed_documents": len(failed_files),
                "success_rate": f"{len(results)/len(file_paths)*100:.1f}%" if file_paths else "0.0%"
            },
            "results": results,
            "failed_documents": failed_files
        }
    
    def _process_single_document(self, file_path: str, config: Dict = None) -> Dict[str, Any]:
        """处理单个文档（用于并发执行）"""
        processor = CitationProcessor(config=config)
        return processor.process(file_path)