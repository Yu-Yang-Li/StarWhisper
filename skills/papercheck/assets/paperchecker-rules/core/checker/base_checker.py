from abc import ABC, abstractmethod
from models.document import Document

class BaseChecker(ABC):
    """所有合规性检查器的基类"""
    
    @abstractmethod
    def check(self, document: Document) -> 'ComplianceResult':
        """
        对文档执行合规性检查
        
        Args:
            document: 要检查的文档对象
            
        Returns:
            ComplianceResult: 检查结果
        """
        pass
    
    @abstractmethod
    def get_check_type(self) -> 'CheckType':
        """
        返回当前检查器的类型
        
        Returns:
            CheckType: 检查类型
        """
        pass
    
    @abstractmethod
    def get_check_name(self) -> str:
        """
        返回检查器的名称
        
        Returns:
            str: 检查器名称
        """
        pass