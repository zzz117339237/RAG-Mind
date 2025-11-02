import os
from typing import Tuple
import fitz  # PyMuPDF


class FileProcessor:
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """从PDF文件提取文本"""
        try:
            document = fitz.open(file_path)
            all_text = ""
            for page_num in range(document.page_count):
                page = document[page_num]
                text = page.get_text("text")
                all_text += text
            return all_text
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"PDF处理失败: {e}")
            return ""

    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """从TXT文件提取文本"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"TXT处理失败: {e}")
            return ""

    @staticmethod
    def extract_text_from_file(file_path: str) -> Tuple[str, str]:
        """根据文件扩展名自动选择提取方法"""
        file_extension = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        if file_extension == '.pdf':
            text = FileProcessor.extract_text_from_pdf(file_path)
            return text, file_name
        elif file_extension == '.txt':
            text = FileProcessor.extract_text_from_txt(file_path)
            return text, file_name
        else:
            import logging
            logging.getLogger(__name__).warning(f"不支持的文件格式: {file_extension}")
            return "", file_name

    @staticmethod
    def get_supported_extensions() -> list:
        """获取支持的文件扩展名"""
        return ['.pdf', '.txt']
