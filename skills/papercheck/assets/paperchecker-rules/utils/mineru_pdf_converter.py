import requests
import time
import os
import zipfile
import tempfile
import datetime
import json


class MineruPDFToMD:
    def __init__(self, config_path="config.json", user_token: str = None):
        """初始化MineruPDFToMD实例，优先从环境变量读取API密钥，其次读取配置文件"""
        self.api_key = self._load_api_key_from_config(config_path)
        self.base_url = "https://mineru.net/api/v4"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # 某些企业/专业版接口需要额外 token 头
        resolved_user_token = user_token or os.getenv("MINERU_USER_TOKEN")
        if resolved_user_token:
            self.headers["token"] = resolved_user_token
    
    def _load_api_key_from_config(self, config_path):
        """从配置文件加载API密钥"""
        env_api_key = os.getenv("MINERU_API_KEY")
        if env_api_key:
            return env_api_key

        if not os.path.exists(config_path):
            raise Exception(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        mineru_config = config.get("mineru_config", {})
        api_key = mineru_config.get("api_key")
        
        if not api_key:
            raise Exception("mineru_api_key not found in config.json. Please add mineru_config section with api_key.")
        
        return api_key

    def create_extraction_task(self, pdf_url, model_version="vlm"):
        """创建解析任务（通过URL）"""
        url = f"{self.base_url}/extract/task"
        data = {
            "url": pdf_url,
            "model_version": model_version
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        
        try:
            response.raise_for_status()
            result = response.json()
            
            if result["code"] != 0:
                raise Exception(f"Failed to create task: {result['msg']}")
            
            return result["data"]["task_id"]
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP Error: {e} - Response: {response.text}")
        except ValueError as e:
            raise Exception(f"JSON Parse Error: {e} - Response: {response.text}")

    def get_task_result(self, task_id):
        """获取单个任务结果"""
        url = f"{self.base_url}/extract/task/{task_id}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        result = response.json()
        
        if result["code"] != 0:
            raise Exception(f"Failed to get task result: {result['msg']}")
        
        return result["data"]

    def wait_for_task_completion(self, task_id, poll_interval=5, max_retries=3):
        """等待单个任务完成，带有重试机制"""
        print(f"Waiting for task {task_id} to complete...")
        retry_count = 0
        
        while True:
            try:
                result = self.get_task_result(task_id)
                state = result["state"]
                
                if state == "done":
                    print("Task completed successfully!")
                    return result
                elif state == "failed":
                    raise Exception(f"Task failed: {result['err_msg']}")
                elif state == "running":
                    progress = result.get("extract_progress", {})
                    extracted = progress.get("extracted_pages", 0)
                    total = progress.get("total_pages", 0)
                    print(f"Processing: {extracted}/{total} pages...")
                elif state == "pending":
                    print("Task is in queue...")
                elif state == "converting":
                    print("Converting format...")
                
                time.sleep(poll_interval)
                retry_count = 0  # 重置重试计数
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    raise Exception(f"Failed to get task status after {max_retries} retries: {str(e)}")
                print(f"Error getting task status: {str(e)}, retrying in {poll_interval} seconds... ({retry_count}/{max_retries})")
                time.sleep(poll_interval)

    def get_batch_upload_urls(self, file_names, model_version="vlm"):
        """获取批量上传URL"""
        url = f"{self.base_url}/file-urls/batch"
        data = {
            "files": [{'name': name} for name in file_names],
            "model_version": model_version
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if result["code"] != 0:
            raise Exception(f"Failed to get upload urls: {result['msg']}")
        
        return result["data"]

    def upload_file(self, upload_url, file_path):
        """上传文件"""
        print(f"Uploading file {file_path} to {upload_url}...")
        
        with open(file_path, 'rb') as f:
            response = requests.put(upload_url, data=f)
        
        response.raise_for_status()
        print(f"File {file_path} uploaded successfully!")

    def get_batch_results(self, batch_id):
        """获取批量任务结果"""
        url = f"{self.base_url}/extract-results/batch/{batch_id}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        result = response.json()
        
        if result["code"] != 0:
            raise Exception(f"Failed to get batch results: {result['msg']}")
        
        return result["data"]

    def wait_for_batch_completion(self, batch_id, poll_interval=5, max_retries=3):
        """等待批量任务完成，带有重试机制"""
        print(f"Waiting for batch {batch_id} to complete...")
        retry_count = 0
        
        while True:
            try:
                result = self.get_batch_results(batch_id)
                extract_results = result.get("extract_result", [])
                
                all_done = True
                any_failed = False
                
                for extract_result in extract_results:
                    state = extract_result["state"]
                    file_name = extract_result["file_name"]
                    
                    if state == "done":
                        print(f"File {file_name}: completed")
                    elif state == "failed":
                        print(f"File {file_name}: failed - {extract_result['err_msg']}")
                        any_failed = True
                    elif state == "running":
                        print(f"File {file_name}: processing")
                        all_done = False
                    elif state == "pending" or state == "waiting-file":
                        print(f"File {file_name}: in queue")
                        all_done = False
                    elif state == "converting":
                        print(f"File {file_name}: converting")
                        all_done = False
                
                if any_failed:
                    raise Exception("Some files in batch failed to process")
                
                if all_done:
                    print("All files in batch completed successfully!")
                    return result
                
                time.sleep(poll_interval)
                retry_count = 0  # 重置重试计数
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    raise Exception(f"Failed to get batch status after {max_retries} retries: {str(e)}")
                print(f"Error getting batch status: {str(e)}, retrying in {poll_interval} seconds... ({retry_count}/{max_retries})")
                time.sleep(poll_interval)

    def download_and_extract_md(self, zip_url, output_dir):
        """下载并提取Markdown文件和images文件夹"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
            temp_zip_path = temp_zip.name
        
        try:
            # 下载zip文件
            print(f"Downloading result from {zip_url}...")
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()
            
            with open(temp_zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 获取当前时间戳
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            
            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                # 获取所有文件列表
                all_files = zip_ref.namelist()
                
                # 找到所有Markdown文件和images文件夹
                md_files = [f for f in all_files if f.endswith(".md")]
                if not md_files:
                    raise Exception("No Markdown file found in the result")
                
                # 检查是否有images文件夹
                has_images = any("images/" in f for f in all_files)
                
                # 处理每个Markdown文件
                extracted_files = []
                for md_file in md_files:
                    # 生成带有时间戳的文件名
                    base_name = os.path.splitext(os.path.basename(md_file))[0]
                    timestamped_name = f"{base_name}_{timestamp}"
                    
                    # 创建与md同名的子文件夹
                    md_subdir = os.path.join(output_dir, timestamped_name)
                    os.makedirs(md_subdir, exist_ok=True)
                    
                    # 读取md文件内容
                    with zip_ref.open(md_file) as f:
                        content = f.read().decode('utf-8')
                    
                    # 保存md文件到子文件夹
                    final_md_name = f"{timestamped_name}.md"
                    md_output_path = os.path.join(md_subdir, final_md_name)
                    with open(md_output_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    
                    # 如果有images文件夹，提取到子文件夹
                    if has_images:
                        image_files = [f for f in all_files if f.startswith("images/")]
                        for image_file in image_files:
                            # 创建images子目录
                            image_subdir = os.path.join(md_subdir, "images")
                            os.makedirs(image_subdir, exist_ok=True)
                            
                            # 提取图片文件
                            zip_ref.extract(image_file, md_subdir)
                            print(f"Extracted image: {image_file}")
                    
                    extracted_files.append(md_output_path)
                    print(f"Processed: {md_output_path}")
                
                return extracted_files
        finally:
            # 删除临时文件
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)

    def convert_pdf_from_url(self, pdf_url, output_dir=".", model_version="vlm"):
        """从URL转换PDF为Markdown"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建解析任务
        task_id = self.create_extraction_task(pdf_url, model_version)
        print(f"Created task with ID: {task_id}")
        
        # 等待任务完成
        task_result = self.wait_for_task_completion(task_id)
        
        # 下载并提取Markdown文件
        zip_url = task_result["full_zip_url"]
        return self.download_and_extract_md(zip_url, output_dir)

    def convert_pdf_from_local(self, file_path, output_dir=".", model_version="vlm"):
        """从本地文件转换PDF为Markdown"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        
        # 检查文件类型
        if not file_path.endswith(".pdf"):
            raise Exception(f"Only PDF files are supported, got: {file_path}")
        
        file_name = os.path.basename(file_path)
        
        # 获取上传URL
        batch_data = self.get_batch_upload_urls([file_name], model_version)
        batch_id = batch_data["batch_id"]
        upload_urls = batch_data["file_urls"]
        
        # 上传文件
        self.upload_file(upload_urls[0], file_path)
        
        # 等待批量任务完成
        batch_result = self.wait_for_batch_completion(batch_id)
        
        # 下载并提取Markdown文件
        extract_results = batch_result.get("extract_result", [])
        if not extract_results:
            raise Exception("No extraction results found")
        
        extracted_files = []
        for result in extract_results:
            if result["state"] == "done":
                zip_url = result["full_zip_url"]
                files = self.download_and_extract_md(zip_url, output_dir)
                extracted_files.extend(files)
                break
        
        return extracted_files

    def convert_pdf_to_md(self, pdf_input, output_dir=".", model_version="vlm"):
        """将PDF转换为Markdown的完整流程（支持URL和本地文件）"""
        # 判断输入是URL还是本地文件
        if pdf_input.startswith("http://") or pdf_input.startswith("https://"):
            return self.convert_pdf_from_url(pdf_input, output_dir, model_version)
        else:
            return self.convert_pdf_from_local(pdf_input, output_dir, model_version)


def convert_pdf_to_markdown(
    pdf_path: str,
    output_dir: str = None,
    config_path: str = None,
    model_version: str = "vlm",
    user_token: str = None
) -> str:
    """
    将PDF文件转换为Markdown格式，使用MinerU API进行转换

    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录，默认为临时目录
        config_path: 配置文件路径

    Returns:
        转换后的Markdown文件路径
    """
    if output_dir is None:
        import tempfile
        output_dir = tempfile.mkdtemp(prefix="mineru_conversion_")

    # 如果未提供配置文件路径，则尝试查找配置文件
    if config_path is None:
        # 首先尝试当前目录
        if os.path.exists("config.json"):
            config_path = "config.json"
        else:
            # 尝试在项目根目录查找
            import pathlib
            project_root = pathlib.Path(__file__).resolve().parent.parent.parent
            project_config_path = project_root / "config.json"

            # 如果项目根目录的config.json不存在，尝试config子目录中的config.json
            if not project_config_path.exists():
                config_in_config_dir = project_root / "config" / "config.json"
                if config_in_config_dir.exists():
                    config_path = str(config_in_config_dir)
                else:
                    # 如果都找不到，抛出异常
                    raise FileNotFoundError("未找到配置文件 config.json，请确保配置文件存在于当前目录、项目根目录或config子目录中")
            else:
                config_path = str(project_config_path)

    # 初始化转换器
    converter = MineruPDFToMD(config_path=config_path, user_token=user_token)

    # 将PDF转换为Markdown
    md_files = converter.convert_pdf_to_md(pdf_path, output_dir, model_version=model_version)

    if not md_files:
        raise Exception("PDF转换失败：未生成任何Markdown文件")

    # 返回第一个转换的Markdown文件路径
    return md_files[0] if md_files else None


def fix_title_levels(md_content):
    """修复Markdown标题层级，使用简单的启发式方法"""
    if md_content is None:
        md_content = ""
    lines = md_content.split('\n')
    
    # 1. 识别所有标题行
    title_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#') and len(stripped) > 1:
            # 提取标题信息
            level = len(stripped.split(' ')[0])
            content = stripped[level:].strip()
            if content:  # 只处理非空标题
                title_lines.append((i, level, content))
    
    if len(title_lines) <= 1:
        return md_content

    # 2. 简单修复标题层级：确保标题层级连续
    fixed_lines = lines.copy()
    
    # 如果第一个标题是H2或更高级别，调整为H1
    if title_lines[0][1] > 1:
        first_line_idx, old_level, content = title_lines[0]
        fixed_title = '#' + ' ' + content
        fixed_lines[first_line_idx] = fixed_title
    
    # 确保后续标题层级不超过前一个标题层级+1
    for i in range(1, len(title_lines)):
        prev_line_idx, prev_level, prev_content = title_lines[i-1]
        curr_line_idx, curr_level, curr_content = title_lines[i]
        
        # 如果当前标题层级比前一个标题层级深超过1级，则调整
        if curr_level > prev_level + 1:
            adjusted_level = min(6, prev_level + 1)  # 限制最大标题层级为H6
            fixed_title = '#' * adjusted_level + ' ' + curr_content
            fixed_lines[curr_line_idx] = fixed_title
    
    return '\n'.join(fixed_lines)

