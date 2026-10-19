import os
from loguru import logger
from datetime import datetime


def run_x_opstep(params: CommandParams):

    today = datetime.now()
    # 格式化为 YYYY-MM-DD
    today_1 = today.strftime("%Y-%m-%d")
    one_month_ago = today - timedelta(days=30)
    today_0 = today.strftime("%Y%m%d")
    one_month_ago_formatted = one_month_ago.strftime("%Y%m%d")
    # 构建命令字符串
    command = f'''
    x-opstep 
    -rawdir /home/pod/shared-nvme/NGSS/data/rawdir/{today_1}
    -reddir /home/pod/shared-nvme/NGSS/data/reddir/{today_1} 
    -template /home/pod/shared-nvme/NGSS/data/template 
    -pdf /home/pod/shared-nvme/NGSS/data/pdf/{today_1} 
    -pm set_date 
    -pdb {one_month_ago_formatted} 
    -pde {today_0} 
    -ps 0.1 -ad /home/pod/shared-nvme/NGSS/astrometry.net/data 
    -ncpu 30
    '''

    # 构建完整的 shell 命令序列
    full_command = [
        "conda activate base",
        command,
        "conda activate observe"]
        # 使用 subprocess 模块在一个连续的 shell 会话中执行命令
    try:
        process = subprocess.Popen(
            [";".join(full_command)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
        )

        stdout, stderr = process.communicate()

        if process.returncode == 0:
            logger.info(f"运行成功，图像相减的证认图位于 /home/pod/shared-nvme/NGSS/data/pdf/{today_1} 路径下，可以进行查看")
            return {"status": "success", "message": stdout.decode(), "pdf_image": f"/home/pod/shared-nvme/NGSS/data/pdf/{today_1}"}
        else:
            logger.error(f"❌Error executing command: {stderr.decode()}")
            raise HTTPException(status_code=500, detail=f"Error executing command: {stderr.decode()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


'''
需增加的内容
1. x-opstep的log
2. 图像相减后的pdf读取与分辨
3. 确认通讯以及处理位置。
'''