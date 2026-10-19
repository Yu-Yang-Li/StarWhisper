import itertools
import json
import os
import shutil
import sys
import uuid
from typing import List, Optional
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from enum import Enum
from ftplib import FTP
from multiprocessing import Manager
from pathlib import Path
from queue import Queue
from typing import Annotated, Any, Iterable, Literal

import dill as pickle
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from icecream import ic
from loguru import logger
from pydantic import BaseModel, Field, validator
import pandas as pd
import hashlib

ic.enable()
os.chdir(ic(Path(__file__).parents[2]))  # 切换到项目根目录作为工作目录
sys.path.append(".")
from src.module.PlanObservation3 import init, main, add_object_to_fine_list, modify_config
from src.module.SearchPath import Searcher
from src.module.UdpConnect import MQTTPublisher
from src.module.transientDetection import pipeline_process
from src.script.daily_update import main as _update_station
from src.util.util import make_and_return_dir


class SessionId(BaseModel):
    main_pid: str = Field(
        description="主进程，即处理请求的fastapi服务进程ID。普通用户不用关心。"
    )
    worker_pid: str = Field(
        description="此次请求规划接口后，在后台生成的进程ID。后续可以凭借此ID查看进程运行状况。普通用户不用关心。"
    )
    uu_id: str = Field(
        description="此次请求规划接口后，在后台生成的session ID。后续可以凭借此ID查看工作日志。"
    )


class SuccessLog(BaseModel):
    uu_id: str = Field(description="请求规划接口后，返回的session ID。")
    debug_log: list[str] = Field(description="记录后台运行的详细debug日志")
    user_log: list[str] = Field(description="面向普通用户的日志")


class ErrMsg(BaseModel):
    err_msg: str = Field(description="接口请求错误原因")


class YesterdayExist(BaseModel):
    msg: str = Field(description="返回消息")
    exist_flag: int = Field(description="昨日数据不存在为0，存在为1")


class BaseOBResp(BaseModel):
    station: str = Field(description="台站")
    query_dt: str = Field(description="查询日期。如果请求接口时为空，默认查询当天。")
    telescope: str = Field(description="望远镜编号")


class OBList(BaseOBResp):
    response: dict[str, dict] = Field(description="观测计划")


class OBListErrMsg(BaseOBResp, ErrMsg):
    pass


class ActionEnum(Enum):
    load_file = "load_file"
    start = "start"
    stop = "stop"


app = FastAPI()


def get_queue_entry(queue: Queue):
    empty = True
    count = 0
    while empty:
        count += 1
        new_empty_status = queue.empty()
        if not new_empty_status:
            empty = False
            logger.info(
                f"Probed queue is not empty, qsize: {str(queue.qsize())}. Accu iter: {str(count)}"
            )
            # 从队列中获取新创建的进程ID
            entry = queue.get()
        else:
            pass
    return entry


def prepare_response(json_files: Iterable, dir_station: str) -> dict[str, dict]:
    response = {}
    for file in json_files:
        file_name = ic(file.name)
        with open(file, mode="r") as f:
            tmp_response = json.load(f)
            response[file_name] = tmp_response
    return {dir_station: response}


def find_latest_log(dt: str) -> tuple[str, Path, Path]:
    dt_path = Path(f"log/{ic(dt)}")
    uuid_hist_log_path = dt_path / "uuid_hist.log"

    try:
        assert dt_path.is_dir()
        assert uuid_hist_log_path.is_file()
    except:
        raise FileNotFoundError
    else:
        with open(uuid_hist_log_path, "r") as f:
            latest_record = f.readlines()[-1]

        ic(latest_record)
        uu_id = latest_record.split(" ")[0]
        log_path = Path(f"log/{dt}/{uu_id}.log")
        debug_log_path = Path(f"log/{dt}/{uu_id}_debug.log")
        return (uu_id, log_path, debug_log_path)


def rename_file(station: str, query_dt: str, telescope_or_filename: str) -> str:
    extension: str = ".ninaTargetSet"
    file_name: str = (
        telescope_or_filename + extension
        if extension not in telescope_or_filename
        else telescope_or_filename
    )
    return f"{station}_{query_dt}_{file_name}"


executor = ProcessPoolExecutor()
queue = Manager().Queue()
executor.submit(init)


@app.get("/update_station", responses={400: {"model": ErrMsg}})
def update_station() -> Any:
    """更新台站数据的接口。由于后续制定观测计划依赖于台站数据的生成，\
    每天必须将此接口的运行放在第一步，以保证【制定观测计划】时不会报错。\
    由于每天此接口只需访问一次，所以如果台站数据已经存在，则不必重新运行。

    Returns:
        str: 台站更新任务是否运行成功的一则消息。一般来讲应该都是成功，不会失败。
    """

    dt = datetime.now().strftime("%Y%m%d")
    if Path(f"data/{dt}").is_dir():
        logger.info("台站数据已经更新！")
        return {"update_msg": "🤗 每日台站数据已存在，不必重新更新。"}
    else:
        try:
            _update_station()  ## 当新增观测台站或者镜子的时候，来这里修改即可。
        except:
            return JSONResponse(
                status_code=400,
                content={"err_msg": "❌ 每日台站数据更新失败！请联系管理员。"},
            )
        else:
            return {"update_msg": "🤗 每日台站数据之前不存在，现已生成！"}


@app.get("/look_config")
async def read_json():
    dt = datetime.now().strftime("%Y%m%d")
    config_dir = Path(f"data/{dt}")
    config_path = config_dir / "observe_config.json"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data


@app.get("/modify_config")
def change_config(   
    key: str = Query(..., description="配置项名"),
    value: str = Query(..., description="新值"),):

    """
    API Endpoint to modify a configuration item.
    
    - **key**: The name of the configuration item.
    - **value**: The new value for the configuration item.
    """
    dt = datetime.now().strftime("%Y%m%d")
    config_dir = Path(f"data/{dt}")
    config_path = config_dir / "observe_config.json"
    try:
        result = modify_config(key, value, config_path)
        return {"status": "success", "message": result}
    except (FileNotFoundError, KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))



#9.6日更新过，增加oblist
@app.get("/plan_observation")
def plan_observation(
    oblist: Optional[List[str]] = Query(None),
    thedate: Optional[str] = Query(None, description="指定日期，格式为 YYYYMMDD"),
    inherit: Optional[bool] = Query(True, description="是否继承前一日或指定日期的观测计划")
) -> SessionId:
    
    """制定当天（即请求这个接口的时间对应的日期）的观测计划。\
    注意：
    1. 请求这个接口后，程序会对所有站点的所有望远镜都指定观测计划；
    2. 此接口的成功运行依赖于更新台站数据接口的成功运行。

    Returns:
        SessionId: 一个json结构体，记录了此次用户触发制定观测计划接口后生成的一些session ID。\
        普通用户只需要关心uu_id，后续可以凭借该ID查看session日志。
    """
    main_pid = os.getpid()
    ic(main_pid)
    uu_id = uuid.uuid4().hex
    ic(type(uu_id))
    ic(oblist)

    logger.info(f"qsize before submit: {str(queue.qsize())}")
    # 将 oblist, thedate, 和 inherit 参数传递给 main 函数
    executor.submit(main, uu_id, queue, oblist, thedate, inherit)
    logger.info(f"qsize immediately after submit: {str(queue.qsize())}")
    sub_pid = get_queue_entry(queue)
    return SessionId(main_pid=str(main_pid), worker_pid=str(sub_pid), uu_id=str(uu_id))


@app.get("/check_log", response_model=SuccessLog, responses={400: {"model": ErrMsg}})
def check_log(uuid: str | None = None) -> Any:
    """随时检查制定观测计划程序运行日志。
    注意：uuid是可选参数。如果参数为空，默认查找最新发起的请求对应的日志。\
        如果uuid非空，则会查找对应session的日志。

    Args:
        uuid (str | None, optional): 发起制定观测计划请求后得到的uuid. Defaults to None.

    Returns:
        200: 返回详细日志信息
        400: 请求接口错误的原因。
    """
    dt = datetime.now().strftime("%Y%m%d")
    if uuid:
        log_path = Path(f"log/{dt}/{uuid}.log")
        debug_log_path = Path(f"log/{dt}/{uuid}_debug.log")
    else:  # uuid为空。默认查找最近一次请求的日志。
        try:
            uuid, log_path, debug_log_path = find_latest_log(dt)
        except:
            err_msg = f"❌ You are checking logs for most recent one. However, there is not any logs produced today. Please firstly call plan_observation to generate a uuid!"
            return JSONResponse(
                status_code=400, content={"err_msg": f"{uuid}: {err_msg}"}
            )

    try:
        assert log_path.is_file()
    except:
        err_msg = f"❌ No logs for {uuid}. Please enter the right uuid!"
        logger.info(err_msg)
        return JSONResponse(status_code=400, content={"err_msg": err_msg})
    else:
        with open(log_path, "r") as f:
            log_txt = f.read().splitlines()
        with open(debug_log_path, "r") as f:
            debug_log_txt = f.read().splitlines()

        ic(uuid)
        return SuccessLog(uu_id=uuid, debug_log=debug_log_txt, user_log=log_txt)


@app.get("/get_oblist", response_model=OBList, responses={400: {"model": OBListErrMsg}})
@logger.catch(reraise=True)
def get_oblist(station: str, query_dt: str | None = None) -> Any:
    """查看观测计划。

    Args:
        station (str): 必选。填写台站拼音（大小写均可），或者填写all，代表查询所有台站计划观测星表。
        query_dt (str | None, optional): 可选。选择日期。如果日期为空，默认为当天。

    Returns:
        200: 返回包含OBList的一个结构体。
        400: 返回错误原因。
    """
    dt = datetime.now().strftime("%Y%m%d") if not query_dt else query_dt
    try:
        searcher = Searcher(dt, None)
        response: dict[str, str | dict] = {"query_dt": dt, "telescope": "NA"}

        if station != "all":
            response["station"] = station
            json_files: Iterable = searcher.find_one_station(station)
            response["response"] = prepare_response(json_files, station)
            ic(response)
            return OBList(**response)
        else:  # 所有站点
            response["station"] = "all"
            json_files_list: list[tuple] = searcher.find_all_station()

            for json_files, dir_station in json_files_list:
                if "response" not in response.keys():
                    response["response"] = prepare_response(json_files, dir_station)
                else:
                    response["response"] |= prepare_response(json_files, dir_station)

            return OBList(**response)
    except FileExistsError as e:
        response["err_msg"] = str(e)
        return JSONResponse(status_code=400, content=response)
    

@app.get("/add_fine_oblist")
@logger.catch(reraise=True)
def add_fine_oblist(objlist: str = None, objname: str = None, ra: float = None, dec: float = None):
    """制定当天（即请求这个接口的时间对应的日期）的观测计划。\
    """
    if objlist is None and objname is None and ra is None and dec is None:
        return {
        "response": f"没有输入任何目标源，跳过执行。"
    }

    sorted_stations = [
    {
        "Name": "YunNan",
        "lat": 23.914,
        "lon": 102.653,
        "num": 1
    },
    {
        "Name": "Gansu",
        "lat": 35.678,
        "lon": 106.848,
        "num": 1
    },
    {
        "Name": "XingLong",
        "lat": 40.393,
        "lon": 117.575,
        "num": 7
    },
    {
        "Name": "XinJiang",
        "lat": 43.522,
        "lon": 88.577,
        "num": 1
    }
    ]
    if objlist is not None:
        [replaced_objs, notfound_object, notobservable_objects] = add_object_to_fine_list(sorted_stations = sorted_stations, objlist = objlist) 
    
    if objname is not None and ra is not None and dec is not None:
        [replaced_objs, notfound_object, notobservable_objects] = add_object_to_fine_list(sorted_stations = sorted_stations, objname = objname, ra = ra, dec=dec)

    if not notfound_object:
        notfound_object = 'No objects'
    
    if not replaced_objs:
        replaced_objs = 'No objects'
    
    if not notobservable_objects:
        notobservable_objects = 'No objects'

    return {
        "response": f"{notfound_object} are not found, {notobservable_objects} are not observable tonight, {replaced_objs} are replaced by the objects you mentioned"
    }



@app.get("/ftp_transfer")
@logger.catch(reraise=True)
#传输nina文件
def ftp_transfer(station: str, query_dt: str, telescope: str):
    station_name_map = {
    'xinglong': 'XingLong',
    'gansu': 'Gansu',
    'xinjiang': 'XinJiang',
    'yunnan': 'YunNan',
    }
    telescope_number = int(telescope)
    station_name = station_name_map.get(station, 'UnknownStation')

    file = f"/home/pod/shared-nvme/NGSS/agent/data/{query_dt}/output_{station_name}/{telescope_number}.ninaTargetSet"
    with open(file, 'rb') as f:
        schedule = f.read()
        payload = schedule
    
    publisher = MQTTPublisher()
    publisher.connect()
    try:
        # 发布消息到 'ftp_transfer' 部分的所有主题
        publisher.publish_to_telescope('ftp_transfer',station, telescope, payload)
        publisher.success_received.wait(timeout=10)  # 无限等待直到收到成功消息
        print("Receive success message, disconnecting.")
    finally:
        publisher.disconnect()
    
    return {
        "response": f"Successfully transferred to remote address {station}{telescope}"
    }



@app.get("/manipulate_nina/{action}", responses={400: {"model": OBListErrMsg}})
@logger.catch(reraise=True)
def manipulate_nina(
    action: ActionEnum,
    station: str | None = None,
    query_dt: str | None = None,
    telescope: str | None = None,
) -> Any:
    publisher = MQTTPublisher()
    publisher.connect()
       
    if action == ActionEnum.load_file:
        try:
            assert station and query_dt and telescope
        except:
            response = {
                "query_dt": query_dt,
                "telescope": telescope,
                "station": station,
                "err_msg": "Missing necessary parameters to load file!",
            }
            return JSONResponse(status_code=400, content=response)
        else:
            renamed: str = ic(rename_file(station, query_dt, telescope))
            #send_msg(f"MSG=LoadFile:{renamed};")
            publisher.publish_to_telescope('nina_action',station, telescope, schedule="MSG=LoadFile:1.ninaTargetSet;")
            publisher.disconnect()
            return {"response": "Success"}
    elif action == ActionEnum.start:
        publisher.publish_to_telescope('nina_action',station, telescope, schedule="MSG=Start;")
        publisher.disconnect()
        return {"response": "Success"}
    elif action == ActionEnum.stop:
        publisher.publish_to_telescope('nina_action',station, telescope, schedule="MSG=Stop;")
        publisher.disconnect()
        return {"response": "Success"}



'''1113新增，数据pipeline'''

class AllSourcesResponse(BaseModel):
    query_dt: str
    station: str
    telescope: int
    response: str

class FilteredCSVDataResponse(BaseModel):
    query_dt: str
    station: str
    telescope: int
    filtered_csv_data: List[dict[str, Any]] = Field(description="暂现源表")

# 获取 all_sources
@app.get("/pipeline_transient", response_model=AllSourcesResponse, responses={400: {"model": OBListErrMsg}})
def pipeline_transient(query_dt: str, station: str, telescope: int) -> AllSourcesResponse:
    try:
        all_sources, _ = pipeline_process(query_dt, station, telescope)
        objlist = ",".join(all_sources)
        ic(objlist)
        sorted_stations = [
        {
            "Name": "YunNan",
            "lat": 23.914,
            "lon": 102.653,
            "num": 1
        },
        {
            "Name": "Gansu",
            "lat": 35.678,
            "lon": 106.848,
            "num": 1
        },
        {
            "Name": "XingLong",
            "lat": 40.393,
            "lon": 117.575,
            "num": 7
        },
        {
            "Name": "XinJiang",
            "lat": 43.522,
            "lon": 88.577,
            "num": 1
        }
        ]

        [replaced_objs, notfound_object, notobservable_objects] = add_object_to_fine_list(sorted_stations = sorted_stations, objlist = objlist)  # 传递oblist给main函数
    
        if not notfound_object:
            notfound_object = 'No objects'
        
        if not replaced_objs:
            replaced_objs = 'No objects'
        
        if not notobservable_objects:
            notobservable_objects = 'No objects'

        msg =  f"{notfound_object} are not found, {notobservable_objects} are not observable tonight, {replaced_objs} are replaced by the objects you mentioned"
        
        response = {
            "query_dt": query_dt,
            "station": station,
            "telescope": telescope,
            "response": msg
        }
        return AllSourcesResponse(**response)
    except Exception as e:
        return JSONResponse(
                status_code=400,
                content={"err_msg": str(e)},
            )

# 获取 filtered_csv_data
@app.get("/get_filtered_csv_data", response_model=FilteredCSVDataResponse, responses={400: {"model": OBListErrMsg}})
def pipeline_csv(query_dt: str, station: str, telescope: int) -> FilteredCSVDataResponse:
    try:
        _, filtered_csv_data = pipeline_process(query_dt, station, telescope)
        if filtered_csv_data.empty:
            return JSONResponse(
                status_code=400,
                content={"err_msg": "No matching data found"},
            ) 
        filtered_csv_data = filtered_csv_data.to_dict(orient='records')
        response = {
            "query_dt": query_dt,
            "station": station,
            "telescope": telescope,
            "filtered_csv_data": filtered_csv_data
        }
        return FilteredCSVDataResponse(**response)
    except Exception as e:
        return JSONResponse(
                status_code=400,
                content={"err_msg": str(e)},
            )