# %%
# 需要import的包
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, time, timedelta
from pathlib import Path
from queue import Queue
from typing import Iterable
from xml.etree.ElementTree import SubElement

import astropy.units as u
import numpy as np
import pandas as pd
from astroplan import (
    AltitudeConstraint,
    FixedTarget,
    MoonSeparationConstraint,
    Observer,
)
from astropy.coordinates import EarthLocation, Longitude, SkyCoord
from astropy.time import Time
from icecream import ic
from loguru import logger
from loguru._logger import Logger
from time import sleep

# 切换到项目根目录作为工作目录
os.chdir(ic(Path(__file__).parents[2]))
sys.path.append(".")

# 导入模块
from src.util.Decorators.Logging.log_func_run_status import LogFuncRun, log_func_run
from src.util.Decorators.Logging.log_iteration_progress import LogIterProgress
from src.util.util import make_and_return_dir
from src.module.observable_calculator import calculate_observable, target_observable

from astropy.utils import iers
iers_file_path = r'/home/pod/miniconda3/envs/observe/finals2000A.all'
iers.conf.iers_file = iers_file_path
iers.conf.auto_download = False
iers.conf.auto_max_age = None
iers.conf.iers_degraded_accuracy = 'ignore'


ic.enable()

debug_logger = logger.bind(level="debug")
user_logger = logger.bind(level="end_user")

def calculate_observable_period(lat0, lon0):
    # 计算可观测时间段
    location = EarthLocation(lat=lat0 * u.deg, lon=lon0 * u.deg, height=0 * u.m)
    observer = Observer(location=location, timezone="Asia/Shanghai")
    date = datetime.utcnow().date()
    # 对应北京时间中午12点
    four_oclock = time(hour=4, minute=0)
    date = datetime.combine(date, four_oclock)
    date = Time(date)

    # 天文昏影与晨光
    twilight_morning = observer.twilight_morning_astronomical(date, which="next")
    twilight_evening = observer.twilight_evening_astronomical(date, which="next")

    return twilight_morning, twilight_evening


# @log_func_run(debug_logger, "")
def calculate_lst_and_corresponding_ra_range(utc_time, latitude_deg, longitude_deg, early_night=0.5, midnight = 2.0,  midmorning=2.0, early_morning = 2.0):
    """
    计算本地恒星时（LST）及对应可转换为赤经的范围。

    参数:
    utc_time (str): UTC时间字符串，格式为'YYYY-MM-DD HH:MM:SS'。
    longitude_deg (float): 地理经度，单位为度（东经为正，西经为负）。
    latitude_deg (float): 地理纬度，单位为度。
    
    early_night（蛇年春节新增）：在傍晚的时候，能够允许ra扩展的前向范围，也就意味着这些天体已经经过中天后多久（h为单位）允许进行筛选和观测。
    midnight， midmorning， early_morning： 分别为常规时候的前向和后向小时数，以及快天亮的时候的允许后向数值。
    命名规则为early意味着需要小一些，刚天黑/快天亮了，mid为常规的。


    返回:
    ra_min (Longitude): RA的估计最小值，基于LST减1小时转换而来。
    ra_max (Longitude): RA的估计最大值，基于LST加1小时转换而来。
    """
    # 将UTC时间转换为Astropy的Time对象
    utc_time_obj = Time(utc_time, format="iso", scale="utc")
    # 计算格林威治平均恒星时（GMST）
    gmst = utc_time_obj.sidereal_time("mean", "greenwich")
    # 转换为本地恒星时
    lst_hour_angle = gmst + Longitude(longitude_deg, unit=u.deg)
    # 确保结果在0-24小时范围内，虽然wrap_at已经处理了这一点，但提及以保持逻辑清晰
    lst_hour_angle.wrap_at("360d", inplace=True)

    # 转换时间差为时角进行减加操作
    '''
    1231 额外新增逻辑，不要观测过于极限的源。故而在设置ra的最大最小值时，采取了如下策略:
    当早于22点时，不要观测更早的源，当晚于3点后，不要观测更晚的源。
    '''
    hour = int(utc_time.split()[1].split(":")[0])
    if hour < 14:
        ra_min = lst_hour_angle - early_night * 15 * u.deg
    else:
        ra_min = lst_hour_angle - midnight * 15 * u.deg

    if hour > 19:
        ra_max = lst_hour_angle + early_morning * 15 * u.deg
    else:
        ra_max = lst_hour_angle + midmorning * 15 * u.deg
    # 2025.01.21更新 if hour > 19注释了，不用这个逻辑了, ra_max直接设定上去
    # if hour > 19:
    #     ra_max = lst_hour_angle + 1 * 15 * u.deg
    # else:
    #     ra_max = lst_hour_angle + 2 * 15 * u.deg

    # ra_max = lst_hour_angle + 2 * 15 * u.deg

    ra_min, ra_max = ra_min.degree, ra_max.degree

    return ra_min, ra_max


# @log_func_run(debug_logger, "")
def is_target_observable_in_interval(obj, interval_time, lat, lon, d_moon = 15):
    """
    从天文观测者的位置来确定是否可观测

    参数：
    obj：原先一直在用的obj，转化为了json格式
    interval_time（int)：12min的那个
    lat0 & lon0:当地经纬度，度为单位

    返回：
    是否可观测，Boolean
    """
    ra = obj["ra"]  # 赤经
    dec = obj["dec"]  # 赤纬

    altconstrain = 40 if lat == 35.678 else 30
    
    observable = target_observable(interval_time, lat, lon, ra, dec, altconstrain, d_moon)
    print(f'observable: {observable}')
    return observable

# 4.7 更新
def load_config(config_path):
    """
    从JSON配置文件中加载配置
    
    参数:
    config_path (str): 配置文件路径
    
    返回:
    dict: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        return json.load(file)

## 12.30更新，增加大量默认值
def create_capture_sequence_xml(obj, config_path=None):
    # 4.7更新
    # 添加默认值
    AutoFocusOnStart = "false"  # 或从配置文件中读取
    AutoFocusOnFinish = "false"
    
    # 从配置文件中读取设置（如果有）
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        AutoFocusOnStart = str(config.get("AutoFocusOnStart", False)).lower()  # 转换为字符串
        AutoFocusOnFinish = str(config.get("AutoFocusOnFinish", False)).lower()  # 转换为字符串
    # ----

    ra = obj["ra"]
    dec = obj["dec"]
    TargetName = obj["objname"]

    if config_path is None:
        ExposureTime = obj.get("ExposureTime", 120)  
        TotalExposureCount = obj.get("TotalExposureCount", 3)
        AutoFocusOnStart = str(obj.get("AutoFocusOnStart", "false")).lower()
        FilterType = str(obj.get("FilterType", ["L"]))
        FilterType = FilterType[0]
    else: # 4.7 更新
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = json.load(file)
        filter_type = config_data.get("FilterType", ["L"])  # 添加默认值
        TotalExposureCount = config_data.get("TotalExposureCount", 3)  # 添加默认值
        ExposureTime = config_data.get("ExposureTime", 120)  # 添加默认值
        FilterType = filter_type[0] if isinstance(filter_type, list) else str(filter_type)  # 确保FilterType被定义
    # else:
    #     with open(config_path, 'r', encoding='utf-8') as file:
    #         config_data = json.load(file)
    #     filter_type = config_data.get("FilterType")
    #     TotalExposureCount = config_data.get("TotalExposureCount")
    #     ExposureTime = config_data.get("ExposureTime")

    # ImageType设定
    # 定义允许的ImageType列表
    allowed_image_types = ["LIGHT", "DARK", "FLAT", "BIAS", "SNAPSHOT"]
    ImageType = obj.get("ImageType", "LIGHT")
    if ImageType not in allowed_image_types:
        ImageType = "LIGHT"
    else:
        ImageType = str(ImageType)  

    # ra, dec to hms or dms
    ra_hours = int(ra / 15)
    ra_minutes = int(((ra / 15) - ra_hours) * 60)
    ra_seconds = (((ra / 15) - ra_hours) * 60 - ra_minutes) * 60
    dec_degrees = int(dec)
    dec_minutes = int((dec - dec_degrees) * 60)
    dec_seconds = ((dec - dec_degrees) * 60 - dec_minutes) * 60

    # 创建根元素
    capture_sequence_list = ET.Element("CaptureSequenceList")
    # 4.7 更新
    capture_sequence_list.set("SlewToTarget", "true")
    capture_sequence_list.set("AutoFocusOnStart", AutoFocusOnStart)  
    capture_sequence_list.set("CenterTarget", "true")
    capture_sequence_list.set("RotateTarget", "false")
    capture_sequence_list.set("StartGuiding", "true")
    capture_sequence_list.set("AutoFocusOnFilterChange", "false")
    capture_sequence_list.set("AutoFocusAfterSetTime", "false")
    # ... 其余类似属性设置 ...
    # ...
    capture_sequence_list.set("TargetName", TargetName)
    capture_sequence_list.set("Mode", "ROTATE")
    capture_sequence_list.set("RAHours", str(ra_hours))
    capture_sequence_list.set("RAMinutes", str(ra_minutes))
    capture_sequence_list.set("RASeconds", str(ra_seconds))
    capture_sequence_list.set("DecDegrees", str(dec_degrees))
    capture_sequence_list.set("DecMinutes", str(dec_minutes))
    capture_sequence_list.set("DecSeconds", str(dec_seconds))
    capture_sequence_list.set("PositionAngle", "350")
    capture_sequence_list.set("Delay", "0")
    capture_sequence_list.set("SlewToTarget", "true")
    capture_sequence_list.set(
        "AutoFocusOnStart", AutoFocusOnStart
    )  
    capture_sequence_list.set("CenterTarget", "true")
    capture_sequence_list.set("RotateTarget", "false")
    capture_sequence_list.set("StartGuiding", "true")
    capture_sequence_list.set("AutoFocusOnFilterChange", "false")
    capture_sequence_list.set("AutoFocusAfterSetTime", "false")
    capture_sequence_list.set("AutoFocusSetTime", "30")
    capture_sequence_list.set("AutoFocusAfterSetExposures", "false")
    capture_sequence_list.set("AutoFocusSetExposures", "10")
    capture_sequence_list.set("AutoFocusAfterTemperatureChange", "false")
    capture_sequence_list.set("AutoFocusAfterTemperatureChangeAmount", "5")
    capture_sequence_list.set("AutoFocusAfterHFRChange", "false")
    capture_sequence_list.set("AutoFocusAfterHFRChangeAmount", "10")

    # 添加多个CaptureSequence元素
    for i, filter_name in enumerate(FilterType, start=1):
        capture_sequence = SubElement(capture_sequence_list, "CaptureSequence")
        SubElement(capture_sequence, "Enabled").text = "true"
        SubElement(capture_sequence, "ExposureTime").text = str(ExposureTime)
        SubElement(capture_sequence, "ImageType").text = "LIGHT"

        filter_type = SubElement(capture_sequence, "FilterType")
        SubElement(filter_type, "Name").text = filter_name
        SubElement(filter_type, "FocusOffset").text = "0"
        SubElement(filter_type, "Position").text = "1"
        SubElement(filter_type, "AutoFocusExposureTime").text = "-1"
        SubElement(filter_type, "AutoFocusFilter").text = "true"

        flatwizard = SubElement(filter_type, "FlatWizardFilterSettings")
        SubElement(flatwizard, "FlatWizardMode").text = "DYNAMICEXPOSURE"
        SubElement(flatwizard, "HistogramMeanTarget").text = "0.5"
        SubElement(flatwizard, "HistogramTolerance").text = "0.1"
        SubElement(flatwizard, "MaxFlatExposureTime").text = "20"
        SubElement(flatwizard, "MinFlatExposureTime").text = "0.01"
        SubElement(flatwizard, "MaxAbsoluteFlatDeviceBrightness").text = "1"
        SubElement(flatwizard, "MinAbsoluteFlatDeviceBrightness").text = "0"
        SubElement(flatwizard, "Gain").text = "-1"
        SubElement(flatwizard, "Offset").text = "-1"

        binning0 = SubElement(flatwizard, "Binning")
        SubElement(binning0, "X").text = "1"
        SubElement(binning0, "Y").text = "1"

        afbinning = SubElement(filter_type, "AutoFocusBinning")
        SubElement(afbinning, "X").text = "1"
        SubElement(afbinning, "Y").text = "1"

        SubElement(filter_type, "AutoFocusGain").text = "-1"
        SubElement(filter_type, "AutoFocusOffset").text = "-1"

        binning1 = SubElement(capture_sequence, "Binning")
        SubElement(binning1, "X").text = "1"
        SubElement(binning1, "Y").text = "1"

        SubElement(capture_sequence, "Gain").text = "-1"
        SubElement(capture_sequence, "Offset").text = "-1"
        SubElement(capture_sequence, "TotalExposureCount").text = str(
            TotalExposureCount
        )
        SubElement(capture_sequence, "ProgressExposureCount").text = "0"
        SubElement(capture_sequence, "Dither").text = "false"
        SubElement(capture_sequence, "DitherAmount").text = "1"

    # 添加Coordinates元素
    coordinates = SubElement(capture_sequence_list, "Coordinates")
    SubElement(coordinates, "RA").text = f"{ra_hours + ra_minutes/60 + ra_seconds/3600}"
    SubElement(coordinates, "Dec").text = (
        f"{dec_degrees + dec_minutes/60 + dec_seconds/3600}"
    )
    SubElement(coordinates, "Epoch").text = "J2000"

    # 添加NegativeDec元素
    SubElement(capture_sequence_list, "NegativeDec").text = "true"

    # 返回构造好的XML
    return capture_sequence_list


'''源单独输入以更改观测列表'''
@debug_logger.catch(reraise=True)
def target_observable_meridian(ra, dec, lat0, lon0):
    """
    判断目标是否可以在今晚观测，并返回是否可观测（observable)和 适合观测的时间（meridian）
    下面一堆被注释掉的print是用于debug的，视情况放出来
    """
    ra0 = ra  # 赤经
    dec0 = dec  # 赤纬

    # 计算可观测时间段
    location = EarthLocation(lat=lat0*u.deg, lon=lon0*u.deg, height=0*u.m)
    observer = Observer(location=location, timezone="Asia/Shanghai")
    date = datetime.utcnow().date()
    # 对应北京时间中午12点
    four_oclock = time(hour=4, minute=0)
    date = datetime.combine(date, four_oclock)
    date = Time(date)
    
    # 天文昏影与晨光
    twilight_morning = observer.twilight_morning_astronomical(date, which='next')
    twilight_evening = observer.twilight_evening_astronomical(date, which='next')

    # 计算目标在今天是否可观测
    coord = SkyCoord(ra0 * u.deg, dec0 * u.deg, frame='icrs')
    target = FixedTarget(coord, name='My target')

    altconstrain = 40 if lat0 == 35.678 else 30

    # 2025.01.21更新 利用de440s.bsp星表解决月距问题
    constraints = [AltitudeConstraint(min= altconstrain *u.deg), MoonSeparationConstraint(min=20*u.deg, ephemeris='/home/pod/shared-nvme/NGSS/agent/src/module/de440s.bsp')]    
    #observable
    applied_constraints = [constraint(observer, target, 
                            time_range=[Time(twilight_evening), Time(twilight_morning)],
                            time_grid_resolution=6*u.minute, #用12以节省时间
                            grid_times_targets=True)
                        for constraint in constraints]
    '''
    # 打印每个约束条件的结果
    for i, result in enumerate(applied_constraints):
        print(f"Constraint {i+1} result shape: {result.shape}")
    print(f"applied_constraints content: {applied_constraints}")
    '''

    constraint_arr = np.logical_and.reduce(applied_constraints)
    '''
    print(f"constraint_arr.shape{constraint_arr.shape}")
    print(f"constraint_arr{constraint_arr}")
    '''
    constraint_arr = constraint_arr.flatten()
    print(f"constraint_arr_flatten.shape{constraint_arr.shape}")
    print(f"constraint_arr_flatten{constraint_arr}")


    # 如果可观测，则计算最接近中天时刻的可观测时间
    if np.any(constraint_arr):
        # 寻找连续的可观测时间段
        observable_times = np.where(constraint_arr)[0]
        observable_segments = []

        start = None
        for i, obs in enumerate(observable_times):
            if start is None and obs:
                start = obs
            elif i == len(observable_times) - 1 or observable_times[i+1] != obs + 1:
                observable_segments.append((start, obs))
                start = None

        # 如果最后一个可观测段没有添加
        if start is not None:
            observable_segments.append((start, observable_times[-1]))

        # 如果只有一个连续段，返回中间的值
        if len(observable_segments) == 1:
            start, end = observable_segments[0]
            mid_time = (start + end) / 2
            #print(f"只有一个连续段,{(twilight_evening + mid_time * 12*u.minute).iso}")
            return True, (twilight_evening + mid_time * 6*u.minute).iso
        else:
            # 如果有多个连续段，比较长度并返回最长的那个的起始或结束时间
            longest_segment = max(observable_segments, key=lambda seg: seg[1] - seg[0])
            if len(longest_segment) == 1:
                # 如果最长段只有一个时间点，返回该时间点
                #print(f"多个连续段，且最长仅有一个时间点,{(twilight_evening + longest_segment[0] * 12*u.minute).iso}")
                return True, (twilight_evening + longest_segment[0] * 6*u.minute).iso
            else:
                start, end = longest_segment
                if end - start >= 1:
                    # 如果最长段有多于一个时间点，返回中间的时间点
                    mid_time = (start + end) / 2
                    #print(f"多个连续段，且最长段有多个时间点,{(twilight_evening + mid_time * 12*u.minute).iso}")
                    return True, (twilight_evening + mid_time * 6*u.minute).iso
                else:
                    # 如果最长段只有一个时间点，返回该时间点
                    return True, (twilight_evening + start * 6*u.minute).iso
    else:
        return False, None

'''
这里就是读取观测列表，并且开始填入观测列表或者修改观测列表了。
输入的replaced_object就是被替换掉的源，如果是从昨天弄下来的观测列表，势必有的昨天观测的源会被挤出去
用这个replaced_object把这些源接住报出来。
'''
@debug_logger.catch(reraise=True)
def update_ob_list(file_path, meridian, row, replaced_object):
    '''
    file_path: 观测筛选出来的那天的json文件的路径
    meridian: 中天时间
    row: row = df.query(f"objname == '{obj}'")，是源的目标名称
    replaced_object: 被替换掉的源，只是一个列表，用来报告的
    '''


    replace_row = None  # 初始化replace_row，如果挤掉了一个源，这个被挤掉的源就是replace_row
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            observation_schedule = json.load(json_file)
    except Exception as e:
        user_logger.error(f"An error occurred while loading the JSON file: {e}")
        #有的时候会报错，没搞懂为什么

    for interval_time in observation_schedule.keys():  # 只遍历时间间隔
        try:
            format_str = "%Y-%m-%d %H:%M:%S.%f"
            interval_datetime = datetime.strptime(interval_time, format_str)
            meridian_datetime = datetime.strptime(meridian, format_str)
            time_diff = abs(interval_datetime - meridian_datetime)
            #print(f"time_diff for interval{interval_datetime} is {time_diff}")
        except Exception as e:
            print(f"An error occurred while loading the JSON file: {e}")
            #同样的傻逼问题，有的时候会报错，没搞懂为什么

        if time_diff.total_seconds() < 6 * 60 + 1:
            '''
            如果观测时间合适了，也就是上面返回的meridian的观测时间和某个时间段基本重合了，我就把他替换掉。
            下面的两种情况分别对应着空的target和已经有了源的target。
            '''
            #print(f"interval_datetime: {interval_datetime};\n meridian_datetime:{meridian_datetime};\n time_diff:{time_diff}")
            if observation_schedule[interval_time]["target"] == "":
                observation_schedule[interval_time]["target"] = row  # 更新字典中的'target'键
                print(f"row {row} is now in observation_schedule {interval_time}")
            else:
                replace_row = observation_schedule[interval_time]["target"]
                observation_schedule[interval_time]["target"] = row  # 更新字典中的'target'键
                print(f"row {row} is now in observation_schedule {interval_time}, and replaced {replace_row}")
            break

    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(observation_schedule, json_file, ensure_ascii=False, indent=4)
    
    if replace_row is not None:
        replaced_object.append(replace_row)
        #如果替换了一个东西，那就把被挤出去的东西记录下来，最后返回。

    return replaced_object

@debug_logger.catch(reraise=True)
def claim_objects(sorted_stations, data_path, objlist):
    '''
    声明需要加入观测列表的源。
    这里声明的源可能不一定能够观测，所以有两个初始化项，不可观测的源notfound_object和原先被挤掉的源replaced_object。
    输入是按照纬度从低到高排序好的台站。需要注意的是这个台站一定是**排序好的**。
    data_path一样的
    objlist是观测者声明的需要观测的天体列表。格式为 string: objlist = 'NGC5068，ESO576-031，UGC04730'
    经过string parse 形成objlist = ['NGC5068' , 'ESO576-031', 'UGC04730']
    '''
    replaced_object = []
    notfound_object =[]
    notobservable_object = []
    
    objects = objlist.split(',')
    for obj in objects:
        found = False
        stations_to_process = sorted_stations[:-1]
        for station in reversed(stations_to_process):
            #注意这里从高纬度往低纬度的找，找到了就可以看
            station_name = station['Name']
            csv_file = os.path.join(data_path, f"{station_name}.csv")
            df = pd.read_csv(csv_file)
            # 检查对象是否存在于CSV文件中
            if obj in df['objname'].values:
                row = df.query(f"objname == '{obj}'")
                # 打印转换后的字典
                ra = row['ra'].values[0]
                dec = row['dec'].values[0]
                user_logger.info(f"{obj} found at {station_name}.csv")
                row_dict = row.to_dict(orient='records')[0] 
                found = True
                lat = station['lat']
                lon = station['lon']
                num = station['num']
                [observable, meridian] = target_observable_meridian(ra, dec, lat, lon)
                user_logger.info(f"{obj} is {observable} at {station}, meridian is {meridian}, found = {found}")
                if observable:
                    file_path = f'{data_path}/output_{station_name}/{num}_Schedule.json'
                    replaced_object = update_ob_list(file_path,meridian, row_dict, replaced_object)
                    df = df[df['objname'] != obj ]
                    pd.DataFrame(df, columns=['objname', 'ra', 'dec', 'distance']).to_csv(csv_file, index=False)
                    user_logger.info(f"the dataframe is saved to path {csv_file}")
                    break

        if not found:
            #not found的原因是这个源在50_bright_03里边
            try:
                # 读取站点的CSV文件
                csv_file = os.path.join(data_path, "50_bright_03.csv")
                df = pd.read_csv(csv_file)
                station = sorted_stations[-1]
                # 检查对象是否存在于CSV文件中，如果不在就直接告诉你notfound了
                if obj in df['objname'].values:
                    row = df.query(f"objname == '{obj}'")
                    ra = row['ra'].values[0]
                    dec = row['dec'].values[0]
                    user_logger.info(f"{obj} found at general catalog")
                    row_dict = row.to_dict(orient='records')[0]
                    for station in reversed(stations_to_process):
                        # 看看那哪个台站能看。
                        station_name = station['Name']
                        lat = station['lat']
                        lon = station['lon']
                        num = station['num'] #需要注意这里的num就直接是台站的数量，3就是3，不是第三个。
                        [observable, meridian] = target_observable_meridian(ra, dec, lat, lon)
                        user_logger.info(f"{obj} is {observable} at {station}, meridian is {meridian}, found = {found}")
                        if observable:
                            # 找到了直接用这个台站的最后一个望远镜观测，
                            # 因为最后一个望远镜是最后分配源的，分到的源应该优先级不高。
                            file_path = f'{data_path}/output_{station_name}/{num}_Schedule.json'
                            user_logger.info(f"processing data in {file_path}")
                            replaced_object = update_ob_list(file_path, meridian, row_dict, replaced_object)
                            print(replaced_object)
                            df = df[df['objname'] != obj ]
                            pd.DataFrame(df, columns=['objname', 'ra', 'dec', 'distance']).to_csv(csv_file, index=False)
                            break
                else:
                    user_logger.info(f"{obj} is not found in the input catalog or it's in the observation list.")
                    notfound_object.append(obj)
            except:
                user_logger.info(f"{obj} 1. Not found, 2. is in the observation list, or 3.not observable for all stations")
                notobservable_object.append(obj)
                
    return replaced_object, notfound_object, notobservable_object


@debug_logger.catch(reraise=True)
def claim_ra_dec_objects(sorted_stations, data_path, objname, ra: float, dec: float):
    '''
    声明需要加入观测列表的源。
    这里声明的源可能不一定能够观测，所以有两个初始化项，不可观测的源notfound_object和原先被挤掉的源replaced_object。
    输入是按照纬度从低到高排序好的台站。需要注意的是这个台站一定是**排序好的**。
    data_path一样的
    objlist是观测者声明的源名称，ra和dec是经纬度，以度为单位
    '''
    replaced_object = []
    found = False
    stations_to_process = sorted_stations[:-1]
    for station in reversed(stations_to_process):
        #注意这里从高纬度往低纬度的找，找到了就可以看
        station_name = station['Name']
        lat = station['lat']
        lon = station['lon']
        num = station['num']
        [observable, meridian] = target_observable_meridian(ra, dec, lat, lon)
        if observable:
            found = True
            formatted_row = {
                "objname": str(objname),  
                "ra": ra,          # 强制转换为浮点数
                "dec": dec,        # 强制转换为浮点数
                "distance": 0             # 添加distance信息
            }
            user_logger.info(f"{obj} is {observable} at {station}, meridian is {meridian}")
            file_path = f'{data_path}/output_{station_name}/{num}_Schedule.json'
            replaced_object = update_ob_list(file_path, meridian, formatted_row, replaced_object)
            break
        else:
            user_logger.info(f"{objname} is not observable at {station_name}.")
    
    if not found:
        user_logger.info(f"{objname} is not observable for all stations")
        notobservable_object = [objname]
                
    return replaced_object, [], notobservable_object


'''将指定的源排序进观测列表的过程完成'''

@LogFuncRun
@debug_logger.catch(reraise=True)
def init_ob_list(station_data, data_path: Path, num: int, debug_logger, **kwargs):

    lat0 = station_data["lat"]
    lon0 = station_data["lon"]
    name = station_data["Name"]

    # 计算可观测时间段
    tw_morning, tw_evening = calculate_observable_period(lat0, lon0)

    # 读取观测config
    config_path = data_path / "observe_config.json"
    with open(config_path, 'r', encoding='utf-8') as file:
        config_data = json.load(file)
    filter_type = config_data.get("FilterType")
    TotalExposureCount = config_data.get("TotalExposureCount")
    ExposureTime = config_data.get("ExposureTime")
    WaitTime = config_data.get("WaitTime")
    ob_time = len(filter_type) * TotalExposureCount * ExposureTime

    # 生成时间间隔
    intervals = []
    current_time = tw_evening
    while current_time < tw_morning:
        intervals.append(current_time)
        current_time += ob_time * u.minute 
        current_time += WaitTime * u.minute  # 2025.1.24更新：额外提供1min冗余

    # 初始化分配结果
    observation_schedule = {
        interval.iso: {"target": ""} for interval in intervals
    }  # 每个时间间隔都是空的，待分配状态

    # 指定要写入的文件路径
    output_path: Path = make_and_return_dir(data_path / f"output_{name}")
    file_path = output_path / f"{str(num+1)}_Schedule.json"

    # 使用json.dump()方法将字典写入文件
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(observation_schedule, json_file, ensure_ascii=False, indent=4)


@LogFuncRun
@debug_logger.catch(reraise=True)
def process_station(station_data, data_path: Path, num: int, debug_logger, **kwargs):
    lat0 = station_data['lat']
    lon0 = station_data['lon']
    name = station_data['Name']

    # 指定要写入的文件路径
    output_path: Path = make_and_return_dir(data_path / f"output_{name}")
    file_path = output_path / f"{str(num+1)}_Schedule.json"
    # 读取观测config
    config_path = data_path / "observe_config.json"

    #读取json下的观测列表
    with open(file_path, 'r', encoding='utf-8') as json_file:
        observation_schedule = json.load(json_file)
    user_logger.success(f"🤗 已完成初始化观测列表")

    try:
        data_path1 = data_path / f"{name}.csv"  # 先从本地观测列表开始
        observation_schedule = assign_object_to_list(
            observation_schedule,
            lat0,
            lon0,
            data_path1,
            config_path,
            log_note={"阶段": "本地观测列表"},
            debug_logger=debug_logger,
        )
        ic(observation_schedule)
    except Exception as e:
        user_logger.warning(
            f"⚠️ Failed to plan observation from local observation list. Error type is: {type(e)}. Error msg is: {e}"
        )
        user_logger.info(f"Begin to use 50 select")
        data_path1 = (
            data_path/ "50_bright_03.csv"  # 如果没有本地观测列表，就直接用50_select
        )
        observation_schedule = assign_object_to_list(
            observation_schedule,
            lat0,
            lon0,
            data_path1,
            config_path,
            log_note={"阶段": "没有本地观测列表，采用50_bright_03"},
            debug_logger=debug_logger,
        )

    # 观测列表可能没有填满，用50_bright_03再过一遍
    user_logger.info("Go through 50 select again.")
    data_path2 = data_path / "50_bright_03.csv"
    observation_schedule = assign_object_to_list(
        observation_schedule,
        lat0,
        lon0,
        data_path2,
        config_path,
        log_note={"阶段": "未填满，采用50_bright_03"},
        debug_logger=debug_logger,
    )
    
    '''这里还需要加一段。如果还有源没有被填满，可以采用重复源进行观测。但是需要体现在log里边'''
    repeat_index = 2
    while True:
        all_targets_assigned = all(schedule["target"] != "" for schedule in observation_schedule.values())

        if all_targets_assigned:
            user_logger.info(f"{name}站点{str(num+1)}号望远镜：今晚所有的观测时间段都已经被分配。")
            break
        else:
            user_logger.info(f"{name}站点{str(num+1)}号望远镜：今晚有的观测时间段未被分配，将进行第{repeat_index}次分配。需注意，这将产生重复观测。")            
            # 第一步：复制最外层的50_bright_03到文件夹内。
            shutil.copy(os.path.join("/home/pod/shared-nvme/NGSS/agent/data/50_bright_03.csv", "50_bright_03.csv"), os.path.join(data_path, "50_copy.csv"))
            user_logger.info(f"文件已从成功复制")
            # 第二步：try
            user_logger.info("Go through 50 select again.")
            data_path2 = data_path / "50_copy.csv"
            observation_schedule = assign_object_to_list(
                observation_schedule,
                lat0,
                lon0,
                data_path2,
                config_path,
                log_note={"阶段": "重复置入观测源。"},
                debug_logger=debug_logger,
            )
            repeat_index = repeat_index+1
                
    
    # 指定要写入的文件路径
    output_path: Path = make_and_return_dir(data_path / f"output_{name}")
    file_path = output_path / f"{str(num+1)}_Schedule.json"

    # 使用json.dump()方法将字典写入文件
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(observation_schedule, json_file, ensure_ascii=False, indent=4)

    pathname = output_path / f"{num+1}.ninaTargetSet"
    to_OBlist(observation_schedule, pathname, config_path, user_logger)
    user_logger.success(f"🤗 已保存targetset至{pathname}")



@LogFuncRun
@debug_logger.catch(reraise=True)
def assign_object_to_list(observation_schedule, lat0, lon0, data_path, config_path, **kwargs):
    """
    现具体的观测源分配逻辑，包括读取CSV、分配、移除已分配源等操作

    参数:
    observation_schedule（list)：这个就是后面需要写的那个了
    data_path：数据的读取路径，是前面创建了新的文件夹，然后往文件夹里放本地的观测可选列表，以及50_select表的路径
    lat0,lon0
    name:台站名字

    返回:
    分配好的observation_schedule
    """

    # 使用Pandas读取CSV文件
    data = pd.read_csv(data_path)
    data["ra"] = data["ra"].astype(float)
    data["dec"] = data["dec"].astype(float)
    data["distance"] = data["distance"].astype(float)
    # 将DataFrame转换为字典列表
    data = data[["objname", "ra", "dec", "distance"]].to_dict(orient="records")

    with open(config_path, 'r', encoding='utf-8') as file:
        config_data = json.load(file)
    early_night = config_data.get("early_night")
    midnight = config_data.get("midnight")
    midmorning = config_data.get("midmorning")
    early_morning = config_data.get("early_morning")
    d_moon = config_data.get("d_moon")


    for index, interval_time in enumerate(
        observation_schedule.keys()
    ):  # 只遍历时间间隔
        user_logger.info(
            f"⏰ Begin to find suitable star for interval: {str(interval_time)}"
        )

        if (
            observation_schedule[interval_time]["target"] == ""
        ):  # 判断条件针对字典中的'target'值
            # 给出计算后的ra的最大最小值，用于筛选ra，加速计算
            ra_min, ra_max = calculate_lst_and_corresponding_ra_range(
                interval_time, lat0, lon0, early_night, midnight,  midmorning, early_morning
            )
            # 使用列表推导式筛选满足条件的字典
            filtered_data = [item for item in data if ra_min <= item["ra"] <= ra_max]
            # dec 从小到大排序
            filtered_data = sorted(filtered_data, key=lambda item: item["dec"], reverse=False)
            user_logger.info(f"🌟 找到了{len(filtered_data)}个源")

            assigned = False
            for obj in filtered_data:
                # 在筛选后的表中依次搜寻是否满足可观测条件
                if is_target_observable_in_interval(obj, interval_time, lat0, lon0, d_moon):
                    observation_schedule[interval_time][
                        "target"
                    ] = obj  # 更新字典中的'target'键
                    assigned = True
                    # 一旦分配成功，跳出循环寻找下一个interval_time的目标
                    break
            if assigned:
                data.remove(obj)  # 从原始数据中移除已分配的目标

    # 覆盖一遍读取的csv文件
    pd.DataFrame(data, columns=["objname", "ra", "dec", "distance"]).to_csv(
        data_path, index=False
    )
    sleep(1)

    return observation_schedule


#12.30 这里有修改，改为在特定位置增加bias和开启自动调焦。
def to_OBlist(OBlist, pathname, config_path, user_logger):
    """
    将上面一大段的create_capture_sequence_xml得到的单目标源汇总为一个TargetSet并且输出

    参数:
    OBlist：观测列表
    path_name：最终targetSet保存路径

    返回:
    无，直接输出了
    """

    # 读取现有XML文档或创建新文档
    try:
        tree = ET.parse(pathname)
        root = tree.getroot()
    except:
        # 如果解析失败，创建一个新的XML文档结构
        root = ET.Element("ArrayOfCaptureSequenceList")
        tree = ET.ElementTree(root)

    null_target = 0
    target_counter = 0  # 新增计数器

    for timestamp, target_info in OBlist.items():
        ic(target_info)
        target = target_info["target"]
        try:
            assert target != ""
        except AssertionError:
            null_target += 1
            user_logger.warning(
                f"⚠️ target is null! Accumulated null target: {str(null_target)}"
            )
        else:
            target_counter += 1  # 增加计数器
            # 每处理15个目标后，在target中添加 "AutoFocusOnStart": True
            if target_counter > 0 and target_counter % 15 == 1:
                target["AutoFocusOnStart"] = True
            
            capture_sequence_list = create_capture_sequence_xml(target, config_path)
            # 将当前捕获序列追加到XML树的根下
            root.append(capture_sequence_list)

            # 每处理30个目标后，添加一个bias目标源
            if target_counter == 30:
                bias_target = {
                    "objname": "bias",
                    "ra": target.get("ra", 0),
                    "dec": target.get("dec", 0),
                    "distance": 0,
                    "ExposureTime": 0,
                    "TotalExposureCount": 10,
                    "ImageType" : "BIAS"
                }
                bias_capture_sequence_list = create_capture_sequence_xml(bias_target)
                root.append(bias_capture_sequence_list)

    # 4.7 更新
    # 确保pathname是字符串
    xml_path = str(pathname) if isinstance(pathname, Path) else pathname
    # 仅在所有遍历和追加操作完成后，一次性写回文件
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    # tree.write(pathname, encoding="utf-8", xml_declaration=True)
    # 读文档的就这样就行


@LogFuncRun
@debug_logger.catch(reraise=True)
def load_from_theday(thedate: str, d_moon, debug_logger, log_note):
    user_logger.info(
            f"🐱：开始继承来自{thedate}的观测计划"
        )
    today = datetime.now().strftime("%Y%m%d")
    # 构造昨天的文件夹路径
    yesterday_folder_path = Path("data") / thedate

    new_folder_path = Path("data") / today
    os.makedirs(new_folder_path, exist_ok=True)

    stations_json_path = r"data/sorted_observatories.json"
    with open(stations_json_path, "r", encoding="utf-8") as file:
        sorted_stations = json.load(file)

    # 遍历并处理每个站点
    for station in sorted_stations:
        station_name = station["Name"]
        lat0 = station["lat"]
        lon0 = station["lon"]
        num_assignments = ic(station["num"])
        # 根据站点的num属性重复处理过程
        # 先生成新的观测列表
        for num in range(num_assignments):

            '''
            检查前文文件，如果存在，将列表读取出来，给新的init_oblist的每个列表分配源。
            如果可观测则保留，不可观测则去掉该目标源。
            '''

            output_path: Path = make_and_return_dir(yesterday_folder_path / f"output_{station_name}")
            file_old = output_path / f"{str(num+1)}_Schedule.json"
            output_path: Path = make_and_return_dir(new_folder_path / f"output_{station_name}")
            file_new = output_path / f"{str(num+1)}_Schedule.json"

            with open(file_old, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # 提取所有objName的值
            objlist = [entry['target'] for entry in data.values()]
            user_logger.info(
            f"🐱：读取到了{thedate}的{station_name}站的{str(num+1)}号望远镜的观测计划")


            with open(file_new, 'r', encoding='utf-8') as file:
                observation_schedule = json.load(file)
            
            for index, interval_time in enumerate(observation_schedule.keys()):  # 只遍历时间间隔
                if index < len(objlist):  # 确保 index 不超过 objlist 的长度
                    if (observation_schedule[interval_time]["target"] == ""):  
                        obj = objlist[index]
                        if is_target_observable_in_interval(obj, interval_time, lat0, lon0, d_moon):
                            observation_schedule[interval_time]["target"] = obj
                            objname = obj["objname"]
                            user_logger.info(f"🐱：继承了时间段为{interval_time}的观测目标。该目标为{objname}.")
                        else:
                            user_logger.info(f"🐱：{interval_time}时段的观测目标未被继承，因为{objname}目标不可见.")
                else:
                    user_logger.warning(f"🐱：今天的观测列表的第 {index+1}项超过了昨天观测列表的长度，无法继续继承观测目标。")
                                            
            
            with open(file_new, "w", encoding="utf-8") as json_file:
                json.dump(observation_schedule, json_file, ensure_ascii=False, indent=4)
            
            user_logger.info(f"🐱：{thedate}的{station_name}站的{str(num+1)}号望远镜的观测计划继承完毕，正在保存中。")
        


def init():
    pass


def create_session_log(uu_id: str, current_date: str):
    date_log = make_and_return_dir(f"log/{current_date}")
    uuid_hist_log_path = date_log / "uuid_hist.log"
    with open(uuid_hist_log_path, "a") as f:
        f.write(f"{uu_id} {str(int(datetime.now().timestamp()))}\n")

    log_path = Path(f"log/{current_date}") / f"{uu_id}.log"
    debug_path = Path(f"log/{current_date}") / f"{uu_id}_debug.log"
    logger.add(log_path, mode="w", filter=lambda x: x["extra"]["level"] == "end_user")
    logger.add(
        debug_path,
        level="TRACE",
        mode="w",
        filter=lambda x: x["extra"]["level"] == "debug",
    )


'''
单独写一个小程序，用于最后返回观测列表，检查后提出新增目标源。直接调用claim_objects即可。
需要多说一句，如果要从昨天继承源，那么不能在预先指定必须要观测的源，这两者冲突。
'''
# 接在 update station后面的，更新对应的配置文件
@debug_logger.catch(reraise=True)
def modify_config(key, value, config_path):
    """
    修改指定配置项的值。
    
    :param key: 配置项名。
    :param value: 新值（会尝试转换为合适的类型）。
    """
    try:
        assert os.path.exists(config_path)
    except:
        user_logger.error("❌ 配置文件不存在!")
    else:
        user_logger.success("✔️ 配置文件存在!")
    
    config_types = {
    "inherit": bool,
    "early_night": float,
    "midnight": float,
    "midmorning": float,
    "early_morning": float,
    "d_moon": float,
    "FilterType": list,  
    "TotalExposureCount": int,
    "ExposureTime": int,
    "WaitTime": float
    }

    def convert_value(value_str, target_type):
        """根据目标类型转换字符串值"""
        if target_type == bool:
            return value_str.lower() in ['true', '1', 't', 'y', 'yes']
        elif target_type == list:
            # 首先尝试使用逗号分割，如果失败则尝试使用空格分割
            items = value_str.split(',')
            if len(items) == 1:  # 如果没有逗号分隔符
                items = value_str.split()  # 使用默认的空白字符作为分隔符（包括空格、制表符等）
            return [item.strip() for item in items]
        else:
            return target_type(value_str)

    with open(config_path, 'r') as file:
        config = json.load(file)
    
    if key not in config:
        user_logger.error(f"KeyError: {key}' 不是有效的配置项。")
        raise KeyError(f"'{key}' 不是有效的配置项。")

    if key not in config_types:
        user_logger.error(f"ValueError: 未定义'{key}'的类型信息，请检查'config_types'映射")
        raise ValueError(f"未定义'{key}'的类型信息，请检查'config_types'映射。")
    
    # 根据预定义的类型转换新值
    new_value = convert_value(value, config_types[key])
    config[key] = new_value
    
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

    user_logger.info(f"配置项 '{key}' 已更新为 {new_value}")
    return f"配置项 '{key}' 已更新为 {new_value}"


# 预先的路径定义
@debug_logger.catch(reraise=True)
def add_object_to_fine_list(sorted_stations:list, objlist: str = None, objname = None, ra: float = None, dec: float = None):
    current_date = datetime.now().strftime('%Y%m%d')
    data_path_base = r'/home/pod/shared-nvme/NGSS/agent/data/'
    data_path_with_date = f"{data_path_base}{current_date}"
    replaced_objs = []
    notfound_object =[]
    notobservable_object =[]

    # 更换目标源即可
    if objlist is not None:
        replaced_objs , notfound_object, notobservable_object = claim_objects(sorted_stations, data_path_with_date, objlist)
        user_logger.info(f"User provide several object which may appears in the 50 mpc catalog.")

    if objname is not None and ra is not None and dec is not None:
        replaced_objs , notfound_object, notobservable_object = claim_ra_dec_objects(sorted_stations, data_path_with_date, objname, ra, dec)
        user_logger.info(f"User provide a single object with RA and Dec.")

    user_logger.info(f"Object list processed complete, the not_found objects are {notfound_object}; the replaced objects are {replaced_objs}; the not observable objects are {notobservable_object}")
    return replaced_objs, notfound_object, notobservable_object
    



@debug_logger.catch(reraise=True)
def main(uu_id: str, q: Queue, objlist: list, thedate: str = None, inherit: bool = True):
    sub_pid = os.getpid()
    q.put(ic(sub_pid))
    debug_logger.info(f"qsize after put: {str(q.qsize())}")

    # 获取当前日期
    current_date = datetime.now().strftime("%Y%m%d")
    create_session_log(uu_id, current_date)
    user_logger.info(f"🏁 Begin to plan observation for {current_date}.")
    data_path_with_date: Path = Path("data") / current_date

    if thedate is None:
        thedate = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    theday_folder_path = Path("data") / thedate

    try:
        assert data_path_with_date.is_dir()
    except:
        user_logger.error("❌ Directory not created yet!")
    else:
        user_logger.success("✔️ Passed check directory existence check!")

    stations_json_path = r"data/sorted_observatories.json"
    with open(stations_json_path, "r", encoding="utf-8") as file:
        sorted_stations = json.load(file)

    '''在这里确定observation mode'''
    config_path = Path("data") / current_date / "observe_config.json"
    with open(config_path, 'r', encoding='utf-8') as file:
        config_data = json.load(file)
    early_night = config_data.get("early_night")
    midnight = config_data.get("midnight")
    midmorning = config_data.get("midmorning")
    early_morning = config_data.get("early_morning")
    d_moon = config_data.get("d_moon")
    filter_type = config_data.get("FilterType")
    TotalExposureCount = config_data.get("TotalExposureCount")
    ExposureTime = config_data.get("ExposureTime")
    WaitTime = config_data.get("WaitTime")

    user_logger.info(f'''
                    🐱🐱🐱确认今日观测的基础信息🐕🐕🐕
                    🐷🐷🐷🐷🐷🐷🐷🐷🐷🐷🐷🐷🐷🐷🐷
                    今日是{current_date}。我们是否选择执行继承观测计划：{inherit}，如果执行，继承{thedate}的观测计划。
                    我们的设定为：
                    傍晚允许的经过中天后的小时数：{early_night} h ;
                    整夜允许的经过中天后的小时数: {midnight} h ;
                    清晨允许的未达中天前的小时数：{early_morning} h ;
                    整夜允许的未达中天前的小时数：{midmorning} h ;
                    月距约束: {d_moon}度 ;
                    观测的波段: {filter_type} ;
                    每个波段的曝光次数: {TotalExposureCount} ;
                    每个波段的曝光时间: {ExposureTime} s ;
                    每个源观测结束后，指向与读出数据提供的额外时间: {WaitTime} min ;
                    ''')

    # 遍历并处理每个站点
    for station in sorted_stations:
        station_name = station["Name"]
        num_assignments = ic(station["num"])
        # 根据站点的num属性重复处理过程
        # 先生成新的观测列表
        for num in range(num_assignments):
            init_ob_list(
                station, data_path_with_date, num, 
                debug_logger=debug_logger,  log_note={"阶段": "初始化观测列表"},  
            ) #  

    if inherit:
        user_logger.info(f"我们开启了继承模式，将会继承{thedate}的观测计划。")
        if theday_folder_path.is_dir():  # 使用新的指定路径进行检查
            user_logger.info(f"{thedate}：彼时观测计划存在，开始继承彼时的观测计划。")
            try:
                load_from_theday(thedate, d_moon, debug_logger = debug_logger, log_note={"阶段": "导入彼时的观测列表"})
                user_logger.info(f"We have successively copied the observation list from the day.")
            except Exception as e:
                user_logger.error(f"Failed to load the day's observation plan: {e}")
    else:
        user_logger.info(f"并未开启继承模式，将会重新制定观测计划。")


        
    # 遍历并处理每个站点
    for station in sorted_stations:
        station_name = station['Name']
        num_assignments = station['num']
        for num in range(num_assignments):
            ic(num)
            process_station(
                station,
                data_path_with_date,
                num,
                debug_logger=debug_logger,
                log_note={"站点": station, "望远镜编号": num + 1},                
            )
            sleep(1)
            


# %%
