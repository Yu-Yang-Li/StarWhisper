from runtime import Args
from typings.Check_observable.Check_observable import Input, Output
import json
from datetime import datetime, time
from astroplan import Observer, FixedTarget, AltitudeConstraint, MoonSeparationConstraint
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
import numpy as np

def handler(args: Args[Input])->Output:  
    input_data = args.input
    try:
        lat = input_data.lat
    except: 
        lat = 40.393
    try:
        lon = input_data.lon
    except: 
        lon = 117.575
    try:
        low_height = input_data.low_height
    except: 
        low_height = 30  
    try:
        TimeZone = input_data.TimeZone
    except: 
        TimeZone = 8     

    List = input_data.List
      

    lat = lat if lat is not None else 40.393  # 选择一个合适的默认值
    lon = lon if lon is not None else 117.575  # 选择一个合适的默认值
    low_height = low_height if low_height is not None else 30
    TimeZone = TimeZone if TimeZone is not None else 8

     
    sunset, sunrise = get_ob_time(lat,lon)
    OBList = []
    for json_str in List:
        obj = json.loads(json_str)
        observable_start, observable_end = is_target_observable(obj['ra'], obj['dec'], lat, lon, sunset, sunrise, low_height)
        if observable_start is not None and observable_end is not None:
            obj['start'] = (observable_start + + TimeZone * u.hour).iso
            obj['end'] = (observable_end + TimeZone * u.hour).iso
            
            OBList.append(obj)
    return {"OBList": str(OBList)}

def get_ob_time(lat0,lon0):
    # 定义观察者的位置
    location = EarthLocation(lat=lat0*u.deg, lon=lon0*u.deg, height=0*u.m)
    observer = Observer(location=location)

    # 定义观测日期
    date = datetime.utcnow().date()
    four_oclock = time(hour=4, minute=0)
    date = datetime.combine(date, four_oclock)
    date = Time(date)

    # 计算今天的日落时间
    sunset = observer.sun_set_time(date, which='next')
    sunrise = observer.sun_rise_time(date, which='next')
    return sunset, sunrise


def is_target_observable(ra0, dec0, lat0, lon0, sunset_today, sunrise_tomorrow, low_height):

    # 解析RA
    ra_h, ra_m, ra_s = map(float, ra0.split(':'))
    ra0 = ra_h * 15 + ra_m / 4 + ra_s / 240  # 15度/小时, 4分钟/度, 240秒/度
    
    # 解析Dec
    sign = 1 if dec0.startswith('+') or not dec0.startswith('-') else -1
    dec_d, dec_m, dec_s = map(float, dec0.lstrip('+-').split(':'))
    dec0= sign * (dec_d + dec_m / 60 + dec_s / 3600)

    # 定义观察者的位置
    location = EarthLocation(lat=lat0*u.deg, lon=lon0*u.deg, height=0*u.m)
    observer = Observer(location=location)

    # 定义观测日期
    date = datetime.utcnow().date()
    four_oclock = time(hour=4, minute=0)
    date = datetime.combine(date, four_oclock)
    date = Time(date)

    coord = SkyCoord(ra0 * u.deg, dec0 * u.deg, frame='icrs')
    target = FixedTarget(coord)

    # 目标高度限制为30度，月距为15度
    constraints = [AltitudeConstraint(min=low_height*u.deg), MoonSeparationConstraint(min = 15*u.deg)]

    #observable
    applied_constraints = [constraint(observer, target, 
                            time_range=[sunset_today, sunrise_tomorrow],
                            time_grid_resolution=10*u.minute, 
                            grid_times_targets=True)
                        for constraint in constraints]
    constraint_arr = np.logical_and.reduce(applied_constraints)
    if not np.any(constraint_arr):
        # 如果没有任何时间点满足所有约束条件
        observable_start = None
        observable_end = None
    else:
        # 找到第一个和最后一个可观测的时间点
        observable_start = sunset_today + np.argwhere(constraint_arr).flatten()[0]*10*u.minute
        observable_end = sunset_today + np.argwhere(constraint_arr).flatten()[-1]*10*u.minute
    
    return observable_start, observable_end