# 此脚本每天运行一次。
# 需要import的包

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from icecream import ic
from loguru import logger

ic.enable()
os.chdir(ic(Path(__file__).parents[2]))  # 切换到项目根目录作为工作目录
sys.path.append(".")


def create_date_folder_and_copy_csv(source_csv_path, target_folder_base):
    # 创建以今天日期命名的文件夹
    today = datetime.now().strftime("%Y%m%d")
    new_folder_path = os.path.join(target_folder_base, today)
    os.makedirs(new_folder_path, exist_ok=True)

    # 复制CSV文件到新文件夹
    destination_csv = os.path.join(new_folder_path, os.path.basename(source_csv_path))
    shutil.copyfile(source_csv_path, destination_csv)
    
    # config也复制过去
    shutil.copyfile(r"data/observe_config.json", os.path.join(new_folder_path, "observe_config.json"))

    return new_folder_path


def process_stations_and_update_catalog(stations_data, original_star_catalog_path):
    """处理所有台站，筛选并保存星表，同时更新原始星表"""
    # 首先读取原始星表
    star_catalog = pd.read_csv(
        os.path.join(original_star_catalog_path, "50_bright_03.csv")
    ).to_dict(orient="records")

    # 确保按纬度升序排列台站，不要删除，如果已经是排列好的，这个速度也是O(1)的，可以接受
    stations_data.sort(key=lambda x: x["lat"])

    # 临时存储已筛选的星体，避免重复
    selected_stars = []

    for i, current_station in enumerate(stations_data[:-1]):
        # 通过本地台站，以及排在后面的高纬度台站，筛选出仅在本地可以观测的目标源
        # 本地的台站优先观测这些源
        next_station = stations_data[i + 1]
        min_lat = float(current_station["lat"])
        max_lat = float(next_station["lat"])

        # 筛选星表，但需检查是否已被筛选过，避免重复
        new_filtered_stars = [
            star
            for star in star_catalog
            if min_lat - 60 <= float(star["dec"]) <= max_lat - 60
            and star not in selected_stars
        ]
        selected_stars.extend(new_filtered_stars)  # 记录已筛选过的星体

        # 构建输出文件名并保存
        output_filename = f"{current_station['Name']}.csv"
        output_path = os.path.join(original_star_catalog_path, output_filename)
        pd.DataFrame(
            new_filtered_stars,
            columns=["objname", "ra", "dec", "distance", "distance_err"],
        ).to_csv(output_path, index=False, encoding="utf-8")

    # 从原始星表中移除已筛选的星体
    remaining_stars = [star for star in star_catalog if star not in selected_stars]

    # 更新原始星表
    pd.DataFrame(
        remaining_stars, columns=["objname", "ra", "dec", "distance", "distance_err"]
    ).to_csv(
        os.path.join(original_star_catalog_path, "50_bright_03.csv"),
        index=False,
        encoding="utf-8",
    )

    logger.success("所有操作已完成，原始星表已更新。")


def main():
    ic(os.getcwd())
    # 将多台站写入json文件，格式为地点，地理位置和望远镜数量。
    # 将可观测的位置分开
    # 将列表按照纬度拆成不同的星表，分为仅在第 i 个区域可以观测的星表。

    # 目前台站数据是常量。
    observatory_str = """
    {
        "Gansu": {
            "lat": "35.678",
            "lon": "106.848",
            "num": 1
        },
        "YunNan": {
            "lat": "23.914",
            "lon": "102.653",
            "num": 1
        },
        "XingLong": {
            "lat": "40.393",
            "lon": "117.575",
            "num": 7
        },
        "XinJiang": {
            "lat": "43.522",
            "lon": "88.577",
            "num": 1
        }
    }
    """

    data = json.loads(observatory_str)

    # 整理数据，确保lat, lon, num为浮点数
    sorted_observatories = [
        {
            "Name": region,
            "lat": float(details["lat"]),
            "lon": float(details["lon"]),
            "num": details["num"],
        }
        for region, details in data.items()
    ]

    # 按纬度(lat)排序
    sorted_observatories = sorted(sorted_observatories, key=lambda x: x["lat"])

    # 打印处理后的结果
    logger.info(
        f"经排序后的台站信息：{json.dumps(sorted_observatories, ensure_ascii=False, indent=2)}"
    )

    stations_json_path = r"data/sorted_observatories.json"
    # 写入JSON文件，这一段可写可不写
    with open(stations_json_path, "w", encoding="utf-8") as json_file:
        json.dump(sorted_observatories, json_file, ensure_ascii=False, indent=2)

    # 修改观测天体列表，分为只有本地可以观测的（dec低）以及最终剩下的列表
    # 最终将会优先观测本地列表，随后从剩下的50_selected里边选
    # 应用，定义路径，这里的路径需要改

    original_star_catalog_path = "data/50_bright_03.csv"

    # 创建日期文件夹并复制CSV，同时获取复制后的CSV路径
    # r'E:/一些可以用于自动观测的程序/程序集'可以修改
    copied_csv_path = create_date_folder_and_copy_csv(
        original_star_catalog_path, r"data"
    )

    # 处理台站数据，注意这里应该使用原始CSV路径读取星表数据，而用copied_csv_path来保存更新后的星表
    process_stations_and_update_catalog(sorted_observatories, copied_csv_path)
    
    # # config也复制过去
    # shutil.copyfile(r"data/observe_config.json", copied_csv_path )

    logger.success("每日定时更新台站数据任务：所有操作已完成。")


if __name__ == "__main__":
    main()
