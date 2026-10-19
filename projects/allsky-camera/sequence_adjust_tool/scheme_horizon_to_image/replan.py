from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from xml.etree.ElementTree import SubElement

import astropy.units as u
import matplotlib.image as mpimg
import numpy as np
from astroplan import (
    AltitudeConstraint,
    FixedTarget,
    MoonSeparationConstraint,
    Observer,
    is_observable,
)
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None


SCHEME_DIR = Path(__file__).resolve().parent
TOOL_DIR = SCHEME_DIR.parent
PIPELINE_ROOT = TOOL_DIR.parent
REPO_ROOT = PIPELINE_ROOT
DEFAULT_BATCH_DIR = TOOL_DIR / "batch_0116_output"
DEFAULT_CONFIG_PATH = TOOL_DIR / "observe_config.json"
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "output" / "replan"
DEFAULT_PARAMETERS_PATH = TOOL_DIR / "parameters.json"
DEFAULT_MASK_PATH = PIPELINE_ROOT / "output" / "mask" / "placeholder.png"
DEFAULT_CSV_PATH = TOOL_DIR / "50_select_bright.csv"
BEIJING_TZ = timezone(timedelta(hours=8))
DEFAULT_STATION = {
    "name": "XingLong",
    "lat": 40.393,
    "lon": 117.575,
}
DEFAULT_POSITION_NAME = "position_2"
DEFAULT_STRATEGY = "csv"


def progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def _configure_astropy_offline_data() -> None:
    # Force Astropy to use local IERS data and avoid network access.
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    local_iers = TOOL_DIR / "finals2000A.all"
    if local_iers.is_file():
        iers.earth_orientation_table.set(iers.IERS_A.open(str(local_iers)))


def _moon_constraint_kwargs() -> dict:
    if importlib.util.find_spec("jplephem") is None:
        progress("未检测到 jplephem，月距约束将回退到 Astropy 内置星历")
        return {}
    eph = os.environ.get("ASTROPY_EPHEMERIS_BSP", "").strip()
    if eph and os.path.isfile(eph):
        return {"ephemeris": eph}
    local = TOOL_DIR / "de440s.bsp"
    if local.is_file():
        return {"ephemeris": str(local)}
    return {}


def _parse_interval_time(interval_time: str) -> Time:
    raw = (interval_time or "").strip()
    try:
        return Time(raw, format="iso", scale="utc")
    except Exception:
        return Time(raw.replace(" ", "T"), format="isot", scale="utc")


def target_observable(
    interval_time: str,
    lat: float,
    lon: float,
    ra: float,
    dec: float,
    altconstrain: float,
    d_moon: float,
) -> bool:
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=0 * u.m)
    observer = Observer(location=location, timezone="Asia/Shanghai")
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    target = FixedTarget(coord, name="interval_check")
    t0 = _parse_interval_time(interval_time)
    moon_kwargs = _moon_constraint_kwargs()
    constraints = [
        AltitudeConstraint(min=altconstrain * u.deg),
        MoonSeparationConstraint(
            min=float(d_moon) * u.deg,
            **moon_kwargs,
        ),
    ]
    kwargs = {
        "time_range": [t0 - 6 * u.minute, t0 + 6 * u.minute],
        "time_grid_resolution": 3 * u.minute,
    }
    try:
        ok = is_observable(
            constraints,
            observer,
            target,
            grid_times_targets=True,
            **kwargs,
        )
    except TypeError:
        try:
            ok = is_observable(constraints, observer, target, **kwargs)
        except ImportError:
            progress("JPL 星历不可用，重试月距约束时改用内置星历")
            fallback_constraints = [
                AltitudeConstraint(min=altconstrain * u.deg),
                MoonSeparationConstraint(min=float(d_moon) * u.deg),
            ]
            ok = is_observable(fallback_constraints, observer, target, **kwargs)
    except ImportError:
        progress("JPL 星历不可用，重试月距约束时改用内置星历")
        fallback_constraints = [
            AltitudeConstraint(min=altconstrain * u.deg),
            MoonSeparationConstraint(min=float(d_moon) * u.deg),
        ]
        try:
            ok = is_observable(
                fallback_constraints,
                observer,
                target,
                grid_times_targets=True,
                **kwargs,
            )
        except TypeError:
            ok = is_observable(fallback_constraints, observer, target, **kwargs)
    return bool(np.asarray(ok, dtype=bool).any())


def create_capture_sequence_xml(obj: dict, config_path: Path) -> ET.Element:
    config = load_config(config_path)
    ra = float(obj["ra"])
    dec = float(obj["dec"])
    target_name = str(obj["objname"])
    filter_type = config.get("FilterType", ["L"])
    if isinstance(filter_type, str):
        filter_names = [filter_type]
    else:
        filter_names = [str(item) for item in filter_type]
    total_exposure_count = int(
        obj.get("TotalExposureCount", config.get("TotalExposureCount", 3))
    )
    exposure_time = int(obj.get("ExposureTime", config.get("ExposureTime", 120)))
    autofocus_on_start = str(obj.get("AutoFocusOnStart", False)).lower()
    image_type = str(obj.get("ImageType", "LIGHT"))

    ra_hours = int(ra / 15)
    ra_minutes = int(((ra / 15) - ra_hours) * 60)
    ra_seconds = (((ra / 15) - ra_hours) * 60 - ra_minutes) * 60
    dec_degrees = int(dec)
    dec_minutes = int((dec - dec_degrees) * 60)
    dec_seconds = ((dec - dec_degrees) * 60 - dec_minutes) * 60

    capture_sequence_list = ET.Element("CaptureSequenceList")
    capture_sequence_list.set("SlewToTarget", "true")
    capture_sequence_list.set("AutoFocusOnStart", autofocus_on_start)
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
    capture_sequence_list.set("TargetName", target_name)
    capture_sequence_list.set("Mode", "ROTATE")
    capture_sequence_list.set("RAHours", str(ra_hours))
    capture_sequence_list.set("RAMinutes", str(ra_minutes))
    capture_sequence_list.set("RASeconds", str(ra_seconds))
    capture_sequence_list.set("DecDegrees", str(dec_degrees))
    capture_sequence_list.set("DecMinutes", str(dec_minutes))
    capture_sequence_list.set("DecSeconds", str(dec_seconds))
    capture_sequence_list.set("PositionAngle", "350")
    capture_sequence_list.set("Delay", "0")

    for filter_name in filter_names:
        capture_sequence = SubElement(capture_sequence_list, "CaptureSequence")
        SubElement(capture_sequence, "Enabled").text = "true"
        SubElement(capture_sequence, "ExposureTime").text = str(exposure_time)
        SubElement(capture_sequence, "ImageType").text = image_type

        filter_type_node = SubElement(capture_sequence, "FilterType")
        SubElement(filter_type_node, "Name").text = filter_name
        SubElement(filter_type_node, "FocusOffset").text = "0"
        SubElement(filter_type_node, "Position").text = "1"
        SubElement(filter_type_node, "AutoFocusExposureTime").text = "-1"
        SubElement(filter_type_node, "AutoFocusFilter").text = "true"

        binning = SubElement(capture_sequence, "Binning")
        SubElement(binning, "X").text = "1"
        SubElement(binning, "Y").text = "1"
        SubElement(capture_sequence, "Gain").text = "-1"
        SubElement(capture_sequence, "Offset").text = "-1"
        SubElement(capture_sequence, "TotalExposureCount").text = str(
            total_exposure_count
        )
        SubElement(capture_sequence, "ProgressExposureCount").text = "0"
        SubElement(capture_sequence, "Dither").text = "false"
        SubElement(capture_sequence, "DitherAmount").text = "1"

    coordinates = SubElement(capture_sequence_list, "Coordinates")
    SubElement(coordinates, "RA").text = str(ra / 15)
    SubElement(coordinates, "Dec").text = str(dec)
    SubElement(coordinates, "Epoch").text = "J2000"
    SubElement(capture_sequence_list, "NegativeDec").text = (
        "true" if dec < 0 else "false"
    )
    return capture_sequence_list


def write_nina_targetset(
    schedule: dict,
    pathname: Path,
    config_path: Path,
) -> None:
    root = ET.Element("ArrayOfCaptureSequenceList")
    tree = ET.ElementTree(root)
    target_counter = 0
    for target_info in schedule.values():
        target = target_info["target"]
        if not target:
            continue
        target_counter += 1
        target_copy = dict(target)
        if target_counter % 15 == 1:
            target_copy["AutoFocusOnStart"] = True
        root.append(create_capture_sequence_xml(target_copy, config_path))
        if target_counter == 30:
            bias_target = {
                "objname": "bias",
                "ra": target_copy.get("ra", 0),
                "dec": target_copy.get("dec", 0),
                "ExposureTime": 0,
                "TotalExposureCount": 10,
                "ImageType": "BIAS",
            }
            root.append(create_capture_sequence_xml(bias_target, config_path))
    tree.write(str(pathname), encoding="utf-8", xml_declaration=True)


@dataclass
class HorizonPoint:
    az: float
    alt: float


@dataclass
class CameraModel:
    cx: float
    cy: float
    E: float
    a0: float
    e: float
    k1: float
    k2: float
    k3: float
    k4: float


@dataclass
class ObservableSkyIndex:
    bins: set[tuple[int, int]]
    resolution_deg: float
    sample_points: list[tuple[float, float]]


@dataclass
class HorizonTriangle:
    name: str
    vertices: list[HorizonPoint]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="方案二：根据语义分割掩码，先做地平 -> 图像投影，再重排批量计划后续目标。",
    )
    parser.add_argument(
        "--batch-dir",
        default=str(DEFAULT_BATCH_DIR),
        help="批量计划目录，默认使用 sequence_adjust_tool/batch_0116_output",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="观测配置 JSON，默认使用 sequence_adjust_tool/observe_config.json",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="单独输出目录根路径",
    )
    parser.add_argument(
        "--mask-path",
        default=str(DEFAULT_MASK_PATH),
        help="语义分割掩码图路径（文件名需包含北京时间）",
    )
    parser.add_argument(
        "--parameters",
        default=str(DEFAULT_PARAMETERS_PATH),
        help="标定参数 JSON 路径（含 x_new/y_new、E/a0/e、k1~k4）",
    )
    parser.add_argument(
        "--position-name",
        default=DEFAULT_POSITION_NAME,
        help="parameters.json 中使用的参数组名，如 position_1",
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        choices=["csv", "tail"],
        help="重排策略：csv=从CSV亮星库替换不可观测目标（默认），tail=从尾部计划中贪心重排",
    )
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_CSV_PATH),
        help="CSV 亮星目标库路径（仅在 strategy=csv 时使用）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若本次输出目录已存在结果文件，则直接复用",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_schedule(schedule_path: Path) -> dict[str, dict]:
    with open(schedule_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_camera_model(parameters_path: Path, position_name: str) -> CameraModel:
    with open(parameters_path, "r", encoding="utf-8") as file:
        items = json.load(file)
    for item in items:
        if str(item.get("position", "")).strip() == position_name:
            return CameraModel(
                cx=float(item["x_new"]),
                cy=float(item["y_new"]),
                E=float(item["E_new"]),
                a0=float(item["a0_new"]),
                e=float(item["e_new"]),
                k1=float(item["k1_new"]),
                k2=float(item["k2_new"]),
                k3=float(item["k3_new"]),
                k4=float(item["k4_new"]),
            )
    raise ValueError(f"未找到参数位置: {position_name}")


def load_binary_mask(mask_path: Path) -> np.ndarray:
    img = mpimg.imread(str(mask_path))
    if img.ndim == 3:
        img = img[..., 0]
    arr = np.asarray(img)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr >= 0.5


def parse_mask_time_beijing(mask_path: Path) -> datetime:
    stem = mask_path.stem

    # 1) 支持紧凑格式：YYYYMMDD_HHMMSS 或 YYYYMMDDHHMMSS
    m = re.search(r"(\d{8})[_-]?(\d{6})", stem)
    if m:
        raw = f"{m.group(1)}{m.group(2)}"
        return datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=BEIJING_TZ)

    # 2) 支持分隔格式：YYYY_MM_DD_HH_MM_SS / YYYY-MM-DD-HH-MM-SS / 混合分隔
    m = re.search(
        r"(\d{4})[_-](\d{2})[_-](\d{2})[_-](\d{2})[_-](\d{2})[_-](\d{2})",
        stem,
    )
    if m:
        y, mo, d, hh, mm, ss = [int(x) for x in m.groups()]
        return datetime(y, mo, d, hh, mm, ss, tzinfo=BEIJING_TZ)

    raise ValueError(f"掩码文件名中未解析到北京时间: {mask_path.name}")


def taylor_distortion(r: float, k1: float, k2: float, k3: float, k4: float) -> float:
    return k1 * r + k2 * (r**3) + k3 * (r**5) + k4 * (r**7)


def solve_radius_from_u(
    u_target: float,
    model: CameraModel,
    image_shape: tuple[int, int] | None = None,
) -> float | None:
    u_target = float(u_target)
    if u_target <= 0.0:
        return 0.0
    coeffs = [model.k4, 0.0, model.k3, 0.0, model.k2, 0.0, model.k1, 0.0]
    coeffs[-1] -= u_target
    roots = np.roots(coeffs)
    positive = sorted(
        float(root.real)
        for root in roots
        if abs(float(root.imag)) < 1e-7 and float(root.real) > 1e-12
    )
    if not positive:
        return None
    if image_shape is not None:
        h, w = image_shape[:2]
        r_cap = float(1.5 * math.hypot(w, h))
        in_canvas = [radius for radius in positive if radius <= r_cap]
        candidates = in_canvas if in_canvas else positive
    else:
        candidates = positive
    radius = min(candidates)
    check = taylor_distortion(radius, model.k1, model.k2, model.k3, model.k4)
    if abs(check - u_target) > 1e-3:
        return None
    return radius


def find_zenith_pixel(
    model: CameraModel,
    image_shape: tuple[int, int],
) -> tuple[int, int] | None:
    if minimize is None:
        return None
    h, w = image_shape[:2]

    def neg_alt(v: np.ndarray) -> float:
        mapped = pixel_to_altaz_xy(float(v[0]), float(v[1]), model)
        if mapped is None:
            return 1e6
        return -float(mapped[1])

    best_xy = None
    best_fun = float("inf")
    trials = [
        (w * 0.5, h * 0.5),
        (model.cx, model.cy),
        (w * 0.35, h * 0.5),
        (w * 0.65, h * 0.5),
        (w * 0.5, h * 0.35),
        (w * 0.5, h * 0.65),
    ]
    for tx, ty in trials:
        result = minimize(
            neg_alt,
            np.array(
                [
                    float(np.clip(tx, 0, w - 1)),
                    float(np.clip(ty, 0, h - 1)),
                ],
                dtype=np.float64,
            ),
            bounds=[(0.0, float(w - 1)), (0.0, float(h - 1))],
            method="L-BFGS-B",
        )
        if result.fun < best_fun:
            best_fun = float(result.fun)
            best_xy = result.x
    if best_xy is None:
        return None
    return int(round(float(best_xy[0]))), int(round(float(best_xy[1])))


def altaz_to_pixel_fast(
    az_deg: float,
    alt_deg: float,
    model: CameraModel,
    image_shape: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """
    与新版 `标注地平坐标系.py` 一致的解析逆映射：
    Az/Alt -> 像素坐标。旧版只用了 r=ze/k1 的一阶近似，这里使用
    E/a0/e/k1..k4 全参数反解。
    """
    az_rad = math.radians(float(az_deg))
    alt_rad = math.radians(float(alt_deg))
    if float(alt_deg) >= 89.98 and image_shape is not None:
        zenith = find_zenith_pixel(model, image_shape)
        if zenith is not None:
            return zenith
    z = math.pi / 2.0 - alt_rad
    a = az_rad % (2.0 * math.pi)

    cos_u0 = (
        math.cos(z) * math.cos(model.e)
        + math.sin(z) * math.sin(model.e) * math.cos(a - model.E)
    )
    cos_u0 = max(-1.0, min(1.0, cos_u0))
    u0 = math.acos(cos_u0)
    if abs(math.sin(u0) * math.sin(model.e)) < 1e-12:
        return int(round(model.cx)), int(round(model.cy))

    b0 = math.atan2(
        math.sin(a - model.E) * math.sin(z) / math.sin(u0),
        -1.0
        * (math.cos(z) - math.cos(u0) * math.cos(model.e))
        / (math.sin(u0) * math.sin(model.e)),
    )
    ca = b0 - model.a0 + model.E
    if ca <= 0.0:
        ca += 2.0 * math.pi

    radius = solve_radius_from_u(u0, model, image_shape=image_shape)
    if radius is None:
        return None

    ang = ca - math.pi
    x_pix = model.cx + radius * math.sin(ang)
    y_pix = model.cy + radius * math.cos(ang)
    if not np.isfinite(x_pix) or not np.isfinite(y_pix):
        return None
    return int(round(x_pix)), int(round(y_pix))


def pixel_to_altaz_xy(x: float, y: float, model: CameraModel) -> tuple[float, float] | None:
    dx = x - model.cx
    dy = y - model.cy
    r = math.hypot(dx, dy)
    u0 = taylor_distortion(r, model.k1, model.k2, model.k3, model.k4)
    if not np.isfinite(u0) or u0 < 0:
        return None

    ca = math.atan2(dx, dy) + math.pi
    X = model.a0 - model.E
    b = X + ca
    cos_cze = math.cos(u0) * math.cos(model.e) - math.cos(b) * math.sin(u0) * math.sin(model.e)
    cos_cze = max(-1.0, min(1.0, cos_cze))
    cze = math.acos(cos_cze)
    amE = math.atan2(
        math.sin(b) * math.sin(u0),
        (math.cos(b) * math.sin(u0) * math.cos(model.e) + math.cos(u0) * math.sin(model.e)),
    )
    caz = amE + model.E
    if caz <= 0:
        caz += 2.0 * math.pi
    calt = math.pi / 2.0 - cze
    az_deg = math.degrees(caz) % 360.0
    alt_deg = math.degrees(calt)
    if alt_deg < 0.0 or alt_deg > 90.0:
        return None
    return az_deg, alt_deg


def build_observable_sky_index(
    mask: np.ndarray,
    model: CameraModel,
    pixel_stride: int = 2,
    resolution_deg: float = 1.0,
) -> ObservableSkyIndex:
    h, w = mask.shape[:2]
    bins: set[tuple[int, int]] = set()
    sample_points: list[tuple[float, float]] = []
    az_size = int(round(360.0 / resolution_deg))
    for yi in range(0, h, max(1, pixel_stride)):
        for xi in range(0, w, max(1, pixel_stride)):
            if not bool(mask[yi, xi]):
                continue
            mapped = pixel_to_altaz_xy(float(xi), float(yi), model)
            if mapped is None:
                continue
            az, alt = mapped
            az_bin = int(round(az / resolution_deg)) % az_size
            alt_bin = int(round(alt / resolution_deg))
            bins.add((az_bin, alt_bin))
            if len(sample_points) < 12000:
                sample_points.append((az, alt))
    return ObservableSkyIndex(
        bins=bins,
        resolution_deg=resolution_deg,
        sample_points=sample_points,
    )


def is_altaz_in_observable_index(
    az: float,
    alt: float,
    sky_index: ObservableSkyIndex,
    tolerance_deg: float = 1.5,
) -> bool:
    if alt < 0.0 or alt > 90.0:
        return False
    res = sky_index.resolution_deg
    az_size = int(round(360.0 / res))
    az_center = int(round(az / res)) % az_size
    alt_center = int(round(alt / res))
    n = max(1, int(math.ceil(tolerance_deg / res)))
    for da in range(-n, n + 1):
        for db in range(-n, n + 1):
            if ((az_center + da) % az_size, alt_center + db) in sky_index.bins:
                return True
    return False


def _angle_diff_deg(a: float, b: float) -> float:
    """角度差（度），范围 [0, 180]。"""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)

def random_beijing_evening() -> datetime:
    month = random.randint(1, 12)
    month = random.randint(1, 12)
    start_day = date(2026, month, 1)
    if month == 12:
        next_month = date(2027, 1, 1)
    else:
        next_month = date(2026, month + 1, 1)
    day_count = (next_month - start_day).days
    day = random.randint(1, day_count)
    hour = random.randint(19, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime(2026, month, day, hour, minute, second, tzinfo=BEIJING_TZ)


def random_triangle(name: str) -> HorizonTriangle:
    min_az_span = 5.0
    min_alt_span = 5.0

    for _ in range(64):
        center_az = random.uniform(25.0, 335.0)
        center_alt = random.uniform(20.0, 70.0)
        vertices: list[HorizonPoint] = []
        for _ in range(3):
            az = center_az + random.uniform(-36.0, 36.0)
            alt = center_alt + random.uniform(-24.0, 24.0)
            az = max(0.0, min(359.0, az))
            alt = max(5.0, min(85.0, alt))
            vertices.append(HorizonPoint(az=az, alt=alt))

        if triangle_area(vertices) < 30.0:
            vertices[1] = HorizonPoint(
                az=max(0.0, min(359.0, center_az + 28.0)),
                alt=max(5.0, min(85.0, center_alt - 16.0)),
            )
            vertices[2] = HorizonPoint(
                az=max(0.0, min(359.0, center_az - 24.0)),
                alt=max(5.0, min(85.0, center_alt + 18.0)),
            )

        if (
            triangle_az_span(vertices) >= min_az_span
            and triangle_alt_span(vertices) >= min_alt_span
        ):
            return HorizonTriangle(name=name, vertices=vertices)

    fallback_center_az = random.uniform(30.0, 330.0)
    fallback_center_alt = random.uniform(20.0, 70.0)
    fallback_vertices = [
        HorizonPoint(
            az=max(0.0, min(359.0, fallback_center_az - 10.0)),
            alt=max(5.0, min(85.0, fallback_center_alt - 8.0)),
        ),
        HorizonPoint(
            az=max(0.0, min(359.0, fallback_center_az + 12.0)),
            alt=max(5.0, min(85.0, fallback_center_alt - 6.0)),
        ),
        HorizonPoint(
            az=max(0.0, min(359.0, fallback_center_az - 6.0)),
            alt=max(5.0, min(85.0, fallback_center_alt + 12.0)),
        ),
    ]
    return HorizonTriangle(name=name, vertices=fallback_vertices)


def triangle_area(vertices: list[HorizonPoint]) -> float:
    a, b, c = vertices
    return abs(
        a.az * (b.alt - c.alt)
        + b.az * (c.alt - a.alt)
        + c.az * (a.alt - b.alt)
    ) / 2.0


def triangle_az_span(vertices: list[HorizonPoint]) -> float:
    az_values = [point.az for point in vertices]
    return max(az_values) - min(az_values)


def triangle_alt_span(vertices: list[HorizonPoint]) -> float:
    alt_values = [point.alt for point in vertices]
    return max(alt_values) - min(alt_values)


def build_random_triangles(count: int = 5) -> list[HorizonTriangle]:
    return [random_triangle(f"triangle_{index + 1}") for index in range(count)]


def mmdd_from_dt(value: datetime) -> str:
    return value.strftime("%m%d")


def parse_mmdd_from_path(path: Path) -> str:
    return path.stem.split("_")[0]


def nearest_schedule_path(batch_dir: Path, target_dt: datetime) -> Path:
    candidates = sorted(batch_dir.glob("*_Schedule.json"))
    if not candidates:
        raise FileNotFoundError(f"未找到任何 Schedule 文件: {batch_dir}")

    target_date = date(2026, target_dt.month, target_dt.day)
    best_path = candidates[0]
    best_gap = 10**9
    for path in candidates:
        mmdd = parse_mmdd_from_path(path)
        candidate_date = datetime.strptime(f"2026{mmdd}", "%Y%m%d").date()
        gap = abs((candidate_date - target_date).days)
        if gap < best_gap:
            best_gap = gap
            best_path = path
    return best_path


def parse_schedule_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f").replace(
        tzinfo=timezone.utc,
    )


def replace_datetime_time(base_dt: datetime, time_dt: datetime) -> datetime:
    return base_dt.replace(
        hour=time_dt.hour,
        minute=time_dt.minute,
        second=time_dt.second,
        microsecond=time_dt.microsecond,
    )


def closest_schedule_index(
    schedule: dict[str, dict],
    target_utc: datetime,
) -> int:
    keys = list(schedule.keys())
    if not keys:
        raise ValueError("schedule 为空，无法匹配当前槽位")

    first_slot = parse_schedule_time(keys[0])
    target_on_schedule_day = replace_datetime_time(first_slot, target_utc)

    diffs = [
        abs(
            (
                parse_schedule_time(item) - target_on_schedule_day
            ).total_seconds()
        )
        for item in keys
    ]
    return min(range(len(keys)), key=lambda idx: diffs[idx])


def unwrap_az(reference: float, value: float) -> float:
    if value - reference > 180.0:
        return value - 360.0
    if value - reference < -180.0:
        return value + 360.0
    return value


def point_in_triangle(
    az: float,
    alt: float,
    triangle: HorizonTriangle,
) -> bool:
    unwrapped = [
        HorizonPoint(
            az=unwrap_az(az, vertex.az),
            alt=vertex.alt,
        )
        for vertex in triangle.vertices
    ]
    point = HorizonPoint(az=az, alt=alt)
    a, b, c = unwrapped
    denominator = (
        (b.alt - c.alt) * (a.az - c.az)
        + (c.az - b.az) * (a.alt - c.alt)
    )
    if abs(denominator) < 1e-9:
        return False

    alpha = (
        (b.alt - c.alt) * (point.az - c.az)
        + (c.az - b.az) * (point.alt - c.alt)
    ) / denominator
    beta = (
        (c.alt - a.alt) * (point.az - c.az)
        + (a.az - c.az) * (point.alt - c.alt)
    ) / denominator
    gamma = 1.0 - alpha - beta
    eps = 1e-9
    return alpha >= -eps and beta >= -eps and gamma >= -eps


def target_altaz(
    target: dict,
    interval_time: str,
    lat: float,
    lon: float,
) -> HorizonPoint:
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=0 * u.m)
    obstime = Time(interval_time, format="iso", scale="utc")
    coord = SkyCoord(
        ra=float(target["ra"]) * u.deg,
        dec=float(target["dec"]) * u.deg,
        frame="icrs",
    )
    altaz = coord.transform_to(AltAz(obstime=obstime, location=location))
    az = float(altaz.az.degree) % 360.0
    alt = float(altaz.alt.degree)
    return HorizonPoint(az=az, alt=alt)


def target_in_random_range(
    target: dict,
    interval_time: str,
    triangles: list[HorizonTriangle],
    lat: float,
    lon: float,
) -> tuple[bool, HorizonPoint]:
    point = target_altaz(target, interval_time, lat, lon)
    if point.alt < 0:
        return False, point
    for triangle in triangles:
        if point_in_triangle(point.az, point.alt, triangle):
            return True, point
    return False, point


def pick_target_for_slot(
    pending_targets: list[dict],
    interval_time: str,
    triangles: list[HorizonTriangle],
    lat: float,
    lon: float,
    d_moon: float,
) -> tuple[dict | None, list[dict]]:
    rejected: list[dict] = []
    for index, target in enumerate(pending_targets):
        in_range, point = target_in_random_range(
            target,
            interval_time,
            triangles,
            lat,
            lon,
        )
        if not in_range:
            rejected.append(
                {
                    "objname": target["objname"],
                    "reason": "不在随机三角形可观测范围内",
                    "az": round(point.az, 3),
                    "alt": round(point.alt, 3),
                }
            )
            continue
        if not target_observable(
            interval_time,
            lat,
            lon,
            float(target["ra"]),
            float(target["dec"]),
            0.0,
            d_moon,
        ):
            rejected.append(
                {
                    "objname": target["objname"],
                    "reason": "月距约束不满足",
                    "az": round(point.az, 3),
                    "alt": round(point.alt, 3),
                }
            )
            continue
        return pending_targets.pop(index), rejected
    return None, rejected


def replan_tail(
    schedule: dict[str, dict],
    current_index: int,
    triangles: list[HorizonTriangle],
    lat: float,
    lon: float,
    d_moon: float,
) -> tuple[dict[str, dict], list[dict]]:
    keys = list(schedule.keys())
    updated = {}
    for idx in range(current_index + 1):
        updated[keys[idx]] = schedule[keys[idx]]

    tail_targets = [
        schedule[key]["target"]
        for key in keys[current_index + 1:]
        if schedule[key].get("target")
    ]
    deferred_targets: list[dict] = []

    for key in keys[current_index + 1:]:
        picked, rejected = pick_target_for_slot(
            tail_targets,
            key,
            triangles,
            lat,
            lon,
            d_moon,
        )
        if picked is None:
            updated[key] = {
                "target": "",
                "note": "随机天空范围下未找到可替换目标",
                "rejected": rejected,
            }
            continue
        updated[key] = {
            "target": picked,
            "note": "该目标由随机三角形天空范围重新排入",
            "rejected": rejected,
        }

    deferred_targets.extend(tail_targets)
    return updated, deferred_targets


def serialize_triangles(triangles: list[HorizonTriangle]) -> list[dict]:
    return [
        {
            "name": triangle.name,
            "vertices": [asdict(point) for point in triangle.vertices],
        }
        for triangle in triangles
    ]


def build_sequence_points(
    schedule: dict[str, dict],
    start_index: int,
    lat: float,
    lon: float,
) -> list[dict]:
    keys = list(schedule.keys())
    points: list[dict] = []
    order = 1
    for key in keys[start_index:]:
        target = schedule[key].get("target")
        if not target:
            continue
        point = target_altaz(target, key, lat, lon)
        if point.alt < 0:
            continue
        points.append(
            {
                "order": order,
                "objname": target["objname"],
                "slot_time": key,
                "az": point.az,
                "alt": point.alt,
            }
        )
        order += 1
    return points


def target_in_mask_range(
    target: dict,
    interval_time: str,
    lat: float,
    lon: float,
    mask: np.ndarray,
    camera_model: CameraModel,
) -> tuple[bool, HorizonPoint]:
    point = target_altaz(target, interval_time, lat, lon)
    if point.alt < 0:
        return False, point
    xy = altaz_to_pixel_fast(point.az, point.alt, camera_model, image_shape=mask.shape)
    if xy is None:
        return False, point
    x, y = xy
    h, w = mask.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return False, point
    return bool(mask[y, x]), point


def pick_target_for_slot_mask(
    pending_targets: list[dict],
    interval_time: str,
    mask: np.ndarray,
    camera_model: CameraModel,
    lat: float,
    lon: float,
    d_moon: float,
    min_alt: float = 30.0,
) -> tuple[dict | None, list[dict]]:
    """为单个时间槽从候选池中挑选目标，依次过三道筛子：掩码可观测、高度角≥min_alt、月距约束。"""
    rejected: list[dict] = []
    for index, target in enumerate(pending_targets):
        in_range, point = target_in_mask_range(
            target=target,
            interval_time=interval_time,
            lat=lat,
            lon=lon,
            mask=mask,
            camera_model=camera_model,
        )
        if not in_range:
            rejected.append(
                {
                    "objname": target["objname"],
                    "reason": "不在掩码可观测范围内",
                    "az": round(point.az, 3),
                    "alt": round(point.alt, 3),
                }
            )
            continue
        # 第三道筛子：高度角 ≥ min_alt（默认 30°）
        if point.alt < min_alt:
            rejected.append(
                {
                    "objname": target["objname"],
                    "reason": f"高度角不足（{round(point.alt, 3)}° < {min_alt}°）",
                    "az": round(point.az, 3),
                    "alt": round(point.alt, 3),
                }
            )
            continue
        if not target_observable(
            interval_time,
            lat,
            lon,
            float(target["ra"]),
            float(target["dec"]),
            0.0,
            d_moon,
        ):
            rejected.append(
                {
                    "objname": target["objname"],
                    "reason": "月距约束不满足",
                    "az": round(point.az, 3),
                    "alt": round(point.alt, 3),
                }
            )
            continue
        return pending_targets.pop(index), rejected
    return None, rejected


def replan_tail_mask(
    schedule: dict[str, dict],
    current_index: int,
    mask: np.ndarray,
    camera_model: CameraModel,
    lat: float,
    lon: float,
    d_moon: float,
) -> tuple[dict[str, dict], list[dict]]:
    keys = list(schedule.keys())
    updated = {}
    for idx in range(current_index + 1):
        updated[keys[idx]] = schedule[keys[idx]]

    tail_targets = [
        schedule[key]["target"]
        for key in keys[current_index + 1:]
        if schedule[key].get("target")
    ]
    deferred_targets: list[dict] = []

    for key in keys[current_index + 1:]:
        picked, rejected = pick_target_for_slot_mask(
            pending_targets=tail_targets,
            interval_time=key,
            mask=mask,
            camera_model=camera_model,
            lat=lat,
            lon=lon,
            d_moon=d_moon,
        )
        if picked is None:
            updated[key] = {
                "target": "",
                "note": "掩码天空范围下未找到可替换目标",
                "rejected": rejected,
            }
            continue
        updated[key] = {
            "target": picked,
            "note": "该目标由掩码天空范围重新排入",
            "rejected": rejected,
        }

    deferred_targets.extend(tail_targets)
    return updated, deferred_targets


def load_csv_targets(csv_path: Path) -> list[dict]:
    """从 CSV 文件加载亮星目标库，返回包含 objname/ra/dec/distance 的字典列表。"""
    targets: list[dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets.append(
                {
                    "objname": row["objname"].strip(),
                    "ra": float(row["ra"]),
                    "dec": float(row["dec"]),
                    "distance": float(row.get("distance", 0)),
                }
            )
    return targets


def replan_from_csv(
    schedule: dict[str, dict],
    current_index: int,
    mask: np.ndarray,
    camera_model: CameraModel,
    csv_targets: list[dict],
    lat: float,
    lon: float,
    d_moon: float,
    min_alt: float = 30.0,
) -> tuple[dict[str, dict], list[dict], list[dict]]:
    """
    CSV 替换策略：整体序列不动。
    仅检查 current_index+1（即紧邻的下一个槽位）的目标是否可观测（三道筛子）；
    如果不可观测，从 CSV 亮星库中按顺序找第一个满足条件的替代品；
    该槽位之后的所有目标原样保留不动。
    返回 (重排后的 schedule, 未使用的 CSV 目标列表, 替换日志)。
    """
    keys = list(schedule.keys())
    updated: dict[str, dict] = {}
    # 当前槽位及之前：保持不变
    for idx in range(current_index + 1):
        updated[keys[idx]] = schedule[keys[idx]]

    replacement_log: list[dict] = []

    # 仅处理紧邻的下一个槽位
    next_idx = current_index + 1
    if next_idx < len(keys):
        key = keys[next_idx]
        target = schedule[key].get("target")

        if target:
            # 检查当前目标是否可观测（三道筛子：掩码、高度角、月距）
            in_range, point = target_in_mask_range(
                target=target,
                interval_time=key,
                lat=lat,
                lon=lon,
                mask=mask,
                camera_model=camera_model,
            )
            alt_ok = point.alt >= min_alt if in_range else False
            moon_ok = False
            if in_range and alt_ok:
                moon_ok = target_observable(
                    key,
                    lat,
                    lon,
                    float(target["ra"]),
                    float(target["dec"]),
                    0.0,
                    d_moon,
                )

            if in_range and alt_ok and moon_ok:
                # 当前目标可观测，原样保留
                updated[key] = schedule[key]
            else:
                # 当前目标不可观测，从 CSV 亮星库中寻找替代
                picked = None
                picked_idx = None
                for csv_idx, csv_target in enumerate(csv_targets):
                    csv_in_range, csv_point = target_in_mask_range(
                        target=csv_target,
                        interval_time=key,
                        lat=lat,
                        lon=lon,
                        mask=mask,
                        camera_model=camera_model,
                    )
                    if not csv_in_range:
                        continue
                    if csv_point.alt < min_alt:
                        continue
                    if not target_observable(
                        key,
                        lat,
                        lon,
                        float(csv_target["ra"]),
                        float(csv_target["dec"]),
                        0.0,
                        d_moon,
                    ):
                        continue
                    picked = csv_target
                    picked_idx = csv_idx
                    break

                if picked is not None:
                    updated[key] = {
                        "target": picked,
                        "note": f"原目标 {target['objname']} 不可观测，从 CSV 亮星库替换为 {picked['objname']}",
                    }
                    replacement_log.append(
                        {
                            "slot_time": key,
                            "original_target": target["objname"],
                            "replacement": picked["objname"],
                        }
                    )
                else:
                    updated[key] = {
                        "target": "",
                        "note": f"原目标 {target['objname']} 不可观测，CSV 亮星库中亦无合适替代",
                    }
        else:
            updated[key] = schedule[key]

    # 该槽位之后的所有目标：原样保留不动
    for idx in range(next_idx + 1, len(keys)):
        updated[keys[idx]] = schedule[keys[idx]]

    # CSV 中未被使用的目标作为 deferred（延后）
    used_csv_indices: set[int] = set()
    for log_entry in replacement_log:
        for i, t in enumerate(csv_targets):
            if t["objname"] == log_entry["replacement"]:
                used_csv_indices.add(i)
                break
    deferred_csv = [
        csv_targets[i]
        for i in range(len(csv_targets))
        if i not in used_csv_indices
    ]
    return updated, deferred_csv, replacement_log


def az_to_theta_deg(az: float) -> float:
    return az % 360.0


def alt_to_radius(alt: float) -> float:
    # 极坐标天图中：天顶在中心，地平线在外圈。
    return 90.0 - alt


def interpolate_az(start_az: float, end_az: float, fraction: float) -> float:
    start = az_to_theta_deg(start_az)
    end = start + unwrap_az(start, az_to_theta_deg(end_az)) - start
    return az_to_theta_deg(start + (end - start) * fraction)


def smooth_triangle_path(
    triangle: HorizonTriangle,
    samples_per_edge: int = 32,
) -> tuple[list[float], list[float]]:
    vertices = triangle.vertices
    thetas: list[float] = []
    radii: list[float] = []
    for index in range(len(vertices)):
        start = vertices[index]
        end = vertices[(index + 1) % len(vertices)]
        for sample_idx in range(samples_per_edge):
            fraction = sample_idx / samples_per_edge
            az = interpolate_az(start.az, end.az, fraction)
            alt = start.alt + (end.alt - start.alt) * fraction
            thetas.append(math.radians(az))
            radii.append(alt_to_radius(alt))
    thetas.append(math.radians(az_to_theta_deg(vertices[0].az)))
    radii.append(alt_to_radius(vertices[0].alt))
    return thetas, radii


def sequence_to_polar(points: list[dict]) -> tuple[list[float], list[float]]:
    thetas = [math.radians(az_to_theta_deg(point["az"])) for point in points]
    radii = [alt_to_radius(point["alt"]) for point in points]
    return thetas, radii


def build_mask_region_polar(
    mask: np.ndarray,
    camera_model: CameraModel,
) -> tuple[list[float], list[float]]:
    thetas: list[float] = []
    radii: list[float] = []
    progress("开始采样掩码对应的天空区域（方案二：Az/Alt -> 像素 -> 查掩码）")
    alt_values = np.arange(0.0, 90.0 + 1e-6, 0.5)
    total_rows = len(alt_values)
    for row_index, alt in enumerate(alt_values, start=1):
        for az in np.arange(0.0, 360.0, 1.0):
            xy = altaz_to_pixel_fast(
                float(az),
                float(alt),
                camera_model,
                image_shape=mask.shape,
            )
            if xy is None:
                continue
            x, y = xy
            h, w = mask.shape[:2]
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            if not bool(mask[y, x]):
                continue
            thetas.append(math.radians(az_to_theta_deg(float(az))))
            radii.append(alt_to_radius(float(alt)))
        if row_index == 1 or row_index == total_rows or row_index % max(1, total_rows // 10) == 0:
            progress(
                "掩码天空区域采样中: "
                f"{row_index}/{total_rows} 层高度角, 已命中样本={len(thetas)}"
            )
    progress(f"掩码天空区域采样完成: 样本数={len(thetas)}")
    return thetas, radii


def build_horizon_overlay_points(
    mask: np.ndarray,
    model: CameraModel,
    az_levels: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315),
    alt_levels: tuple[int, ...] = (15, 30, 45, 60, 75),
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    使用与 `标注地平坐标系.py` 一致的 Az/Alt -> 像素近似映射，
    在掩码图上绘制地平参考网格点。
    返回:
    - az_points: 方位角参考点 (x, y)
    - alt_points: 高度角参考点 (x, y)
    """
    h, w = mask.shape[:2]
    az_points: list[tuple[float, float]] = []
    alt_points: list[tuple[float, float]] = []
    for az in az_levels:
        for alt in np.arange(5.0, 85.0 + 1e-6, 1.0):
            xy = altaz_to_pixel_fast(float(az), float(alt), model, image_shape=mask.shape)
            if xy is None:
                continue
            x, y = xy
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            az_points.append((float(x), float(y)))
    for alt in alt_levels:
        for az in np.arange(0.0, 360.0, 2.0):
            xy = altaz_to_pixel_fast(float(az), float(alt), model, image_shape=mask.shape)
            if xy is None:
                continue
            x, y = xy
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            alt_points.append((float(x), float(y)))
    return az_points, alt_points


def annotate_sequence_points(
    ax,
    points: list[dict],
    prefix: str,
    color: str,
    radial_shift: float,
) -> None:
    for point in points:
        theta = math.radians(az_to_theta_deg(point["az"])) + 0.02
        radius = alt_to_radius(point["alt"])
        label_radius = min(90.0, max(0.0, radius + radial_shift))
        ax.text(
            theta,
            label_radius,
            f"{prefix}{point['order']} {point['objname']}",
            color=color,
            fontsize=7,
            ha="left",
            va="center",
        )


def draw_sky_plot(
    output_dir: Path,
    beijing_time: datetime,
    schedule_path: Path,
    current_slot: str,
    current_target: dict | str,
    mask: np.ndarray,
    camera_model: CameraModel,
    mask_region_thetas: list[float],
    mask_region_radii: list[float],
    original_points: list[dict],
    replanned_points: list[dict],
) -> Path:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "sky_replan_plot.png"
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    ax_mask = fig.add_subplot(1, 2, 2)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_thetagrids(
        range(0, 360, 45),
        labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
    )
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_yticklabels(["90", "75", "60", "45", "30", "15", "0"])
    ax.set_rlabel_position(225)
    ax.set_title(
        "Polar Sky Chart For Tail Sequence Replan\n"
        f"Beijing time: {beijing_time.isoformat()} | "
        "Matched schedule: "
        f"{schedule_path.name} | Current slot: {current_slot}"
    )
    ax.grid(True, alpha=0.3)

    if mask_region_thetas:
        ax.scatter(
            mask_region_thetas,
            mask_region_radii,
            c="limegreen",
            s=12,
            alpha=0.55,
            label="Mask observable region",
        )
    else:
        ax.text(
            0.02,
            0.98,
            "未在天空投影中采样到可观测区域",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="crimson",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        )

    if original_points:
        original_thetas, original_radii = sequence_to_polar(original_points)
        ax.plot(
            original_thetas,
            original_radii,
            color="tab:blue",
            linewidth=1.4,
            alpha=0.85,
        )
        ax.scatter(
            original_thetas,
            original_radii,
            c="tab:blue",
            s=42,
            label="Original sequence after current slot",
        )
        annotate_sequence_points(
            ax,
            original_points,
            prefix="O",
            color="tab:blue",
            radial_shift=-3.0,
        )

    if replanned_points:
        replanned_thetas, replanned_radii = sequence_to_polar(replanned_points)
        ax.plot(
            replanned_thetas,
            replanned_radii,
            color="tab:red",
            linewidth=1.4,
            alpha=0.85,
        )
        ax.scatter(
            replanned_thetas,
            replanned_radii,
            c="tab:red",
            s=42,
            marker="s",
            label="Replanned sequence after current slot",
        )
        annotate_sequence_points(
            ax,
            replanned_points,
            prefix="N",
            color="tab:red",
            radial_shift=3.0,
        )

    current_name = (
        current_target.get("objname", "unknown")
        if isinstance(current_target, dict)
        else str(current_target)
    )
    ax.text(
        0.01,
        0.99,
        f"Current target kept: {current_name}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
    ax.legend(loc="upper right")

    # 右侧显示掩码原图，确保白色可观测区域可直观看到。
    ax_mask.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax_mask.set_title("Mask Image (White = Observable)")
    ax_mask.set_xlabel("x (pixel)")
    ax_mask.set_ylabel("y (pixel)")
    ax_mask.grid(False)

    az_points, alt_points = build_horizon_overlay_points(mask, camera_model)
    if az_points:
        ax_mask.scatter(
            [p[0] for p in az_points],
            [p[1] for p in az_points],
            c="deepskyblue",
            s=4,
            alpha=0.45,
            label="Az lines",
        )
    if alt_points:
        ax_mask.scatter(
            [p[0] for p in alt_points],
            [p[1] for p in alt_points],
            c="orange",
            s=4,
            alpha=0.45,
            label="Alt lines",
        )
    if az_points or alt_points:
        ax_mask.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def export_outputs(
    output_dir: Path,
    metadata: dict,
    schedule: dict[str, dict],
    config_path: Path,
    deferred_targets: list[dict],
    resume: bool,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "scenario.json"
    schedule_path = output_dir / "replanned_schedule.json"
    targetset_path = output_dir / "replanned_sequence.ninaTargetSet"
    deferred_path = output_dir / "deferred_targets.json"

    if (
        resume
        and metadata_path.is_file()
        and schedule_path.is_file()
        and targetset_path.is_file()
        and deferred_path.is_file()
    ):
        return metadata_path, schedule_path, targetset_path

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    with open(schedule_path, "w", encoding="utf-8") as file:
        json.dump(schedule, file, ensure_ascii=False, indent=2)
    with open(deferred_path, "w", encoding="utf-8") as file:
        json.dump(deferred_targets, file, ensure_ascii=False, indent=2)

    if targetset_path.exists():
        targetset_path.unlink()
    write_nina_targetset(schedule, targetset_path, config_path)
    return metadata_path, schedule_path, targetset_path


def build_output_dir(
    output_root: Path,
    beijing_time: datetime,
) -> Path:
    folder_name = beijing_time.strftime("%Y%m%d_%H%M%S_mask")
    return output_root / folder_name


def main() -> None:
    progress("开始执行方案二：地平坐标 -> 图像坐标")
    _configure_astropy_offline_data()
    args = parse_args()

    batch_dir = Path(args.batch_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    parameters_path = Path(args.parameters).expanduser().resolve()
    mask_path = Path(args.mask_path).expanduser().resolve()

    if not batch_dir.is_dir():
        raise FileNotFoundError(f"未找到批量计划目录: {batch_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    if not parameters_path.is_file():
        raise FileNotFoundError(f"未找到参数文件: {parameters_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"未找到掩码图: {mask_path}")

    progress(f"读取配置文件: {config_path}")
    config = load_config(config_path)
    progress(f"读取相机参数: {parameters_path} | position={args.position_name}")
    camera_model = load_camera_model(parameters_path, args.position_name)
    progress(f"读取掩码图: {mask_path}")
    mask = load_binary_mask(mask_path)

    beijing_time = parse_mask_time_beijing(mask_path)
    utc_time = beijing_time.astimezone(timezone.utc)
    progress(f"掩码北京时间: {beijing_time.isoformat()}")
    progress(f"对应 UTC: {utc_time.isoformat()}")
    schedule_path = nearest_schedule_path(batch_dir, beijing_time)
    progress(f"匹配计划文件: {schedule_path}")
    schedule = load_schedule(schedule_path)
    current_index = closest_schedule_index(schedule, utc_time)
    schedule_keys = list(schedule.keys())
    matched_schedule_day_utc = replace_datetime_time(
        parse_schedule_time(schedule_keys[0]),
        utc_time,
    )
    current_slot = schedule_keys[current_index]
    current_target = schedule[current_slot].get("target", "")
    progress(f"命中的当前槽位: {current_slot}")
    original_points = build_sequence_points(
        schedule,
        current_index + 1,
        DEFAULT_STATION["lat"],
        DEFAULT_STATION["lon"],
    )

    progress(f"重排策略: {args.strategy}")
    if args.strategy == "csv":
        # CSV 替换策略：整体序列不动，不可观测目标从 CSV 亮星库替换
        csv_path = Path(args.csv_path).expanduser().resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"未找到 CSV 亮星库: {csv_path}")
        progress(f"读取 CSV 亮星库: {csv_path}")
        csv_targets = load_csv_targets(csv_path)
        progress(f"CSV 亮星库加载完成，共 {len(csv_targets)} 个候选目标")
        replanned_schedule, deferred_targets, replacement_log = replan_from_csv(
            schedule=schedule,
            current_index=current_index,
            mask=mask,
            camera_model=camera_model,
            csv_targets=csv_targets,
            lat=DEFAULT_STATION["lat"],
            lon=DEFAULT_STATION["lon"],
            d_moon=float(config.get("d_moon", 15.0)),
        )
        progress(f"CSV 替换完成：共替换 {len(replacement_log)} 个目标")
    else:
        # tail 贪心重排策略（旧版保留）：从尾部计划中贪心重排
        progress("开始按掩码重排当前槽位之后的目标")
        replanned_schedule, deferred_targets = replan_tail_mask(
            schedule=schedule,
            current_index=current_index,
            mask=mask,
            camera_model=camera_model,
            lat=DEFAULT_STATION["lat"],
            lon=DEFAULT_STATION["lon"],
            d_moon=float(config.get("d_moon", 15.0)),
        )
        replacement_log = []
    replanned_points = build_sequence_points(
        replanned_schedule,
        current_index + 1,
        DEFAULT_STATION["lat"],
        DEFAULT_STATION["lon"],
    )
    mask_region_thetas, mask_region_radii = build_mask_region_polar(mask, camera_model)

    output_dir = build_output_dir(output_root, beijing_time)
    progress(f"绘制天空图并写出到: {output_dir}")
    plot_path = draw_sky_plot(
        output_dir=output_dir,
        beijing_time=beijing_time,
        schedule_path=schedule_path,
        current_slot=current_slot,
        current_target=current_target,
        mask=mask,
        camera_model=camera_model,
        mask_region_thetas=mask_region_thetas,
        mask_region_radii=mask_region_radii,
        original_points=original_points,
        replanned_points=replanned_points,
    )
    metadata = {
        "station": DEFAULT_STATION,
        "generated_beijing_time": beijing_time.isoformat(),
        "generated_utc_time": utc_time.isoformat(),
        "matched_schedule_file": str(schedule_path),
        "requested_mmdd": mmdd_from_dt(beijing_time),
        "matched_mmdd": parse_mmdd_from_path(schedule_path),
        "matched_utc_time_on_schedule_day": (
            matched_schedule_day_utc.isoformat()
        ),
        "current_schedule_index": current_index,
        "current_schedule_time_utc": current_slot,
        "current_target": current_target,
        "mask_path": str(mask_path),
        "position_name": args.position_name,
        "strategy": args.strategy,
        "calibration_source": {
            "type": "json",
            "parameters": str(parameters_path),
            "position_name": args.position_name,
        },
        "mask_region_sample_count": len(mask_region_thetas),
        "sky_plot": str(plot_path),
        "replacement_count": len(replacement_log),
    }
    metadata_path, replanned_path, targetset_path = export_outputs(
        output_dir=output_dir,
        metadata=metadata,
        schedule=replanned_schedule,
        config_path=config_path,
        deferred_targets=deferred_targets,
        resume=args.resume,
    )
    if replacement_log:
        replacement_log_path = output_dir / "replacement_log.json"
        with open(replacement_log_path, "w", encoding="utf-8") as f:
            json.dump(replacement_log, f, ensure_ascii=False, indent=2)
        progress(f"替换日志输出: {replacement_log_path}")
    progress("方案二输出完成")

    print(f"掩码北京时间: {beijing_time.isoformat()}")
    print(f"对应 UTC: {utc_time.isoformat()}")
    print(f"掩码文件: {mask_path}")
    print(f"命中的计划文件: {schedule_path}")
    print(f"映射到该计划当晚的 UTC: {matched_schedule_day_utc.isoformat()}")
    print(f"命中的当前槽位: {current_slot}")
    print(f"当前目标: {current_target}")
    print(f"场景信息输出: {metadata_path}")
    print(f"全天对比图输出: {plot_path}")
    print(f"重排后的计划输出: {replanned_path}")
    print(f"NINA TargetSet 输出: {targetset_path}")
    print(f"延后未排入目标数: {len(deferred_targets)}")


if __name__ == "__main__":
    main()
