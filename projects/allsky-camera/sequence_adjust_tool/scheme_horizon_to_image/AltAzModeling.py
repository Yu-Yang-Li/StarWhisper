"""
全天相机完整标定流程
整合 pipeline1 和 pipeline3，自动完成从初始参数到精确标定的全过程

输入：
    - 观测信息（时间、地点）
    - 相机参数（焦距等）
    - 星点坐标文件 targetlist.txt (num, ra, dec, x, y)
    - 初始图像中心 x0y0.txt

输出：
    - x0y0_new.txt: 精确的图像中心坐标
    - parameters_new.txt: 精确的相机标定参数 (E, a0, e, k1, k2, k3, k4)
"""

import sys

# 导入 all_sky_astro 模块（使用相对路径）
import datetime
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from scipy.optimize import curve_fit
from zoneinfo import ZoneInfo

current_dir = Path(__file__).resolve().parent
# 内联所需的 astrometry 函数，避免外部依赖

import numpy as _np
from astropy.coordinates import AltAz as _AltAz
from astropy.coordinates import SkyCoord as _SkyCoord
from astropy.time import Time as _Time

# 基础数学函数


def zenith(xdata, X, e):
    ca, u = xdata
    b = X + ca
    return _np.arccos(_np.cos(u) * _np.cos(e) - _np.cos(b) * _np.sin(u) * _np.sin(e))


def func_linear(xdata, E):
    return _np.sin(xdata + E)


def taylor(xdata, k1, k2, k3, k4):
    return k1 * xdata + k2 * (xdata**3) + k3 * (xdata**5) + k4 * (xdata**7)


def cosx(x, a, b):
    return a * _np.cos(x / 180 * _np.pi - b)


def sinx(x, c, d):
    return c * _np.sin(x / 180 * _np.pi - d)


# 投影与初值


def bu_from_f(x, y, x0, y0, f):
    ca_b = _np.arctan2(x - x0, y - y0) + _np.pi
    cz_u = 2 * _np.arcsin(_np.sqrt((x - x0) ** 2 + (y - y0) ** 2) / f)
    return ca_b, cz_u


def ini_Ee(ca, cz, az, ze):
    xdata = [ca, cz]
    ydata = ze
    popt, _ = curve_fit(zenith, xdata, ydata)
    X, e = popt
    b0 = X + ca
    amE = _np.arctan2(
        _np.sin(b0) * _np.sin(cz),
        (_np.cos(b0) * _np.sin(cz) * _np.cos(e) + _np.cos(cz) * _np.sin(e)),
    )
    xdata = amE
    ydata = _np.sin(az)
    popt, _ = curve_fit(func_linear, xdata, ydata)
    E = popt[0]
    return E, e


# 迭代与变换


def iteration(x, y, x0, y0, az, ze, E0, e0, mode):
    u = _np.arccos(
        _np.cos(ze) * _np.cos(e0) + _np.sin(ze) * _np.sin(e0) * _np.cos(az - E0)
    )
    r = _np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    popt, _ = curve_fit(taylor, r, u)
    k1, k2, k3, k4 = popt
    cz = taylor(_np.sqrt((x - x0) ** 2 + (y - y0) ** 2), k1, k2, k3, k4)
    ca = _np.arctan2(x - x0, y - y0) + _np.pi
    xdata = [ca, cz]
    ydata = ze
    popt, _ = curve_fit(zenith, xdata, ydata)
    X, e = popt
    Cze = zenith(xdata, X, e)
    b0 = X + ca
    amE = _np.arctan2(
        _np.sin(b0) * _np.sin(cz),
        (_np.cos(b0) * _np.sin(cz) * _np.cos(e) + _np.cos(cz) * _np.sin(e)),
    )
    xdata = amE
    ydata = _np.sin(az)
    popt, _ = curve_fit(func_linear, xdata, ydata)
    E = popt[0]
    a0 = X + E
    Caz = _np.array([val + E if val + E > 0 else val + E + _np.pi * 2 for val in amE])
    if mode == "iter":
        return E, a0, e
    else:
        return Caz, Cze, E, a0, e, k1, k2, k3, k4


def xytoradec(x, y, x0, y0, E, a0, e, k1, k2, k3, k4, time0, loc):
    cz = taylor(_np.sqrt((x - x0) ** 2 + (y - y0) ** 2), k1, k2, k3, k4)
    ca = _np.arctan2(x - x0, y - y0) + _np.pi
    u0 = cz
    X = a0 - E
    b = X + ca
    Cze = _np.arccos(_np.cos(u0) * _np.cos(e) - _np.cos(b) * _np.sin(u0) * _np.sin(e))
    amE = _np.arctan2(
        _np.sin(b) * _np.sin(u0),
        (_np.cos(b) * _np.sin(u0) * _np.cos(e) + _np.cos(u0) * _np.sin(e)),
    )
    if _np.isscalar(amE):
        Caz = amE + E if amE + E > 0 else amE + E + _np.pi * 2
    else:
        Caz = _np.array(
            [val + E if val + E > 0 else val + E + _np.pi * 2 for val in amE]
        )
    Calt = _np.pi / 2 - Cze
    time = _Time(time0)
    altaz_from_xy = _AltAz(obstime=time, location=loc, az=Caz * u.rad, alt=Calt * u.rad)
    skycord_from_xy = _SkyCoord(altaz_from_xy)
    radec_from_xy = skycord_from_xy.transform_to("icrs")
    ra = radec_from_xy.ra.degree
    dec = radec_from_xy.dec.degree
    return (
        ra,
        dec,
        Caz * 180 / _np.pi,
        Cze * 180 / _np.pi,
        u0 * 180 / _np.pi,
        ca * 180 / _np.pi,
    )


def radectoub(Rra, Rdec, x0, y0, E, a0, e, k1, k2, k3, k4, time0, loc):
    time = _Time(time0)
    cord = _SkyCoord(ra=Rra * u.degree, dec=Rdec * u.degree)
    Raltaz = cord.transform_to(_AltAz(obstime=time, location=loc))
    z = _np.pi / 2 - Raltaz.alt.rad
    a = Raltaz.az.rad
    u0 = _np.arccos(_np.cos(z) * _np.cos(e) + _np.sin(z) * _np.sin(e) * _np.cos(a - E))
    b0 = _np.arctan2(
        _np.sin(a - E) * _np.sin(z) / _np.sin(u0),
        -1 * (_np.cos(z) - _np.cos(u0) * _np.cos(e)) / (_np.sin(u0) * _np.sin(e)),
    )
    b = _np.array([val if val > 0 else val + 2 * _np.pi for val in b0])
    ca = _np.array(
        [val - a0 + E if val - a0 + E > 0 else val - a0 + E + 2 * _np.pi for val in b0]
    )
    return a * 180 / _np.pi, z * 180 / _np.pi, u0 * 180 / _np.pi, ca * 180 / _np.pi


def run1(ra1, dec1, x, y, x0, y0, time0, loc, f):
    r = _np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    index = _np.where(r < 0)[0]
    ra1 = _np.delete(ra1, index)
    dec1 = _np.delete(dec1, index)
    x = _np.delete(x, index)
    y = _np.delete(y, index)
    r = _np.delete(r, index)
    cord = _SkyCoord(ra=ra1 * u.degree, dec=dec1 * u.degree)
    time = _Time(time0)
    Waltaz = cord.transform_to(_AltAz(obstime=time, location=loc))
    Waz = Waltaz.az.rad
    Wze = _np.pi / 2 - Waltaz.alt.rad
    ca, cz = bu_from_f(x, y, x0, y0, f)
    E0, e0 = ini_Ee(ca, cz, Waz, Wze)
    return E0, e0, Waz, Wze


def solver(k1, k2, k3, k4, y):
    poly_coeffs = [k4, 0, k3, 0, k2, 0, k1, 0]

    def solve_for_y(poly_coeffs, y):
        pc = poly_coeffs.copy()
        pc[-1] -= y
        return _np.roots(pc)

    R = solve_for_y(poly_coeffs, y)
    for i in R:
        if i.imag == 0:
            r = i.real
            return r
        else:
            continue
    return 0


def findcenter(ra1, dec1, x, y, x0, y0, time0, loc, f, length, step):
    x00 = x0
    y00 = y0
    X, Y, A, C = [[], [], [], []]
    for k in range(0, length):
        x0 = x00 + (k - length / 2) * step
        for j in range(0, length):
            y0 = y00 + (j - length) * step
            E0, e0, Waz, Wze = run1(ra1, dec1, x, y, x0, y0, time0, loc, f)
            for _i in range(20):
                E00 = E0
                E0, a0, e0 = iteration(x, y, x0, y0, Waz, Wze, E0, e0, "iter")
            Caz, Cze, E, a0, e, k1, k2, k3, k4 = iteration(
                x, y, x0, y0, Waz, Wze, E0, e0, ""
            )
            num = -1
            rdif = solver(k1, k2, k3, k4, e0)
            xc = x0 + num * rdif * _np.cos(_np.pi * 2 - (E - a0) - _np.pi / 2)
            yc = y0 + num * rdif * _np.sin(_np.pi * 2 - (E - a0) - _np.pi / 2)
            c2 = 90 - Cze * 180 / _np.pi
            c1 = Caz * 180 / _np.pi
            w2 = 90 - Wze * 180 / _np.pi
            w1 = Waz * 180 / _np.pi
            popt, _ = curve_fit(cosx, c1, (c2 - w2) * _np.pi / 180)
            a, b = popt
            A.append(a)
            popt, _ = curve_fit(sinx, c1, (c1 - w1) * _np.pi / 180)
            c, d = popt
            C.append(c)
            X.append(x0)
            Y.append(y0)
    return X, Y, A, C


def func2d(xy, a0, a1, a2, a3, a4, a5):
    x, y = xy
    return a0 + a1 * x + a2 * y + a3 * x**2 + a4 * x * y + a5 * y**2


def fitcenter(X, Y, A, C, length, step):
    Aabs = abs(_np.array(A))
    Cabs = abs(_np.array(C))
    index1 = _np.where(Aabs > 1)[0]
    index2 = _np.where(Cabs > 1)[0]
    index = _np.union1d(index1, index2)
    X = _np.delete(X, index)
    Y = _np.delete(Y, index)
    A = _np.delete(A, index)
    C = _np.delete(C, index)
    Aabs = _np.delete(Aabs, index)
    Cabs = _np.delete(Cabs, index)
    popt, _ = curve_fit(func2d, (X, Y), Cabs)
    X0 = _np.linspace(min(X), max(X), int(length / step * 10))
    Y0 = _np.linspace(min(Y), max(Y), int(length / step * 10))
    X1, Y1 = _np.meshgrid(X0, Y0)
    Z = func2d((X1, Y1), *popt)
    index = _np.where(Z == _np.min(Z))
    x0_new = X1[index[0][0]][index[1][0]]
    y0_new = Y1[index[0][0]][index[1][0]]
    return x0_new, y0_new


def _load_star_table(targetlist_path):
    """
    读取五列星表: num, RA(度), Dec(度), x, y。
    使用 UTF-8 打开，避免 Windows 默认 GBK 解码失败；.xlsx 为二进制，须改用 targetlist.txt。
    """
    p = Path(targetlist_path)
    if not p.is_file():
        raise FileNotFoundError(f"未找到星表: {p.resolve()}")
    if p.suffix.lower() in (".xlsx", ".xls"):
        raise ValueError(
            "标定需要文本星表（五列: num ra dec x y，度）。"
            "np.loadtxt 无法读取 .xlsx，请使用与本脚本同目录的 targetlist.txt，"
            "或从 Excel 另存为 UTF-8 的 .csv / .txt。"
        )
    with open(p, "r", encoding="utf-8", errors="replace", newline="") as fh:
        try:
            _n, ra1, dec1, x1, y1 = np.loadtxt(fh, dtype=float, unpack=True)
            c = SkyCoord(ra=ra1 * u.degree, dec=dec1 * u.degree, frame="icrs")
        except Exception:
            fh.seek(0)
            _n, ra1, dec1, x1, y1 = np.loadtxt(fh, dtype=str, unpack=True)
            c = SkyCoord(ra=ra1, dec=dec1, frame="icrs")
    ra = c.ra.degree
    dec = c.dec.degree
    x = np.asarray(x1, dtype=float)
    y = np.asarray(y1, dtype=float)
    return ra, dec, x, y


# 全局配置参数（放在程序开头）
CONFIG = {
    # 星点坐标文件（当前脚本目录）
    "targetlist_path": str(current_dir / "targetlist.txt"),
    # 初始图像中心文件（当前脚本目录）
    "x0y0_path": str(current_dir / "x0y0.txt"),
    # 观测时间（北京时间，datetime + Asia/Shanghai；勿用无时区字符串当 UTC）
    "time0": Time(
        datetime.datetime(2022, 2, 5, 1, 14, 47, tzinfo=ZoneInfo("Asia/Shanghai"))
    ),
    # 观测地点
    "latitude": 40.393333,  # 纬度（度）
    "longitude": 117.575000,  # 经度（度）
    "height": 960,  # 海拔（米）
    # 相机参数
    "focal_length_mm": 4.5,  # 焦距（毫米）
    "pixel_per_mm": 1280 / 14.9 * 2,  # x方向pixel数目，ccd尺寸mm，像素密度，jpg文件×2
    # 输出目录：当前脚本目录
    "output_dir": str(current_dir),
}


def calibrate_camera(
    targetlist_path,
    x0y0_path,
    time0,
    latitude,
    longitude,
    height,
    focal_length_mm,
    pixel_per_mm,
    output_dir=None,
):
    """
    完整的相机标定流程

    参数:
        targetlist_path: 星点坐标文件路径 (num, ra, dec, x, y)
        x0y0_path: 初始图像中心文件路径
        time0: 观测时间字符串 "YYYY-MM-DD HH:MM:SS"
        latitude: 纬度（度）
        longitude: 经度（度）
        height: 海拔高度（米）
        focal_length_mm: 焦距（毫米）
        pixel_per_mm: 像素密度（像素/毫米）
        output_dir: 输出目录，默认为 targetlist 所在目录

    返回:
        dict: 包含标定结果的字典
    """

    if output_dir is None:
        output_dir = Path(targetlist_path).parent

    print("=" * 70)
    print("全天相机标定流程")
    print("=" * 70)

    # ==================== 步骤1：读取输入参数 ====================
    print("\n【步骤1】读取输入参数")
    print("-" * 70)

    # 计算焦距（像素单位）
    f = focal_length_mm * pixel_per_mm
    print(f"焦距: {focal_length_mm} mm × {pixel_per_mm} pixel/mm = {f:.2f} pixels")

    # 读取初始图像中心
    with open(x0y0_path, "r", encoding="utf-8", errors="replace") as _xf:
        x0_init, y0_init = np.loadtxt(_xf, dtype=float)
    print(f"初始图像中心: x0 = {x0_init:.2f}, y0 = {y0_init:.2f}")

    # 观测地点和时间
    loc = EarthLocation(
        lat=latitude * u.deg, lon=longitude * u.deg, height=height * u.m
    )
    print(f"观测时间: {time0}")
    print(f"观测地点: 纬度 {latitude}°, 经度 {longitude}°, 海拔 {height}m")

    # 读取星点坐标
    print(f"\n读取星点坐标: {targetlist_path}")
    ra, dec, x, y = _load_star_table(targetlist_path)
    print(f"成功读取 {len(x)} 个星点")

    # ==================== 步骤2：计算地平坐标 ====================
    print("\n【步骤2】将赤道坐标转换为地平坐标")
    print("-" * 70)

    cord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    time = Time(time0)
    altaz = cord.transform_to(AltAz(obstime=time, location=loc))

    ze = np.pi / 2 - altaz.alt.rad  # 天顶距
    az = altaz.az.rad  # 方位角
    print(f"方位角范围: {np.rad2deg(az.min()):.1f}° - {np.rad2deg(az.max()):.1f}°")
    print(f"天顶距范围: {np.rad2deg(ze.min()):.1f}° - {np.rad2deg(ze.max()):.1f}°")

    # ==================== 步骤3：初始标定 (Pipeline1) ====================
    print("\n【步骤3】初始标定 - 计算初始参数")
    print("-" * 70)

    # 计算投影坐标
    b0, proj_u = bu_from_f(x, y, x0_init, y0_init, f)
    print(
        f"投影坐标 b' 范围: {np.rad2deg(b0.min()):.1f}° - {np.rad2deg(b0.max()):.1f}°"
    )
    print(
        f"投影坐标 u 范围: {np.rad2deg(proj_u.min()):.1f}° - {np.rad2deg(proj_u.max()):.1f}°"
    )

    # 获取初始 E 和 e
    E0, e0 = ini_Ee(b0, proj_u, az, ze)
    print(f"\n初始参数:")
    print(f"  E0 = {E0:.6f} rad ({np.rad2deg(E0):.2f}°)")
    print(f"  e0 = {e0:.6f} rad ({np.rad2deg(e0):.2f}°)")

    # 迭代优化
    print("\n开始迭代优化...")
    for i in range(20):
        E00 = E0
        E0, a0, e0 = iteration(x, y, x0_init, y0_init, az, ze, E0, e0, "iter")
        dE_percent = (E00 - E0) / E00 * 100 if E00 != 0 else 0
        print(
            f"  迭代 {i+1:2d}: E={E0:.5f}, a0={a0:.5f}, e={e0:.5f}, dE={dE_percent:.3f}%"
        )

    # 计算最终参数
    Caz, Cze, E_init, a0_init, e_init, k1, k2, k3, k4 = iteration(
        x, y, x0_init, y0_init, az, ze, E0, e0, ""
    )

    print(f"\n初始标定完成:")
    print(f"  E  = {E_init:.6f} rad ({np.rad2deg(E_init):.2f}°)")
    print(f"  a0 = {a0_init:.6f} rad ({np.rad2deg(a0_init):.2f}°)")
    print(f"  e  = {e_init:.6f} rad ({np.rad2deg(e_init):.2f}°)")
    print(f"  k1 = {k1:.10f}")
    print(f"  k2 = {k2:.15e}")
    print(f"  k3 = {k3:.15e}")
    print(f"  k4 = {k4:.15e}")

    # ==================== 步骤4：精确中心搜索 (Pipeline3) ====================
    print("\n【步骤4】精确中心搜索 - 第一轮 (步长 1 像素)")
    print("-" * 70)

    length = 20
    step1 = 1
    step2 = 0.1

    print(f"搜索范围: ±{length} 像素")
    print(f"第一轮步长: {step1} 像素")
    print("这将需要一些时间，请耐心等待...\n")

    X, Y, A, C = findcenter(
        ra, dec, x, y, x0_init, y0_init, time0, loc, f, length, step1
    )
    x0_new1, y0_new1 = fitcenter(X, Y, A, C, length, step1)

    print(f"\n第一轮搜索完成:")
    print(f"  新中心: x0 = {x0_new1:.2f}, y0 = {y0_new1:.2f}")
    print(f"  偏移量: Δx = {x0_new1-x0_init:.2f}, Δy = {y0_new1-y0_init:.2f}")

    # ==================== 步骤5：精确中心搜索 - 第二轮 ====================
    print("\n【步骤5】精确中心搜索 - 第二轮 (步长 0.1 像素)")
    print("-" * 70)

    print(f"第二轮步长: {step2} 像素")
    print("这将需要一些时间，请耐心等待...\n")

    X, Y, A, C = findcenter(
        ra, dec, x, y, x0_new1, y0_new1, time0, loc, f, length, step2
    )
    x0_new, y0_new = fitcenter(X, Y, A, C, length, step2)

    print(f"\n第二轮搜索完成:")
    print(f"  最终中心: x0 = {x0_new:.4f}, y0 = {y0_new:.4f}")
    print(f"  相对初始: Δx = {x0_new-x0_init:.4f}, Δy = {y0_new-y0_init:.4f}")
    print(f"  相对第一轮: Δx = {x0_new-x0_new1:.4f}, Δy = {y0_new-y0_new1:.4f}")

    # ==================== 步骤6：用新中心重新计算参数 ====================
    print("\n【步骤6】用新中心重新计算标定参数")
    print("-" * 70)

    E0, e0, Waz, Wze = run1(ra, dec, x, y, x0_init, y0_init, time0, loc, f)

    print("开始迭代优化...")
    for i in range(20):
        E00 = E0
        E0, a0, e0 = iteration(x, y, x0_new, y0_new, Waz, Wze, E0, e0, "iter")
        dE_percent = (E00 - E0) / E00 * 100 if E00 != 0 else 0
        print(
            f"  迭代 {i+1:2d}: E={E0:.5f}, a0={a0:.5f}, e={e0:.5f}, dE={dE_percent:.3f}%"
        )

    Caz, Cze, E_new, a0_new, e_new, k1_new, k2_new, k3_new, k4_new = iteration(
        x, y, x0_new, y0_new, Waz, Wze, E0, e0, ""
    )

    print(f"\n最终标定参数:")
    print(f"  E  = {E_new:.6f} rad ({np.rad2deg(E_new):.2f}°)")
    print(f"  a0 = {a0_new:.6f} rad ({np.rad2deg(a0_new):.2f}°)")
    print(f"  e  = {e_new:.6f} rad ({np.rad2deg(e_new):.2f}°)")
    print(f"  k1 = {k1_new:.10f}")
    print(f"  k2 = {k2_new:.15e}")
    print(f"  k3 = {k3_new:.15e}")
    print(f"  k4 = {k4_new:.15e}")

    # ==================== 步骤7：保存结果 ====================
    print("\n【步骤7】保存标定结果")
    print("-" * 70)

    output_dir = Path(output_dir)

    # 保存新的图像中心
    x0y0_output = output_dir / "x0y0_new.txt"
    with open(x0y0_output, "w+") as f:
        print(x0_new, y0_new, file=f)
    print(f"✓ 保存图像中心: {x0y0_output}")

    # 保存新的标定参数
    params_output = output_dir / "parameters_new.txt"
    with open(params_output, "w+") as f:
        print(E_new, a0_new, e_new, k1_new, k2_new, k3_new, k4_new, file=f)
    print(f"✓ 保存标定参数: {params_output}")

    # ==================== 完成 ====================
    print("\n" + "=" * 70)
    print("标定完成！")
    print("=" * 70)

    return {
        "x0_new": x0_new,
        "y0_new": y0_new,
        "E": E_new,
        "a0": a0_new,
        "e": e_new,
        "k1": k1_new,
        "k2": k2_new,
        "k3": k3_new,
        "k4": k4_new,
        "x0y0_file": str(x0y0_output),
        "params_file": str(params_output),
    }


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 基于当前脚本位置构造相对路径
    current_dir = Path(__file__).resolve().parent

    # 配置参数
    CONFIG = {
        # 星点坐标：文本五列 num ra dec x y（度）；勿用 .xlsx
        "targetlist_path": str(current_dir / "targetlist.txt"),
        # 初始图像中心文件
        "x0y0_path": str(current_dir / "x0y0.txt"),
        # 观测时间（北京时间）
        "time0": Time(
            datetime.datetime(2022, 2, 5, 1, 14, 47, tzinfo=ZoneInfo("Asia/Shanghai"))
        ),
        # 观测地点
        "latitude": 40.393333,  # 纬度（度）
        "longitude": 117.575000,  # 经度（度）
        "height": 960,  # 海拔（米）
        # 相机参数
        "focal_length_mm": 4.5,  # 焦距（毫米）
        "pixel_per_mm": 1280 / 14.9 * 2,  # 像素密度，jpg文件×2
        # 输出目录（可选，默认与 targetlist 同目录）
        "output_dir": str(current_dir),
    }

    # 运行标定
    result = calibrate_camera(**CONFIG)

    print("\n结果文件:")
    print(f"  {result['x0y0_file']}")
    print(f"  {result['params_file']}")
