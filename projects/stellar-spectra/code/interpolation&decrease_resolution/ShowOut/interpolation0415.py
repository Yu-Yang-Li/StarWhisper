import numpy as np
from astropy.io import fits
import os
from typing import Union, List
import warnings
from scipy import interpolate
from pathlib import Path
import gc
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

def generate_intermediate_points(start: float, end: float, scale_factor: float) -> np.ndarray:
    """生成两个值之间的中间点"""
    step = (end - start) * scale_factor
    return np.arange(start + step, end, step)

def interpolate_3d(teff_points, logg_points, feh_points, teff, logg, feh, base_path):
    """三维线性插值函数"""
    # 找到最近的点的索引
    t_idx = np.searchsorted(teff_points, teff)
    g_idx = np.searchsorted(logg_points, logg)
    f_idx = np.searchsorted(feh_points, feh)
    
    # 确保索引在有效范围内
    t_idx = min(max(t_idx, 1), len(teff_points))
    g_idx = min(max(g_idx, 1), len(logg_points))
    f_idx = min(max(f_idx, 1), len(feh_points))
    
    # 获取周围的点
    t_values = [teff_points[t_idx-1], teff_points[min(t_idx, len(teff_points)-1)]]
    g_values = [logg_points[g_idx-1], logg_points[min(g_idx, len(logg_points)-1)]]
    f_values = [feh_points[f_idx-1], feh_points[min(f_idx, len(feh_points)-1)]]
    
    # 读取所有周围点的数据
    data_points = []
    for f_val in f_values:
        folder = f"Z{f_val:+.1f}"
        for g_val in g_values:
            for t_val in t_values:
                filename = f"lte{int(t_val):05d}-{g_val:.2f}{f_val:+.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
                filepath = os.path.join(base_path, folder, filename)
                with fits.open(filepath) as hdul:
                    data_points.append(hdul[0].data.copy())
    
    # 构建插值点的坐标
    points = np.array([(t, g, f) for f in f_values for g in g_values for t in t_values])
    values = np.array(data_points)
    
    # 执行三维线性插值
    interpolator = interpolate.LinearNDInterpolator(points, values)
    result = interpolator(np.array([teff, logg, feh]))
    
    return result[0]

def generate_intermediate_spectra(
    teff: Union[List[float], float],
    logg: Union[List[float], float],
    feh: Union[List[float], float],
    scale_factor: float
) -> None:
    # 将输入转换为numpy数组并排序
    teff = np.sort(np.array([teff]) if isinstance(teff, (int, float)) else np.array(teff))
    logg = np.sort(np.array([logg]) if isinstance(logg, (int, float)) else np.array(logg))
    feh = np.sort(np.array([feh]) if isinstance(feh, (int, float)) else np.array(feh))
    
    # 基础路径
    base_path = r"/data1/R10000FITS"
    
    # 在主循环开始前强制进行垃圾回收
    gc.collect()
    
    try:
        # 对每个Fe/H区间进行处理，包括最后一个点
        for f_idx in range(len(feh)):
            # 处理非边界点
            if f_idx < len(feh) - 1:
                f1, f2 = feh[f_idx], feh[f_idx+1]
                intermediate_fehs = np.append([f1], generate_intermediate_points(f1, f2, scale_factor))
            else:
                # 处理最后一个点
                intermediate_fehs = [feh[f_idx]]
            
            for f in intermediate_fehs:
                print(f'=========================  Processing FeH = {f}  =================================')
                folder = f"Z{f:+.1f}"
                folder_path = os.path.join(base_path, folder)
                
                # 对每个log g区间进行处理，包括最后一个点
                for g_idx in range(len(logg)):
                    # 处理非边界点
                    if g_idx < len(logg) - 1:
                        g1, g2 = logg[g_idx], logg[g_idx+1]
                        intermediate_loggs = np.append([g1], generate_intermediate_points(g1, g2, scale_factor))
                    else:
                        # 处理最后一个点
                        intermediate_loggs = [logg[g_idx]]
                    
                    for g in intermediate_loggs:
                        print(f'Processing FeH = {f}, logg = {g}')
                        # 对每个TEFF区间进行处理，包括最后一个点
                        for t_idx in range(len(teff)):
                            # 处理非边界点
                            if t_idx < len(teff) - 1:
                                t1, t2 = teff[t_idx], teff[t_idx+1]
                                intermediate_teffs = np.append([t1], generate_intermediate_points(t1, t2, scale_factor))
                            else:
                                # 处理最后一个点
                                intermediate_teffs = [teff[t_idx]]
                            
                            for new_teff in intermediate_teffs:
                                
                                try:
                                    # 创建新的header
                                    new_hdr = fits.Header()
                                    new_hdr['PHXTEFF'] = new_teff
                                    new_hdr['PHXLOGG'] = g
                                    new_hdr['PHXM_H'] = f
                                    
                                    # 如果是原始点，从文件读取数据
                                    if new_teff in teff and g in logg and f in feh:
                                        filename = f"lte{int(new_teff):05d}-{g:.2f}{f:+.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
                                        with fits.open(os.path.join(folder_path, filename)) as hdul:
                                            new_data = hdul[0].data.copy()
                                    else:
                                        # 否则进行三维线性插值
                                        new_data = interpolate_3d(teff, logg, feh, new_teff, g, f, base_path)
                                    
                                    # 保存新的fits文件
                                    new_filename = f"lte{int(new_teff):05d}-{g:.2f}{f:+.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
                                    new_path = os.path.join(folder_path, new_filename)
                                    
                                    # 确保目标文件夹存在
                                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                                    
                                    # 如果文件存在，先尝试删除
                                    if os.path.exists(new_path):
                                        try:
                                            os.remove(new_path)
                                        except PermissionError:
                                            print(f"警告：无法删除已存在的文件 {new_filename}，可能被其他程序占用")
                                            continue
                                    
                                    # 创建并保存新的fits文件
                                    hdu = fits.PrimaryHDU(data=new_data, header=new_hdr)
                                    hdu.writeto(new_path, overwrite=True)
                                    
                                    # print(f"已生成文件: {new_filename}")
                                    
                                except Exception as e:
                                    print(f"处理文件时出错: {str(e)}")
                                    continue
                                
                                finally:
                                    # 确保清理内存
                                    gc.collect()
                        
                        print(f'Finish FeH = {f}, logg = {g}')
                print(f'=========================  Finish FeH = {f}  =================================')
                
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        
    finally:
        # 最后再次进行垃圾回收
        gc.collect()

def read_cube_parameters(csv_file='valid_cubes.csv'):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Process each cube
    cube_params = []
    for cube_id in df['cube_id'].unique():
        cube_data = df[df['cube_id'] == cube_id]
        
        # Extract unique values for each parameter
        teff_values = sorted(cube_data['Teff'].unique())
        logg_values = sorted(cube_data['log_g'].unique())
        feh_values = sorted(cube_data['Fe_H'].unique())
        
        # Store as a parameter set
        cube_params.append({
            'teff': teff_values[:2],  # Only need min and max
            'logg': logg_values[:2],  # Only need min and max
            'feh': feh_values[:2],    # Only need min and max
        })
    
    return cube_params

'''
# Read and format the cube parameters
cube_params = read_cube_parameters('valid_cubes.csv')

# Print parameters in the desired format
for i, params in enumerate(cube_params):
    teff = params['teff']
    logg = params['logg']
    feh = params['feh']
    scale_factor = 1/10
    generate_intermediate_spectra(teff, logg, feh, scale_factor)
'''    
    
def mainfunction(params):
    teff = params['teff']
    logg = params['logg']
    feh = params['feh']
    scale_factor = 1/10
    generate_intermediate_spectra(teff, logg, feh, scale_factor)

# 假设读取参数的函数已经定义好
cube_params = read_cube_parameters(r'/home/wcs/SpecDeNoise/ShowOut/valid_cubes.csv')

# 固定线程数为24
max_workers = 80

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    for params in cube_params:
        executor.submit(mainfunction, params)