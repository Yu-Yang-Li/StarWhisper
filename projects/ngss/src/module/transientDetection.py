import os
from datetime import datetime
import pandas as pd

def pipeline_process(query_dt, station, telescope):
    # 构建目录路径
    directory = os.path.join(r'/home/pod/shared-nvme/NGSS/pdf', query_dt, station, str(telescope))
    # 构建CSV文件路径
    csv_filename = f'{query_dt}_{station}_{telescope}.csv'
    csv_path = os.path.join('/home/pod/shared-nvme/NGSS/AUTOGLASS', csv_filename)

    def check_files_and_split_names():
        large_files = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath) and os.path.getsize(filepath) > 100 * 1024:  # 100KB
                try:
                    _, date_str, source, *_ = filename.split('_')
                    date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    large_files.append((filename, date, source))
                except ValueError:
                    print(f"Could not parse date from file: {filename}")
        return large_files


    def read_csv_and_extract_data():
        # 定义列名，包括 CSV 文件中的所有列
        column_names = ['DATE', 'OBJECT', 'RA', 'DEC', 'X_IMAGE', 'Y_IMAGE', 'SOURCES', 'TEMPLATE_PATH', 'SCIENCE_PATH', 'DIFFERENCE_PATH', 'PDF_PATH', 'JPG_PATH']
        try:
            df = pd.read_csv(csv_path, names=column_names, header=0, on_bad_lines='warn')
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV: {e}")
            return pd.DataFrame(columns=['DATE', 'OBJECT', 'RA', 'DEC', 'PDF_PATH'])
        # 提取所需的列
        extracted_data = df[['DATE', 'OBJECT', 'RA', 'DEC', 'PDF_PATH']]
        return extracted_data

    def get_all_sources(large_files):
        sources = set()
        for _, _, source in large_files:
            sources.add(source)
        return list(sources)

    def filter_csv_data(large_files, csv_data):
        filtered_csv_data = pd.DataFrame(columns=['DATE', 'OBJECT', 'RA', 'DEC', 'PDF_PATH'])
        for filename, date, _ in large_files:
            matched_row = csv_data[(csv_data['DATE'] == str(date))]
            if not matched_row.empty:
                filtered_csv_data = pd.concat([filtered_csv_data, matched_row], ignore_index=True)
        return filtered_csv_data

    # 主处理逻辑
    large_files = check_files_and_split_names()
    csv_data = read_csv_and_extract_data()

    # 获取所有来源
    all_sources = get_all_sources(large_files)

    # 过滤CSV数据
    #filtered_csv_data = filter_csv_data(large_files, csv_data)

    return all_sources, csv_data #filtered_csv_data

