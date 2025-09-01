import h5py
import netCDF4 as nc
import numpy as np
import sys
import os
import time
from joblib import Parallel, delayed
import multiprocessing

# --- 設定 ---
source_filename = 'vvm_output.nc'
output_filename = 'vvm_output_restructured_grads.nc'
TIME_UNITS = 'minutes since 1970-01-01 00:00:00'

# ==============================================================================
# <<< 新增：地理座標設定 >>>
# 請根據您的模擬區域，修改以下數值
# 這裡以台灣周邊區域為例
# ==============================================================================
START_LON = 119.0  # 起始經度 (degrees_east)
START_LAT = 21.5   # 起始緯度 (degrees_north)
DX = 0.05          # X方向網格間距 (經度)
DY = 0.05          # Y方向網格間距 (緯度)
# ==============================================================================


# ==============================================================================
# 優化後的 Worker Function
# ==============================================================================
def process_chunk(chunk_data):
    """
    此函式在一個獨立的 CPU 核心上執行，專門處理資料轉置。
    """
    processed_chunk = {}
    for var_name, data in chunk_data.items():
        if len(data.shape) == 3:
            processed_chunk[var_name] = np.transpose(data, (2, 1, 0))
        elif len(data.shape) == 2:
            processed_chunk[var_name] = np.transpose(data, (1, 0))
        elif len(data.shape) == 1:
            processed_chunk[var_name] = data
    return processed_chunk

# ==============================================================================
# 主程式 (Main Script) - 已優化
# ==============================================================================
def main():
    """主執行函式"""
    start_time = time.time()

    try:
        print(f"正在讀取 {source_filename}...")
        if not os.path.exists(source_filename):
            raise FileNotFoundError(f"錯誤：找不到輸入檔案 '{source_filename}'。")

        # --- 1. 準備階段 (在主程序中執行) ---
        var_metadata = {}
        all_data = {}
        time_values = []

        with h5py.File(source_filename, 'r') as f_in:
            if 'NumSteps' not in f_in.attrs:
                raise ValueError("在 HDF5 檔案的全域屬性中找不到 'NumSteps'。")
            num_steps = int(f_in.attrs['NumSteps'])
            step0_group = f_in['Step0']
            coord_var_names = ['time', 'x', 'y', 'z_mid']
            data_var_names = [name for name in step0_group.keys() if name not in coord_var_names]
            if not data_var_names:
                raise ValueError("在 'Step0' 群組中找不到任何數據變數。")

            # 一次性讀取所有時間步的資料到記憶體
            print("正在將所有時間步的資料讀取到記憶體中...")
            for var_name in data_var_names:
                all_data[var_name] = []
            
            for i in range(num_steps):
                step_group_name = f'Step{i}'
                current_group = f_in[step_group_name]
                time_values.append(current_group['time'][()])
                for var_name in data_var_names:
                    all_data[var_name].append(current_group[var_name][:])
            
            # 將 list 轉換為 numpy array
            for var_name in data_var_names:
                all_data[var_name] = np.array(all_data[var_name])

            dim_map = {'time': num_steps}
            if 'z_mid' in step0_group:
                dim_map['z'] = step0_group['z_mid'].shape[0]
            else:
                raise ValueError("在 'Step0' 群組中找不到必要的 'z_mid' 座標變數。")

            nx, ny = None, None
            for var_name in data_var_names:
                shape = step0_group[var_name].shape
                if len(shape) == 3:
                    nx, ny = shape[0], shape[1]
                    break

            if nx is None or ny is None:
                raise ValueError("在檔案中找不到任何 3D 數據變數來推斷 X 和 Y 的維度。")

            dim_map['x'], dim_map['y'] = nx, ny
            print(f"從資料推斷出維度大小: X={dim_map['x']}, Y={dim_map['y']}, Z={dim_map['z']}, Time={dim_map['time']}")

            z_mid_data = step0_group['z_mid'][:]

            for var_name in data_var_names:
                var_metadata[var_name] = {
                    'shape': step0_group[var_name].shape,
                    'dtype': step0_group[var_name].dtype
                }

        print(f"正在建立 GrADS 相容的 NetCDF 檔案: {output_filename}...")
        with nc.Dataset(output_filename, 'w', format='NETCDF4') as f_out:
            f_out.createDimension('time', dim_map.get('time'))
            f_out.createDimension('z', dim_map['z'])
            f_out.createDimension('y', dim_map['y'])
            f_out.createDimension('x', dim_map['x'])

            times = f_out.createVariable('time', 'f8', ('time',)); times.units = TIME_UNITS
            z_var = f_out.createVariable('z', 'f4', ('z',)); z_var.units = 'meters'; z_var[:] = z_mid_data
            
            lon_values = START_LON + np.arange(dim_map['x']) * DX
            lat_values = START_LAT + np.arange(dim_map['y']) * DY

            y_var = f_out.createVariable('y', 'f4', ('y',))
            y_var.units = 'degrees_north'
            y_var.long_name = 'latitude'
            y_var[:] = lat_values

            x_var = f_out.createVariable('x', 'f4', ('x',))
            x_var.units = 'degrees_east'
            x_var.long_name = 'longitude'
            x_var[:] = lon_values
            
            nc_vars = {}
            for var_name in data_var_names:
                var_shape = var_metadata[var_name]['shape']
                var_dtype = var_metadata[var_name]['dtype']

                if len(var_shape) == 3: dim_names_for_var = ('time', 'z', 'y', 'x')
                elif len(var_shape) == 2: dim_names_for_var = ('time', 'y', 'x')
                elif len(var_shape) == 1: dim_names_for_var = ('time', 'z')
                else: continue
                # 加入 zlib 壓縮
                nc_vars[var_name] = f_out.createVariable(var_name, var_dtype, dim_names_for_var, zlib=True, complevel=4)

            # --- 2. 平行處理階段 (已優化) ---
            num_cores = 4
            print(f"\n--- 開始使用 {num_cores} 個核心進行平行處理 ---")
            
            # 準備要給子程序處理的資料 chunk
            chunks = [{} for _ in range(num_steps)]
            for var_name, data_array in all_data.items():
                for i in range(num_steps):
                    chunks[i][var_name] = data_array[i]

            processed_results = Parallel(n_jobs=num_cores)(
                delayed(process_chunk)(chunk) for chunk in chunks
            )

            # --- 3. 寫入階段 (已優化) ---
            print("\n--- 所有時間步的資料處理完成，正在將結果寫入檔案 ---")
            times[:] = time_values

            for var_name in data_var_names:
                print(f"正在寫入變數: {var_name}")
                # 將所有時間步的資料組合起來
                stacked_data = np.array([res[var_name] for res in processed_results if var_name in res])
                
                # 根據維度寫入 NetCDF
                if var_name in nc_vars:
                    shape_len = len(var_metadata[var_name]['shape'])
                    if shape_len == 3:
                        nc_vars[var_name][:, :, :, :] = stacked_data
                    elif shape_len == 2:
                        nc_vars[var_name][:, :, :] = stacked_data
                    elif shape_len == 1:
                        nc_vars[var_name][:, :] = stacked_data
            
        end_time = time.time()
        print("\n轉換完成！")
        print(f"總執行時間: {end_time - start_time:.2f} 秒")
        print(f"新檔案 '{output_filename}' 現在應該可以被 GrADS 的 sdfopen 正確讀取了。")

    except Exception as e:
        print(f"發生錯誤: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
