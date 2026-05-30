import numpy as np
import netCDF4
import os

OUT_DIR = '../rundata/LS_forcings/'
FILE_PREFIX = 'ls_forcing_'
CONSTANT_FILENAME = 'ls_forcing_constant.nc'

NZ, NY, NX = 44, 192, 192

TIME_VARYING = False
TIMES_TO_GENERATE = [0, 3600, 7200, 10800]

os.makedirs(OUT_DIR, exist_ok=True)

RAW_QV_PROFILE = np.array([
    0.0214213, 0.0203016, 0.019438, 0.01857, 0.0176225, 0.0167084, 
    0.0158519, 0.0151217, 0.0141652, 0.0129944, 0.0117892, 0.010527, 
    0.00915218, 0.00789995, 0.00679199, 0.00578312, 0.00481545, 0.003805, 
    0.0028161, 0.00198373, 0.00131648, 0.000813794, 0.000464324, 0.000245091, 
    0.000119471, 5.302e-05, 2.20153e-05, 9.19223e-06, 4.34173e-06, 2.45438e-06, 
    1.52178e-06, 8.26758e-07, 3.5936e-08, 9.77558e-22, -5.50691e-21, -1.2166e-20, 
    -1.89998e-20, -2.60083e-20, -3.31914e-20, -4.05492e-20, -4.80817e-20, 
    -5.57888e-20, -6.36706e-20, -7.17271e-20
])

def get_qv_forcing_data(time_sec, nz, ny, nx):
    qv = np.zeros((nz, ny, nx), dtype='f8')

    valid_nz = min(nz, len(RAW_QV_PROFILE))
    base_profile = RAW_QV_PROFILE[:valid_nz]

    gradient_x = np.linspace(1.3, 0.7, nx)

    for k in range(valid_nz):
        qv_k = base_profile[k]
        qv[k, :, :] = qv_k * gradient_x
    return qv
    
# def get_th_forcing_data(time_sec, nz, ny, nx):
#     th = np.zeros((nz, ny, nx), dtype='f8')
#     for k in range(nz):
#         th[k, :, :] = 300.0 + k * 3.0
#     return th

def write_netcdf(filename, vars_config, dims_size):
    try:
        with netCDF4.Dataset(filename, 'w', format='NETCDF4') as ncfile:
            print(f"Writing File: {filename}")

            for dim_name, size in dims_size.items():
                ncfile.createDimension(dim_name, size)
            
            for var_name, info in vars_config.items():
                var = ncfile.createVariable(
                    var_name, 
                    info['dtype'], 
                    info['dims']
                )
                
                var.units = info.get('units', '')
                var.description = info.get('desc', '')
                var[:] = info['data']
                
                print(f"  -> Written: {var_name} ({info['dims']})")
                
    except Exception as e:
        print(f"Failed to write NetCDF file {filename}: {e}")

def generate_file_for_time(t, filename):
    dims_size = {'nz': NZ, 'ny': NY, 'nx': NX}
    
    vars_config = {
        'qv': {
            'data': get_qv_forcing_data(t, NZ, NY, NX),
            'dims': ('nz', 'ny', 'nx'),
            'units': 'kg/kg',
            'desc': f'Large-scale forcing for water vapor at t={t}s',
            'dtype': 'f8'
        }
        # 'th': {
        #     'data': get_th_forcing_data(t, NZ, NY, NX),
        #     'dims': ('nz', 'ny', 'nx'),
        #     'units': 'K',
        #     'desc': f'Large-scale forcing for potential temp at t={t}s',
        #     'dtype': 'f8'
        # }
    }
    
    write_netcdf(filename, vars_config, dims_size)


if __name__ == "__main__":
    if TIME_VARYING:
        print(f"Generating time-varying forcing files...")
        for t in TIMES_TO_GENERATE:
            filename = os.path.join(OUT_DIR, f"{FILE_PREFIX}{t:06d}.nc")
            generate_file_for_time(t, filename)
    else:
        print(f"Generating constant forcing file...")
        filename = os.path.join(OUT_DIR, CONSTANT_FILENAME)
        generate_file_for_time(0, filename)
        
    print("Done!")
