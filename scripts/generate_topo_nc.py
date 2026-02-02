import numpy as np
import netCDF4
import os

FILENAME = '../rundata/init.nc'
NZ, NY, NX = 33, 512, 512
DX, DY = 500.0, 500.0

os.makedirs(os.path.dirname(FILENAME), exist_ok=True)

def get_topo_data(ny, nx):
    topo = np.zeros((ny, nx), dtype='i4')
    return topo

def get_tg_data(ny, nx, value=305.0):
    return np.full((ny, nx), value, dtype='f4')

def get_z_levels():
    return np.array([
        46, 160, 301, 470, 667, 891, 1143, 1422, 1729, 2063, 2425, 2815, 
        3232, 3677, 4150, 4650, 5177, 5732, 6315, 6925, 7563, 8229, 8922, 
        9643, 10391, 11167, 11970, 12801, 13660, 14546, 15460, 16401, 17370
    ], dtype='f4')

variables_config = {
    'topo': {
        'data': get_topo_data(NY, NX),
        'dims': ('ny', 'nx'),
        'units': 'index',
        'desc': '2D terrain height as a z-index',
        'dtype': 'i4'
    },
    'Tg': {
        'data': get_tg_data(NY, NX, value=305.0),
        'dims': ('ny', 'nx'),
        'units': 'K',
        'desc': 'Ground Surface Temperature',
        'dtype': 'f8'
    },
    # 'Psfc': {
    #     'data': np.full((NY, NX), 101325.0),
    #     'dims': ('ny', 'nx'),
    #     'units': 'Pa',
    #     'desc': 'Surface Pressure',
    #     'dtype': 'f4'
    # }
}

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
                
        print("Finished")

    except Exception as e:
        print(f"\n[Error] Fail to write: {e}")

if __name__ == "__main__":
    dimensions = {'nz': NZ, 'ny': NY, 'nx': NX}
    write_netcdf(FILENAME, variables_config, dimensions)
