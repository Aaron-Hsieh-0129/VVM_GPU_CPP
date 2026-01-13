import numpy as np
import netCDF4
import os

filename = '../rundata/TOPO.nc' 

nz, ny, nx = 33, 32, 32
DX = 500.0
DY = 500.0

variable_name = 'topo'

z_levels = np.array([
    46, 160, 301, 470, 667, 891, 1143, 1422, 1729, 2063, 2425, 2815, 
    3232, 3677, 4150, 4650, 5177, 5732, 6315, 6925, 7563, 8229, 8922, 
    9643, 10391, 11167, 11970, 12801, 13660, 14546, 15460, 16401, 17370
])

center_y = ny // 2
center_x = nx // 2

h_max = 10000.0
radius_m = 10000.0

radius_x_grid = radius_m / DX
radius_y_grid = radius_m / DY

y = np.arange(ny)
x = np.arange(nx)
xx, yy = np.meshgrid(x, y)

topo_index = np.zeros((ny, nx))
# topo_index[center_y-6:center_y+6, center_x-15:center_x+15] = 15
# topo_index[1:4, 1:4] = 15
# topo_index[6:9, 1:4] = 15
# topo_index[1:4, 6:9] = 15
# topo_index[6:9, 6:9] = 15

try:
    with netCDF4.Dataset(filename, 'w', format='NETCDF4') as ncfile:
        
        ncfile.createDimension('nz', nz) 
        ncfile.createDimension('ny', ny)
        ncfile.createDimension('nx', nx)
        
        var = ncfile.createVariable(variable_name, 'i4', ('ny', 'nx'))
        var.description = '2D terrain height as a z-index (dx=500m, nx=256, r=4km)'
        var.units = 'index'
        
        var[:] = topo_index

except Exception as e:
    print(f"\n{e}")
