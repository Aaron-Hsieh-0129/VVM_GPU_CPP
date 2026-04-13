# ========================================================================================
# This program can read your configuration and generate the corresponding land variables.
# Easily switch between Real Taiwan Topography and Idealized Experiment setup.
# ========================================================================================

import json
import os

import netCDF4 as nc
import numpy as np
from scipy import stats

# ==============================================================================
# Experimental Mode Switch (User Toggle)
# ==============================================================================
# Set to True : Read high-resolution Taiwan topography and perform coarsening
# Set to False: Enter "Idealized Simulation" mode for user-defined ridge & land types
USE_TAIWAN_TOPO = True

CONFIG_PATH = '../rundata/input_configs/default_config.json'
SOURCE_TW_DATA = '../rundata/land/topolsm_TW.nc'

# ==============================================================================
# Helper Functions for Idealized Simulation
# ==============================================================================
def get_ideal_topo_data(ny, nx):
    topo = np.zeros((ny, nx), dtype='i4')
    mountain_width = nx // 4
    start_col = nx - mountain_width
    peak_col = start_col + (mountain_width // 2)
    max_height = 7
    
    cols_rise = np.arange(start_col, peak_col + 1)
    cols_fall = np.arange(peak_col + 1, nx)
    
    rise_vals = np.round((cols_rise - start_col) * max_height / (peak_col - start_col))
    fall_vals = np.round((nx - cols_fall) * max_height / (nx - peak_col))
    
    topo[:, cols_rise] = rise_vals.astype('i4')
    topo[:, cols_fall] = fall_vals.astype('i4')
    return topo

def get_ideal_vegtype_data(ny, nx):
    vegtype = np.ones((ny, nx), dtype='i4')
    vegtype[:, :nx//2] = 17         # IGBP 17 = Water Bodies (Aligned with standard)
    vegtype[:, nx//2:nx//4*3] = 13  # IGBP 13 = Urban
    vegtype[:, nx//4*3:] = 2        # IGBP 2 = Evergreen Broadleaf
    return vegtype

def get_ideal_soiltype_data(ny, nx):
    soiltype = np.ones((ny, nx), dtype='i4') # STATSGO 1 = Sand
    soiltype[:, :nx//2] = 14                 # STATSGO 14 = Water
    soiltype[:, nx//2:] = 13                 # STATSGO 13 = Organic Material
    return soiltype

def get_ideal_slopetype_data(ny, nx):
    slopetype = np.ones((ny, nx), dtype='i4')
    slopetype[:,:] = 1
    return slopetype

def get_ideal_tg_data(ny, nx, value=305.0):
    return np.full((ny, nx), value, dtype='f8')

# ==============================================================================
# Auto-read Configuration and Create Pure Physical Grid
# ==============================================================================
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

NX = config['grid']['nx']
NY = config['grid']['ny']
NZ = config['grid']['nz']
DX = config['grid']['dx']
DY = config['grid']['dy']
DZ = config['grid']['dz']
DZ1 = config['grid']['dz1']
HALO = config['grid']['n_halo_cells']

FILENAME = config['netcdf_reader']['source_file']
os.makedirs(os.path.dirname(FILENAME), exist_ok=True)

# Initialize 1D longitude and latitude arrays
lon_1d = np.zeros(NX, dtype='f4')
lat_1d = np.zeros(NY, dtype='f4')

# Calculate stretched vertical grid heights (z_up) - starting from 0 meters
domain = 15000.0
cz2 = (DZ - DZ1) / (DZ * (domain - DZ))
cz1 = 1.0 - cz2 * domain

z_up = np.zeros(NZ, dtype='f8')
z_up[0] = 0.0  # Layer 0 is the surface (physical height 0)
for k in range(1, NZ):  
    z_up[k] = z_up[k-1] + DZ
for k in range(0, NZ):  
    z_up[k] = z_up[k] * (cz1 + cz2 * z_up[k])

# Initialize default surface arrays (Using IGBP classification standard)
topo   = np.zeros((NY, NX), dtype='i4')
lu     = np.full((NY, NX), 17, dtype='i4') # IGBP 17 = Water bodies / Ocean
soil   = np.full((NY, NX), 14, dtype='i4') # STATSGO 14 = Default soil for water
slope  = np.zeros((NY, NX), dtype='i4')
albedo = np.full((NY, NX), 8.0, dtype='f4')
gvf    = np.zeros((NY, NX), dtype='f4')
lai    = np.zeros((NY, NX), dtype='f4')
shdmax = np.zeros((NY, NX), dtype='f4')
shdmin = np.zeros((NY, NX), dtype='f4')
Tg     = np.full((NY, NX), 300.0, dtype='f8')

# ==============================================================================
# Logic Branching: Real Topography vs. Idealized Simulation
# ==============================================================================
if USE_TAIWAN_TOPO and os.path.exists(SOURCE_TW_DATA):
    print(f"--- Mode: Real Topography ({SOURCE_TW_DATA}) ---")
    
    def get_mode_2d(arr, factor):
        m, n = arr.shape[0] // factor, arr.shape[1] // factor
        reshaped = arr[:m*factor, :n*factor].reshape(m, factor, n, factor)
        mode_val, _ = stats.mode(reshaped, axis=(1, 3), keepdims=False)
        return mode_val

    def get_mean_2d(arr, factor):
        m, n = arr.shape[0] // factor, arr.shape[1] // factor
        reshaped = arr[:m*factor, :n*factor].reshape(m, factor, n, factor)
        return reshaped.mean(axis=(1, 3))

    # Create USGS (1-27) to IGBP (1-20) mapping array
    usgs_to_igbp = np.zeros(30, dtype='i4')
    usgs_to_igbp[1] = 13     # Urban
    usgs_to_igbp[2:5] = 12   # Croplands
    usgs_to_igbp[5:7] = 14   # Cropland/Natural vegetation mosaic
    usgs_to_igbp[7] = 10     # Grasslands
    usgs_to_igbp[8:10] = 7   # Open Shrublands
    usgs_to_igbp[10] = 9     # Savannas
    usgs_to_igbp[11] = 4     # Deciduous Broadleaf Forest
    usgs_to_igbp[12] = 3     # Deciduous Needleleaf Forest
    usgs_to_igbp[13] = 2     # Evergreen Broadleaf Forest
    usgs_to_igbp[14] = 1     # Evergreen Needleleaf Forest
    usgs_to_igbp[15] = 5     # Mixed Forests
    usgs_to_igbp[16] = 17    # Water Bodies (Crucial Conversion)
    usgs_to_igbp[17:19] = 11 # Permanent wetlands
    usgs_to_igbp[19] = 16    # Barren or Sparsely Vegetated
    usgs_to_igbp[20] = 19    # Mixed Tundra
    usgs_to_igbp[21] = 18    # Wooded Tundra
    usgs_to_igbp[22] = 19    # Mixed Tundra
    usgs_to_igbp[23] = 20    # Barren Tundra
    usgs_to_igbp[24] = 15    # Snow and Ice
    usgs_to_igbp[25:28] = 16 # Other specific terrains -> Barren

    COARSE_FACTOR = max(1, int(DX / 500.0))  
    temp_height = np.zeros((NY, NX), dtype='f4')  

    with nc.Dataset(SOURCE_TW_DATA, 'r') as ds:
        raw_h = ds.variables['height'][:]
        raw_lon = ds.variables['lon'][:]
        raw_lat = ds.variables['lat'][:]
        
        m_tw, n_tw = raw_h.shape[0] // COARSE_FACTOR, raw_h.shape[1] // COARSE_FACTOR
        
        # Keep the exact same 0, 0 offset as the original Fortran code
        XS_TW, YS_TW = 0, 0
        ex, ey = min(XS_TW + m_tw, NX), min(YS_TW + n_tw, NY)
        sx, sy = XS_TW, YS_TW
        dx_len, dy_len = ex - sx, ey - sy

        # Process longitude and latitude
        lon_tw = raw_lon[:m_tw * COARSE_FACTOR].reshape(-1, COARSE_FACTOR).mean(axis=1)
        lat_tw = raw_lat[:n_tw * COARSE_FACTOR].reshape(-1, COARSE_FACTOR).mean(axis=1)
        
        lon_1d[sx:ex] = lon_tw[:dx_len]
        lat_1d[sy:ey] = lat_tw[:dy_len]
        
        dum_lon = lon_1d[sx + 1] - lon_1d[sx] if dx_len > 1 else DX / 6.37E6 / (2. * np.pi) * 360.
        for i in range(sx - 1, -1, -1): lon_1d[i] = lon_1d[i + 1] - dum_lon
        for i in range(ex, NX): lon_1d[i] = lon_1d[i - 1] + dum_lon
            
        dum_lat = lat_1d[sy + 1] - lat_1d[sy] if dy_len > 1 else DY / 6.37E6 / (2. * np.pi) * 360.
        for j in range(sy - 1, -1, -1): lat_1d[j] = lat_1d[j + 1] - dum_lat
        for j in range(ey, NY): lat_1d[j] = lat_1d[j - 1] + dum_lat

        # Coarsening and Variable Conversion
        temp_height[sy:ey, sx:ex] = get_mean_2d(raw_h, COARSE_FACTOR)[:dy_len, :dx_len]
        temp_lu = get_mode_2d(ds.variables['lu'][:], COARSE_FACTOR)[:dy_len, :dx_len]
        lu[sy:ey, sx:ex]    = usgs_to_igbp[temp_lu] # Apply USGS -> IGBP conversion
        
        soil[sy:ey, sx:ex]  = get_mode_2d(ds.variables['soil'][:], COARSE_FACTOR)[:dy_len, :dx_len]
        slope[sy:ey, sx:ex] = get_mode_2d(ds.variables['slope'][:], COARSE_FACTOR)[:dy_len, :dx_len]
        gvf[sy:ey, sx:ex]   = get_mean_2d(ds.variables['gvf'][:], COARSE_FACTOR)[:dy_len, :dx_len]
        lai[sy:ey, sx:ex]   = get_mean_2d(ds.variables['lai'][:], COARSE_FACTOR)[:dy_len, :dx_len]
        albedo[sy:ey, sx:ex] = get_mean_2d(ds.variables['albedo'][:], COARSE_FACTOR)[:dy_len, :dx_len]
        shdmax[sy:ey, sx:ex] = get_mean_2d(ds.variables['shdmax'][:], COARSE_FACTOR)[:dy_len, :dx_len]
        shdmin[sy:ey, sx:ex] = get_mean_2d(ds.variables['shdmin'][:], COARSE_FACTOR)[:dy_len, :dx_len]

    # Calculate discrete height grid index (topo)
    for j in range(NY):
        for i in range(NX):
            if temp_height[j, i] > 9000.0:
                temp_height[j, i] = 0.0

            # Distinguish land and sea
            is_land = (lu[j, i] != 17) or (temp_height[j, i] > 0.0)

            if is_land:
                topo[j, i] = HALO # Base offset for land
                if temp_height[j, i] > 0.0:
                    best_k = int(np.argmin(np.abs(z_up - temp_height[j, i])))
                    calculated_topo = best_k + (HALO - 1)
                    topo[j, i] = max(topo[j, i], calculated_topo)
            else:
                topo[j, i] = 0

    # Remove single-grid caves
    for j in range(1, NY - 1):
        for i in range(1, NX - 1):
            c = topo[j, i]
            if (topo[j, i-1] > c and topo[j, i+1] > c and topo[j-1, i] > c and topo[j+1, i] > c):
                topo[j, i] = min(topo[j, i-1], topo[j, i+1], topo[j-1, i], topo[j+1, i])

else:
    print("--- Mode: Idealized Simulation (User-Defined Ridge & Land Types) ---")
    
    # Initialize idealized coordinates
    lon_1d = (np.arange(1, NX + 1) * DX - 0.5 * (DX * NX)) / 6.37E6 / (2. * np.pi) * 360.
    lat_1d = (np.arange(1, NY + 1) * DY - 0.5 * (DY * NY)) / 6.37E6 / (2. * np.pi) * 360.
    
    # Generate idealized fields
    base_topo   = get_ideal_topo_data(NY, NX)
    lu_ideal    = get_ideal_vegtype_data(NY, NX)
    soil_ideal  = get_ideal_soiltype_data(NY, NX)
    slope_ideal = get_ideal_slopetype_data(NY, NX)
    Tg_ideal    = get_ideal_tg_data(NY, NX, value=305.0)

    # Assign generated fields and handle HALO offset
    for j in range(NY):
        for i in range(NX):
            lu[j, i]     = lu_ideal[j, i]
            soil[j, i]   = soil_ideal[j, i]
            slope[j, i]  = slope_ideal[j, i]
            Tg[j, i]     = Tg_ideal[j, i]
            albedo[j, i] = 15.0 # Typical land albedo
            
            is_land = (lu[j, i] != 17) # Check if it is land (not water body)
            
            if is_land:
                # Land base starts at HALO, plus any extra mountain height
                topo[j, i] = base_topo[j, i] + HALO
            else:
                # Sea is always 0
                topo[j, i] = 0

# ==============================================================================
# Shared Post-processing: Auto-derive physical height & generate variables
# ==============================================================================
height = np.zeros((NY, NX), dtype='f4')
mask   = np.ones((NZ, NY, NX), dtype='i1')
sea_land_ice_mask = np.ones((NY, NX), dtype='i4')

for j in range(NY):
    for i in range(NX):
        k_cpp = topo[j, i]
        
        if k_cpp >= HALO:
            k_phys = int(k_cpp - (HALO - 1))
            safe_k = min(k_phys, NZ - 1)
            # Convert back to km for height output
            height[j, i] = z_up[safe_k] / 1000.0
            mask[:safe_k, j, i] = 0

        # Create sea_land_ice_mask based on final 'lu' assignment
        if lu[j, i] == 17:  # IGBP 17 = Water bodies
            sea_land_ice_mask[j, i] = 0

lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

# ==============================================================================
# Output to NetCDF
# ==============================================================================
variables_config = {
    'lon': {
        'data': lon_2d, 'dims': ('ny', 'nx'), 'units': 'degrees_east', 'dtype': 'f4',  
        'long_name': 'Longitude (2D matrix)'
    },
    'lat': {
        'data': lat_2d, 'dims': ('ny', 'nx'), 'units': 'degrees_north', 'dtype': 'f4',  
        'long_name': 'Latitude (2D matrix)'
    },
    'topo': {
        'data': topo.astype('f8'), 'dims': ('ny', 'nx'), 'units': 'grid', 'dtype': 'f8',  
        'long_name': 'Topography index (Vertical level index of the surface)'
    },
    'mask': {
        'data': mask, 'dims': ('nz', 'ny', 'nx'), 'units': 'T/F', 'dtype': 'i1',  
        'long_name': '3D atmospheric mask (0 = terrain inside, 1 = free atmosphere)'
    },
    'height': {
        'data': height, 'dims': ('ny', 'nx'), 'units': 'km', 'dtype': 'f4',  
        'long_name': 'Physical terrain height above sea level'
    },
    'sea_land_ice_mask': {
        'data': sea_land_ice_mask, 'dims': ('ny', 'nx'), 'units': 'flag', 'dtype': 'i4',  
        'long_name': 'Sea/Land/Ice mask (0 = water bodies, 1 = land or ice)'
    },
    'vegtype': {
        'data': lu, 'dims': ('ny', 'nx'), 'units': 'category', 'dtype': 'i4',  
        'long_name': 'Vegetation / Land-use category (IGBP / MODIS NOAH 20-category)'
    },
    'soiltype': {
        'data': soil, 'dims': ('ny', 'nx'), 'units': 'category', 'dtype': 'i4',  
        'long_name': 'Soil texture category (STATSGO 19-category)'
    },
    'slopetype': {
        'data': slope, 'dims': ('ny', 'nx'), 'units': 'category', 'dtype': 'i4',  
        'long_name': 'Slope category'
    },
    'Tg': {
        'data': Tg, 'dims': ('ny', 'nx'), 'units': 'K', 'dtype': 'f8',  
        'long_name': 'Deep soil temperature / Ground temperature'
    },
    'albedo': {
        'data': albedo, 'dims': ('ny', 'nx'), 'units': '%', 'dtype': 'f4',  
        'long_name': 'Surface background snow-free albedo'
    },
    'gvf': {
        'data': gvf, 'dims': ('ny', 'nx'), 'units': '%', 'dtype': 'f4',  
        'long_name': 'Green vegetation fraction'
    },
    'lai': {
        'data': lai, 'dims': ('ny', 'nx'), 'units': 'm2/m2', 'dtype': 'f4',  
        'long_name': 'Leaf area index'
    },
    'shdmax': {
        'data': shdmax, 'dims': ('ny', 'nx'), 'units': '%', 'dtype': 'f4',  
        'long_name': 'Maximum areal fractional coverage of green vegetation'
    },
    'shdmin': {
        'data': shdmin, 'dims': ('ny', 'nx'), 'units': '%', 'dtype': 'f4',  
        'long_name': 'Minimum areal fractional coverage of green vegetation'
    }
}

with nc.Dataset(FILENAME, 'w', format='NETCDF4') as ds:
    print(f"\nWriting Initialization Data to: {FILENAME} ...")
    
    ds.createDimension('nx', NX)
    ds.createDimension('ny', NY)
    ds.createDimension('nz', NZ)

    for var_name, info in variables_config.items():
        var = ds.createVariable(var_name, info['dtype'], info['dims'])
        
        if 'units' in info:
            var.units = info['units']
            
        if 'long_name' in info:
            var.long_name = info['long_name']
            
        var[:] = info['data']

print("Initialization file generated successfully with detailed variable names!")


# ==============================================================================
# Noah LSM Vegetation Types (vegtype)
# ==============================================================================
# --- IGBP / MODIS Classification (20 types) [Used when ivegsrc = 1] ---
#  1: Evergreen Needleleaf Forest
#  2: Evergreen Broadleaf Forest
#  3: Deciduous Needleleaf Forest
#  4: Deciduous Broadleaf Forest
#  5: Mixed Forests
#  6: Closed Shrublands
#  7: Open Shrublands
#  8: Woody Savannas
#  9: Savannas
# 10: Grasslands
# 11: Permanent Wetlands
# 12: Croplands
# 13: Urban and Built-Up
# 14: Cropland/Natural Vegetation Mosaic
# 15: Snow and Ice
# 16: Barren or Sparsely Vegetated
# 17: Water
# 18: Wooded Tundra
# 19: Mixed Tundra
# 20: Barren Tundra

# ==============================================================================
# Noah LSM Soil Types (soiltyp)
# ==============================================================================
# --- STATSGO Classification (19 types) [Used when isot = 1] ---
#  1: Sand
#  2: Loamy Sand
#  3: Sandy Loam
#  4: Silt Loam
#  5: Silt
#  6: Loam
#  7: Sandy Clay Loam
#  8: Silty Clay Loam
#  9: Clay Loam
# 10: Sandy Clay
# 11: Silty Clay
# 12: Clay
# 13: Organic Material
# 14: Water
# 15: Bedrock
# 16: Other (Land-Ice)
# 17: Playa
# 18: Lava
# 19: White Sand

# ==============================================================================
# Noah LSM Slope Types (slopetyp)
# ==============================================================================
#  1: 0 - 8%      (Flat)
#  2: 8 - 16%     (Gently Sloping)
#  3: 16 - 30%    (Sloping)
#  4: 30 - 45%    (Steep)
#  5: 45 - 60%    (Very Steep)
#  6: 60 - 90%  
#  7: 90 - 120%
#  8: 120 - 150%
#  9: > 150%      (Cliff)
