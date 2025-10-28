#!/usr/bin/env python
# coding: utf-8

# Created by Justin Cooke -- October 2025
# 
# The purpose of this script is to analyze the first 18-yr cycle of the GOM HYCOM Nature Run (see Dukhovskoy et al. 2015 in Deep-Sea Research I)
# 
# This notebook will go as follows:
#     Load HYCOM data from ./hycom_data/*.nc
#     Calculate eta ref to find deep mesoscale eddies for the Eastern half of the Gulf
#     Calculate LC Length and Northern Ext over the 18 year period
#     Conduct a CEOF analysis on the deep eddy dataset
#     Identify prominent modes and when they peak

# In[ ]:


# Import modules

# Sci computing
import numpy as np
import scipy as sp
import seawater as sw
import scipy.sparse.linalg as sla

# Parallel comupting
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

# For Data
import netCDF4 as nc
import xarray as xr

# Plotting stuff
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grdspc
import cmocean as cm

# Gen stuff
from datetime import date
today = date.today()


# In[ ]:


# Load using XArray the lat lon and depth data first

ds_latlon = xr.open_dataset("./hycom_data/hycom_latlon.nc")
ds_depth = xr.open_dataset("./hycom_data/hycom_depth.nc")

# Now get the lat, lon, and depth for 22N to 28N and -90W to -83W and depth down to 2000m
ds_lon = ds_latlon['Longitude'][:]
ds_lat = ds_latlon['Latitude'][:]
ds_z = ds_depth['Depth'][:] 

# finding the index in lon array that corresponds to 90W
ind90 = list(np.where(ds_lon >= -90)) 
ind83 = list(np.where(ds_lon >= -83))
nlon90 = ind90[0][0]
nlon83 = ind83[0][0] + 1
lon = ds_lon[nlon90:nlon83]

# finding the indices in lat array that correspond to 22W and 28W to only grab data from this region
ind22 = list(np.where(ds_lat >= 22))
nlat22 = ind22[0][0] 
ind28 = list(np.where(ds_lat >= 28))
nlat28 = ind28[0][0] + 1
lat = ds_lat[nlat22:nlat28]

ind2k = list(np.where(ds_z >= 2000))
n2k = ind2k[0][0] + 1
depth = ds_z[:n2k]


# In[ ]:


# Next, we can load the potential temperature and salinity

ds_theta = xr.open_dataset("./hycom_data/hycom_temp.nc", chunks={'MT': 540, 'Depth': 13, 'Latitude': 83, 'Longitude': 88})
ds_sal = xr.open_dataset("./hycom_data/hycom_sal.nc", chunks={'MT': 540, 'Depth': 13, 'Latitude': 83, 'Longitude': 88})
ds_ssh = xr.open_dataset("./hycom_data/hycom_ssh.nc", chunks={'MT': 540, 'Latitude': 83, 'Longitude': 88})


# In[ ]:


# Now, we will mask all points that do not reach 2000 meters depth

# Lazily load potential temp and salinity
theta = ds_theta['temperature']
sal = ds_sal['salinity']

# Assign coordinates to the lat, lon, and depth dimensions
theta['Longitude'] = lon
theta['Latitude'] = lat
theta['Depth'] = depth
theta['MT'] = ds_theta['MT']

sal['Longitude'] = lon
sal['Latitude'] = lat
sal['Depth'] = depth
sal['MT'] = ds_sal['MT']


# Find the points that have valid points at each depth over all lat lon
depth_mask = theta.notnull().any(dim='MT')

# Here we are multiplying our boolean by depth and taking this at the last depth level (the maximum)
deepest_valid_depth = (depth_mask * depth).max(dim='Depth')

# This is our mask for points that reach at least 2000m
has_2000m = deepest_valid_depth >= 2000

# Now we are applying our mask
masked_theta = theta.where(has_2000m)
masked_sal = sal.where(has_2000m)

# We are storing the masked potential temperature 
ds_theta['masked_theta'] = masked_theta
ds_sal['masked_salinity'] = masked_sal


# In[ ]:


# Example to downsample and look at a single time instance and depth instance (surface)
# This selects the 2d field at time 0 and the surface (depth 0)
#slice_2d = ds_theta['masked_theta'].isel(MT=0,Depth=0)

# Now this loads in the 2d slice from above
#myfield = slice_2d.compute()


# In[ ]:


# Now we need to convert depth to pressure

# First we need to define a function dpth which is converts pressure to depth in m

def dpth(pres_dbar,lat_deg):
    x = np.sin(np.radians(lat_deg))**2
    g = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * x) * x)

    depth_in_m = (((-1.82e-15 * pres_dbar + 2.279e-10) * pres_dbar - 2.2512e-5) * pres_dbar + 9.72659) * pres_dbar / g

    return depth_in_m

def prs(depth_meters, latitude_deg, tol=0.001):
    # Iteratively compute pressure [dbar] from depth [m] and latitude [deg] 

    # Parameters
    # depth_meters : float or np.ndarray
        # Depth in meters
    # latitude_deg : float or np.ndarray
        # Latitude degrees north (-90 to 90)
    # tol          : float, optional

    # Returns
    # pressure : np.ndarray 
        # Pressure in decibar (dbar)
    # iterations : int
        # Number of iterations used to converge

    # Convert inputs to arrays

    depth_meters = np.atleast_1d(depth_meters).astype(float)
    latitude_deg = np.atleast_1d(latitude_deg).astype(float)

    # Broadcast latitude to depth shape if needed
    if latitude_deg.size == 1:
        latitude_deg = np.full_like(depth_meters,latitude_deg)
    elif latitude_deg.shape != depth_meters.shape:
        if latitude_deg.shape[0] == depth_meters.shape[1]:
            latitude_deg = np.tile(latitude_deg, (depth_meters.shape[0],1))
        else: 
            raise ValueError("Latitude and Depth must have compatible dimensions")

    # Initialization
    pressure = 1.01 * depth_meters
    converged = False
    max_iters = 20
    iters = 1

    while not converged and iters <= max_iters:
        d = dpth(pressure,latitude_deg)
        new_pressure = pressure + (depth_meters - d) * 1.01
        delta = np.abs(new_pressure - pressure)

        if np.max(delta) < tol:
            converged = True

        pressure = new_pressure
        iters += 1

    if not converged:
        pressure[:] = np.nan

    return pressure.squeeze()


# In[ ]:


# Converting depth to pressure (dbar)

# Need to create an mxn array where m = [depth] and n = [latitude] to be used in prs kinda like meshgrid
depth_2d, lat_2d = xr.broadcast(depth,lat)

# make depth and lat lazy
depth_2d = depth_2d.chunk({'Depth': -1, 'Latitude': 50})
lat_2d = lat_2d.chunk({'Depth': -1, 'Latitude': 50})

# This wrapper (apply_ufunc) allows converting depth to pressure
# pressure is the name of our output variable
pressure = xr.apply_ufunc( 
    prs, # the actual function we are calling
    depth_2d, # input one which is our mxn depth array
    lat_2d, # input two which is our mxn latitude array
    input_core_dims=[['Depth', 'Latitude'], ['Depth', 'Latitude']], # input and output explicitly tells xarray both inputs and outputs share the same 2d struct
    output_core_dims=[['Depth', 'Latitude']],
    dask='parallelized', # executes lazily with Dask
    vectorize=True, # ensure pres is applied elementwise across the 2D arrays
    output_dtypes=[float], # ensure consistent output type
    dask_gufunc_kwargs={'allow_rechunk': True},
)

# Zero out top row
pressure[0,:] = 0.0

# Want to create repeating matrices of pressure
pressure_full = pressure.expand_dims({
    'Longitude': ds_theta['Longitude'],
    'MT': ds_theta['MT']
}).transpose('MT', 'Depth', 'Latitude', 'Longitude')

pressure_full_masked = pressure_full.where(has_2000m)

# Ref pressure now
pref = xr.zeros_like(pressure_full)
pref_masked = pref.where(has_2000m)

pressure_full_masked = pressure_full.chunk({'MT': 540, 'Depth': 13, 'Latitude': 83, 'Longitude': 88})
pref_masked = pref_masked.chunk({'MT': 540, 'Depth': 13, 'Latitude': 83, 'Longitude': 88})


# In[ ]:


# Now we can handle converting potential temp to temp

this_theta = ds_theta['masked_theta']
this_sal = ds_sal['masked_salinity']

temperature = xr.apply_ufunc(
    sw.eos80.temp,
    this_sal,
    this_theta,
    pressure_full_masked,
    pref_masked,
    input_core_dims=[['MT','Depth','Latitude','Longitude']]*4,
    output_core_dims=[['MT','Depth','Latitude','Longitude']],
    dask='parallelized',
    vectorize=True,
    output_dtypes=[float],
    dask_gufunc_kwargs={'allow_rechunk': True}
)

temp_masked = temperature.where(has_2000m)
temp_masked = temp_masked.chunk({'MT': 540, 'Depth': 13, 'Latitude': 83, 'Longitude': 88})
ds_theta['actual_temp_masked'] = temp_masked


# In[ ]:


# Next will be to calculate specific volume anomaly, will have to follow the above I'm sure 
# Can create constant single value xarrays for s35 and T0 like:

s35 = xr.full_like(this_sal, fill_value=35.0)  # e.g., reference salinity
T0  = xr.zeros_like(this_theta)  # e.g., reference temp

s35_masked = s35.where(has_2000m)
T0_masked = T0.where(has_2000m)

s35_masked = s35_masked.chunk({'MT': 540, 'Depth': 13, 'Latitude': 83, 'Longitude': 88})
T0_masked  = T0_masked.chunk({'MT': 540, 'Depth': 13, 'Latitude': 83, 'Longitude': 88})


# In[ ]:


# Find density using standard salinity and temperature
dens_350p = xr.apply_ufunc(
    sw.eos80.dens,
    s35_masked,
    T0_masked,
    pressure_full_masked,
    input_core_dims=[['MT','Depth','Latitude','Longitude']]*3,
    output_core_dims=[['MT','Depth','Latitude','Longitude']],
    dask='parallelized',
    vectorize=True,
    output_dtypes=[float],
    dask_gufunc_kwargs={'allow_rechunk': True}
)

# Chunk it
dens_350p = dens_350p.chunk({'MT': 540, 'Depth': 13, 'Latitude': 83, 'Longitude': 88})

this_temp = ds_theta['actual_temp_masked']
this_sal = ds_sal['masked_salinity']

# Now find density using salinity and temp
dens_stp = xr.apply_ufunc(
    sw.eos80.dens,
    this_sal,
    this_temp,
    pressure_full_masked,
    input_core_dims=[['MT','Depth','Latitude','Longitude']]*3,
    output_core_dims=[['MT','Depth','Latitude','Longitude']],
    dask='parallelized',
    vectorize=True,
    output_dtypes=[float],
    dask_gufunc_kwargs={'allow_rechunk': True}
)

# Chunk it
dens_stp = dens_stp.chunk({'MT': 540, 'Depth': 13, 'Latitude': 83, 'Longitude': 88})


# In[ ]:


# Now calculate specific volume anomaly

sv_350p = 1/dens_350p
sv_stp  = 1/dens_stp

svan = sv_stp - sv_350p


# In[ ]:


# Now the tricky part, integrate over the water column to find geopotential anomaly

# First, convert from dbar to Pascals
p_PA = pressure * 1e4

g_real = 9.81

# Now we need to do the integration for each latitude slice
p_PA_brdcst = p_PA.broadcast_like(svan)

gpan = xr.apply_ufunc(
    np.trapezoid,
    svan,
    p_PA_brdcst,
    input_core_dims=[['Depth']]*2,
    output_core_dims=[[]],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[float],
    dask_gufunc_kwargs={'allow_rechunk': True}
)


# In[ ]:


# Calculate eta steric and eta ref

# Eta Steric first
eta_steric = gpan/g_real
eta_steric_masked = eta_steric.where(has_2000m)

# Now Eta Ref
this_ssh = ds_ssh['ssh']
this_ssh_masked = this_ssh.where(has_2000m)
eta_ref = this_ssh - eta_steric_masked

# Remove common mode
eta_ref = eta_ref - xr.DataArray.mean(eta_ref,dim=["Latitude","Longitude"],skipna=True)


# In[ ]:


slice2d = eta_ref.isel(MT=0)
with ProgressBar():
    this_eta_field = slice2d.compute()


# In[ ]:


plt.contourf(lon,lat,this_eta_field,cmap=cm.cm.balance)
plt.savefig('./test_fig.png',format='png')


# In[ ]:


# Write eta ref to netcdf file so we don't need to repeat all this b.s.
#eta_ref.to_netcdf("./hycom_data/hycom_etaref.nc")

