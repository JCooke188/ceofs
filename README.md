Created by Justin Cooke, 2026

These python notebooks are for use with the manuscript titled: ''Insights into Deep Mesoscale Eddy Variability in the Gulf of Mexico'' by Justin P. Cooke, Kathleen A. Donohue, and D. Randolph Watts

The goal of our study was to improve our understanding on the variability of deep (>1000 m) mesoscale eddies in the Eastern Gulf of Mexico. 

**Files and Descriptions**

hycom.ipynb
- Use this file to grab the temperature, salinity, and ssh files from the repository on hycom.org.
- Also fixes an issue with ssh when downloaded, where extra time values were included. 
- The data are taken for the first 18 years of the 54 year HYCOM nature-run, in the Eastern Gulf (defined as 22N to 28N and 83W to 90W), for all available depths, and saved in netcdf files.

get_eta_ref.ipynb
- Use this file to generate eta ref (deep streamfunction) for the 18 year HYCOM nature-run, using the temperature, salinity, and ssh netcdf files downloaded with hycom.ipynb
- Eta ref is calculated in three time chunks for reducing computational load, operations are handled using xarray and dask for parallelization.

ceof_calc.ipynb
- Main notebook used to generate the data/figures in the study. 

cpies_variance.ipynb
- Finds variance of eta ref in the different frequency bands using eta ref observed by CPIES

supp_variance_testing.ipynb
- Notebook used to understand why deep variance is higher in the model versus observations 

gem_calc.ipynb
- Notebook to investigate stratification of model versus observatations (CTD casts) in the Gulf to understand why deep variance is higher in the model versus observations 

