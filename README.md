# BGC-obs-trends
Scripts to calculate ocean biogeochemical trends based on oceanic observations

This repository contains scripts that were used to produce the results presented in manuscript Jutras et al., (in prep), some required data, and processed results.

List of scripts:

* **gridded_trends_from_obs.py** : Running this script requires downloading the BGC-Argo, GLODAP and WOD datasets. This script can:
   1) Read different observational datasets (for now Argo data, GLODAP, and WOD)
   2) Put this data on a lat-lon-neutral density grid
   3) Calculate long-term trends within each grid cell using a linear fit, testing for the statistical validity of the trend

* **utils_gridding.py** : This script contains functions used in gridded_trends_from_obs.py.

List of input files used in the scripts:

* **Maximum_seasonal_MLD.nc** : This file contains the maximum seasonal mixed layer depth over 20 years, calculated from the RG-gridded Argo product (Roemmich & Gilson, 2009). 
It serves to remove the data located within the mixed layer from the analysis.

* **Climatologies_gamma_2.5.nc** : This file contains climatologies of depth, nitrate, oxygen, DIC, pH, temperature and salinity on a 2.5 degree grid and on 0.1 kg/m³ neutral density layers.
Nitrate, oxygen, temperature and salinity climatologies are interpolated from the World Ocean Atlas climatologies (Garcia et al., 2024). The DIC climatology is obtained from the NCEI Mapped Observation-Based Oceanic (MOBO) DIC product (Keppler et al., 2020). The pH climatology is obtained from the GLODAPv2 dataset (Olsen et al., 2016). The data was interpolated on a density grid using a climatology of the RG-Argo depth on neutral density layers.
This dataset serves for reference values from which anomalies are computed.

List of result files:

* **nitrate_trends_on_grid.nc** : This file contains the nitrate trends on a 2.5 x 2.5 degree grid and 0.1 kg/m³ density bins, based on observations.

* **nitrate_trends_per_region.nc** : This file contains the time series of nitrate anomalies on a 2.5 x 2.5 degree grid and for groups of density layers.

