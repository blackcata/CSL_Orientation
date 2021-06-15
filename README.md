# CSL_Orientation
Basic Python Code for calculating statistics and drawing plots. To start the climate researches, the abilities 
to calculate several statistics, such as linear regression, correlation coefficient, composite and bootstrap, 
and draw various types of plots, such as time series, scatter plot, contour map, is needed for CSL students. 
In this code, the basic functions for calculating statistics and examples for drawing basic plots are provided with python and jupyter code. 
If a graduate freshman could draw these plots on themselves, the students are ready to start the climate research. 


### 1. Orientation Purpose
  - Calculating Climate Statistics (xarray, scipy, sklearn MODULES)
    - Read and write with the netCDF4 format data
    - Extract the index from given target area 
    - Understand and calculate climatology and anomaly
    - Remove linear trend from each dataset
    - Calculate linear-regression, correlation coefficients, composite.
    - Calculate significance by using bootstrap method
    
  - Drawing Various Plots (matplotlib.pyplot, basemap MODULES)
    - Draw x,y time-series plot 
    - Draw contour-map 
    - Draw subplots 
    - Draw scatter plots
    - Save figures
  
### 2. Provided Data 
Monthly SST (Sea Surface Temperature) Reanalysis data from the ERSST reanalysis was used for calculating climate statistics.
Additionally, the monthly South-Korea station data was used to estimate the ENSO impacts on the Korea Penisulla.
You can download these reanalysis and station data from below URL.
  - URL : https://www.dropbox.com/sh/nbt8zer5r0yex92/AAAVw-MYUggHOBwCWbzUGKrWa?dl=0


### 3. Code composition
  - CSL_base.py [basic python functions for calculating climate statistics]
  - Orientation_Nino34.ipynb [Jupyter codes for drawing plots]
  
Before start the codes, you need to download the dataset to DATA folder, which is located same path on python and jupyter code.

### 4. Ouput files
  - Time Series : TS_Nino34.pdf
  - Regression Map : Map_Reg_Nino34_SST.pdf
  - Correlation Map : Map_Corr_Nino34_SST.pdf
  - Composite Map : Map_Comp_Nino34_SST.pdf
  - Scatter Map : Map_Scat_Nino34_t2m_prcp_Korea.pdf
  - Scatter Plot : Scat_mean_Nino34_t2m_prcp.pdf
 
All output pdf files will be saved on the RESULT directory.
  
