# Technical Report

Large scale cropyield estimation from satellite mosaics produced by the Finnish Environment Institute and the Finnish Meteorological Institute for Finnish Geospatial Platform.

## Remote sensing data

Name:   Sentinel-2 image index mosaics (S2ind)

Format: Geotiff

Indeces: Normalized Difference Vegetation Index (NDVI), Tillage Index (NDTI), and NDMI (Moisture index).

NDVI = (B8-B4)/(B8+B4)
NDTI = (B11-B12)/(B11+B12)
NDMI = (B8-B11)/(B8+B11)

Available in a data repository for users at the governmental CSC IT Centre for Science.

Full meta data: http://metatieto.ymparisto.fi:8080/geoportal/catalog/search/resource/details.page?uuid=%7B33E3D2A0-AC5B-443B-BFC7-F8A4C2A08A0D%7D

## Vector data

For observing arable land, we use IACS LPIS data set that contains geometries of parcels. As our reference data is aggregated to farm level, we need to combine our parcels to farm level multipolygons representing the land under certain crop production. 

## Reference data

As a reference data, we will have historical crop yields per hectare on farm-level from around 6,000 Finnish agricultural holdings participating the annual Finnish crop production survey. Note that this is condidential information and available only for statistical production purposes with disclosure agreement. The crop species included in this study are winter wheat, spring wheat, oat, malting barley, feed barley, and autumn rye.

## Processing steps

### From Mosaics to Analysis Ready Data (ARD)

Prerequisites: For machine learning classification, the vector data set should be diveded into training and test set. Vector data should be in ESRI shapefile format. Shapefile should include farm ID as variable 'farmID'.

#### makeS2IndexHisto

First we run Python program makeS2IndexHisto.py. Inputs are 1) a path to a directory where the Sentinel-2 indeces are as 30-day mosaics  (argument --in_s2_tif); 2) a path to the vector data (shapefile, argument --in_aoi_shapefile); 3) a path to directory for saving the results (argument --output_dir).

NOTE: shapefile naming should follow this convention: [something]-[year]-[dataset ID].shp

NOTE: Sentinel files should follow this convention: pta_sjp_s2ind_ndvi_20200801_20200831.tif

RUN:

python makeS2IndexHisto.py -s /Users/myliheik/Documents/GISdata/sentinel2/fromPTA/AOI2/mosaics/ \
    -a /Users/myliheik/Documents/GISdata/satotutkimus/shpfiles/satotilat-2019-train1110.shp \
    -o /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/ \
    --extractIntensities --calculatePopulations --make_histograms

WHERE: 

1. argument extractIntensities runs the function to extract all pixel values of parcels from the mosaic of a given year.
1. argument calculatePopulations runs the function to calculate population percentiles from all the pixel values. This is needed to set the range for calculating histograms. Optionally, argument show-plots shows population distributions as plots on current display device.
1. argument make_histograms runs the function to calculate normalized histograms from pixels values of (multi)polygons.

Arguments should be run in this order.

#### makeARD

Next, we modify histograms into 2D ARD format which is suitable for machine learner. 

Inputs are 1) a path to the directory where histogram results are saved from makeS2IndexHisto.py. Output of this function will use the same path to save the results there; 2) a path to the directory where target values (crop yields per farm) are. Target values should be saved in a comma separated values (.csv) file. The first row should indicate the variables. Farm ID is mandatory ('farmID') and a variable ending with 'ha' indicating the yield.; 3) a path to the directory where meteorological data is stored (--meteo_dir). Meteorological data should be in Python pickle format. It can include any variables, but 'farmID' is mandatory.

RUN: 

python makeARD.py -i /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/test1120 -t /Users/myliheik/Documents/myCROPYIELD/data -l -s -n

WHERE: 

1. argument input_dir (-i) takes the path to the directory where histogram results are stored.
1. argument target_dir (-t) takes the path to the directory where target values (crop yields per farm) are stored.
1. argument meteo_dir (-k) takes the path to the directory where meteorological data is stored.
1. argument make_2D (-l) reshapes histograms into 2D analysis ready format.
1. argument mergeTarget (-s) merges ARD with target values (crop yields). 
1. argument make_2Dmeteo (-n) merges ARD and target values with meteorological data.


#### runClassifier

Lastly, we fit the data to a classifier, in this case random forest.

RUN:

runClassifier.py -i /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/train1400 \
-t /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/test1400 -l 30

Argument -m to use also meteorological features:

runClassifier.py -i /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/train1400 \
-t /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/test1400 -l 30 -n -m


WHERE:

1. argument train_dir (-i) takes the path to the directory where the train set ARD file are stored.
1. argument test_dir (-t) takes the path to the directory where the test set ARD file are stored.
1. argument importance (-l) sets the number of the most important features to show, default 20.
1. argument savePreds (-n) takes the path to a directory where to save the test set predictions,optionally.
1. argument use_2Dmeteo (-m) if to use also meteorological features.

The function prints statistics (R2, RMSE, MSE) and list of important features as an output. Test set predictions can be optionally saved into the test_dir.

