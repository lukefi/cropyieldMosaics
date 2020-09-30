#!/usr/bin/env python
# coding: utf-8

"""
2020-09-16 MY
env: myGIS

From Sentinel-2 mosaics to analysis ready data with rasterstats. Indeces 'NDMI', 'NDTI' and 'NDVI'.

INPUTS:
/Users/myliheik/Documents/GISdata/satotutkimus/shpfiles
/Users/myliheik/Documents/GISdata/sentinel2/fromPTA/AOI2/mosaics/
binsize

OUTPUT:
train set range files + plots of distributions
histograms per parcel, .pkl

- ensin vivulla --extractIntensities saadaan intensiteettitiedostot, jokaiselta vuodelta ('fullArrayIntensities' + vuosi + '.pkl')
	- tallentaa myöskaikkien pikselien arvot ( populationNDMIIntensities*.pkl)
- sitten vielä aja lopuksi vivulla --calculatePopulations, saadaan koko train populaation jakaum('populationStats.pkl')
	- ja kuvia 
- kaiken tämän jälkeen aja vivulla --make_histograms, joka tekee histogrammit per par (histograms.pkl)

Next, run makeAR.py (2D or 3D).	

RUN:

python makeS2IndexHisto.py -s /Users/myliheik/Documents/GISdata/sentinel2/fromPTA/AOI2/mosaics/ \
    -a /Users/myliheik/Documents/GISdata/satotutkimus/shpfiles/satotilat-2019-train1110.shp \
    -o /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/ \
    --extractIntensities --calculatePopulations --make_histograms

PUHTI:
python makeS2IndexHisto.py -s /appl/data/geo/sentinel/s2/ \
    -a /scratch/project_2001253/cropyieldMosaics/shpfiles/satotilat-2019-train1110.shp \
    -o /scratch/project_2001253/cropyieldMosaics \
    --extractIntensities

"""
import argparse
import os
import pathlib
import pickle
import textwrap
import pandas as pd

import glob
import os.path
import pandas as pd
import pickle

from pathlib import Path

import argparse
import textwrap

import rasterio as rio
from rasterstats import zonal_stats
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns



# PARAMETERS:
sns.set_style('darkgrid')
nrbins = 16

# FUNCTIONS:

def plot_histogram(values, title:str, show=False, save_in=None):
    sns.distplot(values, axlabel = "Pixel counts per parcel")
    plt.title(title)

    if show:
        plt.show()

    if save_in:
        print(f'Saving plot as {save_in} ...')
        plt.savefig(save_in)
    plt.clf()

def calculateRange(allvalues, feature):                    
    # Calculate range of population values:
    # We need to know the distribution per feature to get fixed range for histogram:

    perc = [0,1,2,5,10,20,25,40,50,60,80,90,95,98,99,100]
    
    # removing zeros:
    
    allstripped = np.concatenate(allvalues)
    allstripped2 = allstripped[allstripped > 0]
    
    # Pickle in dict:
    stats = {'feature': feature, 'min': allstripped2.min(), 'max': allstripped2.max(),
             'n_pixels': len(allstripped), 
             'n_nonzero_pixels': len(allstripped2),
             'percentiles': perc,
             'intensities': np.percentile(allstripped2, perc)}
    
    return stats

def save_intensities(filename, arrayvalues):
    with open(filename, 'wb+') as outputfile:
        pickle.dump(arrayvalues, outputfile)
        
        

def extractarray(shpfile, datadir, outputdir, savePopulations=False):
    # List all files:
    list_of_files = glob.glob(os.path.join(datadir, '*.tif'))
    
    myarrays = []
    populationndmi = []
    populationndti = []
    populationndvi = []
    
    setti = shpfile.replace(".shp", "").split('-')[-1]
    vuosi = shpfile.replace(".shp", "").split('-')[-2]

    output = os.path.join(outputdir,setti)
    Path(output).mkdir(parents=True, exist_ok=True)

    # Loop mosaic files:
    for filename in list_of_files:
        #print('Sentinel file: ' + filename.split('/')[-1])
        filename_parts = filename.replace(".tif", "").split('_')
        
        #date = "".join(filename_parts[2:3])        
        #year = int("".join(list(date)[:4]))        
        #feature = "".join(filename_parts[1:2])
        # PUHTI:
        date0 = "".join(filename_parts[4:5])        
        year = int("".join(list(date0)[:4])) 
        date = "".join(list(date0)[4:])
        feature = "".join(filename_parts[3:4])

        if year == int(vuosi) and feature in ['ndmi', 'ndti', 'ndvi']:
            print('\nProcessing Sentinel file: ' + filename.split('/')[-1])
            print('Feature: ' + "".join(filename_parts[3:4]))
            print('Mosaic year: ' + "".join(list(date0)[:4]))
            print('Mosaic starting date: ' + "".join(filename_parts[4:5]))
            
            # Read all pixels within a parcel:
            parcels = zonal_stats(shpfile, filename, stats="count", geojson_out=True, all_touched = False, 
                                  raster_out = True, nodata=np.nan)
            print("Length : %d" % len (parcels))


            for x in parcels:
                myarray = x['properties']['mini_raster_array'].compressed()

                myid = [x['properties']['farmID'], date, feature]
                arr = myarray.tolist()
                myid.extend(arr)
                arr = myid

                if feature == "ndmi":
                    populationndmi.append(myarray.tolist())
                if feature == "ndti":
                    populationndti.append(myarray.tolist())
                if feature == "ndvi":
                    populationndvi.append(myarray.tolist())

                myarrays.append(arr)


    
    filename2 = os.path.join(output, 'fullArrayIntensities' + vuosi + '.pkl')
    
    # Save all pixel values: 
    print('Saving all pixels values into ' + filename2 + '...')
    save_intensities(filename2, myarrays)
    
    if savePopulations:
        
        filename4 = os.path.join(output, 'populationNDMIIntensities_' + vuosi + '.pkl')
        save_intensities(filename4, np.concatenate(populationndmi))
        filename4 = os.path.join(output, 'populationNDTIIntensities_' + vuosi + '.pkl')
        save_intensities(filename4, np.concatenate(populationndti))
        filename4 = os.path.join(output, 'populationNDVIIntensities_' + vuosi + '.pkl')
        save_intensities(filename4, np.concatenate(populationndvi))
    



def calculatePopulations(outputdirsetti, plotPopulations=False, show=False):

    populationndmi = []
    populationndti = []
    populationndvi = []
    
    # Find all population intensities files in resultsdir:
    ndmifiles = glob.glob(os.path.join(outputdirsetti, 'populationNDMIIntensities*.pkl'))
    ndtifiles = glob.glob(os.path.join(outputdirsetti, 'populationNDTIIntensities*.pkl'))
    ndvifiles = glob.glob(os.path.join(outputdirsetti, 'populationNDVIIntensities*.pkl'))
    
    for filename in ndmifiles:
        with open(filename, "rb") as f:
            tmp = pickle.load(f)
            populationndmi.append(tmp)
            
    for filename in ndtifiles:
        with open(filename, "rb") as f:
            tmp = pickle.load(f)
            populationndti.append(tmp)
            
    for filename in ndvifiles:
        with open(filename, "rb") as f:
            tmp = pickle.load(f)
            populationndvi.append(tmp)

    
    populationStats = [calculateRange(populationndmi, 'ndmi'),
                       calculateRange(populationndti, 'ndti'),
                       calculateRange(populationndvi, 'ndvi')]

    filename3 = os.path.join(outputdirsetti, 'populationStats.pkl')

    print('Saving population statistics into ' + filename3 + '...')
    save_intensities(filename3, populationStats)
    
    if plotPopulations:
        print('Distibution of NDMI population:')
        otsikko = 'Distribution of Feature NDMI'
        plot_histogram(np.concatenate(populationndmi), otsikko, show=show, save_in=os.path.join(outputdirsetti, 'population_distribution_NDMI.png'))
        print('Distibution of NDTI population:')
        otsikko = 'Distribution of Feature NDTI'
        plot_histogram(np.concatenate(populationndti), otsikko, show=show, save_in=os.path.join(outputdirsetti, 'population_distribution_NDTI.png'))
        print('Distibution of NDVI population:')
        otsikko = 'Distribution of Feature NDVI'
        plot_histogram(np.concatenate(populationndvi), otsikko, show=show, save_in=os.path.join(outputdirsetti, 'population_distribution_NDVI.png'))


def decideBinSeq(outputdirsetti):
    # Read population statistics:
    filename = os.path.join(outputdirsetti, 'populationStats.pkl')
    
    with open(filename, "rb") as f:
        tmp = pickle.load(f)
    
    for i in range(3):
        print(f'\nFeature:  {list(tmp[i].values())[0]}')
        print(f'Percentiles:  {list(tmp[i].values())[5]}') 
        print(f'Values:  {list(tmp[i].values())[6]}')
    
    print('We choose percentiles 1% and 100% for range.')
    
    ndmirange = [list(tmp[0].values())[6][1], list(tmp[0].values())[6][15]]
    ndtirange = [list(tmp[1].values())[6][1], list(tmp[1].values())[6][15]]
    ndvirange = [list(tmp[2].values())[6][1], list(tmp[2].values())[6][15]]
    
    return ndmirange, ndtirange, ndvirange

def makeHisto(outputdirsetti, bins):
    print('\nDecide bin sequences for range...')
    ndmirange, ndtirange, ndvirange = decideBinSeq(outputdirsetti)
      
    histlist = []
    nrbins = bins

    # Find all population intensities files in resultsdir:
    rawfiles = glob.glob(os.path.join(outputdirsetti, 'fullArrayIntensities20*.pkl'))
    print('\nCalculate histograms...')
    for filename in rawfiles:
        with open(filename, "rb") as f:
            tmp = pickle.load(f)
    
            for elem in tmp:
                myid = elem[0:3]
                line = elem[3:]
                inx = elem[2]

                if inx == 'ndmi':
                    maximum = ndmirange[1]
                    minimum = ndmirange[0]
                    bin_seq = np.linspace(minimum,maximum,nrbins+1)

                if inx == 'ndti':
                    maximum = ndtirange[1]
                    minimum = ndtirange[0]
                    bin_seq = np.linspace(minimum,maximum,nrbins+1)

                if inx == 'ndvi':
                    maximum = ndvirange[1]
                    minimum = ndvirange[0]
                    bin_seq = np.linspace(minimum,maximum,nrbins+1)

                if min(line) >= maximum:
                    hist = [float(0)]*(nrbins-1); hist.append(float(1)) 
                elif max(line) <= minimum:
                    hist = [1]; hist.extend([0]*(nrbins-1))
                else:
                    density, _ = np.histogram(line, bin_seq, density=False)
                    hist = density / float(density.sum())

                myid.extend(hist)
                hist2 = myid
                #print(hist2)
                histlist.append(hist2)

    print('Saving histograms into histograms.pkl...')
    save_intensities(os.path.join(outputdirsetti, 'histograms.pkl'), histlist)


    print('Done.')            
    return histlist


# HERE STARTS MAIN:

def main(args):
    try:
        if not args.in_s2_tif or not args.in_aoi_shapefile:
            raise Exception('Missing raster file or shapefile argument. Try --help .')

        print(f'\n\nmakeS2IndexHisto.py')
        print(f'\nSentinel-2 raster files in: {args.in_s2_tif}')
        print(f'ESRI shapefile: {args.in_aoi_shapefile}\n')
        out_dir_path = pathlib.Path(os.path.expanduser(args.output_dir))
        out_dir_path.mkdir(parents=True, exist_ok=True)
        
        setti = args.in_aoi_shapefile.replace(".shp", "").split('-')[-1]
        vuosi = args.in_aoi_shapefile.replace(".shp", "").split('-')[-2]

        out_dir_results = os.path.join(out_dir_path, 'results')
        Path(out_dir_results).mkdir(parents=True, exist_ok=True)
        
        outputdirsetti = os.path.join(out_dir_results, setti)
        Path(outputdirsetti).mkdir(parents=True, exist_ok=True)
        
        if args.extractIntensities:
            extractarray(args.in_aoi_shapefile, args.in_s2_tif, out_dir_results, savePopulations=True)
        
        if args.calculatePopulations:
            calculatePopulations(outputdirsetti, show=False, plotPopulations=True)
            
        if args.make_histograms:
            histlist = makeHisto(outputdirsetti, nrbins)
            
        # Convert to RF ARD
        
        #output_stats(stats, out_stats_file_path, shape_properties, debug=args.debug)

        print(f'\nDone.')

    except Exception as e:
        print('\n\nUnable to read input or write out statistics. Check prerequisites and see exception output below.')
        parser.print_help()
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent(__doc__))
    parser.add_argument('-s', '--in_s2_tif',
                        type=str,
                        help='Directory of Sentinel-2 GeoTIFF image mosaics (.tif)')
    parser.add_argument('-a', '--in_aoi_shapefile',
                        type=str,
                        help='ESRI shapefile containing a set of polygons (.shp with its auxiliary files)')
    parser.add_argument('-o', '--output_dir',
                        help='Directory for output pickle files',
                        type=str,
                        default='.')
    parser.add_argument('-p', '--extractIntensities',
                        help='Extract all pixel values from parcels.',
                        default=False,
                        action='store_true')

    parser.add_argument('-t', '--show-plots',
                        help='Show population distributions as plots on current display device.',
                        default=False,
                        action='store_true')

    parser.add_argument('-l', '--calculatePopulations',
                        help='Calculate population percentiles.',
                        default=False,
                        action='store_true')

    parser.add_argument('-m', '--make_histograms',
                        help="Extract histograms of (multi)parcel's pixel profile.",
                        default=False,
                        action='store_true')

    parser.add_argument('--debug',
                        help='Verbose output for debugging.',
                        action='store_true')

    args = parser.parse_args()
    main(args)
