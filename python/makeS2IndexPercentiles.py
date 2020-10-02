#!/usr/bin/env python
# coding: utf-8

"""
2020-10-02 MY
env: myGIS

From Sentinel-2 mosaics to percentiles with rasterstats. Indeces 'NDMI', 'NDTI' and 'NDVI'.

INPUTS:
/Users/myliheik/Documents/GISdata/satotutkimus/shpfiles
/Users/myliheik/Documents/GISdata/sentinel2/fromPTA/AOI2/mosaics/

OUTPUT:
percentiles per parcel per year per set, .pkl


RUN:

python makeS2IndexPercentiles.py -s /Users/myliheik/Documents/GISdata/sentinel2/fromPTA/AOI2/mosaics/ \
    -a /Users/myliheik/Documents/GISdata/satotutkimus/shpfiles/satotilat-2019-train1110.shp \
    -o /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/ 

PUHTI:

python makeS2IndexPercentiles.py -s /appl/data/geo/sentinel/s2/ \
    -a /scratch/project_2001253/cropyieldMosaics/shpfiles/satotilat-2019-train1110.shp \
    -o /scratch/project_2001253/cropyieldMosaics 

"""
import argparse
import pickle
import textwrap

import glob
import os.path
from pathlib import Path

from rasterstats import zonal_stats
import numpy as np

# PARAMETERS:


# FUNCTIONS:

def save_intensities(filename, arrayvalues):
    with open(filename, 'wb+') as outputfile:
        pickle.dump(arrayvalues, outputfile)    

def makePerc(shpfile, datadir, outputdir):
    # List all files:
    list_of_files = glob.glob(os.path.join(datadir, '*.tif'))
    # stats to calculate:
    stats = ['mean', 'percentile_10', 'percentile_20', 'percentile_30', 'percentile_40', 'percentile_50',
            'percentile_60', 'percentile_70', 'percentile_80', 'percentile_90']
    
    
    myarrays = []

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
            parcels = zonal_stats(shpfile, filename, stats=stats, geojson_out=True, all_touched = False, 
                                  raster_out = False, nodata=np.nan)
            print("Length : %d" % len (parcels))


            for x in parcels:
                myarray = [x['properties']['farmID'], date, feature, x['properties']['mean'], x['properties']['percentile_10'], x['properties']['percentile_20'], x['properties']['percentile_30'], x['properties']['percentile_40'], x['properties']['percentile_50'], x['properties']['percentile_60'], x['properties']['percentile_70'], x['properties']['percentile_80'], x['properties']['percentile_90']]

                myarrays.append(myarray)


    
    filename2 = os.path.join(output, 'percentiles' + vuosi + '.pkl')
    
    # Save all pixel values: 
    print('Saving all pixels values into ' + filename2 + '...')
    save_intensities(filename2, myarrays)
    
    


# HERE STARTS MAIN:

def main(args):
    try:
        if not args.in_s2_tif or not args.in_aoi_shapefile:
            raise Exception('Missing raster file or shapefile argument. Try --help .')

        print(f'\n\makeS2IndexPercentiles.py')
        print(f'\nSentinel-2 raster files in: {args.in_s2_tif}')
        print(f'ESRI shapefile: {args.in_aoi_shapefile}\n')
        out_dir_path = Path(os.path.expanduser(args.output_dir))
        out_dir_path.mkdir(parents=True, exist_ok=True)
        
        setti = args.in_aoi_shapefile.replace(".shp", "").split('-')[-1]
        vuosi = args.in_aoi_shapefile.replace(".shp", "").split('-')[-2]

        out_dir_results = os.path.join(out_dir_path, 'results')
        Path(out_dir_results).mkdir(parents=True, exist_ok=True)
        
        outputdirsetti = os.path.join(out_dir_results, setti)
        Path(outputdirsetti).mkdir(parents=True, exist_ok=True)
            
        makePerc(args.in_aoi_shapefile, args.in_s2_tif, out_dir_results)

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

    parser.add_argument('--debug',
                        help='Verbose output for debugging.',
                        action='store_true')

    args = parser.parse_args()
    main(args)
