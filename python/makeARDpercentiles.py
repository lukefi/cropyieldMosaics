#!/usr/bin/env python
# coding: utf-8

"""
2020-10-02 MY
env: myGIS

From Sentinel-2 percentiles to analysis ready data (ARD) with pandas. Indeces 'NDMI', 'NDTI' and 'NDVI'.

INPUTS:

- Input directory (also results are saved here)
- Target directory (merge with y)
- Directory for meteorological data
- Number of dimensions in ARD data

OUTPUT:
ARD per parcel, ard2DPercentiles.pkl

- With option  --makeARD (2D or 3D) reshapes the time series into ARD either 2D or 3D.
- Option --dimensions decides the dimensions in ARD data.

RUN:

Reshape percentiles into 2D analysis ready format and merge ARD with target values, 2D only:
python makeARDpercentiles.py -i /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/test1120 \
-t /Users/myliheik/Documents/myCROPYIELD/data -n 2 -a

Merge ARD with meteo features, 2D only:
python makeARDpercentiles.py -i /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/test1120 \
-k /Users/myliheik/Documents/myCROPYIELD/copyof-shpfiles/ -n 2 -m




"""
import argparse
import os
import pathlib
import pickle
import textwrap
import pandas as pd

import glob
import os.path
import pickle

from pathlib import Path

import numpy as np

# PARAMETERS:


# FUNCTIONS:

def save_intensities(filename, arrayvalues):
    with open(filename, 'wb+') as outputfile:
        pickle.dump(arrayvalues, outputfile)        
    
    
def joinPerc(in_dir_path):    
    
    # Find all percentile files in resultsdir:
    rawfiles = glob.glob(os.path.join(in_dir_path, 'percentiles20*.pkl'))
    print('\nJoin all years...')
    
    perclist = []
    
    for filename in rawfiles:
        with open(filename, "rb") as f:
            tmp = pickle.load(f)

            for elem in tmp:
                perclist.append(elem)

    print('Saving all years into percentiles.pkl...')
    save_intensities(os.path.join(in_dir_path, 'percentiles.pkl'), perclist)

def makeARD(in_dir_path):
    print('\nMerging percentile files from all years...')

    # Read in percentiles:
    histofile = os.path.join(in_dir_path, 'percentiles.pkl')
    with open(histofile, "rb") as f:
        data = pickle.load(f)
        
    df = pd.DataFrame(data)
    print(f"\nShape of dataframe: {df.shape}")    
    df.columns = ['farmID','date', 'feature', 'mean', 'percentile_10', 'percentile_20', 'percentile_30', 'percentile_40', 'percentile_50', 'percentile_60', 'percentile_70', 'percentile_80', 'percentile_90']
    
    dfndmi = df[df['feature'] == 'ndmi']
    dfndti = df[df['feature'] == 'ndti']
    dfndvi = df[df['feature'] == 'ndvi']
    
    print(f"Lengths of subsets ndmi, ndti, ndvi: {len(dfndmi), len(dfndti), len(dfndvi)}")
    
    dfndmipivot = dfndmi.pivot(index='farmID', columns='date', values=['mean', 'percentile_10', 'percentile_20', 'percentile_30', 'percentile_40', 'percentile_50', 'percentile_60', 'percentile_70', 'percentile_80', 'percentile_90'])
    dfndmipivot.columns = ['_ndmi_'.join(col).strip() for col in dfndmipivot.columns.values]

    dfndtipivot = dfndti.pivot(index='farmID', columns='date', values=['mean', 'percentile_10', 'percentile_20', 'percentile_30', 'percentile_40', 'percentile_50', 'percentile_60', 'percentile_70', 'percentile_80', 'percentile_90'])
    dfndtipivot.columns = ['_ndti_'.join(col).strip() for col in dfndtipivot.columns.values]

    dfndvipivot = dfndvi.pivot(index='farmID', columns='date', values=['mean', 'percentile_10', 'percentile_20', 'percentile_30', 'percentile_40', 'percentile_50', 'percentile_60', 'percentile_70', 'percentile_80', 'percentile_90'])
    dfndvipivot.columns = ['_ndvi_'.join(col).strip() for col in dfndvipivot.columns.values]

    all_dfs = pd.concat(
        (iDF for iDF in [dfndmipivot, dfndtipivot, dfndvipivot]),
        axis=1, join='inner'
    ).reset_index()

    print(f"Shape of dataframe: {all_dfs.shape}")

    print('Saving percentiles into ardPercentiles.pkl...')
    save_intensities(os.path.join(in_dir_path, 'ardPercentiles.pkl'), all_dfs)


def mergeTarget(targetdir, in_dir_path):
    
    # Read in ARD and merge with target y by farmID:
    ardfile = os.path.join(in_dir_path, 'ardPercentiles.pkl')
    with open(ardfile, "rb") as f:
        data = pickle.load(f)
        
    setti = str(in_dir_path).split('/')[-1]
    targetsetti = setti.split('1')[0] + 'y' + setti[-4:]
    targetfilebase = os.path.join(targetdir, targetsetti)
    
    df = pd.concat(map(pd.read_csv, glob.glob(targetfilebase + '*.csv')), sort=False)


    dfnew = df[['farmID', df.columns[df.columns.str.endswith('ha')][0]]]
    dfnew.columns = ['farmID', 'y']

    print('Saving into ard2Dpercentiles.pkl...')
    save_intensities(os.path.join(in_dir_path, 'ard2Dpercentiles.pkl'), data.merge(dfnew, how = 'left', on = 'farmID'))

def make2Dmeteo(in_dir_path, meteo_dir):
    ardfile = os.path.join(in_dir_path, 'ard2Dpercentiles.pkl')
    
    setti = str(in_dir_path).split('/')[-1]
    with open(ardfile, "rb") as f:
        data = pickle.load(f)
    
    meteodf = pd.concat(map(pd.read_pickle, glob.glob(os.path.join(meteo_dir, 'ard-*') + setti + '.pkl')), sort=False)
    
    print('Saving into ard2DpercentilesMeteo.pkl...')
    newdata = data.merge(meteodf, how = 'left', on = 'farmID').dropna()
    
    if len(data) != len(newdata):
        print(f'\n Numbers before and after joining dont match: {len(data)} and {len(newdata)}')
    print(f'\n Any NAs: {newdata.isna().any().any()}')
    save_intensities(os.path.join(in_dir_path, 'ard2DpercentilesMeteo.pkl'), newdata)
    

# HERE STARTS MAIN:

def main(args):
    try:
        if not args.input_dir:
            raise Exception('Missing input directory argument. Try --help .')

        print(f'\n\makeARDpercentiles.py')
        print(f'\nPercentile files in: {args.input_dir}')
      
        inputdirsetti = pathlib.Path(os.path.expanduser(args.input_dir))       
        nrdim = args.dimensions
        
        if nrdim == 3:
            print(f'\nARD in 3D is under construction...')
            

            
        if args.makeARD:
            if nrdim == 2:
                joinPerc(inputdirsetti)
                makeARD(inputdirsetti)
                if not args.target_dir:
                    print(f'\nCannot merge with target variable. Please, give target directory (-t) where target files are stored!')
                else:    
                    targetdir = pathlib.Path(os.path.expanduser(args.target_dir))
                    mergeTarget(targetdir, inputdirsetti)
            else:
                print(f'\nARD in 3D is under construction...')
                
        if args.mergeMeteo:
            if not args.meteo_dir:
                print(f'\nCannot merge with meteo data. Please, give meteo directory (-k) where meteo files are stored!')
            else:    
                meteodir = pathlib.Path(os.path.expanduser(args.meteo_dir))
                make2Dmeteo(inputdirsetti, meteodir)

        print(f'\nDone.')

    except Exception as e:
        print('\n\nUnable to read input or write out statistics. Check prerequisites and see exception output below.')
        parser.print_help()
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent(__doc__))
    parser.add_argument('-i', '--input_dir',
                        help='Directory for input pickle files',
                        type=str)
    
    parser.add_argument('-t', '--target_dir',
                        help='Directory for target directory.',
                        type=str)
    
    parser.add_argument('-k', '--meteo_dir',
                        help='Directory for meteorological directory.',
                        type=str)
    
    parser.add_argument('-a', '--makeARD',
                        help="Reshape the time series into ARD.",
                        default=False,
                        action='store_true')
    
    parser.add_argument('-m', '--mergeMeteo',
                        help="Reshape percentiles into 2D analysis ready format and add meteorological data.",
                        default=False,
                        action='store_true')
 
    parser.add_argument('-n', '--dimensions',
                        help='Number of dimensions in ARD (2D or 3D), default 2.',
                        type=int,
                        default=2)

    parser.add_argument('--debug',
                        help='Verbose output for debugging.',
                        action='store_true')

    args = parser.parse_args()
    main(args)
