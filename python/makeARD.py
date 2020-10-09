#!/usr/bin/env python
# coding: utf-8

"""
2020-09-16 MY
2020-10-07 added 3D
2020-10-08 added ELY to test set predictions. Needed for regional forecasts!

env: ihan sama, pandas

RUN:

Reshape histograms into 2D analysis ready format:
python makeARD.py -i /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/test1120 -l

Reshape histograms into 3D analysis ready format and make target y as well:
python makeARD.py -i /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/test1120 -m

Merge ARD with target values, 2D only:
python makeARD.py -i /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/test1120 \
-t /Users/myliheik/Documents/myCROPYIELD/data -s

Merge ARD with meteo features, 2D only:
python makeARD.py -i /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/test1120 \
-k /Users/myliheik/Documents/myCROPYIELD/copyof-shpfiles/ -n

"""
import glob
import os.path
import pandas as pd
import pickle

from pathlib import Path

import argparse
import textwrap


# FUNCTIONS:

def save_intensities(filename, arrayvalues):
    with open(filename, 'wb+') as outputfile:
        pickle.dump(arrayvalues, outputfile)

def load_intensities(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def make2D(in_dir_path):
    
    # Read in histograms:
    histofile = os.path.join(in_dir_path, 'histograms.pkl')
    with open(histofile, "rb") as f:
        data = pickle.load(f)
        
    df = pd.DataFrame(data)
    print(f"Shape of dataframe: {df.shape}")
    
    old_names = df.columns.tolist()  
    new_names = ['farmID','date', 'feature', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5', 
             'bin6', 'bin7', 'bin8', 'bin9', 'bin10', 'bin11', 'bin12', 'bin13', 'bin14', 'bin15', 'bin16'] 

    df = df.rename(columns=dict(zip(old_names, new_names))) 
    
    #split df to feature wise dfs:
    dfndmi = df[df['feature'] == 'ndmi']
    dfndti = df[df['feature'] == 'ndti']
    dfndvi = df[df['feature'] == 'ndvi']
    
    print(f"Lengths of subsets ndmi, ndti, ndvi: {len(dfndmi), len(dfndti), len(dfndvi)}")
    
    dfndmipivot = dfndmi.pivot(index='farmID', columns='date', values=['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 
             'bin6', 'bin7', 'bin8', 'bin9', 'bin10', 'bin11', 'bin12', 'bin13', 'bin14', 'bin15', 'bin16'])
    dfndmipivot.columns = ['_ndmi_'.join(col).strip() for col in dfndmipivot.columns.values]

    dfndtipivot = dfndti.pivot(index='farmID', columns='date', values=['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 
                 'bin6', 'bin7', 'bin8', 'bin9', 'bin10', 'bin11', 'bin12', 'bin13', 'bin14', 'bin15', 'bin16'])
    dfndtipivot.columns = ['_ndti_'.join(col).strip() for col in dfndtipivot.columns.values]

    dfndvipivot = dfndvi.pivot(index='farmID', columns='date', values=['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 
                 'bin6', 'bin7', 'bin8', 'bin9', 'bin10', 'bin11', 'bin12', 'bin13', 'bin14', 'bin15', 'bin16'])
    dfndvipivot.columns = ['_ndvi_'.join(col).strip() for col in dfndvipivot.columns.values]

    all_dfs = pd.concat(
        (iDF for iDF in [dfndmipivot, dfndtipivot, dfndvipivot]),
        axis=1, join='inner'
    ).reset_index()

    print(f"Shape of dataframe: {all_dfs.shape}")

    print('Saving histograms into ard.pkl...')
    save_intensities(os.path.join(in_dir_path, 'ard.pkl'), all_dfs)

    return all_dfs

def merge3DTarget(data, targetdir, in_dir_path):    
    # Read in target y and merge with 3D data by farmID:        
    setti = in_dir_path.split('/')[-1]
    targetsetti = setti.split('1')[0] + 'y' + setti[-4:]
    targetfilebase = os.path.join(targetdir, targetsetti)
    
    df = pd.concat(map(pd.read_csv, glob.glob(targetfilebase + '*.csv')), sort=False)

    dfnew = df[['farmID', df.columns[df.columns.str.endswith('ha')][0]]]
    dfnew.columns = ['farmID', 'y']
    data2 = data.merge(dfnew, how = 'left', on = 'farmID')[['farmID', 'y']].drop_duplicates()
    print(f"Shape of y: {data2['y'].to_numpy().shape}")
    print('Saving y into y3D.pkl...')
    save_intensities(os.path.join(in_dir_path, 'y3DfarmID.pkl'), data2['farmID'])
    save_intensities(os.path.join(in_dir_path, 'y3D.pkl'), data2['y'].to_numpy())
    
    
def make3D(inputdir, targetdir):
    histofile = os.path.join(inputdir, 'histograms.pkl')
    with open(histofile, "rb") as f:
        tmp = pickle.load(f)
    print(f"Length of histogram file: {len(tmp)}")
    
    # to pandas:
    all_files_df = pd.DataFrame(tmp)
    
    print(f"Length of dataframe: {len(all_files_df)}")
    
    old_names = all_files_df.columns.tolist()  
    new_names = ['farmID','date', 'feature', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5', 
             'bin6', 'bin7', 'bin8', 'bin9', 'bin10', 'bin11', 'bin12', 'bin13', 'bin14', 'bin15', 'bin16'] 
    all_files_df = all_files_df.rename(columns=dict(zip(old_names, new_names))) 
    
    dfpivot = all_files_df.pivot(index=['farmID', 'date'], columns='feature', values=['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 
             'bin6', 'bin7', 'bin8', 'bin9', 'bin10', 'bin11', 'bin12', 'bin13', 'bin14', 'bin15', 'bin16'])
    dfpivot.columns = ['_'.join(col).strip() for col in dfpivot.columns.values]
    
    print(f"Any missing dates: {(dfpivot.reset_index().groupby('farmID')['date'].nunique() != 13).any()}")
    
    merge3DTarget(dfpivot, targetdir, inputdir)
    
    x_data = dfpivot.to_numpy()
    x_array = x_data.reshape(int(x_data.shape[0]/13),13,48)
    
    print(f"Shape of X in 2D: {x_data.shape}")
    print(f"Shape of X in 3D: {x_array.shape}")
    
    print(f"Saving X into ard3D.pkl...")
    save_intensities(os.path.join(inputdir, 'ard3D.pkl'), x_array)



def mergeTarget(targetdir, in_dir_path):
    
    # Read in ARD and merge with target y by farmID:
    ardfile = os.path.join(in_dir_path, 'ard.pkl')
    with open(ardfile, "rb") as f:
        data = pickle.load(f)
        
    setti = in_dir_path.split('/')[-1]
    targetsetti = setti.split('1')[0] + 'y' + setti[-4:]
    targetfilebase = os.path.join(targetdir, targetsetti)
    
    df = pd.concat(map(pd.read_csv, glob.glob(targetfilebase + '*.csv')), sort=False)


    dfnew = df[['farmID', df.columns[df.columns.str.endswith('ha')][0]]]
    dfnew.columns = ['farmID', 'y']

    print('Saving into ard2D.pkl...')
    save_intensities(os.path.join(in_dir_path, 'ard2D.pkl'), data.merge(dfnew, how = 'left', on = 'farmID'))

def make2Dmeteo(in_dir_path, meteo_dir):
    ardfile = os.path.join(in_dir_path, 'ard2D.pkl')
    #with open(ardfile, "rb") as f:
    #    tmp = pickle.load(f)
    
    setti = in_dir_path.split('/')[-1]
    #targetsetti = setti.split('1')[0] + 'y' + setti[-4:]
    #targetfilebase = os.path.join(targetdir, targetsetti)
    
    #print(ardfile, setti, targetsetti, targetfilebase)

    with open(ardfile, "rb") as f:
        data = pickle.load(f)
    
    meteodf = pd.concat(map(pd.read_pickle, glob.glob(os.path.join(meteo_dir, 'ard-*') + setti + '.pkl')), sort=False)
    
    print('Saving into ard2Dmeteo.pkl...')
    newdata = data.merge(meteodf, how = 'left', on = 'farmID').dropna()
    
    if len(data) != len(newdata):
        print(f'\n Numbers before and after joining dont match: {len(data)} and {len(newdata)}')
    print(f'\n Any NAs: {newdata.isna().any().any()}')
    save_intensities(os.path.join(in_dir_path, 'ard2Dmeteo.pkl'), newdata)
    
# HERE STARTS MAIN:

def main(args):
    try:
        if not args.input_dir:
            raise Exception('Missing input directory argument. Try --help .')

        print(f'\n\nmakeARD.py')
        print(f'\nHistogram file in: {args.input_dir}')
        print(f'\nTarget file in: {args.target_dir}')
        
        in_dir_path = args.input_dir
        target_dir_path = args.target_dir
        
        if args.make_2D:
            arddata = make2D(in_dir_path)
           
        
        if args.make_3D:
            print(f'\n\nBuilding ARD in 3D for neural networks...')
            make3D(in_dir_path, target_dir_path)
            
            
        if args.mergeTarget:
            print(f'\n\nMerge with target, for 2D only.')
            if not args.input_dir or not args.target_dir:
                raise Exception('Missing input or target directory argument. Try --help .')
            else:
                mergeTarget(target_dir_path, in_dir_path)
                
        if args.make_2Dmeteo:
            print(f'\n\nMerge ARD + target with meteorological data, for 2D only.')
            if not args.input_dir or not args.meteo_dir:
                raise Exception('Missing input or meteo directory argument. Try --help .')
            else:
                make2Dmeteo(in_dir_path, args.meteo_dir)

        print(f'\nDone.')

    except Exception as e:
        print('\n\nUnable to read input or write out statistics. Check prerequisites and see exception output below.')
        parser.print_help()
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent(__doc__))

    parser.add_argument('-i', '--input_dir',
                        help='Directory for input directory.',
                        type=str,
                        default='.')

    parser.add_argument('-t', '--target_dir',
                        help='Directory for target directory.',
                        type=str,
                        default='.')
    
    parser.add_argument('-k', '--meteo_dir',
                        help='Directory for meteorological directory.',
                        type=str,
                        default='.')
        
    parser.add_argument('-l', '--make_2D',
                        help='Reshape histograms into 2D analysis ready format.',
                        default=False,
                        action='store_true')
    
    parser.add_argument('-n', '--make_2Dmeteo',
                        help='Reshape histograms into 2D analysis ready format and add meteorological data.',
                        default=False,
                        action='store_true')

    parser.add_argument('-m', '--make_3D',
                        help="Reshape histograms into 3D analysis ready format.",
                        default=False,
                        action='store_true')
    
    parser.add_argument('-s', '--mergeTarget',
                        help="Merge ARD with target values, 2D only.",
                        default=False,
                        action='store_true')

    parser.add_argument('--debug',
                        help='Verbose output for debugging.',
                        action='store_true')

    args = parser.parse_args()
    main(args)