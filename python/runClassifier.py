"""
2020-09-22

RUN:

runClassifier.py -i /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/train1400 \
-t /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/test1400 -l 30

# Use -n to save predictions:
runClassifier.py -i /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/train1400 \
-t /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/test1400 -l 30 -n

# Use -m to use also meteorological features:
runClassifier.py -i /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/train1400 \
-t /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/test1400 -l 30 -n -m


"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

import pandas as pd
import numpy as np
import pickle
import os.path
from pathlib import Path
import argparse
import textwrap

# FUNCTIONS:

def classify(trainfile: str, testfile: str, predfile: str, learner: str, important: int, savePred=False):

    train = pd.read_pickle(trainfile)
    test = pd.read_pickle(testfile)
    
    X_train = train.drop(['y', 'farmID'], axis = 1)
    y_train = train['y']
    X_test = test.drop(['y', 'farmID'], axis = 1)
    y_test = test['y']
    
    if learner == 'rf':
        
        rf = RandomForestRegressor(n_estimators=200, random_state=42, max_features = 'sqrt')
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        if savePred:  
            print(f'Saving predictions on test set into {predfile}...\n')
            with open(predfile, 'wb+') as outputfile:
                pickle.dump(y_pred, outputfile)
        
        mse = mean_squared_error(y_test, y_pred)
        print(f"RMSE: {sqrt(mse)}")
        print(f"MSE: {mse}")
        print(f"R2: {r2_score(y_test, y_pred)}")

        print("Calculating feature importances ... \n")
        feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',
                                                                        ascending=False)
        print(feature_importances[1:important])
        
# HERE STARTS MAIN:

def main(args):
    try:
        if not args.train_dir or not args.test_dir :
            raise Exception('Missing train set or test set directory argument. Try --help .')

        print(f'\nrunClassifier.py')
        print(f'\nARD train set in: {args.train_dir}')
        print(f'\nARD test set in: {args.test_dir}')


        if args.use_2Dmeteo:
            print(f'\nUsing ARD + meteorological features, for 2D only.')
            trainfile = os.path.join(args.train_dir, 'ard2Dmeteo.pkl')
            testfile = os.path.join(args.test_dir, 'ard2Dmeteo.pkl')
            predfile = os.path.join(args.test_dir, 'ard2DmeteoPreds.pkl')
        else:
            trainfile = os.path.join(args.train_dir, 'ard2D.pkl')
            testfile = os.path.join(args.test_dir, 'ard2D.pkl')
            predfile = os.path.join(args.test_dir, 'ard2DPreds.pkl')

        classify(trainfile, testfile, predfile, learner = 'rf', important = 30, savePred=args.savePreds)
        
        print(f'\nDone.')

    except Exception as e:
        print('\n\nUnable to read input or write out statistics. Check prerequisites and see exception output below.')
        parser.print_help()
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent(__doc__))

    parser.add_argument('-i', '--train_dir',
                        help='Directory for input directory.',
                        type=str,
                        default='.')

    parser.add_argument('-t', '--test_dir',
                        help='Directory for target directory.',
                        type=str,
                        default='.')
        
    parser.add_argument('-l', '--importance',
                        help='Number of the most important features to show, default 20.',
                        type=int,
                        default=20)

    parser.add_argument('-m', '--use_2Dmeteo',
                        help='Use additional meteorological features.',
                        default=False,
                        action='store_true')

    parser.add_argument('-n', '--savePreds',
                        help='Save test set predictions.',
                        default=False,
                        action='store_true')

    parser.add_argument('--debug',
                        help='Verbose output for debugging.',
                        action='store_true')

    args = parser.parse_args()
    main(args)
