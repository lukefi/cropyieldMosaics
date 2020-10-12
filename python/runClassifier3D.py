"""
2020-10-09 MY

python runClassifier3D.py -i /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/train1120 -n 


"""
import pandas as pd
import numpy as np
import pickle
import os.path
from pathlib import Path
import argparse
import textwrap
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping



# FUNCTIONS:

def load_intensities(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def save_intensities(filename, arrayvalues):
    with open(filename, 'wb+') as outputfile:
        pickle.dump(arrayvalues, outputfile)


def classifier(traindir, savePred=False):
    
    inputdir = traindir
    ydata = load_intensities(os.path.join(inputdir, 'y3D.pkl'))
    xdata = load_intensities(os.path.join(inputdir, 'ard3D.pkl'))
    
    basepath = inputdir.split('/')[:-1]
    setti = inputdir.split('/')[-1][-4:]

    testpath = os.path.join(os.path.sep, *basepath, 'test' + setti)
    
    ydatatest = load_intensities(os.path.join(testpath, 'y3D.pkl'))
    xdatatest = load_intensities(os.path.join(testpath, 'ard3D.pkl'))
    
    predfile = os.path.join(testpath, 'ard3DPreds.pkl')
    print(f"Shape of training set: {xdata.shape}")
    
    # LSTM
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape = (xdata.shape[1],
        xdata.shape[2]), activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))

    # monitor validation progress
    early = EarlyStopping(monitor = "val_loss", mode = "min", patience = 7)
    callbacks_list = [early]

    model.compile(loss = 'mean_squared_error',
                  optimizer = 'adam',
                  metrics = ['mse'])

    # and train the model
    history = model.fit(xdata, ydata, 
        epochs=50, batch_size=25, verbose=0, 
        validation_split = 0.20,
        callbacks = callbacks_list)
    
    test_predictions = model.predict(xdatatest)
    
    # Evaluate the model on the test data 
    results = model.evaluate(xdatatest, ydatatest, batch_size=128)
    print("test loss, test acc:", results)
    print("RMSE: ", math.sqrt(results[1]))
    
    dfPreds = pd.DataFrame(test_predictions[:, :, 0])
    # Predictions to compare with forecasts: 15.5.-15.6. eli 4. mosaic, that is pythonic 3.
    # and 5. -> 15.6.-15.7.
    # and 7. -> 15.7.-15.8.
    # and 12. -> the final state
    dfPredsFinal = dfPreds.iloc[:,[3, 5, 7, 12]]
    
    print(f'Saving predictions on test set into {predfile}...\n')
    save_intensities(predfile, dfPredsFinal)
    
    
# HERE STARTS MAIN:

def main(args):
    try:
        if not args.train_dir :
            raise Exception('Missing train set directory argument. Try --help .')

        print(f'\nrunClassifier3D.py')
        print(f'\nARD 3D train set in: {args.train_dir}')

        classifier(args.train_dir, savePred=args.savePreds)
        
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
        
    parser.add_argument('-n', '--savePreds',
                        help='Save test set predictions.',
                        default=False,
                        action='store_true')

    parser.add_argument('--debug',
                        help='Verbose output for debugging.',
                        action='store_true')

    args = parser.parse_args()
    main(args)