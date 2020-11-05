"""
2020-10-09 MY

Printing plots of RMSE from EO predictions.

RUN:
python combineResults.py -i /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/test1320 


"""
import argparse
import textwrap

import subprocess
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")

import os.path
from pathlib import Path

def drawPlots(pred_dir):
    
    setti = pred_dir.split('/')[-1][-4:]
    basepath = pred_dir.split('/')[1:-2]
    img = "RMSE-results-" + setti + ".png"
    imgpath = os.path.join(os.path.sep, *basepath, 'img')
    imgfp = Path(os.path.expanduser(imgpath))
    imgfp.mkdir(parents=True, exist_ok=True)
    imgfile = os.path.join(imgfp, img)
    
    # save all RMSEs to temporary file tmp1.txt
    settii = "cat /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/results-RF-" + setti + "* | grep 'RMSE' > tmp1.txt"
    subprocess.call(settii, shell=True)
    settii = "cat /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/results-RF-" + setti + "* | grep 'Saving' > tmp2.txt"
    subprocess.call(settii, shell=True)
    settii = "cat /Users/myliheik/Documents/myCROPYIELD/cropyieldMosaics/results/results-LSTM-" + setti + ".txt | grep 'RMSE' > tmp3.txt"
    subprocess.call(settii, shell=True)
    
    
    tmp1 = pd.read_csv("tmp1.txt", sep = ":", header = None)
    tmp2 = pd.read_csv("tmp2.txt", sep = "/", header = None)
    tmp3 = pd.read_csv("tmp3.txt", sep = ":", header = None)

    tmp1['Method'] = tmp2[8].str.split('.').str[0]
    tmp1['Month'] = tmp1['Method'].str.split('Preds').str[1]
    tmp1['Month'] = tmp1['Month'].replace('', 'Final')
    tmp1.columns = ['mittari', 'RMSE', 'Method', 'Month']
    dictionary = {'June': 1, 'July': 2, 'August': 3, 'Final': 4}
    tmp1['kk'] = tmp1['Month'].copy()
    for key in dictionary.keys():
        tmp1['kk'] = tmp1['kk'].replace(key, dictionary[key])
    tmp = tmp1.sort_values('kk')
    
    tmp3['mittari'] = 'RMSE'
    tmp3['Method'] = 'LSTM'
    tmp3['Month'] = 'Final'
    tmp3['kk'] = 4
    tmp3['RMSE'] = tmp3[1]
    tmp = tmp.append(tmp3[['mittari', 'RMSE', 'Method', 'Month', 'kk']])
    
    # Draw a nested barplot 
    g = sns.catplot(
        data=tmp, kind="bar",
        x="Month", y="RMSE", hue="Method",
        ci="sd", palette="Paired", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set(ylim=(200, 2000))
    g.set_axis_labels("", "RMSE (kg/ha)")
    g.legend.set_title("")
    g.set(title = "Crop " + setti)
    print('\nSaving the bar plot into', imgfile)
    g.savefig(imgfile)
    
    
# HERE STARTS MAIN:

def main(args):
    try:
        if not args.pred_dir :
            raise Exception('Missing predictions directory argument. Try --help .')

        print(f'\ncombineResults.py')
        print(f'\nPredictions in: {args.pred_dir}')


        drawPlots(args.pred_dir)
        
        
        print(f'\nDone.')

    except Exception as e:
        print('\n\nUnable to read input or write out statistics. Check prerequisites and see exception output below.')
        parser.print_help()
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent(__doc__))

    parser.add_argument('-i', '--pred_dir',
                        help='Directory for predictions directory.',
                        type=str,
                        default='.')
        
    parser.add_argument('--debug',
                        help='Verbose output for debugging.',
                        action='store_true')

    args = parser.parse_args()
    main(args)


