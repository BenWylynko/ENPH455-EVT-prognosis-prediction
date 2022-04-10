import sys
from pathlib import Path
sys.path.append(str(Path(Path(".").resolve(), 'cnn'))) #gross

from cnn_seg import main
from argparse import Namespace
from itertools import product

#default args
def_args = {'train': True, 'plot': False, 'prep_train': False, 'results': False, 
    'gen_split': False, 'k_fold': True}

def run_main(masked, test_size, n_layers):
    args = Namespace(n_layers = n_layers, 
        test_size = test_size, 
        masked = masked, 
        **def_args)

    main(args)


if __name__ == "__main__":
    masked_vals = [False, True] #False runs are done
    test_size_vals = [0.25, 0.2]
    n_layers_vals = [2, 3]

    for i, tup in enumerate(product(masked_vals, test_size_vals, n_layers_vals)):
        run_main(*tup)

















