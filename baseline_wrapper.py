from baseline import main
from argparse import Namespace
from itertools import product

def_args = {'infile': "failed_evt\encoded_usable_final.xlsx",
    'mode': 'cat', 
    'gen_split': False}

def run_main(model, test_size):
    args = Namespace(model = model, 
        test_size = test_size, 
        **def_args)

    main(args)


if __name__ == "__main__":
    model_vals = ['nb', 'kn', 'svm']
    test_size_vals = [0.25, 0.2]

    for i, tup in enumerate(product(model_vals, test_size_vals)):
        run_main(*tup)