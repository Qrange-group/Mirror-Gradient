# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start, quick_eval
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--not_mg', action="store_true", help='whether to not use Mirror Gradient, default is False')
    parser.add_argument('--resume', type=str, default=None)

    config_dict = {
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()

    if args.resume == None:
        quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True, mg=not args.not_mg)
    else:
        quick_eval(model=args.model, resume=args.resume, dataset=args.dataset, config_dict=config_dict)
        


