import os
import warnings
import argparse
from Utils.csvread import CsvReader
import numpy as np
from DataPreprocessor.datacleaner import Cleaner

def init_arg_parser():
    parser = argparse.ArgumentParser(description="Automatically generates score table.")

    parser.add_argument('-train', '--trainingdata', help="Get the training data", required=True)
    parser.add_argument('-s', '--strategy', help="Data clean strategy", required=True)
    parser.add_argument('-o', '--output', help="Output path", required=True)
    return parser

if __name__ == "__main__":
    parser = init_arg_parser()
    args = parser.parse_args()

    csvreader = CsvReader()
    data = csvreader.read(args.trainingdata)

    cleaner = Cleaner(data)
    df = cleaner.clean(args.strategy, 'df')

    df.to_csv(args.output)
