import numpy as np
import heapq
import sys
import copy

parser = argparse.ArgumentParser(description='Process filename.')
parser.add_argument('-f','--filename', type=str, help='filename', required=True)
parser.add_argument('-s','--stopwords', type=str, help='stopword', required=True)
parser.add_argument('-c','--corpus', type=str, help='corpus', required=True)

args = parser.parse_args()

filename = args.filename

