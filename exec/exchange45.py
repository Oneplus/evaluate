#!/usr/bin/env python
from __future__ import print_function
import sys


def main():
    dataset = open(sys.argv[1], 'r').read().strip().split('\n\n')

    fpo = open(sys.argv[2], 'w')
    for data in dataset:
        lines = data.splitlines()
        for line in lines:
            if line.startswith('#'):
                print(line, file=fpo)
            else:
                tokens = line.split('\t')
                tokens[4], tokens[5] = tokens[5], tokens[4]
                print('\t'.join(tokens), file=fpo)
        print('', file=fpo)


if __name__ == "__main__":
    main()
