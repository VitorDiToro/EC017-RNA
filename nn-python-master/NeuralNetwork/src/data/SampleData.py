'''
Created on 5 de dez de 2017

@author: marcelovca90
'''

import numpy as np
import os, sys
from numpy.random import sample

class SampleData:
    
    @staticmethod
    def read(folder, filename, flatten = False):
        filename_abs = os.path.join(os.path.dirname(__file__), folder, filename)
        data = []
        with open(filename_abs) as file:
            for line in file:
                data_string = line.strip().split(',')
                data_float = map(float, data_string)
                data.append(data_float)
        return [item for sublist in data for item in sublist] if flatten else data

# https://en.wikipedia.org/wiki/AND_gate
class LOGIC_GATE_AND:
    input = SampleData.read('logic-gate-and', 'input.txt')
    output = SampleData.read('logic-gate-and', 'output.txt', True)

# https://en.wikipedia.org/wiki/OR_gate
class LOGIC_GATE_OR:
    input = SampleData.read('logic-gate-or', 'input.txt')
    output = SampleData.read('logic-gate-or', 'output.txt', True)

# https://en.wikipedia.org/wiki/XOR_gate
class LOGIC_GATE_XOR:
    input = SampleData.read('logic-gate-xor', 'input.txt')
    output = SampleData.read('logic-gate-xor', 'output.txt', True)

# https://archive.ics.uci.edu/ml/datasets/ionosphere
class IONOSPHERE:
    input = SampleData.read('ionosphere', 'input.txt')
    output = SampleData.read('ionosphere', 'output.txt', True)

# https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
class TIC_TAC_TOE_ENDGAME:
    input = SampleData.read('tic-tac-toe-endgame', 'input.txt')
    output = SampleData.read('tic-tac-toe-endgame', 'output.txt', True)
