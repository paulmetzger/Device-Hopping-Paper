from enum import Enum
import sys


class Operator(Enum):
    ADD = 1
    SUB = 2
    MUL = 3


def operator_to_str(o: Operator) -> str:
    if o == Operator.ADD:
        return '+'
    elif o == Operator.SUB:
        return '-'
    elif o == Operator.MUL:
        return '*'
    else:
        print('Error: Unknown reduction operator.')
        sys.exit(1)