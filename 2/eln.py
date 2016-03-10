import math


class NegativeInputException(Exception):
    pass


def exp(x):
    return 0 if math.isnan(x) else math.exp(x)


def ln(x):
    if x == 0:
        return math.nan
    if x > 0:
        return math.log(x)
    raise NegativeInputException


def sum(eln_x, eln_y):
    if math.isnan(eln_x):
        return eln_y
    if math.isnan(eln_y):
        return eln_x

    if eln_x >= eln_y:
        return eln_x + ln(1 + math.exp(eln_y - eln_x))
    return eln_y + ln(1 + math.exp(eln_x - eln_y))


def product(eln_x, eln_y):
    return math.nan if math.isnan(eln_x) or math.isnan(eln_y) else eln_x + eln_y


def greater(eln_x, eln_y):
    if math.isnan(eln_x):
        return False
    if math.isnan(eln_y):
        return True
    return eln_x > eln_y


def print_matrix(matrix, caption=None):
    if caption:
        print('==================')
        print(caption)
    for i in range(len(matrix[0])):
        print('\t\t%d' % i, end='')
    print()

    i = 0
    for row in matrix:
        print(i, end='')
        i += 1
        for x in row:
            print('\t%f' % exp(x), end='')
        print()
