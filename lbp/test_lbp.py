import numpy as np
import cv2
from matplotlib import pyplot as plt

file_name = '0002_01.jpg'
img = cv2.imread(file_name, 0)
print(img)
print(img.shape)
replicate = cv2.copyMakeBorder(img, 0, 3, 0, 13, cv2.BORDER_REPLICATE)
# cv2.imshow(replicate)
# plt.imshow(replicate, cmap='gray', interpolation='bicubic')
# plt.show()
print(replicate.shape)
GRID_COL = 7
GRID_ROW = 7
blocks = []
width = replicate.shape[0]
height = replicate.shape[1]
block_width = int(width / GRID_COL)
block_height = int(height / GRID_ROW)
print(block_width, block_height)
for i in range(GRID_ROW):
    blocks.append([].copy())
    for j in range(GRID_COL):
        block = replicate[i * block_width:(i + 1) * block_width, j * block_height:(j + 1) * block_height]
        blocks[i].append(block)

import itertools


def is_uniform(pattern):
    num_change = 0
    for i in range(len(pattern) - 1):
        if pattern[i + 1] != pattern[i]:
            num_change += 1
    if pattern[0] != pattern[-1]:
        num_change += 1
    return num_change <= 2


def kbits(n, k):
    result = []
    for bits in itertools.combinations(range(n), k):
        s = ['0'] * n
        for bit in bits:
            s[bit] = '1'
        result.append(''.join(s))
    return result


def generate_all_patterns(p=8):
    all_uniform_patterns = []  # to contain all uniform patterns
    is_pattern_uniform = {}  # to check if a pattern is uniform
    for i in range(p + 1):
        patterns = kbits(p, i)
        for pat in patterns:
            if is_uniform(pat):
                is_pattern_uniform[pat] = True
                all_uniform_patterns.append(pat)
            else:
                is_pattern_uniform[pat] = False
    all_uniform_patterns = sorted(all_uniform_patterns)
    return all_uniform_patterns, is_pattern_uniform


all_uniform_patterns, is_pattern_uniform = generate_all_patterns()


def threshold_submatrix(submatrix, p=8, r=2):
    """
    Thresholding a submatrix to return an LBP descriptor for each pixel.
    In this example, neighborhood of (8, 2) is used
    :param r: radius = 2
    :param p: total number of points, right now fix = 8
    :param submatrix:
    :return:
    """
    # print(submatrix)
    width = submatrix.shape[0]
    height = submatrix.shape[1]
    center_x = int(width / 2)
    center_y = int(height / 2)
    center = submatrix[center_x, center_y]
    neighbors = np.asarray([
        submatrix[center_x, 0],
        np.average(submatrix[center_x + 1:width, 0:center_y]),
        submatrix[width - 1, center_y],
        np.average(submatrix[center_x + 1:width, center_y + 1:height]),
        submatrix[center_x, height - 1],
        np.average(submatrix[0: center_x, center_y + 1: height]),
        submatrix[0, center_y],
        np.average(submatrix[0:center_x, 0:center_y])
    ])
    neighbors = (neighbors < center).astype(int)
    neighbors = ''.join(np.char.mod('%d', neighbors))
    return neighbors


def sort_patterns(patterns):
    """
    Put patterns into bins (hash map) and count the appearance of each pattern
    :param patterns:
    :return:
    """
    NON_UNIFORM_PATTERN = '0'
    result = {}
    global all_uniform_patterns, is_pattern_uniform
    for pat in patterns:
        if pat in is_pattern_uniform:
            result[pat] = result.get(pat, 0) + 1
        else:
            result[NON_UNIFORM_PATTERN] = result.get(NON_UNIFORM_PATTERN, 0) + 1

    for pat in all_uniform_patterns:
        if pat not in result:
            result[pat] = 0

    return result


def process_block(block):
    """
    Process each block and return a LBP descriptor
    :param block:
    :return:
    """
    r = 2
    width = block.shape[0]
    height = block.shape[1]
    patterns = []
    for i in range(r, width - r):
        for j in range(r, height - r):
            sub_matrix = block[i - r:i + r + 1, i - r:i + r + 1]
            threshold_matrix = threshold_submatrix(sub_matrix)
            patterns.append(threshold_matrix)
    # print(patterns)
    sorted_patterns = sort_patterns(patterns)
    return sorted_patterns

all_histograms = []
for i in range(len(blocks)):
    for j in range(len(blocks[0])):
        block = blocks[i][j]
        histogram = process_block(block)
        all_histograms.append(histogram)

print(len(all_histograms))
print(all_histograms)
