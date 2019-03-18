#!/usr/bin/env python3


### Hacker module: win_the_game ###
### ref link: https://www.hackerearth.com/practice/algorithms/dynamic-programming/introduction-to-dynamic-programming-1/practice-problems/algorithm/win-the-game/

from sys import stdin, stdout, stderr, setrecursionlimit

verbose = 0

def get_int_array():
    return list(map(int, stdin.readline().split()))


def count_p_red(R, G, depth=1):
    assert R > 0

    dstr = '  ' * depth

    if verbose:
        print("{} count_p_red : R -> {} G -> {} depth -> {}".format(dstr, R, G, depth))
    p_red = R / (R + G)
    p_other = 1 - p_red

    if verbose:
        print("{} depth={} -> p_red -> {} p_other -> {}".format(dstr, depth, p_red, p_other))

    if G == 0:
        if depth % 2 == 1:
            if verbose:
                print("{} return (depth={}) -> {}".format(dstr, depth, 1))
            return 1.0
        else:
            if verbose:
                print("{} return (depth={}) -> {}".format(dstr, depth, 0))
            return 0.0

    if depth % 2 == 1:
        result = p_red + p_other * count_p_red(R, G - 1, depth + 1)
        if verbose:
            print("{} return (depth={}) -> {}".format(dstr, depth, result))
        return result
    else:
        result = p_other * count_p_red(R, G - 1, depth + 1)
        if verbose:
            print("{} return2 (depth={}) -> {}".format(dstr, depth, result))
        return result


def solve(R, G):
    if R == 0 or G == 0:
        return 1.0
    return count_p_red(R, G)


if 1:
    setrecursionlimit(10000)

    T = int(input())

    for _ in range(T):
        (R, G) = get_int_array()
        print("{:.6f}".format(solve(R, G)))
