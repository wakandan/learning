import random

import mnist_loader
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

TRACE_LEVEL = 5


def debug(*args, **kwargs):
    logger.debug(args[0].format(args[1:]))


def trace(*args, **kwargs):
    logger.log(TRACE_LEVEL, args[0].format(map(str, *args[1:])))


def info(*args):
    logger.info(args[0].format(args[1:]))


class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        """
        Call nan_to_num so that numpy deals correctly with ln number, which is very close to zero
        :param a:
        :param y:
        :return:
        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return a - y


class QuadraticCost:
    @staticmethod
    def fn(a, y):
        # return 0.5 * np.linalg.norm(a - y) ** 2
        return 0.5 * (a - y).dot(a - y)

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)


def sigmoid(z):  # sigmoid function
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):  # sigmoid prime
    return sigmoid(z) * (1.0 - sigmoid(z))
