# Meaning Representations from Trajectories in Autoregressive Models
# https://github.com/tianyu139/meaning-as-trajectories
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from scipy.stats import entropy


class BaseDistanceMetric:
    def __init__(self):
        pass

    def compute(self, a, b):
        # Takes as input two log probabilities
        raise NotImplementedError


class L1Distance(BaseDistanceMetric):
    def compute(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b), ord=1)

    def __str__(self):
        return "l1"

class L2Distance(BaseDistanceMetric):
    def compute(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b), ord=2)

    def __str__(self):
        return "l2"

class LInfDistance(BaseDistanceMetric):
    def compute(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b), ord=np.inf)

    def __str__(self):
        return "linf"


class SymmetricKLDivergence(BaseDistanceMetric):
    def __init__(self, temp):
        self.temp = temp
    def compute(self, a, b):
        a = np.exp(np.array(a) * self.temp)
        a /= sum(a)
        b = np.exp(np.array(b) * self.temp)
        b /= sum(b)
        return entropy(a, qk=b) + entropy(b, qk=a)
    def __str__(self):
        return f"symkl-temp={self.temp}"

class TotalVariationDistance(BaseDistanceMetric):
    def __init__(self, temp):
        self.temp = temp

    def compute(self, a, b):
        a = np.exp(np.array(a) * self.temp)
        a /= sum(a)
        b = np.exp(np.array(b) * self.temp)
        b /= sum(b)

        return np.linalg.norm(a - b, ord=np.inf)

    def __str__(self):
        return f"tv-temp={self.temp}"

class HellingerDistance(BaseDistanceMetric):
    def __init__(self, temp):
        self.temp = temp

    def compute(self, a, b):
        a = np.exp(np.array(a) * self.temp)
        a /= sum(a)
        b = np.exp(np.array(b) * self.temp)
        b /= sum(b)

        a = np.sqrt(a)
        b = np.sqrt(b)
        return np.linalg.norm(a - b, ord=2) / np.sqrt(2)

    def __str__(self):
        return f"hel-temp={self.temp}"

class CosineSimilarity(BaseDistanceMetric):
    # Note: not really a metric
    def compute(self, a, b):
        EPS=1e-6

        a = np.array(a)
        b = np.array(b)
        return -(np.dot(a,b) / np.max([EPS, np.linalg.norm(a) * np.linalg.norm(b)])) + 1

    def __str__(self):
        return "cossim"


class CosineSimilarityProb(BaseDistanceMetric):
    # Note: not really a metric
    def __init__(self, temp):
        self.temp = temp

    def compute(self, a, b):
        EPS=1e-6

        a = np.exp(np.array(a) * self.temp)
        a /= np.sum(a)
        b = np.exp(np.array(b) * self.temp)
        b /= np.sum(b)
        return -(np.dot(a,b) / np.max([EPS, np.linalg.norm(a) * np.linalg.norm(b)])) + 1

    def __str__(self):
        return f"cossim-temp={self.temp}"
