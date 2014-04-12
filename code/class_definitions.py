# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:30:28 2014

@author: ogavril

class definitions
"""

class INDEP_VARIABLE:
    def __init__(self,name):
        self.name = name
        self.estimated_range = None
        self.gap = None
        self.CI_starts = []
        self.CI_ends = []
        self.means = []

class DISCRETE_MODEL:
    def __init__(self,name):
        self.name = name
        self.sensitivity = None
        self.specificity = None
        self.precision = None
        self.accuracy = None