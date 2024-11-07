import numpy as np
import pandas as pd
import networkx as nx
import cv2
import os
import logging
import json
import uuid
import shutil
from io import StringIO
import itertools
import re
import argparse
import sys
import tempfile

from copy import deepcopy
from random import randint
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from warnings import filterwarnings as fw

from openpyxl import Workbook
from openpyxl.styles import Font, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment

from plotly import graph_objects as go
from plotly import express as px
import matplotlib.pyplot as plt
import colorsys

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances

from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from scipy.interpolate import splprep, splev

from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree, KDTree
from scipy.spatial import distance_matrix
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from scipy.ndimage import distance_transform_edt
from scipy.special import binom
from scipy.interpolate import interp1d
from scipy.stats import bootstrap
from scipy.optimize import minimize

from shapely.geometry import Polygon, MultiPoint, LinearRing
from shapely.ops import cascaded_union
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

from collections import deque
from prettytable import PrettyTable

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import OPTICS
from sklearn.neighbors import KernelDensity
from sklearn.cluster import MeanShift
from sklearn.svm import OneClassSVM
from sklearn.covariance import MinCovDet
from shapely.geometry import Polygon, Point