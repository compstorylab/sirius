exec(open("./scripts/imports.py").read())
from .args import argSetup
from .data import loadData, getTypes, computeBandwidth
from .viz import vizDD, vizDC, vizCC
from .mi import miDD, miDC, miCC, calcMI
from .matrix import matrixHeatmap, layoutGraph, drawGraph