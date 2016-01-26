#testing adding a comment then commiting to git
"""
Tool for OpenFOAM simulation post-processing analysis to get Strouhal Numbers and other analysis
"""

from os import path
import re, sys
from pylab import *
import matplotlib.pyplot as py
from scipy import *
import numpy as np
from scipy.signal import argrelmax
import PyFoam
# import PyFoam.Basics, PyFoam.Infrastructure, PyFoam.Applications
#
# from PyFoam.RunDictionary.FileBasis import FileBasisBackup
# from PyFoam.Basics.PlyParser import PlyParser
#
# from PyFoam.Basics.FoamFileGenerator import FoamFileGenerator
# from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile, ParsedBoundaryDict, ParsedFileHeader
# from PyFoam.Basics.DataStructures import Vector,Field,Dimension,DictProxy,TupleProxy,Tensor,SymmTensor,Unparsed,UnparsedList,Codestream,DictRedirection,BinaryBlob,BinaryList,BoolProxy
from PyFoam.Basics.LineReader import LineReader
from PyFoam.RunDictionary.FileBasis import FileBasis
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile,ParsedBoundaryDict
from PyFoam.RunDictionary.SolutionFile import SolutionFile
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParameterFile import ParameterFile


import json

# define a probe class for easier handling of multiple probes
class Probe:

    def __init__(self, filepath, type, num_probes):

        self.filePath = filepath
        self.type = type
        self.Ux = None
        self.Uy = None
        self.Uz = None
        self.t = None
        self.dt = None
        self.L = None
        self.p = None
        self.Cm = None
        self.Cl = None
        self.Cd = None
        self.location = None
        self.headerlength = None
        self.num_probes =  num_probes # the offset is the number of probes readings contained in one file.
        raw = []
        if type == 'velocity':
            offset = 0
            loc_offset = 0
            if self.num_probes == 0:
                offset = 0
                loc_offset = 0
            elif self.num_probes == 1:
                offset = 3
                loc_offset = 1
            elif self.num_probes == 2:
                offset = 6
                loc_offset = 2

            with open(self.filePath, 'r+') as velocityFile:
                for line in velocityFile:
                    temp = line.split()
                    raw.append(temp)
            self.L = len(raw)
            self.headerlength = 4
            self.t = np.zeros(self.L - self.headerlength)
            self.Ux = np.zeros(self.L - self.headerlength)
            self.Uy = np.zeros(self.L - self.headerlength)
            self.Uz = np.zeros(self.L - self.headerlength)
            self.location = (raw[0][2 + loc_offset], raw[1][2 + loc_offset], raw[2][2 + loc_offset])
            for i in range(4,self.L):
                j = i - 4 # eliminate the offset for the new array
                self.t[j] = float(raw[i][0])
                self.Ux[j] = float(raw[i][1 + offset][1:])
                self.Uy[j] = float(raw[i][2 + offset])
                self.Uz[j] = float(raw[i][3 + offset][:-1])
        elif type == 'pressure':
            print 'Pressure probe reader not written yet'
            exit(-1)
        elif type == 'force':
            print 'Reading force probe'
            with open(self.filePath, 'r+') as forceFile:
                for _ in xrange(8):
                    next(forceFile)
                for line in forceFile:
                    temp = line.split()
                    raw.append(temp)
            self.headerlength = 0   # header has already been stripped
            self.L = len(raw)
            self.t = np.zeros(self.L)
            self.Cm = np.zeros(self.L)
            self.Cd = np.zeros(self.L)
            self.Cl = np.zeros(self.L)
            for i in range(self.L):
                self.t[i] = float(raw[i][0])
                self.Cm[i] = float(raw[i][1])
                self.Cd[i] = float(raw[i][2])
                self.Cl[i] = float(raw[i][3])
        else:
            print 'Type ' + self.type + ' is either invalid or not yet implemented'
            exit(-1)
        self.dt = (self.t[self.L-1 - self.headerlength] - self.t[0]) / (self.L - 1 - self.headerlength)
        print 'delta T = ', self.dt


def getFFT(probe, type, show):

    fs = 1/probe.dt
    L = probe.L
    dataBar = None
    dataPrime = None
    if type == 'force':
        dataBar = np.mean(probe.Cl)
        dataPrime = probe.Cl - dataBar
    elif type == 'Ux':
        dataBar = np.mean(probe.Ux)
        dataPrime = probe.Ux - dataBar
    elif type == 'Uy':
        dataBar = np.mean(probe.Uy)
        dataPrime = probe.Uy - dataBar
    else:
        print 'Type ' + type + ' is either invalid or not yet implemented'
        exit(-1)
    Z = np.fft.fft(dataPrime, L)
    Q = abs(Z)/probe.L
    halfQ = Q[0:L/2-1]
    dF = fs/L
    f = np.arange(0, fs, dF)
    index = np.argmax(Q[0:probe.L/2 - 1])   # get absolute maxima
    localMax = argrelmax(halfQ)             # get local maxima
    print 'Absolute Maxima at: ', f[index]
    for j in range(10):
        print 'Local Maxima at: ', f[localMax[0][j]]

    if show:
        fig = py.figure()
        py.plot(f[0:probe.L/2-1], Q[0:probe.L/2 -1], color='red', lw=1)
        py.xscale('log')
        py.yscale('log')
        py.xlabel('Frequency (Hz)')
        py.ylabel('Magnitude')
        py.show()

def compareProbes(probe1,probe2):

    # TODO: write probe comparator for pressure
    # TODO: get proper average


    L1 = probe1.L
    L2 = probe2.L

    L = None
    U1 = None
    U2 = None
    t1 = None
    t2 = None
    tStable = 250000

    if L1 < L2:
        L = L1
        # print 'The first probe is shorter'
        U1 = (probe1.Ux[tStable:], probe1.Uy[tStable:], probe1.Uz[tStable:])
        U2 = (probe2.Ux[tStable + (L2-L1):], probe2.Uy[tStable + (L2-L1):], probe2.Uz[tStable + (L2-L1):])
        t1 = probe1.t[tStable:]
        t2 = probe2.t[tStable + (L2-L1):]

    else:
        L = L2
        # print 'The second probe is shorter'
        U2 = (probe2.Ux[tStable:], probe2.Uy[tStable:], probe2.Uz[tStable:])
        U1 = (probe1.Ux[tStable + (L1-L2):], probe1.Uy[tStable + (L1-L2):], probe1.Uz[tStable + (L1-L2):])
        t1 = probe2.t[tStable:]
        t2 = probe1.t[tStable +(L1-L2):]
    # get relative difference in each of the signals

    # print 'Test length of shorter array is ', len(U1[0][:])
    # print 'Altered length of longer array is ', len(U2[0][:])
    # print U1[0][:]
    dUx = np.zeros(L)
    dUx = 100*(np.mean(U1[0][:]) - np.mean(U2[0][:]))/np.mean(U1[0][:])
    dUy = 100*(np.mean(U1[1][:]) - np.mean(U2[1][:]))/np.mean(U1[1][:])
    dUz = 100*(np.mean(U1[2][:]) - np.mean(U2[2][:]))#/np.mean(U1[2][:])

    mean_diff = (dUx, dUy, dUz)

    print 'At, ', probe1.location, ', relative difference average: ', np.around(mean_diff,3)

    return np.around(mean_diff,3)

    # f = open('probeDifferenceTest', 'w+')
    # json.dump(dUx.tolist(),f)
    # # json.dump(dUz.tolist(),f)
    # # json.dump(dUy.tolist(),f)
    # f.close()

###################
#       MAIN
###################

# TODO: add subsetter for multiple probes in 1 file
# TODO: proper openfoam file reader?
# load in probe
path = '/Users/Mark/Documents/Data/testCaseValidationRe100_nOrtho0/postProcessing/'


probe1_1 = Probe('/Volumes/Data2/cases/testCaseValidationRe100_nOrtho0/postProcessing/' + 'probes1/60/U', 'velocity', 2)
probe2_1 = Probe('/Volumes/Data2/cases/testCaseValidationRe100_nOrtho1/postProcessing/' + 'probes1/10/U', 'velocity', 2)
compareProbes(probe1_1, probe2_1)


# fname='/Users/Mark/Documents/Data/testCaseValidationRe100_nOrtho0/500/Umean' #sys.argv[
# path = '/Users/Mark/Documents/Data/testCaseValidationRe100_nOrtho0/500/'
# file='/Users/Mark/Documents/Data/testCaseValidationRe100_nOrtho0/500/'
# name='/Users/Mark/Documents/Data/testCaseValidationRe100_nOrtho0/500/Umean'
# #
# # print "Parsing: "+fname
# #
# f=ParsedParameterFile(fname)
#
# print "\nHeader:"
# print f.header
#
#
# for i in range(50):
#     print 'INTERNAL ',  f["internalField"][i]
#
# for b in f["boundaryField"]:
#     print 'BOUNDARY VALUE:', b
#     test = f["boundaryField"][b]
#
#
# vals = f.getValueDict()
#
# temp = ParsedBoundaryDict(f)

# b = vals.get('boundaryField')["topWall"]
# print b
# temp = str(b)
# bvals = temp.split('(')
#
# print bvals
# for l in range(3):
#     print bvals[l]

# print foo

# temp = f.getValueDict()
# test=temp.has_key('internalField')
# test1 = temp.get('internalField')
# print 'internal field: ', test
# #
# temp1 = temp.get('boundaryField')
# # print temp1
#
# recast1 = str(temp1)
# recast = recast1.strip('{}')
#
# print recast
# print '1: ', recast[0]
# print '2: ', recast[1]
# print '3: ',recast[2]
# print foo
# foo = t.getValueDict()
# foob = foo.get('topWall')
# print 'top wall: ', foob
#
# print 'boundary field: ', btest
# print btest1




# print "\nContent:"
# print f.content
# print "\nReconstructed: "
# print str(f)
#
#
# print "Writing to file"
# tmpFile='/Users/Mark/Documents/Data/testCaseValidationRe100_nOrtho0/500/UmeanNEW'
# o=open(tmpFile,"w")
# o.write(str(f))
# o.close()
# print "Reparsing"
# g=ParsedParameterFile(tmpFile,listLengthUnparsed=100)
# print g.content
# print
# if g.content==f.content:
#     print "Reparsed content is equal to original"
# else:
#     print "Reparsed content differs"
#




# getFFT(probe, 'force', True)
# getFFT(probe, 'Uy', True)
# dt = 0.001
# fs = 1/dt
# Cl_bar = np.mean(probe.Cl)
# L = probe.L
# Cl_prime = probe.Cl - Cl_bar
# Z = np.fft.fft(Cl_prime, L)
# Q = abs(Z)/probe.L
# halfQ = Q[0:L/2-1]
# dF = fs/probe.L
# f = np.arange(0, fs, dF)
#
# index = np.argmax(Q[0:probe.L/2 - 1])
# localMax = argrelmax(halfQ)
# print localMax
# print 'test', localMax[0][0]
# for j in range(4):
#     print 'Local Maxima at: ', f[localMax[0][j]]
#


# print f[index]
# print index






#probeFile1 = open(path + 'probes1/60/U', 'r+')
#probeFile2 = open(path + 'probes2/60/U', 'r+')
#probeFile2 = open(path + 'probes3/60/U', 'r+')

