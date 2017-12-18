from __future__ import division
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg
import scipy.special as special
import numpy as np
import math
import scipy.special

from scipy import integrate
import scipy.stats.mstats
import operator
from operator import add
from operator import itemgetter
import scipy.optimize
from scipy.optimize import minimize_scalar
import collections
from compiler.ast import flatten

# Draw figure
import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm

########################################
## Exact answer for surface area
ExactAns=53.0*58.0
#JOG = 2741788.93275 #+ SUB 1683697.02838

# Initial value of fraction area for optimization
IFA= [1000]
# Percentage of temperature data used for fitting
#PTD = [1,0.5,0.25,0.1]
PTD = [0.8]
# number of fraction
maxF=10
minF=1
numF=maxF-minF+1

###### Output results ##################
outputdir='./171211-1.0'
#Reduce plotted data points
deltatempdata = 1
########################################

for nPTD in range(0, len(PTD)):
	avePTD=0
	#for nPTD in range(0, 1):
	PercentageTempData = PTD[nPTD]
	OptAnswer_EachArea=np.zeros([numF*len(IFA),numF])
	OptAnswer_Average=np.zeros([numF*len(IFA)])
	OptAnswer_STD=np.zeros([numF*len(IFA)])
	OptAnswer_Error=np.zeros([numF*len(IFA)])

	fig, ax = plt.subplots( nrows=0, ncols=0 )  # create figure & 1 axis
	x = np.arange(1,numF*len(IFA)+1)
	y = np.ones(numF*len(IFA))*ExactAns

	#### Optimized Conditions #############
	for NumFraction in range (minF,maxF+1):
		for Num_A  in range (NumFraction,NumFraction+1):
			OptAnswer_nIFA=np.zeros([len(IFA)])
			#Change initial fraction area: IFA
			for nIFA in range(0,len(IFA)):
				Init_Frac_A=IFA[nIFA]
				resultdir=outputdir+'/Result_frac3_temp'+str(PercentageTempData)+'_Init'+str(Init_Frac_A)
				dirname=resultdir + '/NumFrac'+str(NumFraction)+'_Num_A'+str(Num_A)
				ans=np.genfromtxt(dirname+'Optimization.csv',delimiter=',')
				OptAnswer_nIFA[nIFA] = ans[-1]
				for nfrac in range (1,NumFraction+1):
					OptAnswer_EachArea[(NumFraction-1)*len(IFA)+nIFA ,nfrac-1]=ans[-1-NumFraction+nfrac-1]

				if (sum(OptAnswer_EachArea[(NumFraction-1)*len(IFA)+nIFA ,:])>1e10):
					OptAnswer_EachArea[(NumFraction-1)*len(IFA)+nIFA ,:]=0

			#print PercentageTempData,Init_Frac_A, NumFraction, OptAnswer_EachArea[nIFA,:]
	Int_EachArea = np.zeros(numF*len(IFA))
	for nbar in range (minF, maxF+1):
		#print nbar, OptAnswer_EachArea[:,nbar]
		plt.bar(x,OptAnswer_EachArea[:,nbar-1], bottom=Int_EachArea,color=cm.spectral(nbar/NumFraction))
		Int_EachArea=Int_EachArea+OptAnswer_EachArea[:,nbar-1]
	nnn=0
	for nn in range (0,len(Int_EachArea)):
		if(Int_EachArea[nn]>0):
			nnn=nnn+1
	print nnn, sum(Int_EachArea)
	np.savetxt(outputdir+'/Area.csv',Int_EachArea)

	plt.xlabel('Number of fraction')
	plt.ylabel('Estimated surface area')
	plt.xticks(x)
	plt.plot(x,y,'+')
	plt.ylim([0,4e3])
	fig.savefig(outputdir + '/EstimatedSurfaceArea1000.png')
	plt.close(fig)
