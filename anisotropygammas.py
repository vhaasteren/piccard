#!/usr/bin/env python
# Chiara M. F. Mingarelli, mingarelli@gmail.com
# First section presents the various Gammas up to quadrupole (i.e. l=2) in the "computational frame".
# The second part gives the Gammas in the cosmic frame, i.e. after rotation, for dipole and qudrupole gammas.
# The final function reports the real values of the rotated gammas.

import math
import matplotlib.pyplot as plt
import numpy as np
from math import factorial, sqrt, sin, cos, tan, acos, atan, pi, log
from cmath import exp
import scipy
from scipy.integrate import quad, dblquad
from scipy import special as sp
import random


def Gamma00(zeta):
    """
    l=0, m=0. Isotropic solution.
    Pulsar term doubling at zero.
    Normalised so that Gamma00=1 when zeta=0.
    """
    b=(1.-cos(zeta))
    norm = 3./(4*pi)
    c00=sqrt(4*pi)
    if zeta==0: return 2*norm*c00*0.25*sqrt(pi*4)*(1+(cos(zeta)/3.))
    newans= 0.25*c00*norm*sqrt(pi*4)*(1+(cos(zeta)/3.)+4*b*log(sin(zeta/2)))
    return newans

def dipole_Gammas(m,zeta):
    a=1.+cos(zeta)
    b=1.-cos(zeta)
    norm = 3./(4*pi)
    if m==0:
        if zeta==0: return -2*0.5*norm*(sqrt(pi/3.))*a
        ans01=-0.5*norm*(sqrt(pi/3.))*(a+3*b*(a+4*log(sin(zeta/2.))))
        return ans01
    if m==1:
        if zeta==0: return 0.
        if zeta==pi: return 0.
        ans11=norm*0.5*sqrt(pi/6.)*sin(zeta)*(1.+3*b*(1+(4./a)*log(sin(zeta/2.))))
        return ans11
    if m==-1:
        if zeta==0: return 0.
        if zeta==pi: return 0.
        ans11_m=-1*norm*0.5*sqrt(pi/6.)*sin(zeta)*(1+3*b*(1+(4./a)*log(sin(zeta/2.))))
        return ans11_m
        
def quadrupole_Gammas(m,zeta):
    norm = 3./(4*pi)
    a=1.+cos(zeta)
    b=1.-cos(zeta)
    if zeta == 0 and m!=0: return 0.
    if zeta == pi and m!=0: return 0.
    if m==2:
        ans22=-1*norm*sqrt(5*pi/6.)/4.*b/a*(a*(cos(zeta)**2+4*cos(zeta)-9.)-24*b*log(sin(zeta/2.)))
        return ans22
    if m==1:
        ans21=norm*(0.25*sqrt(2*pi/15.))*sin(zeta)*(5*cos(zeta)**2+15.*cos(zeta)-21.-60*(b/a)*log(sin(zeta/2)))
        return ans21
    if m==0:
        if zeta==0: return 2*0.25*norm*(4./3)*(sqrt(pi/5))*cos(zeta)
        ans20=norm*(1./3)*sqrt(pi/5)*(cos(zeta)+(15./4)*(1-cos(zeta))*(cos(zeta)**2+4*cos(zeta)+3.+8.*log(sin(zeta/2.))))
        return ans20
    if m==-1:
        return -1*norm*(0.25*sqrt(2*pi/15.))*sin(zeta)*(5*cos(zeta)**2+15.*cos(zeta)-21.-60*(b/a)*log(sin(zeta/2)))
    if m==-2:
        return -1*norm*sqrt(5*pi/6.)/4.*b/a*(a*(cos(zeta)**2+4*cos(zeta)-9.)-24*b*log(sin(zeta/2.)))
        
# Part2: rotation functions (from T. Sidery's Ylm.py file, corrected)
    
def dlmk(l,m,k,theta1):
    """returns value of d^l_mk as defined in allen, ottewill 97.
    Called by Dlmk"""
    if m>=k:
        factor = sqrt(factorial(l-k)*factorial(l+m)/factorial(l+k)/factorial(l-m))
        part2 = (cos(theta1/2))**(2*l+k-m)*(-sin(theta1/2))**(m-k)/factorial(m-k)
        part3 = sp.hyp2f1(m-l,-k-l,m-k+1,-(tan(theta1/2))**2)
        return factor*part2*part3
    else:
        return (-1)**(m-k)*dlmk(l,k,m,theta1)

def Dlmk(l,m,k,phi1,phi2,theta1,theta2):
    """returns value of D^l_mk as defined in allen, ottewill 97."""
    return exp(complex(0.,-m*phi1))*dlmk(l,m,k,theta1)*exp(complex(0.,-k*gamma(phi1,phi2,theta1,theta2)))

def gamma(phi1,phi2,theta1,theta2):
    """calculate third rotation angle
    inputs are angles from 2 pulsars
    returns the angle.
    """
    gamma = atan( sin(theta2)*sin(phi2-phi1)/(cos(theta1)*sin(theta2)*cos(phi1-phi2) - sin(theta1)*cos(theta2)))
    if (cos(gamma)*cos(theta1)*sin(theta2)*cos(phi1-phi2) + sin(gamma)*sin(theta2)*sin(phi2-phi1) - cos(gamma)*sin(theta1)*cos(theta2)) >= 0:
        return gamma
    else:
        return pi + gamma

# Part 3: Rotated Gammas: Dipole

def rotated_dipole(m,phi1,phi2,theta1,theta2):
    l=1
    zeta=acos(sin(theta1)*sin(theta2)*cos(phi1-phi2) + cos(theta1)*cos(theta2))
    dipole_gammas=[dipole_Gammas(-1,zeta),dipole_Gammas(0,zeta),dipole_Gammas(1,zeta)]
    rotated_gamma=0
    for i in range(2*l+1):
        rotated_gamma += Dlmk(l,m,i-l,phi1,phi2,theta1,theta2).conjugate()*dipole_gammas[i] #as per eq 73 in Allen&Ottewill'97
    return rotated_gamma

def rotated_quadrupole(m,phi1,phi2,theta1,theta2):
    l=2
    zeta=acos(sin(theta1)*sin(theta2)*cos(phi1-phi2) + cos(theta1)*cos(theta2))
    quad_gammas=[quadrupole_Gammas(-2,zeta),quadrupole_Gammas(-1,zeta),quadrupole_Gammas(0,zeta),quadrupole_Gammas(1,zeta),quadrupole_Gammas(2,zeta)]
    rotated_gamma=0
    for i in range(2*l+1):
        rotated_gamma += Dlmk(l,m,i-l,phi1,phi2,theta1,theta2).conjugate()*quad_gammas[i]
    return rotated_gamma
    

def any_Gamma_comp(phi,theta,m,l,phi1,phi2,theta1,theta2):
    """
    Evaluation of any gamma in the *computational frame*. phi and theta are the variables being integrated over
    whereas phi1,phi2,theta1,theta2 are the coordinates of the pulsar pairs and are just used to
    compute zeta. Normalisation such that c00*\Gamma00=1 at zeta=0.
    """
    zeta=acos(sin(theta1)*sin(theta2)*cos(phi1-phi2) + cos(theta1)*cos(theta2)) #angular separation
    ylm=sp.sph_harm(m,l,phi,theta) #anisotropy
    norm = 3./(4*pi) #to be consistent with previous gammas
    numerator=-0.25*sin(theta)*(1.-cos(theta))*(sin(zeta)*sin(zeta)*sin(phi)*sin(phi)-sin(zeta)*sin(zeta)*cos(theta)*cos(theta)*cos(phi)*cos(phi)-cos(zeta)*cos(zeta)*sin(theta)*sin(theta)+2*sin(zeta)*cos(zeta)*sin(theta)*cos(theta)*cos(phi))
    deno=1.+sin(zeta)*sin(theta)*cos(phi)+cos(zeta)*cos(theta)
    integrand=norm*numerator*ylm/deno
    return integrand.real #this answer is necessarily real-valued, as it is calculated in comp. frame.

def int_Gamma_lm(m,l,phi1,phi2,theta1,theta2):
    """
    Integrates any_Gamma_comp function from 0..pi and 0..2pi. Special cases with analytical solutions
    (l=0,1,2) are handled separately to not waste computing time.
    """
    zeta=acos(sin(theta1)*sin(theta2)*cos(phi1-phi2) + cos(theta1)*cos(theta2)) #angular separation
    if l==0:
        return Gamma00(zeta)
    if l==1:
        return dipole_Gammas(m,zeta)
    if l==2:
        return quadrupole_Gammas(m,zeta)
    else: result=dblquad(any_Gamma_comp,0,pi,lambda x: 0,lambda x: 2*pi,args=(m,l,phi1,phi2,theta1,theta2))[0]
    return  result  

def rotated_Gamma_ml(m,l,phi1,phi2,theta1,theta2,gamma_ml):
    """
    This function takes any gamma in the computational frame and rotates it to the
    cosmic frame. Special cases exist for dipole and qudrupole, as these have
    been found analytically.
    """
    rotated_gamma = 0
    for i in range(2*l+1):
        rotated_gamma += Dlmk(l,m,i-l,phi1,phi2,theta1,theta2).conjugate()*gamma_ml[i]
    return rotated_gamma

def real_rotated_Gammas(m,l,phi1,phi2,theta1,theta2,gamma_ml):
    if m>0:
        ans=(1./sqrt(2))*(rotated_Gamma_ml(m,l,phi1,phi2,theta1,theta2,gamma_ml)+(-1)**m*rotated_Gamma_ml(-m,l,phi1,phi2,theta1,theta2,gamma_ml))
        return ans.real
    if m==0:
        return rotated_Gamma_ml(0,l,phi1,phi2,theta1,theta2,gamma_ml).real
    if m<0:
        ans=(1./sqrt(2)/complex(0.,1))*(rotated_Gamma_ml(-m,l,phi1,phi2,theta1,theta2,gamma_ml)-(-1)**m*rotated_Gamma_ml(m,l,phi1,phi2,theta1,theta2,gamma_ml))
        return ans.real
        
    
        
#testing
    
if __name__ == "__main__":

    l=1

    phi1 = 0.3
    phi2 = 0.7
    theta1 = 0.2
    theta2 = 1.0
    p1 = np.array([sin(theta1)*cos(phi1), sin(theta1)*sin(phi1),cos(theta1)])
    p2 = np.array([sin(theta2)*cos(phi2), sin(theta2)*sin(phi2),cos(theta2)])
    rot_Gs=[]
    cosmic_Gs=[]
    plus_gamma_ml = [] #this will hold the list of gammas evaluated at a specific value of phi1,2, and theta1,2.
    neg_gamma_ml = []
    gamma_ml = []

    #pre-calculate all the gammas so this gets done only once. Need all the values to execute rotation codes.
    for i in range(l+1):
        intg_gamma=int_Gamma_lm(i,l,phi1,phi2,theta1,theta2)
        neg_intg_gamma=(-1)**(i)*intg_gamma  # just (-1)^m Gamma_ml since this is in the computational frame
        plus_gamma_ml.append(intg_gamma)     #all of the gammas from Gamma^-m_l --> Gamma ^m_l
        neg_gamma_ml.append(neg_intg_gamma)  #get the neg m values via complex conjugates 
    neg_gamma_ml=neg_gamma_ml[1:]            #this makes sure we don't have 0 twice
    rev_neg_gamma_ml=neg_gamma_ml[::-1]      #reverse direction of list, now runs from -m .. 0
    gamma_ml=rev_neg_gamma_ml+plus_gamma_ml
    #print gamma_ml    #just 1 list from -m..m, this concatenates the lists.

    
    for m in range(-l,l+1):
        rot_Gs.append(real_rotated_Gammas(m,l,phi1,phi2,theta1,theta2,gamma_ml))
        result_file = open("testForCPPcodeLis"+str(l)+".txt", "a") # the a+ allows you to create the file and write to it.
        result_file.write('{0} {1} {2}  \n'.format(m, l, rot_Gs[m+l])) #writes data to 0th, 1st and 2nd column, resp.
        result_file.close()

