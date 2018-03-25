#!/usr/bin/python2

import math
import numpy as np

global m,n,ka,A,z,g,H           # g is gradient, and H is hessian of \psi(z)
global ns,nd,ls                 # ns is newton step in descent direction. ls is the step size.
n=20                            # dimension of x, the parameter
m=100                           # Total number of sensors
sgm=1/math.sqrt(math.sqrt(n))   # variance for randomly choosing a1,a2,...,am
ka=0.001                        # Quality of approximation, kappa
#sgm=1

A=sgm*np.random.randn(m,n)

def comp_gH():
    global m,n,ka,A,z,g,H

    g=np.zeros(m,)
    H=np.zeros((m,m))

    W=np.zeros((n,n))
    for i in xrange(m):
        W=W+(z[i]*(A[i].reshape(n,1)).dot(A[i].reshape(1,n)))
    W=np.linalg.inv(W)

    for i in xrange(m):
        g[i]=A[i].dot(W.dot(A[i]))
        g[i]=g[i]+(ka/z[i])-(ka/(1.0-z[i]))
    #g=-g

    t=A.dot(W.dot(A.T))
    H=-(t*t)
    for i in xrange(m):
        H[i,i]=H[i,i]-ka*((1.0/(z[i]**2))+(1.0/((1-z[i])**2)))
    #H=-H

def psi(z):                     # The function \psi(z)
    global A,m,n,ka
    W=np.zeros((n,n))

    for i in xrange(m):
        W=W+(z[i]*(A[i].reshape(n,1)).dot(A[i].reshape(1,n)))

    p=math.log(np.linalg.det(W))

    for i in xrange(m):
        #print i
        #print z[i]
        p=p+ka*(math.log(z[i])+math.log(1.0-z[i]))

    return p

def comp_ns():                  # Compute newton step
    global m,n,ka,ls
    global z,ns,nd,g,H

    Hi=np.linalg.inv(H)
    ns=-Hi.dot(g)
    t=np.ones((1,m)).dot(-ns)
    t=t/(np.ones((1,m)).dot(Hi.dot(np.ones((m,1)))))
    t=t*Hi.dot(np.ones((m,1)))
    t=t.reshape(m,)
    ns=ns+t

    nd=math.sqrt(g.dot(ns))


    #def comp_ls():             # Compute step size and newton decrement
    #global m,n,ka,ls
    #global z,ns,g,H

    ls=1.0                        # initial lambda
    b=0.0001                      # beta
    t=0.5                       # tau
    p=psi(z)
    while(not(np.all(z+ls*ns>0) and np.all(z+ls*ns<1))):
        ls=t*ls
        #print ls,z+ls*ns
    while(psi(z+ls*ns)<p+ls*b*(g.dot(ns))):
        ls=t*ls

k=30.0                            # No. of sesors to select from m sensors
z=(float(k)/m)*np.ones(m)              # Initial guess of feasible z
comp_gH()
comp_ns()
#z=z+ls*ns
#nd=float('inf')
ctr=0
while(nd>0.0001):
    print "ctr,ls,ns,psi:",ctr,ls,psi(z)
    z=z+ls*ns
    comp_gH()
    comp_ns()
    #z=z+ls*ns
    ctr+=1

print ctr
