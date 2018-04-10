#!/usr/bin/python2

import math
import numpy as np
import cvxpy as cvx
import matplotlib as mlp
import matplotlib.pyplot as plt

global n,m,T,A,W,Zp,N,K,lmd,mu,R,m2v,v2m

# Cvx
def cvxsolve():
    global n,m,T,A,W,Zp,N,K,lmd,mu,R,m2v,v2m
    Z=cvx.Variable(T,m)
    #for v in N.keys():
    #    i,j=v
    #    Z[i,j]=0
    #for v in K.keys():
    #    i,j=v
    #    Z[i,j]=1

    zsm=0
    for i in xrange(T):
        zsm+=Z[i,:]
    obf=lmd*cvx.max_entries(zsm)
    for i in xrange(T):
        obf+=(W[i].reshape(1,m))*(Z[i,:].T)
    obj=cvx.Minimize(obf)
    const=[]
    const+=[
            0<=Z,
            Z<=1,
            zsm>=1
            ]
    for v in N.keys():
        i,j=v
        const+=[Z[i,j]==0]
    for v in K.keys():
        i,j=v
        const+=[Z[i,j]==1]
    for i in xrange(T):
        const+=[cvx.log_det((A.T)*cvx.diag(Z[i,:])*A)>=mu]
    
    prob=cvx.Problem(obj,const)
    result=prob.solve(solver='SCS',verbose=True,eps=1e-1)

    return Z.value

def solve():
    global n,m,T,A,W,Zp,N,K,lmd,mu,R,m2v,v2m
    eps=0.00001
    W=np.ones((T,m))
    Zp=np.zeros((T,m))
    #Zp=np.random.random()
    N={}
    K={}

    ctr=0
    while(len(N)+len(K)<m*T):
        Z=np.array(cvxsolve())
        for i in xrange(T):
            for j in xrange(m):
                if K.has_key((i,j)):
                    Z[i,j]=1
                elif N.has_key((i,j)):
                    Z[i,j]=0
                elif Z[i,j]<=eps:
                    Z[i,j]=0
                    #if not N.has_key((i,j)):
                    N[(i,j)]=-1
                elif Z[i,j]>=1-eps:
                    Z[i,j]=1
                    #if not K.has_key((i,j)):
                    K[(i,j)]=-1
        cr=((Z-Zp)**2).sum()
        if cr<0.1:
            mx=-float('inf')
            mx2=-float('inf')
            for i in xrange(T):
                for j in xrange(m):
                    if (not(K.has_key((i,j)) or N.has_key((i,j)))) and Z[i,j]>mx:
                        mx=Z[i,j]
                        a,b=i,j
            K[(a,b)]=-1
            print "new addition to list:",Z[(a,b)]
            Z[a,b]=1
        Zp=Z
        #W[i]=np.ones(m)-Zp[i,:].reshape(m,)
        W=1-Z
        print "ctr:",ctr
        #print Z
        print "cr:",cr
        print "N:",len(N)
        print "K:",len(K)
        print "N+K",len(N)+len(K)
        ctr+=1

n=30
m=100
T=10

A=np.random.randn(m,n)
rho=0.1
mu=rho*math.log(np.linalg.det(A.T.dot(np.eye(m).dot(A))))
lmd=1

W=np.ones((T,m))
Zp=np.zeros((T,m))
N={}
K={}

#N[(1,1)]=-1
#K[(2,2)]=-1

solve()
#Z=cvxsolve()
