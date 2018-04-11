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
    #Z=cvx.Variable(T,m)
    Z=np.zeros((T,m))
    V=cvx.Variable(len(R))
    #for v in N.keys():
    #    i,j=v
    #    Z[i,j]=0
    #for v in K.keys():
    #    i,j=v
    #    Z[i,j]=1

    #zsm=np.zeros((m,1))
    zsm=[]
    for i in xrange(m):
        zsm.append(0)
    for i in xrange(T):
        for j in xrange(m):
            if K.has_key((i,j)):
                zsm[j]+=1
            elif R.has_key((i,j)):
                zsm[j]+=V[m2v[(i,j)]]
        #zsm+=Z[i,:]
    obf=lmd*cvx.max_elemwise(zsm)
    for i in xrange(T):
        for j in xrange(m):
            if K.has_key((i,j)):
                obf+=W[i,j]
            elif R.has_key((i,j)):
                obf+=W[i,j]*V[m2v[(i,j)]]

    obj=cvx.Minimize(obf)
    const=[]
    const+=[
            0<=V,
            V<=1,
            ]
    for i in xrange(m):
        if not((type(zsm[i])==int) or (type(zsm[i])==float)):
            const+=[zsm[i]>=1]
    #for v in N.keys():
    #    i,j=v
    #    const+=[Z[i,j]==0]
    #for v in K.keys():
    #    i,j=v
    #    const+=[Z[i,j]==1]
    for i in xrange(T):
        nc=0
        for j in xrange(m):
            if K.has_key((i,j)):
                nc+=A[j,:].reshape(n,1).dot(A[j,:].reshape(1,n))
            elif R.has_key((i,j)):
                nc+=V[m2v[(i,j)]]*(A[j,:].reshape(n,1).dot(A[j,:].reshape(1,n)))
        if not((type(cvx.log_det(nc))==int) or (type(cvx.log_det(nc))==float)):
            const+=[cvx.log_det(nc)>=mu]
        #const+=[cvx.log_det((A.T)*cvx.diag(Z[i,:])*A)>=mu]
    
    prob=cvx.Problem(obj,const)
    result=prob.solve(solver='SCS',verbose=True,eps=1e-1)

    for i in xrange(T):
        for j in xrange(m):
            if K.has_key((i,j)):
                Z[i,j]=1
            elif N.has_key((i,j)):
                Z[i,j]=0
            elif R.has_key((i,j)):
                Z[i,j]=V.value[m2v[(i,j)]]

    #print "len(V):",V.shape
    return Z

def solve():
    global n,m,T,A,W,Zp,N,K,lmd,mu,R,m2v,v2m
    eps=0.00001
    W=np.ones((T,m))
    Zp=np.zeros((T,m))
    #Zp=np.random.random()
    N={}
    K={}
    R={}
    k=0
    for i in xrange(T):
        for j in xrange(m):
            R[(i,j)]=-1

    ctr=0
    while(len(N)+len(K)<m*T+4):

        ###
        k=0
        for i,j in R.keys():
            m2v[(i,j)]=k
            v2m[k]=(i,j)
            k+=1
        ###

        Z=np.array(cvxsolve())
        for i in xrange(T):
            for j in xrange(m):
                if K.has_key((i,j)):
                    Z[i,j]=1
                elif N.has_key((i,j)):
                    Z[i,j]=0
                elif not R.has_key((i,j)):
                    "somethin fishy going"
                elif Z[i,j]<=eps:
                    Z[i,j]=0
                    #if not N.has_key((i,j)):
                    N[(i,j)]=-1
                    del R[(i,j)]
                elif Z[i,j]>=1-eps:
                    Z[i,j]=1
                    #if not K.has_key((i,j)):
                    K[(i,j)]=-1
                    del R[(i,j)]
        cr=((Z-Zp)**2).sum()
        if cr<0.01:
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
            del R[(a,b)]
        Zp=Z
        #W[i]=np.ones(m)-Zp[i,:].reshape(m,)
        W=1-Z
        print "ctr:",ctr
        #print Z
        print "cr:",cr
        print "N:",len(N)
        print "K:",len(K)
        print "R:",len(R)
        print "N+K",len(N)+len(K)
        ctr+=1

n=10
m=30
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

R={}
for i in xrange(T):
    for j in xrange(m):
        R[(i,j)]=-1

m2v={}
v2m={}
k=0
for i,j in R.keys():
    m2v[(i,j)]=k
    v2m[k]=(i,j)
    k+=1

solve()
#Z=cvxsolve()
