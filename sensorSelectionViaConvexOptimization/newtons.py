#!/usr/bin/python2

import math
import numpy as np
import cvxpy as cvx
import matplotlib as mpl
import matplotlib.pyplot as plt

global m,n,ka,A,z,g,H,zs        # g is gradient, and H is hessian of \psi(z)
global ns,nd,ls,k               # ns is newton step in descent direction. ls is the step size.
global ub,lb,lloc,rlloc           # variables for data and plots
ub=[]
lb=[]
lloc=[]
rlloc=[]
n=20                            # dimension of x, the parameter
m=100                           # Total number of sensors
sgm=1/(math.sqrt(n))   # variance for randomly choosing a1,a2,...,am
ka=0.001                        # Quality of approximation, kappa
#sgm=1

k=20                            # No. of sensors to select

A=sgm*np.random.randn(m,n)

### Solving the problem using cvx
z=cvx.Variable(m)
obj=cvx.Maximize(cvx.log_det((A.T)*cvx.diag(z)*A))
const=[np.ones((1,m))*z==k,0<=z,z<=1]
prob=cvx.Problem(obj,const)

result=prob.solve()
#print "cvx method:",result
###

### Solving the problem using newton's method
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

#k=30                            # No. of sesors to select from m sensors
def solve():
    global m,n,ka,ls,k
    global z,ns,nd,g,H,zs,A
    global ub,lb,lloc,rlloc
    z=(float(k)/m)*np.ones(m)              # Initial guess of feasible z
    comp_gH()
    comp_ns()
    ctr=0
    while(nd>0.0001):
        #print "ctr,ls,ns,psi:",ctr,ls,psi(z)
        z=z+ls*ns
        comp_gH()
        comp_ns()
        ctr+=1
    #print ctr
    #print "newtons method:",psi(z)

    ### Now we have z*. Let's estimate z^ from z*.
    z=z.tolist()
    zs={}
    for i in xrange(m):
        z[i]=(z[i],i)
        zs[i]=z[i]
    z.sort(reverse=True)

    sh=np.zeros((n,n))
    for i in xrange(m):
        sh=sh+(z[i][0])*(A[z[i][1]].reshape(n,1)).dot(A[z[i][1]].reshape(1,n))
    sh=np.linalg.inv(sh)

    ub.append(math.log(np.linalg.det(np.linalg.inv(sh)))+2*m*ka)
    #print "Upper bound, U:",ub[-1]

    sh=np.zeros((n,n))
    for i in xrange(k):
        sh=sh+(A[z[i][1]].reshape(n,1)).dot(A[z[i][1]].reshape(1,n))
    sh=np.linalg.inv(sh)

    lb.append(math.log(np.linalg.det(np.linalg.inv(sh))))
    #print "Lower bound, L:",lb[-1]

    #zh=[]
    #for i in xrange(m):
    #    zh.append(0)
    #for i in z[:k]:
    #    zh[i[1]]=1

    #print z
    #print zh
###

def chk_swp(s,u,sh,mn=0,mx=1):           # Checks if a swap is possible and returns i,j
    global m,n,ka,ls,k
    global z,ns,nd,g,H,zs

    SH=sh
    
    for i in xrange(k):
        for j in xrange(m-k):
            i1=i
            j1=j
            i=s[i][1]
            j=u[j][1]
            S=( ( np.eye(2) + np.vstack([A[i],A[j]]).dot(SH.dot(np.hstack([-A[i].reshape(n,1),A[j].reshape(n,1)]))) ) )
            #if np.linalg.det(S)<0:
            #    print "Hello, There"
            #    print i1,j1
            if s[i1][0]<=mx and mn<=u[j1][0]:
                if np.linalg.det(S)>1:
                    SH=SH-(SH.dot(np.hstack([-A[i].reshape(n,1),A[j].reshape(n,1)]).dot(np.linalg.inv(S).dot(np.vstack([A[i],A[j]]).dot(SH)))))
                    #print "printing i,j:",i,j 
                    #sh=SH
                    if not(np.all(sh==SH)):
                        sh=SH
                    return i1,j1,sh
            i=i1
            j=j1
    if not(np.all(sh==SH)):
        sh=SH
    return -1,-1,sh

def locopt():               # perform local optimization ignoring order
    #print "################# Starting loc"
    global m,n,ka,ls,k
    global z,ns,nd,g,H,zs
    global ub,l,lloc,rlloc

    s=z[:k]
    u=z[k:]
    u.reverse()

    SH=np.zeros((n,n))
    for i in xrange(k):
        SH=SH+(A[z[i][1]].reshape(n,1)).dot(A[z[i][1]].reshape(1,n))
    SH=np.linalg.inv(SH)

    sh=np.zeros((n,n))
    for i in xrange(k):
        sh=sh+(A[z[i][1]].reshape(n,1)).dot(A[z[i][1]].reshape(1,n))
    sh=np.linalg.inv(sh)

    i,j,sh=chk_swp(s,u,sh)

    ctr=1
    while(i!=-1):
        #print "swap in progress"
        t=u[j]
        u[j]=s[i]
        s[i]=t
        #print "swaps:",ctr
        ctr+=1
        i,j,sh=chk_swp(s,u,sh)

    lloc.append(math.log(np.linalg.det(np.linalg.inv(sh))))
    #print "Local optimization lower bound, Lloc:",lloc[-1]
    #print "local swaps:",ctr


def rlocopt():               # perform restricted local optimization ignoring order
    #print("################# Starting rloc")
    global m,n,ka,ls,k
    global z,ns,nd,g,H,zs
    global ub,l,lloc,rlloc

    s=z[:k]
    s.reverse()
    u=z[k:]

    
    sh=np.zeros((n,n))
    for i in xrange(k):
        sh=sh+(A[z[i][1]].reshape(n,1)).dot(A[z[i][1]].reshape(1,n))
    sh=np.linalg.inv(sh)

    mn=0.1                  # restriction intervals
    mx=0.9

    i,j,sh=chk_swp(s,u,sh,mn,mx)

    ctr=1
    while(i!=-1):
        #print "swap in progress"
        t=u[j]
        u[j]=s[i]
        s[i]=t
        #print "swaps:",ctr
        ctr+=1
        i,j,sh=chk_swp(s,u,sh,mn,mx)

    rlloc.append(math.log(np.linalg.det(np.linalg.inv(sh))))
    #print "restricted local optimization lower bound, Rlloc:",rlloc[-1]
    #print "rlocal swaps:",ctr


### main
k=20
while(k<=40):
    solve()
    locopt()
    rlocopt()
    print k
    k+=1
    #print ub
    #print 
    #print lb
    #print
    #print lloc
    #print
    #print rlloc
###

### Plots

x=np.linspace(20,40,21)
fig1=plt.figure()
plt.plot(x,ub,label="Upper bound")
plt.plot(x,lb,label="Lower bound")
plt.plot(x,lloc,label="Basic local optimization")
plt.plot(x,rlloc,label="Restricted local optimization")
plt.legend()
fig1.show()

for i in xrange(21):
    lb[i]=ub[i]-lb[i]
    lloc[i]=ub[i]-lloc[i]
    rlloc[i]=ub[i]-rlloc[i]
fig2=plt.figure()
plt.plot(x,lb,label="Basic delta")
plt.plot(x,lloc,label="Basic local optimization")
plt.plot(x,rlloc,label="Restricted local optimization")
plt.legend()
fig2.show()



raw_input()
###
