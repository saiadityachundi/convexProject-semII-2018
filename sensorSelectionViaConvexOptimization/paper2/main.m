% matlab code for paper2

global n
global m
global T
global A
global W
global Zp
global N
global K
global lmd
global rho
global mu
global nl
global kl

n=10;
m=30;
T=10;
A=randn(m,n);
rho=0.6;
mu=rho*log(det((A')*eye(m)*A));
lmd=10;
nl=0;
kl=0;

W=ones(T,m);
Zp=zeros(T,m);
N=zeros(T,m);
K=zeros(T,m);

% N(1,1)=1;
% K(2,2)=1;
% nl=1;
% kl=1;

%Z=cvxsolve()
%solve();
%{
Z1=Zp;
rho=0.5;
solve();
Z2=Zp;
rho=0.9;
solve();
Z3=Zp;
%}


Zdata1=zeros(T,m,9);
rho=0;
for i=1:9
    rho=rho+0.1;
    N=zeros(T,m);
    K=zeros(T,m);
    solve();
    Zdata1(:,:,i)=Zp;
    %figy(i)=K;
end

lmd=20;
Zdata2=zeros(T,m,9);
rho=0;
for i=1:9
    rho=rho+0.1;
    N=zeros(T,m);
    K=zeros(T,m);
    solve();
    Zdata2(:,:,i)=Zp;
    %figy(i)=K;
end

%{
lmd=10;
Zdata3=zeros(T,m,9);
rho=0;
for i=1:9
    rho=rho+0.1;
    N=zeros(T,m);
    K=zeros(T,m);
    solve();
    Zdata3(:,:,i)=Zp;
    %figy(i)=K;
end
%}
lmd=10;
rho=0.6;
solve();

function solve()
global n
global m
global T
global A
global W
global Zp
global N
global K
global lmd
global rho
global mu
global nl
global kl

eps=0.0000001;
W=ones(T,m);
Zp=zeros(T,m);
N=zeros(T,m);
nl=0;
K=zeros(T,m);
kl=0;
mu=rho*log(det((A')*eye(m)*A));

ctr=0;
while nl+kl<m*T
    Zr=cvxsolve();
    cr=norm(Zr-Zp,'fro')^2;
    if cr<0.1
        
        mx=-inf;
        for i=1:T
            for j=1:m
                if (K(i,j)~=1) && Zr(i,j)>mx
                    mx=Zr(i,j);
                    a=i;
                    b=j;
                end
            end
        end
        K(a,b)=1;
        kl=kl+1;
        Zr(a,b)=1;
        

        %cr=norm(Zr-Zp,'fro')^2;
        %if cr<0.001
        
    end
    
        for i=1:T
            for j=1:m
                if Zr(i,j)<=eps
                    if N(i,j)~=1
                        N(i,j)=1;
                        nl=nl+1;
                        Zr(i,j)=0;
                    end
                end
                if Zr(i,j)>=1-eps
                    if K(i,j)~=1
                        K(i,j)=1;
                        kl=kl+1;
                        Zr(i,j)=1;
                    end
                end
            end
        end
    
    Zp=Zr;
    W=1-Zr;
    Zp
    lmd
    rho
    mu
    ctr
    cr
    nl
    kl
    nl+kl
    ctr=ctr+1;
end
end

function Zrn=cvxsolve()
global n
global m
global T
global A
global W
global Zp
global N
global K
global lmd
global rho
global mu
global nl
global kl

cvx_begin
%if nl+kl>(m*T/2)
    cvx_precision low
    %cvx_precision medium
%end
%variable Z(T,m);
variable V(m*T-(nl+kl));
expression Z(T,m);
ctr=1;
for i=1:T
    for j=1:m
        if K(i,j)==1
            Z(i,j)=1;
        elseif N(i,j)==1
            Z(i,j)=0;
        else
            %Z(i,j)=variable(1);
            Z(i,j)=V(ctr);
            ctr=ctr+1;
        end
    end
end
%for i=1:T
%    for j=1:m
%        if N(i,j)==1
%            Z(i,j)=0;
%        end
%        if K(i,j)==1
%            Z(i,j)=1;
%        end
%    end
%end

minimize ( trace(W*Z') + lmd*max(sum(Z)) );
subject to
sum(Z)>=1;
0<=Z;
Z<=1;

for i=1:T
    mc=(A')*diag(Z(i,:))*A;
    %             if all(N(i,:)+K(i,:)==1)
    %                 mc=(A')*diag(K(i,:))*A;
    %                 mc=nearestSPD(mc);
    %             end
    if not(all(N(i,:)+K(i,:)==1))
    %if det((A')*diag(Zp(i,:))*A)
    %    i_am_zero=0;
    %end
        log_det(mc)>=mu;
    end
end

%         for i=1:T
%             for j=1:m
%                 if N(i,j)==1
%                     Z(i,j)==0;
%                 end
%                 if K(i,j)==1
%                     Z(i,j)==1;
%                 end
%             end
%         end

cvx_end
Zrn=Z
%     for i=1:T
%         for j=1:m
%             Zrn(i,j)=Z(i,j)
%         end
%     end
end



%{
function Ahat = nearestSPD(A)
% nearestSPD - the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
% usage: Ahat = nearestSPD(A)
%
% From Higham: "The nearest symmetric positive semidefinite matrix in the
% Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
% where H is the symmetric polar factor of B=(A + A')/2."
%
% http://www.sciencedirect.com/science/article/pii/0024379588902236
%
% arguments: (input)
%  A - square matrix, which will be converted to the nearest Symmetric
%    Positive Definite Matrix.
%
% Arguments: (output)
%  Ahat - The matrix chosen as the nearest SPD matrix to A.

if nargin ~= 1
error('Exactly one argument must be provided.')
end

% test for a square matrix A
[r,c] = size(A);
if r ~= c
error('A must be a square matrix.')
elseif (r == 1) && (A <= 0)
% A was scalar and non-positive, so just return eps
Ahat = eps;
return
end

% symmetrize A into B
B = (A + A')/2;

% Compute the symmetric polar factor of B. Call it H.
% Clearly H is itself SPD.
[U,Sigma,V] = svd(B);
H = V*Sigma*V';

% get Ahat in the above formula
Ahat = (B+H)/2;

% ensure symmetry
Ahat = (Ahat + Ahat')/2;

% test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
p = 1;
k = 0;
while p ~= 0
[R,p] = chol(Ahat);
k = k + 1;
if p ~= 0
% Ahat failed the chol test. It must have been just a hair off,
% due to floating point trash, so it is simplest now just to
% tweak by adding a tiny multiple of an identity matrix.
mineig = min(eig(Ahat));
Ahat = Ahat + (-mineig*k.^2 + eps(mineig))*eye(size(A));
end
end
end
%}