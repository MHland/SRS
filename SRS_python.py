#!/usr/bin/env python
# encoding: utf-8
"""
@name: SRS algorithm
@author: Wei Haoshan
@Time: 2022/1/17
@Update: 2022/11/19
@dependencies: numpy
"""
# Refer to paper:

import random
import numpy as np
import time

class SRS:
    def __init__(self,n,bl,bu,p=3,sp=None,deps=12,delta=0.01,Vectorization=False,\
                num=None,MAX=False,OptimalValue=None,ObjectiveLimit=None,eps=4,\
                ShortLambda=0.02,LongLambda=0.2,InitialLt=3,Lt=2):
        '''
        n:  The dimension of the objective function.
        bl: The lower bound of the parameter to be determined.
        bu: The upper bound of the parameter to be determined
        *args (Contains mainly adjustable parameters):
        | name          | type    | defult      | describe 
        | p             | int     | 3           | p is the key parameter, and the value is generally 3-20, 
        |               |         |             | which needs to be given according to the specific situation
        | sp            | int     | sp=p(p<=5)  | Its range is [3, p]
        |               |         | sp=5(5<p<12)|
        |               |         | sp=12(p<=12)|
        | deps          | float   | 10           | Its range is (0, +infty),
        |               |         |             | It is a key parameter for adjusting the precision,
        |               |         |             | The larger the value, the higher the precision and the longer the time
        | delta         | float   | 0.01         | Its range is (0, 0.5),
        |               |         |             | It is a key parameter for adjusting the precision,
        |               |         |             | the larger the value, the higher the precision and the longer the time
        | Vectorization | bool    | False       | Whether the objective function satisfies the vectorization condition
        | num           | int     | 2000        | if Vectorization=True: num=2000 else: num=20000 (defult).
        |               |         | 20000       | The key parameter, representing the maximum number of 
        |               |         |             | times the target function is called. When testing, the accuracy
        |               |         |             | can be improved by increasing num.
        | MAX           | bool    | False       | Whether to find the maximum value of the objective function.
        | OptimalValue  | float   | None        | The optimal value of the objective function.
        | ObjectiveLimit| float   | None        | When the optimal value is known, the algorithm terminates
        |               |         |             | within ObjectiveLimit of the optimal value.
        | eps           | float   | 4           | Its range is (0, +infty), 
        |               |         |             | it is not critical, and adjustment is not recommended.
        | ShortLambda   | float   | 0.02        | Its range is (0, 0.1), 
        |               |         |             | it is not critical, and adjustment is not recommended.
        | LongLambda    | float   | 0.2         | Its range is (0.1, 1), 
        |               |         |             | it is not critical, and adjustment is not recommended.
        | InitialLt     | int     | 3           | Its range is (0, 10), 
        |               |         |             | it is not critical, and adjustment is not recommended.
        | Lt            | int     | 2           | Its range is (0, 10), 
        |               |         |             | it is not critical, and adjustment is not recommended.
        '''
        blp,bup = bl,bu
        bl,bu = np.zeros((n,1)),np.zeros((n,1))
        for i in range(n):
            bl[i,0],bu[i,0] = blp[i],bup[i]
        if Vectorization:
            if num is None:
                num = 2000
            else:
                num = num
        else:
            if num is None:
                num = 20000
            else:
                num = num
        # set p
        if p < 3:
            p = 3
        # set sp
        if sp != None:
            if sp > p:
                sp = p
            elif sp < 3:
                sp = 3
        else:
            if p < 5:
                sp = p
            elif p < 12:
                sp = 5
            else:
                sp = 12
        # set delta
        if delta >= 0.5 or delta <= 0:
            delta = 0.3
        if ObjectiveLimit != None:
            OLindex = True
        else:
            OLindex = False
        # set bound
        bln, bun = np.ones((n,1)), np.ones((n,1))
        for i in range(n):
            bln[i,0], bun[i,0] = bl[i], bu[i]
        bl ,bu = bln, bun
        self.n = n
        self.bl = bl
        self.bu = bu
        self.p = p
        self.sp = sp
        self.deps = deps
        self.delta = delta
        self.eps = eps
        self.num = num
        self.OptimalValue = OptimalValue
        self.ShortLambda = ShortLambda
        self.LongLambda = LongLambda
        self.InitialLt = InitialLt
        self.Lt = Lt
        self.MAX = MAX
        self.Vectorization = Vectorization
        self.OLindex = OLindex
        if self.OLindex:
            self.ObjectiveLimit = ObjectiveLimit
        if self.MAX:
            A = -1
        else:
            A = 1
        self.A = A
        self.s = 0
        self.fe = 0
        self.feS = 0
        self.MM = 0
        self.EachParFE = np.zeros((n,num),dtype=np.int64)
        self.BestValueFE = np.zeros((num,1))
        self.EachParFE = np.empty(shape=[n,0])
        self.BestValueFE = np.empty(shape=[0,1])
    
    def __fitness(self,func,*args):
        return self.A * func(self.x,*args)
    
    def __calculate_function_all(self,x,func,*args):
        if self.Vectorization:
            self.x = x.T
            y = self.__fitness(func,*args)
            y = np.array(y)
            IndexFE = np.argmin(y)
            self.BestValueFE = np.vstack((self.BestValueFE,y[IndexFE]))
            self.EachParFE = np.hstack((self.EachParFE,x[:,IndexFE:IndexFE+1]))
            self.fe = self.fe+1
            self.feS = self.feS+self.MM
        else:
            y = np.zeros((self.MM,1))
            for iii in range(self.MM):
                self.x = x[:,iii:iii+1].T
                y[iii,0] = self.__fitness(func,*args)
                self.fe = self.fe+1
            min_y = np.min(y)
            IndexFE = np.argmin(y)
            self.BestValueFE = np.vstack((self.BestValueFE,min_y*np.ones((self.MM,1))))
            self.EachParFE = np.hstack((self.EachParFE,x[:,IndexFE:IndexFE+1]*np.ones((self.n,self.MM))))
        return y

    def __calculate_function(self,x,func,*args):
        if self.Vectorization:
            self.x = x.T
            y = self.__fitness(func,*args)
            y = np.array(y)
            self.fe = self.fe+1
            self.feS = self.feS+self.MM
        else:
            y = np.zeros((self.MM,1))
            for iii in range(self.MM):
                self.x = x[:,iii:iii+1].T
                y[iii,0] = self.__fitness(func,*args)
                self.fe = self.fe+1
        return y
    
    def __calculate_function_try(self,x,func,*args):
        if self.Vectorization:
            self.x = x.T
            y = self.__fitness(func,*args)
            y = np.array(y)
        else:
            y = np.zeros((self.MM,1))
            for iii in range(self.MM):
                self.x = x[:,iii:iii+1].T
                y[iii,0] = self.__fitness(func,*args)
        return y
    
    def __calculate_FE(self):
        if self.Vectorization:
            self.BestValueFE = np.vstack((self.BestValueFE,np.min(self.BestValue[self.s,:])))
            self.EachParFE = np.hstack((self.EachParFE,self.EachPar[:,self.s:self.s+1]))
        else:
            self.EachParFE = np.hstack((self.EachParFE,self.EachPar[:,self.s:self.s+1]*np.ones((self.n,self.MM))))
            self.BestValueFE = np.vstack((self.BestValueFE,np.min(self.BestValue[self.s,:])*np.ones((self.MM,1))))
    
    def __select_points(self,Xp,Xb,x,j,p1,pp):
        ra1 = [i for i in range(p1)]
        random.shuffle(ra1)
        ra1 = np.array(ra1,dtype=np.int64)
        ra = [np.mod(j+1,p1),np.mod(j,p1)]
        xx = np.zeros((self.n,1))
        xx[:,0] = np.min(np.append(Xp[:,j:j+1]-self.bl,self.bu-Xp[:,j:j+1],axis=1),axis=1)/4
        xxx = np.random.randn(self.n,pp-9)*xx
        x[:,j*pp:(j+1)*pp-9] = Xp[:,j:j+1]*np.ones((1,pp-9))+xxx

        x[:,(j+1)*pp-9] = Xp[:,ra1[2]]-Xp[:,ra1[0]]+Xp[:,ra1[1]]
        x[:,(j+1)*pp-8] = (2*Xp[:,ra1[0]]-Xp[:,ra1[2]]-Xp[:,ra1[1]])/2

        x[:,(j+1)*pp-7] = Xb[:,0] - (Xp[:,ra[1]]+Xp[:,ra[0]])/2
        x[:,(j+1)*pp-6] = Xp[:,ra[1]] - Xb[:,0] + Xp[:,ra[0]]
        x[:,(j+1)*pp-5] = x[:,(j+1)*pp-6]-(Xp[:,ra[1]]-2*Xb[:,0]+Xp[:,ra[0]])/2
        x[:,(j+1)*pp-4] = Xb[:,0] + (Xp[:,ra[1]]-2*Xb[:,0]+Xp[:,ra[0]])/4

        x[:,(j+1)*pp-3] = (Xp[:,j]+Xb[:,0])/2
        x[:,(j+1)*pp-2] = 2*Xb[:,0]-Xp[:,j]
        x[:,(j+1)*pp-1] = 2*Xp[:,j]-Xb[:,0]
        return x

    def SRS_run(self,func,*args):
        FE = []
        iter_best_global = []
        T1 = time.time()
        n = self.n
        p = self.p
        p1 = self.sp
        n1 = 3*n+3
        m1 = int(max(np.floor(n1*p/p1)+1,9))
        num = self.num
        eps = self.eps
        popsize = m1*p1*np.ones((n,1),dtype=np.int64)
        psize = popsize/p1
        psize = np.array(psize,dtype=np.int64)
        BL = np.copy(self.bl)
        BU = np.copy(self.bu)
        Mbounds = BU-BL
        M = Mbounds
        

        k = (self.bu-self.bl)/(psize-1)
        
        self.MM = 1
        x = (BU+BL)/2+(BU-BL)*(np.random.uniform(-1,1,size=(n,self.MM)))/2
        try:
            y = self.__calculate_function_try(x,func,*args)
        except (OSError,TypeError,NameError,IndexError) as reason:
            print('Check objective function!')
            print('Error:%s'%str(reason))
            if IndexError:
                print('Pay attention, each variable (x) of the objective function is a column vector, '+\
                'that\'s mean,the 1st variable = x[:,0:1] or x[0,0], the 2nd variable = x[:,1:2] or x[0,1],'+\
                '..., the n-th variable = x[:,n-1:n] or x[0,n-1]. '
                'if it is a vectorized objective function, the i-th variable = x[:,i-1:i], '+\
                'otherwise the i-th variable = x[0,i-1], and please never use x[i] here.')

        Index = 0
        self.MM = m1*p
        x = (BU+BL)/2+(BU-BL)*(np.random.uniform(-1,1,size=(n,self.MM)))/2
        y = self.__calculate_function_all(x,func,*args)
        indexY = np.argsort(y,axis=0)
        yps = y[indexY[:,0]]
        indexY = indexY.T
        yps = yps[0:p,:]

        Xp = x[:,indexY[0,0:p]]
        Xb = np.zeros((self.n,1))
        Xb[:,0] = x[:,indexY[0,len(indexY.T)-1]]
        self.EachPar = np.zeros((n,num))
        self.BestValue = np.zeros((num,p))
        self.BY = np.array([])
        neps = eps
        sss = 0
        while 1:
            if self.eps>neps+2:
                lam = self.ShortLambda
                lt = self.Lt
            else:
                lam = self.LongLambda
                lt = self.Lt
            if sss == 0:
                lt = self.InitialLt
            x = np.zeros((n,n1*p))
            Bb = BL*np.ones((1,n1))
            Be = BU*np.ones((1,n1))
            for i in range(p):
                r1 = np.random.randn(self.n,self.n)
                r1[r1>0],r1[r1<=0]=1,-1
                xx1 = Xp[:,i:i+1]*np.ones((1,n))+lam*Mbounds*np.eye(n)
                xx2 = Xp[:,i:i+1]*np.ones((1,n))-lam*Mbounds*np.eye(n)
                xx3 = Xp[:,i:i+1]*np.ones((1,n))-lam*Mbounds*r1
                xb1 = (Xp[:,i:i+1]+Xb)/2
                xb2 = 2*Xb-Xp[:,i:i+1]
                xb3 = 2*Xp[:,i:i+1]-Xb
                xx = np.concatenate((xx1,xx2,xx3,xb1,xb2,xb3),axis=1)
                xx[xx<Bb] = Bb[xx<Bb]
                xx[xx>Be] = Be[xx>Be]
                x[:,i:n1*p:p] = xx
            self.MM = n1*p
            y = self.__calculate_function(x,func,*args)
            for i in range(p):
                yp = y[i:n1*p:p]
                yp = np.append(yp,yps[i:i+1,:],axis=0)
                indexY = np.argmin(yp)
                yps[i,:] = np.copy(yp[indexY])
                xp = np.append(x[:,i:n1*p:p],Xp[:,i:i+1],axis=1)
                Xp[:,i] = xp[:,indexY]
            indexYb = np.argmax(y)
            Xb[:,0] = x[:,indexYb]
            FE.append(self.fe)
            iter_best_global.append(min(y)[0])
            
            s = self.s
            Index = Index+1
            self.BestValue[s,:] = yps[:,0]
            self.BY = np.append(self.BY,np.min(yps))
            indeX = np.argsort(yps,axis=0)
            yps = yps[indeX[:,0]]
            self.EachPar[:,s:s+1] = Xp[:,indeX[0]]
            self.__calculate_FE()
            self.s = self.s+1
            if self.OLindex and self.OptimalValue!=None:
                if np.abs(self.OptimalValue-np.min(self.BestValue[self.s-1,:]))<np.abs(self.ObjectiveLimit):
                    break
            if Index>lt:
                if np.abs(np.min(np.log10(Mbounds/M)))>self.deps or self.fe>self.num:
                    break
                ineed = np.abs(np.min(self.BY[self.s-lt])-np.min(self.BY[self.s-1]))
                if np.abs(np.log10(np.max([ineed,10**(-self.eps-1)])))>=self.eps:
                    sss = 1
                    bb,be = np.zeros((self.n,1)),np.zeros((self.n,1))
                    bb[:,0] = np.min(Xp,axis=1)
                    be[:,0] = np.max(Xp,axis=1)
                    bb[:,0] = np.min(np.append(self.bl,bb-k,axis=1),axis=1)
                    be[:,0] = np.max(np.append(self.bu,be+k,axis=1),axis=1)
                    self.bl[:,0] = np.max(np.append(bb,BL,axis=1),axis=1)
                    self.bu[:,0] = np.min(np.append(be,BU,axis=1),axis=1)
                    k = (self.bu-self.bl)/(psize-1)
                    Mbounds = self.bu-self.bl
                    self.MM = m1*p1
                    x = np.zeros((self.n,self.MM))
                    Xp1 = Xp[:,indeX[0:p1,0]]

                    BestX = Xp1
                    for i in range(p1):
                        x[:,i*np.max(psize):(i+1)*np.max(psize)] = Xp1[:,i:i+1]*np.ones((1,np.max(psize)))
                    Pi = np.zeros((self.n,p1))
                    l = [i for i in range(self.n)]
                    for i in range(p1):
                        random.shuffle(l)
                        Pi[:,i] = l
                    Pi = np.array(Pi,dtype=np.int64)
                    LL = np.zeros((self.n,np.max(psize)))
                    for i in range(self.n):
                        LL[i,:] = [self.bl[i,0]+k[i,0]*j for j in range(np.max(psize))]
                    Index1 = 0
                    BestY = np.zeros((self.n+1,p1))
                    BestY[0,:] = yps[0:p1].T
                    BX = self.EachPar[:,s]
                    for i in range(self.n):
                        for j in range(p1):
                            xneed = LL[Pi[i,j],0:np.max(psize)-1]+k[Pi[i,j],0]*np.random.uniform(0,1,size=(1,psize[Pi[i,j],0]-1))
                            x[Pi[i,j],j*psize[Pi[i,j],0]+1:(j+1)*psize[Pi[i,j],0]] = xneed
                            x[Pi[i,j],j*psize[Pi[i,j],0]] = x[Pi[i,j],0]+k[Pi[i,j],0]*(np.random.rand(1)[0]*2-1)
                            if x[Pi[i,j],j*psize[Pi[i,j],0]]<self.bl[Pi[i,j],0]:
                                x[Pi[i,j],j*psize[Pi[i,j],0]] = self.bl[Pi[i,j],0]+k[Pi[i,j],0]*(np.random.rand(1)[0])
                            elif x[Pi[i,j],j*psize[Pi[i,j],0]]>self.bu[Pi[i,j],0]:
                                x[Pi[i,j],j*psize[Pi[i,j],0]] = self.bu[Pi[i,j],0]-k[Pi[i,j],0]*(np.random.rand(1)[0])
                        y = self.__calculate_function(x,func,*args)
                        Index1 = Index1+1
                        for j in range(p1):
                            index = np.argmin(y[j*psize[Pi[i,j],0]:(j+1)*psize[Pi[i,j],0]])
                            nash = y[j*psize[Pi[i,j],0]+index]
                            BestY[Index1,j] = nash
                            x[Pi[i,j],j*psize[Pi[i,j],0]:(j+1)*psize[Pi[i,j],0]] = x[Pi[i,j],j*psize[Pi[i,j],0]+index]*np.ones((1,np.max(psize)))
                            if nash == np.min(BestY[0:Index1+1,j]):
                                BestX[:,j] = x[:,j*psize[Pi[i,j],0]+index]
                            B1 = np.min(BestY[0:Index1,:])
                            B2 = np.min(BestY[Index1,0:j+1])
                            if nash == np.min([B1,B2]):
                                BX = x[:,j*psize[Pi[i,j],0]+index]
                    s = self.s
                    self.BestValue[s,0:p1] = np.min(BestY[Index1-self.n:Index1+1,:],axis=0)
                    self.BY = np.append(self.BY,np.min(self.BestValue[s,0:p1]))
                    self.EachPar[:,s] = BX
                    self.__calculate_FE()
                    FE.append(self.fe)
                    iter_best_global.append(min(self.BestValue[s, 0:p1]))
                    
                    Xp[:,0:p1] = BestX

                    bb[:,0] = np.min(Xp,axis=1)
                    be[:,0] = np.max(Xp,axis=1)
                    self.bl[:,0] = np.max(np.append(self.bl,bb-Mbounds*self.delta,axis=1),axis=1)
                    self.bu[:,0] = np.min(np.append(self.bu,be+Mbounds*self.delta,axis=1),axis=1)
                    k = (self.bu-self.bl)/(psize-1)

                    x = np.zeros((self.n,self.MM))

                    for j in range(p1):
                        x = self.__select_points(Xp,Xb,x,j,p1,m1)
                    
                    N = (BU-BL)*np.random.rand(self.n,self.MM)+BL
                    x[x<BL] = N[x<BL]
                    x[x>BU] = N[x>BU]
                    y = self.__calculate_function(x,func,*args)
                    yps[0:p1,0] = np.min(BestY,axis=0)
                    y = np.append(y,yps,axis=0)
                    x = np.append(x,Xp,axis=1)
                    indexY = np.argsort(y,axis=0)
                    yps = y[indexY,0]
                    yps = np.copy(yps[0:p,:])
                    xneed = np.abs(yps[0,0]-self.BY[s])
                    if np.abs(np.log10(np.max([xneed,10**(-self.eps-1)])))>=self.eps:
                        self.eps = self.eps+1
                    
                    Xp = x[:,indexY[0:p,0]]
                    Xb[:,0] = x[:,indexY[len(indexY)-1,0]]
                    heihei1 = np.zeros((self.n,1))*False
                    heihei2 = np.zeros((self.n,1))*False
                    nx1 = np.copy(BU)
                    nx2 = np.copy(BL)
                    for i in range(p):
                        nx = Xp[:,i:i+1]
                        heihei1 = np.logical_or(heihei1,nx>=BU)
                        heihei2 = np.logical_or(heihei2,nx<=BL)
                        nx1[heihei1[:,0],0] = np.min(np.append(nx[heihei1[:,0]],nx1[heihei1[:,0]],axis=1),axis=1)
                        nx2[heihei2[:,0],0] = np.max(np.append(nx[heihei2[:,0]],nx2[heihei2[:,0]],axis=1),axis=1)
                    self.bu[heihei1[:,0],0] = np.min(np.append(nx1[heihei1[:,0]]+k[heihei1[:,0]],BU[heihei1[:,0]],axis=1),axis=1)
                    self.bl[heihei2[:,0],0] = np.max(np.append(nx2[heihei2[:,0]]-k[heihei2[:,0]],BL[heihei2[:,0]],axis=1),axis=1)
                    self.__calculate_FE()
                    self.s = self.s+1
        T2 = time.time()
        Result = []
        ResultName = []
        self.s = self.s-1
        Result.append(self.A*self.BY[self.s])
        ResultName.append('Best target function value')
        Result.append(self.EachPar[:,s])
        ResultName.append('Best parameter value')
        Result.append(self.s+1)
        ResultName.append('Generation')
        Result.append(self.fe)
        ResultName.append('Function evaluations')
        if self.Vectorization:
            Result.append(self.feS)
            ResultName.append('Function evaluations (scalar)')
        if self.OptimalValue != None:
            Result.append(np.abs(self.OptimalValue-Result[0]))
            ResultName.append('Absolute error')
        elif self.OLindex and self.OptimalValue == []:
            Result.append(np.abs(0-Result[0]))
            ResultName.append('Absolute error (note: the default optimal value is 0)')
        # Result.append(self.A*np.min(self.BestValue[0:self.s,:],axis=1))
        # ResultName.append('Optimal value')
        Result.append(T2-T1)
        ResultName.append('Time (s)')
        self.result = Result
        self.result_name = ResultName
        FE.append(self.fe+1)
        iter_best_global.append(self.A*self.BY[self.s])
        return Result[0], Result[1], iter_best_global, FE
    
    def type_result(self):
        ResultName = self.result_name
        Result = self.result
        for i in range(len(ResultName)):
            print(ResultName[i] + ': '+ str(Result[i]))
    
    def get_result(self):
        return self.result_name, self.result

    def Booth(x):
        x1 = x[:,0:1]
        x2 = x[:,1:2]
        y = (x1+2*x2-5)**2+(2*x1+x2-7)**2
        return y

    def GoldsteinPrice(x):
        x1 = x[:,0:1]
        x2 = x[:,1:2]
        y = (1+((x1+x2+1)**2)*(19-14*x1+3*x1**2-14*x2+6*x1*x2+3*x2**2))\
            *(30+(2*x1-3*x2)**2*(18-32*x1+12*x1**2+48*x2-36*x1*x2+27*x2**2))
        return y
    
    def Schaffer(x):
        x1 = x[:,0:1]
        x2 = x[:,1:2]
        y = 0.5+((np.sin((x1**2+x2**2)**0.5))**2-0.5)/(1+0.001*(x1**2+x2**2))**2
        return y
    
    def Tripod(x):
        x1 = x[:,0:1]
        x2 = x[:,1:2]
        p1 = np.ones(x1.shape)
        p2 = np.ones(x2.shape)
        p1[x1<0] = 0
        p2[x2<0] = 0
        y = p2*(1+p1)+np.abs(x1+50*p2*(1-2*p1))+np.abs(x2+50*(1-2*p2))
        return y
    
    def UrsemWaves(x):
        x1 = x[:,0:1]
        x2 = x[:,1:2]
        y = -(0.3*x1)**3+(x2**2-4.5*x2**2)*x1*x2+4.7*np.cos(3*x1-x2**2*(2+x1))*np.sin(2.5*np.pi*x1)
        return y

    def Wolfe(x):
        x1 = x[:,0:1]
        x2 = x[:,1:2]
        x3 = x[:,2:3]
        y = 4/3*((x1**2+x2**2-x1*x2)**0.75)+x3
        return y
    
    def Rastrigin(x,A):
        y = np.zeros((x.shape[0],1))
        y[:,0] = A*10 + np.sum(x**2 - A*np.cos(2*np.pi*x),axis=1)
        return y
    
    def Rosenbrock(x):
        y = 0
        for i in range(x.shape[1]-1):
            y = y + 100*(x[:,i+1:i+2]-x[:,i:i+1]**2)**2+(1-x[:,i:i+1])**2
        return y
    
    def Schwefel(x):
        y1 = 0
        y2 = 0
        for i in range(x.shape[1]):
            y1 = y1+x[:,i:i+1]**2
            y2 = y2+x[:,i:i+1]*i
        y = y1+(0.5*y2)**2+(0.5*y2)**4
        return y

    def Zakharov(x):
        y1 = 0
        y2 = 0
        for i in range(x.shape[1]):
            y1 = y1 + x[:,i:i+1]**2
            y2 = y2 + x[:,i:i+1]*(i+1)
        y = y1 + (0.5*y2)**2 + (0.5*y2)**4
        return y

if __name__=='__main__':
    print('*********************************************')
    print('Test function 1: Booth')
    n = 2
    bl = [-100,-100]
    bu = [100,100]
    srs = SRS(n,bl,bu,MAX=False,Vectorization=False)
    srs.SRS_run(SRS.Booth)
    srs.type_result()
    print('*********************************************')

    print('Test function 2: GoldsteinPrice')
    n = 2
    bl = [-10,-10]
    bu = [10,10]
    srs = SRS(n,bl,bu,MAX=False,Vectorization=False,OptimalValue=3)
    srs.SRS_run(SRS.GoldsteinPrice)
    srs.type_result()
    print('*********************************************')

    print('Test function 3: Schaffer')
    n = 2
    bl = [-100,-100]
    bu = [100,100]
    srs = SRS(n,bl,bu,MAX=False,Vectorization=False)
    srs.SRS_run(SRS.Schaffer)
    srs.type_result()
    print('*********************************************')

    print('Test function 4: Tripod')
    n = 2
    bl = [-100,-100]
    bu = [100,100]
    srs = SRS(n,bl,bu,p=10,sp=10,MAX=False,Vectorization=False)
    srs.SRS_run(SRS.Tripod)
    srs.type_result()
    print('*********************************************')

    print('Test function 5: UrsemWaves')
    n = 2
    bl = [-0.9,-1.2]
    bu = [1.2,1.2]
    srs = SRS(n,bl,bu,p=12,sp=3,MAX=False,Vectorization=False,OptimalValue=-7.306999)
    srs.SRS_run(SRS.UrsemWaves)
    srs.type_result()
    print('*********************************************')
    
    print('Test function 6: Wolfe')
    n = 3
    bl = [0,0,0]
    bu = [2,2,2]
    srs = SRS(n,bl,bu,MAX=False,Vectorization=True)
    srs.SRS_run(SRS.Wolfe)
    srs.type_result()
    print('*********************************************')

    print('Test function 7: Rastrigin')
    n = 10
    bl = [-5 for i in range(n)]
    bu = [5 for i in range(n)]
    srs = SRS(n,bl,bu,p=3,sp=3,MAX=False,Vectorization=True)
    srs.SRS_run(SRS.Rastrigin,10)
    srs.type_result()
    print('*********************************************')

    print('Test function 8: Rosenbrock')
    n = 10
    bl = [-10 for i in range(n)]
    bu = [10 for i in range(n)]
    srs = SRS(n,bl,bu,p=3,sp=3,MAX=False,Vectorization=True,num=3000)
    srs.SRS_run(SRS.Rosenbrock)
    srs.type_result()
    print('*********************************************')

    print('Test function 9: Schwefel')
    n = 10
    bl = [-100 for i in range(n)]
    bu = [100 for i in range(n)]
    srs = SRS(n,bl,bu,p=3,sp=3,MAX=False,Vectorization=True)
    srs.SRS_run(SRS.Schwefel)
    srs.type_result()
    print('*********************************************')

    print('Test function 10: Zakharov')
    n = 20
    bl = [-5 for i in range(n)]
    bu = [10 for i in range(n)]
    srs = SRS(n,bl,bu,p=3,sp=3,delta=0.3,deps=5,MAX=False,Vectorization=True,num=1000,OptimalValue=0,ObjectiveLimit=10**-5)
    srs.SRS_run(SRS.Zakharov)
    srs.type_result()
    print('*********************************************')