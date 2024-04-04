#!/usr/bin/python
#coding:utf-8

import sys
import math
import numpy
import csv
import collections

In_noise_variance = 10 ** -6
W_noise_variance = 10 ** -6


def clem_convert_DT(Dm, Dn, phi, theta):
    Dm_ = Dn-phi-theta-math.pi
    Dn_ = Dn-theta-math.pi
    phi_ = Dm-Dn
    result = collections.namedtuple('result', 'Dm, Dn, phi')
    return result(Dm=Dm_, Dn=Dn_, phi=phi_)



def clem_find_theta_UTM(U):
    MATRIX_SIZE = U.shape[0]
    theta = numpy.zeros(int(MATRIX_SIZE*(MATRIX_SIZE-1)/2))
    phi = numpy.zeros(int(MATRIX_SIZE*(MATRIX_SIZE-1)/2))
    alpha = numpy.zeros(MATRIX_SIZE)

    counter = 0
    for i in range(1, MATRIX_SIZE):
        if i & 1: #奇数の場合（ビット演算）
            for j in range(1,i+1):
                if U[MATRIX_SIZE-j][i-j] == 0:
                    theta[counter] = math.pi
                    phi[counter] = 0 #do not care
                else:
                    theta[counter] = 2 * numpy.arctan(numpy.abs(-U[MATRIX_SIZE-j][i-j+1]/U[MATRIX_SIZE-j][i-j]))
                    phi[counter] = -numpy.angle(-U[MATRIX_SIZE-j][i-j+1]/U[MATRIX_SIZE-j][i-j])

                T = numpy.identity(MATRIX_SIZE, dtype=numpy.complex128)
                T[i-j][i-j] = numpy.exp((theta[counter]+math.pi)/2*1j) * numpy.exp(phi[counter]*1j) * numpy.sin(theta[counter]/2)
                T[i-j][i-j+1] = numpy.exp((theta[counter]+math.pi)/2*1j) * numpy.cos(theta[counter]/2)
                T[i-j+1][i-j] = numpy.exp((theta[counter]+math.pi)/2*1j) * numpy.exp(phi[counter]*1j) * numpy.cos(theta[counter]/2)
                T[i-j+1][i-j+1] = numpy.exp((theta[counter]+math.pi)/2*1j) * -1 * numpy.sin(theta[counter]/2)
                U = numpy.dot(U, numpy.conjugate(T.T)) #ユニタリ共役：numpy.conjugate(A.T)

                counter+=1
                ##print("hogehoge_odd:",i,j,i-j)
        else:
            for j in range(1,i+1):
                if U[MATRIX_SIZE+j-i-1][j-1] == 0:
                    theta[counter] = math.pi
                    phi[counter] = 0 #do not care
                else:
                    theta[counter] = 2 * numpy.arctan(numpy.abs(U[MATRIX_SIZE+j-i-2][j-1]/U[MATRIX_SIZE+j-i-1][j-1]))
                    phi[counter] = - numpy.angle(U[MATRIX_SIZE+j-i-2][j-1]/U[MATRIX_SIZE+j-i-1][j-1])

                T = numpy.identity(MATRIX_SIZE, dtype=numpy.complex128)
                T[MATRIX_SIZE+j-i-2][MATRIX_SIZE+j-i-2] = numpy.exp((theta[counter]+math.pi)/2*1j) * numpy.exp(phi[counter]*1j) * numpy.sin(theta[counter]/2)
                T[MATRIX_SIZE+j-i-2][MATRIX_SIZE+j-i-1] = numpy.exp((theta[counter]+math.pi)/2*1j) * numpy.cos(theta[counter]/2)
                T[MATRIX_SIZE+j-i-1][MATRIX_SIZE+j-i-2] = numpy.exp((theta[counter]+math.pi)/2*1j) * numpy.exp(phi[counter]*1j) * numpy.cos(theta[counter]/2)
                T[MATRIX_SIZE+j-i-1][MATRIX_SIZE+j-i-1] = numpy.exp((theta[counter]+math.pi)/2*1j) * -1 * numpy.sin(theta[counter]/2)
                U = numpy.dot(T, U) 

                counter+=1
                ##print("hogehoge_even:",i,j,MATRIX_SIZE+j-i-2)

    for i in range(0, MATRIX_SIZE):
        alpha[i] = numpy.angle(U[i][i])
 
    for i in range(int((MATRIX_SIZE-1)/2),0,-1):
        for j in range(0, 2*i): # 2i 回実行
            result = clem_convert_DT(alpha[MATRIX_SIZE-2-j], alpha[MATRIX_SIZE-1-j], phi[2*i*i+i-j-1], theta[2*i*i+i-j-1])
            alpha[MATRIX_SIZE-2-j] = result.Dm
            alpha[MATRIX_SIZE-1-j] = result.Dn
            phi[2*i*i+i-j-1] = result.phi

    phi = numpy.mod(phi,2*math.pi)
    theta = numpy.mod(theta,2*math.pi)
    alpha = numpy.mod(alpha,2*math.pi)

    result = collections.namedtuple('result', 'phi, theta, alpha')
    return result(phi=phi, theta=theta, alpha=alpha)



def clem_find_theta_LTM(U):
    MATRIX_SIZE = U.shape[0]
    theta = numpy.zeros(int(MATRIX_SIZE*(MATRIX_SIZE-1)/2))
    phi = numpy.zeros(int(MATRIX_SIZE*(MATRIX_SIZE-1)/2))
    alpha = numpy.zeros(MATRIX_SIZE)

    counter = 0
    for i in range(1, MATRIX_SIZE):
        if i & 1: #奇数の場合（ビット演算）
            for j in range(1,i+1):
                if U[j-1][MATRIX_SIZE-i+j-1] == 0:
                    theta[counter] = math.pi
                    phi[counter] = 0 #do not care
                else:
                    theta[counter] = 2 * numpy.arctan(numpy.abs(U[j-1][MATRIX_SIZE-i+j-2]/U[j-1][MATRIX_SIZE-i+j-1]))
                    phi[counter] = numpy.angle(U[j-1][MATRIX_SIZE-i+j-2]/U[j-1][MATRIX_SIZE-i+j-1])

                T = numpy.identity(MATRIX_SIZE, dtype=numpy.complex128)
                T[MATRIX_SIZE-i+j-2][MATRIX_SIZE-i+j-2] = numpy.exp((theta[counter]+math.pi)/2*1j) * numpy.exp(phi[counter]*1j) * numpy.sin(theta[counter]/2)
                T[MATRIX_SIZE-i+j-2][MATRIX_SIZE-i+j-1] = numpy.exp((theta[counter]+math.pi)/2*1j) * numpy.cos(theta[counter]/2)
                T[MATRIX_SIZE-i+j-1][MATRIX_SIZE-i+j-2] = numpy.exp((theta[counter]+math.pi)/2*1j) * numpy.exp(phi[counter]*1j) * numpy.cos(theta[counter]/2)
                T[MATRIX_SIZE-i+j-1][MATRIX_SIZE-i+j-1] = numpy.exp((theta[counter]+math.pi)/2*1j) * -1 * numpy.sin(theta[counter]/2)
                U = numpy.dot(U, numpy.conjugate(T.T)) #ユニタリ共役：numpy.conjugate(A.T)

                counter+=1
        else:
            for j in range(1,i+1):
                if U[i-j][MATRIX_SIZE-j] == 0:
                    theta[counter] = math.pi
                    phi[counter] = 0 #do not care
                else:
                    theta[counter] = 2 * numpy.arctan(numpy.abs(-U[i-j+1][MATRIX_SIZE-j]/U[i-j][MATRIX_SIZE-j]))
                    phi[counter] = numpy.angle(-U[i-j+1][MATRIX_SIZE-j]/U[i-j][MATRIX_SIZE-j])

                T = numpy.identity(MATRIX_SIZE, dtype=numpy.complex128)
                T[i-j][i-j] = numpy.exp((theta[counter]+math.pi)/2*1j) * numpy.exp(phi[counter]*1j) * numpy.sin(theta[counter]/2)
                T[i-j][i-j+1] = numpy.exp((theta[counter]+math.pi)/2*1j) * numpy.cos(theta[counter]/2)
                T[i-j+1][i-j] = numpy.exp((theta[counter]+math.pi)/2*1j) * numpy.exp(phi[counter]*1j) * numpy.cos(theta[counter]/2)
                T[i-j+1][i-j+1] = numpy.exp((theta[counter]+math.pi)/2*1j) * -1 * numpy.sin(theta[counter]/2)
                U = numpy.dot(T, U) 

                counter+=1

    for i in range(0, MATRIX_SIZE):
        alpha[i] = numpy.angle(U[i][i])
 
    for i in range(int((MATRIX_SIZE-1)/2),0,-1):
        for j in range(0, 2*i): # 2i 回実行
            result = clem_convert_DT(alpha[j], alpha[j+1], phi[2*i*i+i-j-1], theta[2*i*i+i-j-1])
            alpha[j] = result.Dm
            alpha[j+1] = result.Dn
            phi[2*i*i+i-j-1] = result.phi

    phi = numpy.mod(phi,2*math.pi)
    theta = numpy.mod(theta,2*math.pi)
    alpha = numpy.mod(alpha,2*math.pi)

    result = collections.namedtuple('result', 'phi, theta, alpha')
    return result(phi=phi, theta=theta, alpha=alpha)



def clem_simulate_U_UTM(MATRIX_SIZE, x, phi, theta, alpha):

    for i in range(1, int(MATRIX_SIZE/2)+1):
        for j in range(0, 2*i-1): # 2(i-1)+1 回実行
            T = numpy.identity(MATRIX_SIZE, dtype=numpy.complex128)
            T[2*i-2-j][2*i-2-j] = numpy.exp((theta[2*i*i-3*i+1+j]+math.pi)/2*1j) * numpy.exp(phi[2*i*i-3*i+1+j]*1j) * numpy.sin(theta[2*i*i-3*i+1+j]/2)
            T[2*i-2-j][2*i-1-j] = numpy.exp((theta[2*i*i-3*i+1+j]+math.pi)/2*1j) * numpy.cos(theta[2*i*i-3*i+1+j]/2)
            T[2*i-1-j][2*i-2-j] = numpy.exp((theta[2*i*i-3*i+1+j]+math.pi)/2*1j) * numpy.exp(phi[2*i*i-3*i+1+j]*1j) * numpy.cos(theta[2*i*i-3*i+1+j]/2)
            T[2*i-1-j][2*i-1-j] = numpy.exp((theta[2*i*i-3*i+1+j]+math.pi)/2*1j) * -1 * numpy.sin(theta[2*i*i-3*i+1+j]/2)
            x = T.dot(x)

    for i in range(int((MATRIX_SIZE-1)/2),0,-1):
        for j in range(0, 2*i): # 2i 回実行
            T = numpy.identity(MATRIX_SIZE, dtype=numpy.complex128)
            T[MATRIX_SIZE-2-j][MATRIX_SIZE-2-j] = numpy.exp((theta[2*i*i+i-j-1]+math.pi)/2*1j) * numpy.exp(phi[2*i*i+i-j-1]*1j) * numpy.sin(theta[2*i*i+i-j-1]/2)
            T[MATRIX_SIZE-2-j][MATRIX_SIZE-1-j] = numpy.exp((theta[2*i*i+i-j-1]+math.pi)/2*1j) * numpy.cos(theta[2*i*i+i-j-1]/2)
            T[MATRIX_SIZE-1-j][MATRIX_SIZE-2-j] = numpy.exp((theta[2*i*i+i-j-1]+math.pi)/2*1j) * numpy.exp(phi[2*i*i+i-j-1]*1j) * numpy.cos(theta[2*i*i+i-j-1]/2)
            T[MATRIX_SIZE-1-j][MATRIX_SIZE-1-j] = numpy.exp((theta[2*i*i+i-j-1]+math.pi)/2*1j) * -1 * numpy.sin(theta[2*i*i+i-j-1]/2)
            x = T.dot(x)

    #前段Uでは，位相合わせと標準出力しない
    #D = numpy.diag(numpy.exp(alpha*1j))
    #x = D.dot(x) 
    #print ("output power:",numpy.power(numpy.abs(x),2))
    #print ("output phase:",numpy.degrees(numpy.angle(x)))

    return x



def clem_simulate_U_LTM(MATRIX_SIZE, x, phi, theta, alpha):

    for i in range(1, int(MATRIX_SIZE/2)+1):
        for j in range(0, 2*i-1): # 2(i-1)+1 回実行
            T = numpy.identity(MATRIX_SIZE, dtype=numpy.complex128)
            T[MATRIX_SIZE-2*i+j][MATRIX_SIZE-2*i+j] = numpy.exp((theta[2*i*i-3*i+1+j]+math.pi)/2*1j) * numpy.exp(phi[2*i*i-3*i+1+j]*1j) * numpy.sin(theta[2*i*i-3*i+1+j]/2)
            T[MATRIX_SIZE-2*i+j][MATRIX_SIZE-2*i+1+j] = numpy.exp((theta[2*i*i-3*i+1+j]+math.pi)/2*1j) * numpy.cos(theta[2*i*i-3*i+1+j]/2)
            T[MATRIX_SIZE-2*i+1+j][MATRIX_SIZE-2*i+j] = numpy.exp((theta[2*i*i-3*i+1+j]+math.pi)/2*1j) * numpy.exp(phi[2*i*i-3*i+1+j]*1j) * numpy.cos(theta[2*i*i-3*i+1+j]/2)
            T[MATRIX_SIZE-2*i+1+j][MATRIX_SIZE-2*i+1+j] = numpy.exp((theta[2*i*i-3*i+1+j]+math.pi)/2*1j) * -1 * numpy.sin(theta[2*i*i-3*i+1+j]/2)
            x = T.dot(x)

    for i in range(int((MATRIX_SIZE-1)/2),0,-1):
        for j in range(0, 2*i): # 2i 回実行
            T = numpy.identity(MATRIX_SIZE, dtype=numpy.complex128)
            T[j][j] = numpy.exp((theta[2*i*i+i-j-1]+math.pi)/2*1j) * numpy.exp(phi[2*i*i+i-j-1]*1j) * numpy.sin(theta[2*i*i+i-j-1]/2)
            T[j][j+1] = numpy.exp((theta[2*i*i+i-j-1]+math.pi)/2*1j) * numpy.cos(theta[2*i*i+i-j-1]/2)
            T[j+1][j] = numpy.exp((theta[2*i*i+i-j-1]+math.pi)/2*1j) * numpy.exp(phi[2*i*i+i-j-1]*1j) * numpy.cos(theta[2*i*i+i-j-1]/2)
            T[j+1][j+1] = numpy.exp((theta[2*i*i+i-j-1]+math.pi)/2*1j) * -1 * numpy.sin(theta[2*i*i+i-j-1]/2)
            x = T.dot(x)

    D = numpy.diag(numpy.exp(alpha*1j))
    x = D.dot(x)
    #print ("output power [W]:",numpy.power(numpy.abs(x),2))
    #print ("output phase [deg]:",numpy.degrees(numpy.angle(x)))

    return x



def reck_find_theta_BU(U):
    MATRIX_SIZE = U.shape[0]
    theta = numpy.zeros((MATRIX_SIZE,MATRIX_SIZE))
    phi = numpy.zeros((MATRIX_SIZE,MATRIX_SIZE))
    alpha = numpy.zeros(MATRIX_SIZE)

    for N in range(MATRIX_SIZE,1,-1):
        cos_sum = 1.0
        phase_sum = numpy.array([1+0j])
        tmp_u = numpy.array([0+0j])
        for i in range(0,N):
            if cos_sum != 0:
                if i == N-1: 
                    tmp_u = U[N-1][N-1-i] / (numpy.power(1j,3+i+1) * cos_sum)
                else:
                    tmp_u = U[N-1][N-1-i] / (numpy.power(1j,3+i) * cos_sum)
            else:
                tmp_u = 0+0j

            if i == N-1: 
                pass
            else:
                # Rounding of numerical values for arcsin()
                if numpy.abs(tmp_u) > 1.0:
                    theta[N-1][N-1-i] = 2 * numpy.arcsin(1.0)
                elif numpy.abs(tmp_u) < -1.0:
                    theta[N-1][N-1-i] = 2 * numpy.arcsin(-1.0)
                else: 
                    theta[N-1][N-1-i] = 2 * numpy.arcsin(numpy.abs(tmp_u))

                phase_sum *= numpy.exp(theta[N-1][N-1-i]*1j/2)

            if i == 0:
                alpha[N-1] = numpy.angle(tmp_u/phase_sum) #alpha setting
                phase_sum *= numpy.exp(alpha[N-1]*1j)
            else:
                phi[N-1][N-1-(i-1)] = numpy.angle(tmp_u/phase_sum)

                if N == 2 and i == 1: #finding theta on 2.2 unitary matrix multiplier
                    alpha[0] = numpy.angle(U[0][1]/(1j*numpy.cos(theta[1][1]/2)*numpy.exp(theta[1][1]*1j/2)))
                else:
                    phase_sum *= numpy.exp(phi[N-1][N-1-(i-1)]*1j)

            cos_sum *= numpy.cos(theta[N-1][N-1-i]/2)

        for i in range(0,N-1):
            T = numpy.identity(MATRIX_SIZE, dtype=numpy.complex128)
            T[i][i] = numpy.exp((theta[N-1][i+1]+math.pi)/2*1j) * numpy.exp(phi[N-1][i+1]*1j) * numpy.sin(theta[N-1][i+1]/2)
            T[i][i+1] = numpy.exp((theta[N-1][i+1]+math.pi)/2*1j) * numpy.cos(theta[N-1][i+1]/2)
            T[i+1][i] = numpy.exp((theta[N-1][i+1]+math.pi)/2*1j) * numpy.exp(phi[N-1][i+1]*1j) * numpy.cos(theta[N-1][i+1]/2)
            T[i+1][i+1] = numpy.exp((theta[N-1][i+1]+math.pi)/2*1j) * -1 * numpy.sin(theta[N-1][i+1]/2)
 
            U = numpy.dot(U, numpy.conjugate(T.T)) #Unitary matrix conjugate and transpose：numpy.conjugate(A.T)

    result = collections.namedtuple('result', 'phi, theta, alpha')
    return result(phi=phi, theta=theta, alpha=alpha)



def reck_find_theta_TD(U):
    MATRIX_SIZE = U.shape[0]
    theta = numpy.zeros((MATRIX_SIZE,MATRIX_SIZE))
    phi = numpy.zeros((MATRIX_SIZE,MATRIX_SIZE))
    alpha = numpy.zeros(MATRIX_SIZE)

    for N in range(MATRIX_SIZE,1,-1):
        cos_sum = 1.0
        phase_sum = numpy.array([1+0j])
        tmp_u = numpy.array([0+0j])
        for i in range(0,N):

            if cos_sum != 0:
                if i == N-1: # last i in iteration
                    tmp_u = U[MATRIX_SIZE-N][MATRIX_SIZE-N+i] / (numpy.power(1j,i) * cos_sum)
                else:
                    tmp_u = U[MATRIX_SIZE-N][MATRIX_SIZE-N+i] / (numpy.power(1j,i+1) * cos_sum)
            else:
                tmp_u = 0+0j

            if i == N-1: # last i in iteration
                pass
            else:
                # Rounding of numerical values for arcsin()
                if numpy.abs(tmp_u) > 1.0:
                    theta[N-1][N-1-i] = 2 * numpy.arcsin(1.0)
                elif numpy.abs(tmp_u) < -1.0:
                    theta[N-1][N-1-i] = 2 * numpy.arcsin(-1.0)
                else: 
                    theta[N-1][N-1-i] = 2 * numpy.arcsin(numpy.abs(tmp_u))
                #Accumulate exp(theta)
                phase_sum *= numpy.exp(theta[N-1][N-1-i]*1j/2)

            if i == N-1:
                alpha[MATRIX_SIZE-N] = numpy.angle(tmp_u/phase_sum) #alpha setting 
                for j in range(0,N-1): # modified phi
                    phi[N-1][N-1-j] = phi[N-1][N-1-j] - alpha[MATRIX_SIZE-N]

                if N == 2 and i == 1: #finding theta on 2.2 unitary matrix multiplier
                    alpha[MATRIX_SIZE-1] = numpy.angle(U[MATRIX_SIZE-1][MATRIX_SIZE-1]/(-1j*numpy.sin(theta[1][1]/2)*numpy.exp(theta[1][1]*1j/2)))

            else:
                phi[N-1][N-1-i] = numpy.angle(tmp_u/phase_sum)

            cos_sum *= numpy.cos(theta[N-1][N-1-i]/2)

        for i in range(0,N-1):
            T = numpy.identity(MATRIX_SIZE, dtype=numpy.complex128)
            T[MATRIX_SIZE-2-i][MATRIX_SIZE-2-i] = numpy.exp((theta[N-1][i+1]+math.pi)/2*1j) * numpy.exp(phi[N-1][i+1]*1j) * numpy.sin(theta[N-1][i+1]/2)
            T[MATRIX_SIZE-2-i][MATRIX_SIZE-1-i] = numpy.exp((theta[N-1][i+1]+math.pi)/2*1j) * numpy.cos(theta[N-1][i+1]/2)
            T[MATRIX_SIZE-1-i][MATRIX_SIZE-2-i] = numpy.exp((theta[N-1][i+1]+math.pi)/2*1j) * numpy.exp(phi[N-1][i+1]*1j) * numpy.cos(theta[N-1][i+1]/2)
            T[MATRIX_SIZE-1-i][MATRIX_SIZE-1-i] = numpy.exp((theta[N-1][i+1]+math.pi)/2*1j) * -1 * numpy.sin(theta[N-1][i+1]/2)
 
            U = numpy.dot(U, numpy.conjugate(T.T)) #Unitary matrix conjugate and transpose：numpy.conjugate(A.T)

    result = collections.namedtuple('result', 'phi, theta, alpha')
    return result(phi=phi, theta=theta, alpha=alpha)



def reck_simulate_BU(MATRIX_SIZE, x, phi, theta, alpha):
    for N in range(MATRIX_SIZE,1,-1):
        for i in range(1,N):
            T = numpy.identity(MATRIX_SIZE, dtype=numpy.complex128)
            T[i-1][i-1] = numpy.exp((theta[N-1][i]+math.pi)/2*1j) * numpy.exp(phi[N-1][i]*1j) * numpy.sin(theta[N-1][i]/2)
            T[i-1][i] = numpy.exp((theta[N-1][i]+math.pi)/2*1j) * numpy.cos(theta[N-1][i]/2)
            T[i][i-1] = numpy.exp((theta[N-1][i]+math.pi)/2*1j) * numpy.exp(phi[N-1][i]*1j) * numpy.cos(theta[N-1][i]/2)
            T[i][i] = numpy.exp((theta[N-1][i]+math.pi)/2*1j) * -1 * numpy.sin(theta[N-1][i]/2)
            x = T.dot(x)
 
    #D = numpy.diag(numpy.exp(alpha*1j))
    #x = D.dot(x)
    #print ("output power:",numpy.power(numpy.abs(x),2))
    #print ("output phase:",numpy.degrees(numpy.angle(x)))

    return x



def reck_simulate_TD(MATRIX_SIZE, x, phi, theta, alpha):
    for N in range(MATRIX_SIZE,1,-1):
        for i in range(1,N):
            T = numpy.identity(MATRIX_SIZE, dtype=numpy.complex128)
            T[MATRIX_SIZE-i-1][MATRIX_SIZE-i-1] = numpy.exp((theta[N-1][i]+math.pi)/2*1j) * numpy.exp(phi[N-1][i]*1j) * numpy.sin(theta[N-1][i]/2)
            T[MATRIX_SIZE-i-1][MATRIX_SIZE-i] = numpy.exp((theta[N-1][i]+math.pi)/2*1j) * numpy.cos(theta[N-1][i]/2)
            T[MATRIX_SIZE-i][MATRIX_SIZE-i-1] = numpy.exp((theta[N-1][i]+math.pi)/2*1j) * numpy.exp(phi[N-1][i]*1j) * numpy.cos(theta[N-1][i]/2)
            T[MATRIX_SIZE-i][MATRIX_SIZE-i] = numpy.exp((theta[N-1][i]+math.pi)/2*1j) * -1 * numpy.sin(theta[N-1][i]/2)
            x = T.dot(x)

    D = numpy.diag(numpy.exp(alpha*1j))
    x = D.dot(x)
    #print ("output power [W]:",numpy.power(numpy.abs(x),2))
    #print ("output phase [deg]:",numpy.degrees(numpy.angle(x)))

    return x



#############################################################################

# Read an input file
filenamer1="opti_inputs.csv"
fr1 = open(filenamer1, "rU")
cr1 = csv.reader(fr1)
next(cr1)
VMM_type = next(cr1)
next(cr1)
power_adjust = [float(elm) for elm in next(cr1)]
next(cr1)
x = [float(elm) for elm in next(cr1)]
next(cr1)
tmpA = [[float(elm) for elm in v] for v in cr1]
A = numpy.zeros((len(tmpA),int(len(tmpA[0])/2)),dtype=complex)
for i in range(0,len(tmpA)):
    for j in range(0,int(len(tmpA[0])/2)):
        A[i][j] = tmpA[i][2*j] + 1j*tmpA[i][2*j+1]


#Power Adjustment factor
x = numpy.dot(x,power_adjust[0])

# check for contradiction form an input file 
if len(A[0]) != len(x):
    print("Dimensions mismatch in \""+str(filenamer1)+"\"")
    sys.exit(1)


print ("+++++Theoretical result+++++")
print ("A=\n", A)
print ("x=\n", x)
print ("")

print ("Ax=\n", A.dot(x))
print ("\n")


################
U,s,V = numpy.linalg.svd(A, full_matrices=True)	

if "clements" in VMM_type: 
    resultV = clem_find_theta_UTM(V)
elif "reck" in VMM_type: 
    resultV = reck_find_theta_BU(V)
else:
    print("Inappropriate setting for VMM type in \""+str(filenamer1)+"\"")
    sys.exit(1)

sigma = numpy.zeros(s.shape[0])
for i in range(0,s.shape[0]):
    sigma[i] = numpy.dot(numpy.log10(s[i]),20)

if len(V) >= len(U):
    D = numpy.diag(numpy.exp(resultV.alpha*1j))
    for i in range(0,abs(len(V)-len(U))):
        D = numpy.delete(D, len(V)-i-1,1)
        D = numpy.delete(D, len(V)-i-1,0)
else:
    D = numpy.diag(numpy.exp(numpy.append(resultV.alpha, numpy.zeros(len(U) - len(V)))*1j))

if "clements" in VMM_type: 
    resultU = clem_find_theta_LTM(numpy.dot(U,D))
elif "reck" in VMM_type: 
    resultU = reck_find_theta_TD(numpy.dot(U,D))
else:
    print("Inappropriate setting for VMM type in \""+str(filenamer1)+"\"")
    sys.exit(1)


print ("++++Optical simulation++++")
r = numpy.random.normal(0, In_noise_variance, len(x))
x = x + r
#print ("x + noise =\n", x)

r = numpy.random.normal(0, W_noise_variance, len(resultV.phi))
phi = resultV.phi + r
#print(phi)
r = numpy.random.normal(0, W_noise_variance, len(resultV.theta))
theta = resultV.theta + r
#print(theta)

if "clements" in VMM_type: 
    x = clem_simulate_U_UTM(len(V), x, phi, theta, resultV.alpha)
elif "reck" in VMM_type: 
    x = reck_simulate_BU(len(V), x, phi, theta, resultV.alpha)
else:
    print("Inappropriate setting for VMM type in \""+str(filenamer1)+"\"")
    sys.exit(1)

for i in range(0,len(sigma)):
    x[i] = numpy.dot(x[i],numpy.power(10,sigma[i]/20))

if len(U) > len(V):
    x = numpy.append(x, numpy.zeros(len(U) - len(V)))
elif len(U) < len(V):
    for i in range(len(V),len(U),-1):
        x = numpy.delete(x,i-1,0)

r = numpy.random.normal(0, W_noise_variance, len(resultU.phi))
phi = resultU.phi + r
#print(phi)
r = numpy.random.normal(0, W_noise_variance, len(resultU.theta))
theta = resultU.theta + r
#print(theta)

if "clements" in VMM_type: 
    x = clem_simulate_U_LTM(len(U), x, phi, theta, resultU.alpha)
elif "reck" in VMM_type: 
    x = reck_simulate_TD(len(U), x, phi, theta, resultU.alpha)
else:
    print("Inappropriate setting for VMM type in \""+str(filenamer1)+"\"")
    sys.exit(1)

print ("Sim. output: Ax=\n",x)
print ("\n")

fr1.close()
