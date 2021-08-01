# Python function to add noise OpenFOAM results
# Developer: HanGao (hgao1@nd.edu)
###############################################################################
# system import
import numpy as np

global unitTest 
unitTest = False;
def fivePointPC(UMatrix):
    """ Function is to get value of U from the openFoam U files
        
    Args:
    UMatrix: Numpy matrix read from U file in OpenFoam

    Returns:
    [s1, s2, s3, s4, s5]: complex matrix to introduce phase error and magnitude error
    """    
    rows, cols = UMatrix.shape
    # define 4-point scheme matrix for future use
    A1 = np.matrix([[-1, -1, -1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]])
    # define 5-point scheme matrix
    A0 = np.matrix([[-1, -1, -1], [1, 1, -1], [1, -1, 1], [-1, 1, 1], [0, 0, 0]])
    # define s1 s2 s3 s4 s5
    s1 = np.zeros((rows), dtype = complex)
    s2 = np.zeros((rows), dtype = complex)
    s3 = np.zeros((rows), dtype = complex)
    s4 = np.zeros((rows), dtype = complex)
    s5 = np.zeros((rows), dtype = complex)
    # define image magnitude
    imMag = np.ones(rows)
    # hardcoding (to be known)
    gamma = 1
    # mean of the velocity magnitude
    M1 = 0
    ph0 = np.zeros(rows)
    for i in range(0, rows): 
        M1 = M1 + np.sqrt((UMatrix[i,0])**2 + (UMatrix[i,1])**2 + (UMatrix[i,2])**2) 
    M1 = M1 / rows
    # calculate the s1 to s5
    for i in range(0, rows):
        s1[i] = imMag[i]*np.exp(1j*((A0[0,0]*UMatrix[i,0]+A0[0,1]*UMatrix[i,1]+A0[0,2]*UMatrix[i,2])*gamma*M1+ph0[i]))
        s2[i] = imMag[i]*np.exp(1j*((A0[1,0]*UMatrix[i,0]+A0[1,1]*UMatrix[i,1]+A0[1,2]*UMatrix[i,2])*gamma*M1+ph0[i]))
        s3[i] = imMag[i]*np.exp(1j*((A0[2,0]*UMatrix[i,0]+A0[2,1]*UMatrix[i,1]+A0[2,2]*UMatrix[i,2])*gamma*M1+ph0[i]))
        s4[i] = imMag[i]*np.exp(1j*((A0[3,0]*UMatrix[i,0]+A0[3,1]*UMatrix[i,1]+A0[3,2]*UMatrix[i,2])*gamma*M1+ph0[i]))
        s5[i] = imMag[i]*np.exp(1j*((A0[4,0]*UMatrix[i,0]+A0[4,1]*UMatrix[i,1]+A0[4,2]*UMatrix[i,2])*gamma*M1+ph0[i]))        
    return [s1, s2, s3, s4, s5], M1


def addRandomNoiseOnfivePointPC(s1, s2, s3, s4, s5, UMatrix ,NL):
    #np.random.seed(123)
    """ Function is to get value of U from the openFoam U files
        
    Args:
    UMatrix: truth field
    s1, s2, s3, s4, s5: Numpy complex matrix
    NL: scaler, noise level in (0, 1)

    Returns:
    [s1, s2, s3, s4, s5]: Numpy complex matrix add random noise
    
    """    
    rows = len(s1)
    # define noise scaler k
    k = NL/1.4142135
    # add noise to s1 s2 s3 s4 s5
    for i in range(0, rows):
        U_magnitude_X = np.sqrt((UMatrix[i,0])**2) 
        U_magnitude_Y = np.sqrt((UMatrix[i,1])**2) 
        U_magnitude_Z = np.sqrt((UMatrix[i,2])**2)   
        U_magnitude = np.sqrt( U_magnitude_X**2 + U_magnitude_Y**2 + U_magnitude_Z**2 )
        s1[i] = s1[i] + k * U_magnitude *  (np.random.normal(0, 1, 1) + 1j * np.random.normal(0, 1, 1) ) 
        s2[i] = s2[i] + k * U_magnitude *  (np.random.normal(0, 1, 1) + 1j * np.random.normal(0, 1, 1) )
        s3[i] = s3[i] + k * U_magnitude *  (np.random.normal(0, 1, 1) + 1j * np.random.normal(0, 1, 1) )
        s4[i] = s4[i] + k * U_magnitude *  (np.random.normal(0, 1, 1) + 1j * np.random.normal(0, 1, 1) )
        s5[i] = s5[i] + k * U_magnitude *  (np.random.normal(0, 1, 1) + 1j * np.random.normal(0, 1, 1) )        
    return [s1, s2, s3, s4, s5]


def invfivePointPC(s1, s2, s3, s4, s5, M1):
    """ Function is to get value of U from the openFoam U files
        
    Args:
    s1, s2, s3, s4, s5: Numpy complex matrix(velocity encodes)
    M1: first moment

    Returns:
    UMatrix: Numpy matrix read from U file in OpenFoam
    """    
    gamma = 1
    rows = len(s1)
    # define 4-point scheme matrix for future use
    A1 = np.matrix([[-1, -1, -1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]])
    # define 5-point scheme matrix
    A0 = np.matrix([[-1, -1, -1], [1, 1, -1], [1, -1, 1], [-1, 1, 1], [0, 0, 0]])
    # define Ani
    Ani = np.matmul(A1, np.linalg.pinv(A0))
    
    A1inv = np.linalg.pinv(A1)
    
    # define ph1 ph2 ph3 ph4 ph5
    ph1 = np.zeros(rows)
    ph2 = np.zeros(rows)
    ph3 = np.zeros(rows)
    ph4 = np.zeros(rows)
    ph5 = np.zeros(rows)
    # define ni1, ni2, ni3, ni4
    ni1 = np.zeros(rows)
    ni2 = np.zeros(rows)
    ni3 = np.zeros(rows)
    ni4 = np.zeros(rows)
    
    # define UMatrix
    UMatrix = np.zeros((rows, 3))
    # define imMag
    imMag = np.zeros(rows)
    # do the decode invFivePoint scheme
    for i in range(0, rows):
        ph1[i] = np.angle(s1[i])
        ph2[i] = np.angle(s2[i])
        ph3[i] = np.angle(s3[i])
        ph4[i] = np.angle(s4[i])
        ph5[i] = np.angle(s5[i])
        ni1[i] = 2 * 3.1415926 * np.round( (1/2/3.1415926) * (Ani[0,0]*ph1[i]+Ani[0,1]*ph2[i]+Ani[0,2]*ph3[i]+Ani[0,3]*ph4[i]+Ani[0,4]*ph5[i]-ph1[i]) )
        ni2[i] = 2 * 3.1415926 * np.round( (1/2/3.1415926) * (Ani[1,0]*ph1[i]+Ani[1,1]*ph2[i]+Ani[1,2]*ph3[i]+Ani[1,3]*ph4[i]+Ani[1,4]*ph5[i]-ph2[i]) )
        ni3[i] = 2 * 3.1415926 * np.round( (1/2/3.1415926) * (Ani[2,0]*ph1[i]+Ani[2,1]*ph2[i]+Ani[2,2]*ph3[i]+Ani[2,3]*ph4[i]+Ani[2,4]*ph5[i]-ph3[i]) )
        ni4[i] = 2 * 3.1415926 * np.round( (1/2/3.1415926) * (Ani[3,0]*ph1[i]+Ani[3,1]*ph2[i]+Ani[3,2]*ph3[i]+Ani[3,3]*ph4[i]+Ani[3,4]*ph5[i]-ph4[i]) )
        imMag[i] = ( np.absolute(s1[i]) + np.absolute(s2[i]) + np.absolute(s3[i]) + np.absolute(s4[i]) + np.absolute(s5[i]) ) / 5
        UMatrix[i,0] = imMag[i] * 1 / (gamma * M1) * ( A1inv[0,0] * (ph1[i]+ni1[i]) + A1inv[0,1] * (ph2[i]+ni2[i]) + A1inv[0,2] * (ph3[i]+ni3[i]) + A1inv[0,3] * (ph4[i]+ni4[i]) )
        UMatrix[i,1] = imMag[i] * 1 / (gamma * M1) * ( A1inv[1,0] * (ph1[i]+ni1[i]) + A1inv[1,1] * (ph2[i]+ni2[i]) + A1inv[1,2] * (ph3[i]+ni3[i]) + A1inv[1,3] * (ph4[i]+ni4[i]) )
        UMatrix[i,2] = imMag[i] * 1 / (gamma * M1) * ( A1inv[2,0] * (ph1[i]+ni1[i]) + A1inv[2,1] * (ph2[i]+ni2[i]) + A1inv[2,2] * (ph3[i]+ni3[i]) + A1inv[2,3] * (ph4[i]+ni4[i]) )        
    return UMatrix

def addMRINoise(UMatrix, NL):
    [s1, s2, s3, s4, s5], M1=fivePointPC(UMatrix)
    [s1, s2, s3, s4, s5]=addRandomNoiseOnfivePointPC(s1, s2, s3, s4, s5, UMatrix ,NL)
    UMatrix=invfivePointPC(s1, s2, s3, s4, s5, M1)
    return UMatrix


    
