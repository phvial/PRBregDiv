"""
Functions 

Phase retrieval with Bregman divergences and application to audio signal recovery
Pierre-Hugo Vial, Paul Magron, Thomas Oberlin, Cédric Févotte

October 2020
"""

# %% Packages import

import numpy as np
import librosa as l
import scipy.special as ss
from tqdm import tqdm
from pystoi import stoi
import scipy.signal as sp
import datetime as dt
import os

# %% Time-frequency functions

def stft(x, fr_len=1024, fr_step=512, n_fft=1024):
    """
    Computes the STFT of input signal x
    Analysis window is a fr_len samples-long sine bell (root of the Hann window)
    --
    x: input signal
    fr_len: length of analysis window
    fr_step: hop-size
    n_fft: size of DFT
    """
    w_hann = sp.get_window('hann', fr_len)
    w_mlt = np.sqrt(w_hann)
    return l.stft(x, n_fft=n_fft, hop_length=fr_step, win_length=fr_len, window = w_mlt)


def istft(X,fr_len=1024, fr_step=512):
    """
    Computes the inverse STFT of input coefficients X
    Synthesis window is a fr_len samples-long sine bell (root of the Hann window)
    --
    X: input time-freq. coefficients
    fr_len: length of analysis window
    fr_step: hop-size
    """
    w_hann = sp.get_window('hann', fr_len)
    w_mlt = np.sqrt(w_hann)
    return  l.istft(X,hop_length=fr_step, win_length=fr_len, window = w_mlt)


def Pa(X,M):
    """
    Projection of X on the set of time-freq. coefficients whose magnitude is M
    --
    X: input time-freq. coefficients
    M: magnitude measurements
    """
    p = np.angle(X)
    return np.multiply(M, np.exp(1j*p))

def Pc(X):
    """
    Projection of X on the set of consistent time-freq. coefficients
    --
    X: iput time-freq. coefficients
    """
    return stft(istft(X))

def init_random(M, seed=None):
    """
    Computes initialization with given magnitude and random phase
    --
    M: magnitude measurements
    seed: random seed
    """
    if seed !=None:
        np.random.seed(seed=seed)
    p0=np.random.uniform(-np.pi, np.pi, size=[np.size(M, axis=0), np.size(M, axis=1)])
    X0=np.complex64(np.multiply(M, np.exp(1j*p0)))
    return X0       
    
def noiseDenoise_Wiener(x0, SNRdb):
    """
    Initialisation for PR from modified spectrograms
    Produces modified magnitude measurements by adding gaussian noise and denoising via oracle Wiener filtering
    --
    x0: ground-truth signal
    SNRdb: SNR(dB) value for the noising process
    """
    #1. Noising step
    P0 = np.mean(np.square(x0))
    X0 = stft(x0)
    noise= np.random.normal(size=x0.size)
	#Computing noise power
    Pnoise = np.mean(np.square(noise))
    noise = noise * np.sqrt(P0/Pnoise) * np.power(10,-SNRdb/10)
    #Adding noise
    x0n=x0+noise
    X0n=stft(x0n)
    Noise = stft(noise)
    #2. Wiener filtering with oracle mask
    mask_wiener = np.square(np.abs(X0))/ (np.square(np.abs(X0)) + np.square(np.abs(Noise))+ 1e-8)
    M = mask_wiener * np.abs(X0n)
    return M

def computeGrad(x, R, d, grad_dist, direction, beta=0.5, eps=1e-8):
    """
    Returns gradient for Gradient descent algorithm
    --
    x: 
    R: measurements
    d: value of d (d=1 for magnitude spec., d=2 for power spec.)
    grad_dist: loss function
    direction: 'right' for right PR, 'left' for left PR
    beta: value of beta for general beta-divergence
    """
    R0 = R + eps
    Ax = stft(x)
    Ax_mod = np.abs(Ax)
    Rx = np.power(Ax_mod,d) + eps
    if direction =='left':
        if grad_dist in ['KL', 'kl']:
            z = np.log(Rx)-np.log(R0)
        elif grad_dist in ['is','IS']:
            z = 1/(R0) - 1/(Rx)
        elif grad_dist in ['beta', 'b']:
            z = 1/(beta-1)*(np.power(Rx,beta-1)-np.power(R0,beta-1))
        elif grad_dist in ['beta05', 'b05']:
            z = 2*(np.power(R0,-1/2)-np.power(Rx,-1/2))
        else: #Quadratic
            z = 2*(Rx-R0)
    else:
        if grad_dist in ['KL', 'kl']:
            z = np.multiply(np.power(Rx,-1), Rx-R0+eps)
        elif grad_dist in ['is','IS']:
            z = np.multiply(np.power(Rx,-2), Rx-R0+eps)
        elif grad_dist in ['beta', 'b']:
            z = np.multiply(np.power(Rx,beta-2), Rx-R0+eps)
        elif grad_dist in ['beta05', 'b05']:
            z = np.multiply(np.power(Rx,-1/2), Rx-R0+eps)
        else: #Quadratic
            z = 2*(Rx-R0)
    
    grad = d/2*istft(np.power(Ax_mod+eps, d-2) * Ax * z)
    return grad

def computeProx(Y, R, rho, opt_dist, direction, eps=1e-8):
    """
    Returns proximal operator of loss function evaluated in Y for ADMM algorithm
    --
    Y: input entry of proximal operator
    R: measurements
    rho: penalty parameter
    opt_dist: loss function
    direction: 'right' for right PR, 'left' for left PR    
    """
    if opt_dist in ['2', 'l2', 'euc', 'L2']:
            v=(rho*Y+2*R)/(rho+2)
    elif opt_dist in ['is', 'IS']:
            b=1/(R+eps)-rho*Y
            delta = np.square(b)+4*rho
            v = (-b+np.sqrt(delta))/(2*rho)        
    elif opt_dist in ['kl', 'KL']:
        if direction=='left':
            v = 1/rho * ss.lambertw(rho * R * np.exp(rho * Y))
        else:
            delta = 4*rho*R + np.square(1-Y)
            v = (Y - 1 + np.sqrt(delta))/(2*rho)
    return v


# %% Phase retrieval algorithms
    
def GLA(X0, M, it=200):
    """
    Griffin-Lim Algorithm
    --
    X0: initialisation value
    M: magnitude measurements (M=R^{1/d})
    it: number of iterations
    """
    Xi = X0
    for i in tqdm(range(it)):
        Xi = Pc(Pa(Xi,M))
    return Xi

def FGLA(X0, M, it=200, gamma=0.99):
    """
    Fast Griffin-Lim Algorithm
    --
    X0: initialisation value
    M: magnitude measurements (M=R^{1/d})
    it: number of iterations
    gamma: acceleration parameter
    """
    Ti = X0
    Ti1 = X0
    Xi = X0
    for i in tqdm(range(it)):
        Ti = Pc(Pa(Xi,M))
        Xi = Ti + gamma*(Ti-Ti1)
        Ti1 = Ti
    return Xi

def GLADMM(X0, R, it=200):
    """
    Griffin-Lim like ADMM
    --
    X0: initialisation value
    M: magnitude measurements (M=R^{1/d})
    it: number of iterations
    """
    Zi = X0
    Xi = X0
    Ui = np.zeros(X0.shape)
    for i in tqdm(range(it)):
        Xi = Pa(Zi-Ui, R)
        Zi = Pc(Xi+Ui)
        Ui = Ui + Xi - Zi
    return Xi
    
def GradDesc(x0, R, d, grad_dist, direction, mu,
             acc=True, gamma=0.99, beta=0.5, it=200, normalize_gradient=True):
    """
    Gradient descent algorithm for phase retrieval with optionnal acceleration
    --
    x0: initialisation signal
    R: measurements
    d: value of d (d=1 for magnitude spec., d=2 for power spec.)
    grad_dist: loss function
    direction: 'right' for right PR, 'left' for left PR
    mu: step size
    acc: True for accelerated gradient algorithm, False for standard gradient algorithm
    gamma: acceleration parameter
    beta: value of beta for general beta-divergence
    it: number of iterations
    """
    xi = x0
    yi = x0
    if normalize_gradient:
        init_norm = np.square(np.linalg.norm(x0))
    else:
        init_norm = 1
    try:
        for i in tqdm(range(it)):
            yi1 = yi
			#Gradient descent step
            yi = xi - mu/init_norm*computeGrad(xi, R, d, grad_dist, direction, beta)
			#Optional acceleration
            if acc:
                xi = yi + gamma*(yi-yi1)
            else:
                xi = yi
    except l.ParameterError:
        print('Error ! Audio buffer is not finite everywhere')
    return xi

def ADMM(x0, R, d, opt_dist, direction, rho=0.1, it=200, eps=1e-8):
    """
    ADMM algorithm for phase retrieval
    --
    x0: initialisation signal
    R: measurements
    d: value of d (d=1 for magnitude spec., d=2 for power spec.)
    opt_dist: loss function
    direction: 'right' for right PR, 'left' for left PR
    rho: penalty parameter
    it: number of iterations
    """
    lbda = np.zeros(R.shape, dtype=np.complex64)
    xi = x0
    try:
        for i in tqdm(range(it)):
            Xi = stft(xi)
            H = np.power(Xi, d) + lbda/rho
			#Update auxiliary variable U
            U = computeProx(np.abs(H), R, rho, opt_dist, direction)
			#Update auxiliary variable Theta
            Theta = np.angle(H)
            Z = np.multiply(U, np.exp(1j*Theta))
			#Update signal estimation x
            xi = istft(np.power(Z-lbda/rho, 1/d))
			#Update the Lagrange multipliers lbda
            lbda = lbda + rho*(stft(xi)-Z)
    except l.ParameterError:
        print('Error ! Audio buffer is not finite everywhere')
    return xi
    

# %% Divergences measures
    
def scale_lag_invariant_snr(ref_sig, out_sig, eps=1e-8):
    """
    Calcuate Scale- and Lag-Invariant Source-to-Noise Ratio (SI-SNR)
    --
    ref_sig: numpy.ndarray (nsamples,)
    out_sig: numpy.ndarray (nsamples,)
    """
    assert len(ref_sig) == len(out_sig)
    # Center the signals
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    # Lag-invariant part: get the delay that maximizes the correlation between the ref
    # and the estimate and modify the reference accordingly
    corr = sp.correlate(ref_sig, out_sig)
    delay = np.argmax(corr) - ref_sig.shape[0] + 1
    delayed_ref = ref_sig[np.arange(delay, ref_sig.shape[0] + delay) % ref_sig.shape[0]]
    # Scale-invariant part: project the (delayed) ref on the subspace spanned by the estimate
    ref_energy = np.sum(delayed_ref ** 2) + eps
    proj = np.sum(delayed_ref * out_sig) * delayed_ref / ref_energy
    noise = out_sig - proj
    sli_snr = 10 * np.log(np.sum(proj ** 2) / (np.sum(noise ** 2) + eps) + eps) / np.log(10.0)
    
    return sli_snr

def computeDiv(x_ref, x_est, div, sr=22050):
    """
    Computes a metric of discrepancy between a reference signal and its estimate
    --
    x_ref: reference signal
    x_est: estimated signal
    div: chosen discrepancy measure
    """
    if div in ['stoi']:
        dm = stoi.stoi(x_ref, x_est, sr, extended=False) 
    elif div in ['sli_snr']:
        dm= scale_lag_invariant_snr(x_ref, x_est)
    elif div in ['sc']: #Spectral convergence
        X_ref = stft(x_ref)
        X_est = stft(x_est)  
        S_ref = np.abs(X_ref)
        S_est = np.abs(X_est)
        dm = np.linalg.norm(S_ref - S_est) / np.linalg.norm(S_ref)
    return dm

def measuresDiv(x_ref, x_est, divList):
    """
    Performs measure according to divergences/distances in divList
    --
    x_ref: reference signal
    x_est: estimated signal
    divList = [div1, div2, ...]
    """
    meas = []
    for div in divList:
        dm = computeDiv(x_ref, x_est, div=div)
        meas.append(dm)
    return meas
        
# %% misc
    
def savepath(ad_text=[], fd=''):
    """
    Creates a new directory for saving generated signals and returns path
    """
    td = str(dt.datetime.today())
    td0=td.replace(' ','_')
    td0=td0.replace(':','h')
    td0 = td0[:16]
    ad=''
    for w in ad_text:
        ad+='_'+w
    current_path = os.getcwd()
    current_path = current_path.replace('\\','/')
    if fd!='':
        current_path+=r'/%s'%fd
    newpath = current_path+'/'+td0+ad+r'/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

            