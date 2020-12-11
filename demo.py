"""
Demonstration script

Phase retrieval with Bregman divergences and application to audio signal recovery
Pierre-Hugo Vial, Paul Magron, Thomas Oberlin, Cédric Févotte

October 2020
"""

# %% Packages import 

import sys, getopt
import librosa as l
import numpy as np
import glob
import soundfile as sf
import matplotlib
import matplotlib.pyplot as plt

from functions import measuresDiv
from functions import stft, istft, GLA, FGLA, GLADMM, GradDesc, ADMM, init_random, savepath

def main(argv):
    n_iter = 1000
    gamma = 0.99
    rho = 1e-1
    alg_type = 'gd'
    alg_dir = 'left'
    alg_d = 1
    alg_loss = 'KL'
    alg_step=1e-1
    file_path = r'datasets/tarantella.wav'
    
    
    #Parse command line arguments
    try:
        opts, args = getopt.getopt(argv, 'hf:n:a:', ['filepath=', 'niter=','algorithm='])
    except getopt.GetoptError:
        print('demo.py -f <filepath> -n <number of iterations> -a <algorithm>')
        sys.exit(2)
    for opt, arg in opts:
        if opt =='-h':
            print('demo.py -f <filepath> -n <number of iterations> -a <algorithm>')
            sys.exit()    
        elif opt in ['-f', '--filepath']:
            file_path = arg
        elif opt in ['-n', '--niter']:
            n_iter = int(arg)
        elif opt in ['-a', '--algorithm']:
            alg_type = arg
            
    newpath=savepath(['PR', 'demo'])  
      
    #Execution:

	#Loading original signal and preprocessing
    xo, sr = l.load(file_path) 
    xo = istft(stft(xo))
    M = np.abs(stft(xo))
    
	#Initialisation with original magnitude of STFT and random phase
    X_init = init_random(M)
    x_init = istft(X_init)
    
	#Processing algorithm
    if alg_type in ['gradient', 'gradient descent', 'GD', 'gd']:
        R = np.power(M,alg_d)
        x_gen = GradDesc(x_init, R, alg_d, alg_loss, alg_dir, alg_step, it=n_iter, gamma=gamma)
        sf.write(newpath+'generated.wav', x_gen, sr, subtype = 'PCM_24')
    
	#Processing ADMM algorithms
    elif alg_type in ['ADMM', 'admm']:
        R = np.power(M,alg_d)
        x_gen = ADMM(x_init, R, alg_d, alg_loss, alg_dir, it=n_iter, rho=rho)
        sf.write(newpath+'generated.wav', x_gen, sr, subtype = 'PCM_24')

    elif alg_type in ['GLA', 'gla', 'gl', 'GL', 'griffin-lim', 'Griffin-Lim']:
        X_gla = GLA(X_init, M, it=n_iter)
        x_gen = istft(X_gla)
        sf.write(newpath+'generated.wav', x_gen, sr, subtype = 'PCM_24')
    
    elif alg_type in ['FGLA', 'fgla', 'FGL', 'fgl']:
        X_fgla = FGLA(X_init, M, it=n_iter, gamma=gamma)
        x_fgla = istft(X_fgla)
        sf.write(newpath+'generated.wav', x_fgla, sr, subtype = 'PCM_24')
        
    elif alg_type in ['GLADMM', 'GL-ADMM']:
        X_gladmm = GLADMM(X_init, M, it=n_iter)
        x_gladmm = istft(X_gladmm)
        sf.write(newpath+'generated.wav', x_gladmm, sr, subtype = 'PCM_24')
    
# %%    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    