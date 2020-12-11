"""
Phase retrieval with Bregman divergences and application to audio signal recovery
Pierre-Hugo Vial, Paul Magron, Thomas Oberlin, Cédric Févotte

December 2020
"""

#Packages import 
import sys, getopt
import librosa as l
import numpy as np
import glob
import soundfile as sf
import matplotlib
import matplotlib.pyplot as plt


from functions import stft, istft, GLA, FGLA, GLADMM, GradDesc, ADMM, init_random, savepath, measuresDiv, noiseDenoise_Wiener

def main(argv):
    database = 'speech'
    n_files = 10
    n_iter = 2500
    len_sec = 2
    gamma = 0.99
    rho = 1e-1
    mod_spec=False
    
    #Parse command line arguments
    try:
        opts, args = getopt.getopt(argv, 'hd:n:f:s:', ['dataset=', 'niter=', 'snr='])
    except getopt.GetoptError:
        print('main.py -d <dataset> -n <number of iterations> -f <number of files> -s <input SNR in dB>')
        sys.exit(2)
    for opt, arg in opts:
        if opt =='-h':
            print('main.py -d <dataset> -n <number of iterations> -f <number of files> -s <input SNR in dB>')
            sys.exit()    
        elif opt in ['-d', '--dataset']:
            database = arg
        elif opt in ['-n', '--niter']:
            n_iter = int(arg)
        elif opt in ['-f']:
            n_files = int(arg)
        elif opt in ['-s']:
            snr = int(arg)
            mod_spec=True
    
    # Database import
    if database=='music':
        dataset_path = r'datasets/music'
        files = glob.glob(dataset_path + r'/*.mp3')
    elif database=='speech':
        dataset_path = r'datasets/speech'
        files = glob.glob(dataset_path + r'/*.wav')
    for i in range(len(files)):
        files[i]=files[i].replace('\\','/')
    files = files[:n_files]
    newpath=savepath(['PR', database])
    
    #Procedure:
    #Gradient descent algorithms (loss, direction, step size, d)
    GD_algs = [('IS','right',1e-7,2),
               ('KL','right',1e-4,1),('KL','left',1e-2,1),
               ('b05','right',1e-1,1),('b05','left',1e-6,1),
               ('l2','left',1e-1,1),
               ('b05','right',1e-3,2),('b05','left',1e-6,2),
               ('KL', 'right', 1e-1,2),('KL', 'left', 1e-3,2),
               ('l2','left',1e-5,2)]
    #ADMM algorithms (optimized cost, direction, d)
    ADMM_algs = [('IS','left',1), ('KL', 'left',1), ('l2', 'left',1)]
    
    #Execution:
    c_file=0
    for file_path in files:
    	#Loading original signal and preprocessing
        xo, sr = l.load(file_path) 
        xo = xo[:int(len_sec*sr)]
        xo = istft(stft(xo))
        if mod_spec:
            M = noiseDenoise_Wiener(xo, snr) #Producing modified magnitude spectrogram
            divs=['stoi']
        else:
            M = np.abs(stft(xo))
            divs=['sc','stoi']
        sf.write(newpath+'%i_ORIGINAL.wav'%c_file, xo, sr, subtype = 'PCM_24')
        np.save(newpath+'%i_ORIGINAL.npy'%c_file, xo)
        print('File %i/%i'%(c_file+1, n_files))
        
    	#Initialisation with original/modified magnitude of STFT and random phase
        X_init = init_random(M)
        x_init = istft(X_init)
        sf.write(newpath+'%i_INIT.wav'%(c_file), x_init, sr, subtype = 'PCM_24')
        np.save(newpath+'%i_INIT.npy'%(c_file), x_init)
        
    	#Processing gradient descent algorithms
        for alg in GD_algs:
            d=alg[3]
            print('Gradient descent: %s, %s, d=%i'%(alg[0], alg[1], d))
            R = np.power(M,d)
            x_gen = GradDesc(x_init, R, d, alg[0], alg[1], alg[2], it=n_iter, gamma=gamma)
            sf.write(newpath+'%i_GD_%s_%s_d%i.wav'%(c_file, alg[0], alg[1], d), x_gen, sr, subtype = 'PCM_24')
            np.save(newpath+'%i_GD_%s_%s_d%i.npy'%(c_file, alg[0], alg[1], d), x_gen)
        
    	#Processing ADMM algorithms
        for alg in ADMM_algs:
            d=alg[2]
            print('ADMM: %s, %s, d=%i'%(alg[0], alg[1], d))
            R = np.power(M,d)
            x_gen = ADMM(x_init, R, d, alg[0], alg[1], it=n_iter, rho=rho)
            sf.write(newpath+'%i_ADMM_%s_%s_d%i.wav'%(c_file, alg[0], alg[1], d), x_gen, sr, subtype = 'PCM_24')
            np.save(newpath+'%i_ADMM_%s_%s_d%i.npy'%(c_file, alg[0], alg[1], d), x_gen)
        
    	#Processing GLA
        print('Griffin-Lim Algorithm')    
        X_gla = GLA(X_init, M, it=n_iter)
        x_gla = istft(X_gla)
        sf.write(newpath+'%i_GLA.wav'%(c_file), x_gla, sr, subtype = 'PCM_24')
        np.save(newpath+'%i_GLA.npy'%(c_file), x_gla)
        
    	#Processing FGLA
        print('Fast Griffin-Lim Algorithm')
        X_fgla = FGLA(X_init, M, it=n_iter, gamma=gamma)
        x_fgla = istft(X_fgla)
        sf.write(newpath+'%i_FGLA.wav'%(c_file), x_fgla, sr, subtype = 'PCM_24')
        np.save(newpath+'%i_FGLA.npy'%(c_file), x_fgla)
        
    	#Processing GLADMM
        print('GLADMM')
        X_gladmm = GLADMM(X_init, M, it=n_iter)
        x_gladmm = istft(X_gladmm)
        sf.write(newpath+'%i_GLADMM.wav'%(c_file), x_gladmm, sr, subtype = 'PCM_24')
        np.save(newpath+'%i_GLADMM.npy'%(c_file), x_gladmm)
            
        c_file+=1

    matplotlib.use('TKAgg')
   
    other_algs=['GLA', 'FGLA', 'GLADMM', 'INIT']
    algs = GD_algs + ADMM_algs + other_algs
    
     
    #Compute metrics in divs
    distmat=np.zeros((n_files, len(algs), len(divs)))
    for i in range(n_files):
        original, sr = l.load(newpath+'%i_ORIGINAL.wav'%i)
        for j in range(len(GD_algs)):
            j_gd = GD_algs[j]
            x_gen = np.load(newpath+'%i_GD_%s_%s_d%i.npy'%(i, j_gd[0], j_gd[1], j_gd[3]))
            distmat[i,j,:] = measuresDiv(original, x_gen, divs)
        
        for j in range(len(ADMM_algs)):
            j_admm = ADMM_algs[j]
            x_gen = np.load(newpath+'%i_ADMM_%s_%s_d%i.npy'%(i,j_admm[0], j_admm[1], j_admm[2]))
            distmat[i,j+len(GD_algs),:] = measuresDiv(original, x_gen, divs)
            
        for j in range(len(other_algs)):
            j_other = other_algs[j]
            x_gen = np.load(newpath+'%i_%s.npy'%(i,j_other))
            distmat[i,j+len(GD_algs)+len(ADMM_algs),:] = measuresDiv(original, x_gen, divs)
            
    plt.close('all')
    labels_g1 = [ '0.5', 'KL',
          'QD', 'KL', '0.5']
    labels_g2=['0.5', 'KL',
          'QD', 'KL', '0.5', 'IS']
    labels_a =['IS', 'KL', 'QD']
    labels_o=['GLA', 'FGLA', 'GLADMM', 'INIT']
    titles = ['Gradient descent\n$d=1$', 'Gradient descent\n$d=2$', 'ADMM\n$d=1$', 'Baselines\n']
    plt.rcParams.update({'font.weight':'regular','font.size':19})
    
    #Create figure for every metric in divs
    for k in range(len(divs)):
        fig, ax = plt.subplots(1,4, sharey=True, figsize=(18,9))
        fig.subplots_adjust(wspace=0.05)
    
        bplot_g1 = ax[0].boxplot(distmat[:,0:5,k], patch_artist=True, showfliers=False,
                           showmeans=True, labels=labels_g1, 
                           medianprops={'color':'black'}, 
                           meanprops={'markerfacecolor':'black',
                                      'markeredgecolor':'black',
                                       'marker':'o'})
        plt.sca(ax[0])
        plt.xticks(rotation=30, fontweight='demi')
        plt.yticks(fontweight='demi')
        ax[0].set_title(titles[0])
        ax[0].text(0.25, 0.95, '(left)', horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
        ax[0].text(0.75, 0.95, '(right)', horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
        plt.axvline(x=3, color="black", linestyle=":")
        
        for patch in bplot_g1['boxes']:
            patch.set_facecolor('mediumturquoise')
        
        bplot_g2 = ax[1].boxplot(distmat[:,5:11,k], patch_artist=True, showfliers=False,
                           showmeans=True, labels=labels_g2, 
                           medianprops={'color':'black'}, 
                           meanprops={'markerfacecolor':'black',
                                      'markeredgecolor':'black',
                                       'marker':'o'})
        ax[1].set_title(titles[1])
        ax[1].text(0.2, 0.95, '(left)', horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
        ax[1].text(0.7, 0.95, '(right)', horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
        plt.sca(ax[1])
        plt.xticks(rotation=30, fontweight='demi')
        plt.axvline(x=3, color="black", linestyle=":")
        for patch in bplot_g2['boxes']:
            patch.set_facecolor('darkcyan')
            
        bplot_a = ax[2].boxplot(distmat[:,11:14,k], patch_artist=True, showfliers=False,
                           showmeans=True, labels=labels_a, 
                           medianprops={'color':'black'}, 
                           meanprops={'markerfacecolor':'black',
                                      'markeredgecolor':'black',
                                       'marker':'o'})
        ax[2].text(0.5, 0.95, '(left)', horizontalalignment='center', verticalalignment='center', transform=ax[2].transAxes)      
        plt.sca(ax[2])
        plt.xticks(rotation=30, fontweight='demi')
        
        ax[2].set_title(titles[2])
        for patch in bplot_a['boxes']:
            patch.set_facecolor('orangered')
       
        bplot_o = ax[3].boxplot(distmat[:,14:,k], patch_artist=True, showfliers=False,
                           showmeans=True, labels=labels_o, 
                           medianprops={'color':'black'}, 
                           meanprops={'markerfacecolor':'black',
                                      'markeredgecolor':'black',
                                       'marker':'o'})
        plt.sca(ax[3])
        plt.xticks(rotation=30, fontweight='demi')
        
        ax[3].set_title(titles[3])
        for patch in bplot_o['boxes']:
            patch.set_facecolor('gold')   
        ax[0].yaxis.grid(True)
        ax[1].yaxis.grid(True)
        ax[2].yaxis.grid(True)
        ax[3].yaxis.grid(True)
        btt = np.nanmin(distmat[:,-1,k])
        plt.ylim(bottom= btt - 0.1*np.abs(btt))
        fig.savefig(newpath+'%s_%s.pdf'%(database, divs[k]), bbox_inches='tight', pad_inches=0)

# %%    
if __name__ == '__main__':
    main(sys.argv[1:])