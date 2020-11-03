"""
Evaluation and figures for phase retrieval from exact spectrograms

Phase retrieval with Bregman divergences and application to audio signal recovery
Pierre-Hugo Vial, Paul Magron, Thomas Oberlin, Cédric Févotte

October 2020
"""

# %% Packages import 

import sys, getopt
import librosa as l
import matplotlib
import matplotlib.pyplot as plt
from functions import measuresDiv
import numpy as np

def main(argv):
    matplotlib.use('TKAgg')
    labels = [ 'G$\\cdot$05$\\cdot$R1', 'G$\\cdot$05$\\cdot$L1', 'G$\\cdot$KL$\\cdot$R1',
              'G$\\cdot$KL$\\cdot$L1', 'G$\\cdot$QD$\\cdot$1',
              'G$\\cdot$IS$\\cdot$R2', 'G$\\cdot$05$\\cdot$R2', 'G$\\cdot$05$\\cdot$L2',
              'G$\\cdot$KL$\\cdot$R2', 'G$\\cdot$KL$\\cdot$L2', 'G$\\cdot$QD$\\cdot$2',
              'A$\\cdot$IS$\\cdot$L1', 'A$\\cdot$KL$\\cdot$L1', 'A$\\cdot$QD$\\cdot$1',
              'GLA', 'FGLA', 'GLADMM', 'INIT'         
            ]
    GD_algs = [('b05','right',1e-1,1),('b05','left',1e-6,1),
           ('KL','right',1e-4,1),('KL','left',1e-1,1),
           ('l2','left',1,1),
           ('IS','right',1e-8,2),
           ('b05','right',1e-3,2),('b05','left',1e-5,2),
           ('KL', 'right', 1e-1,2),('KL', 'left', 1e-1,2),
           ('l2','left',1e-5,2)]
    ADMM_algs = [('IS','left',1), ('KL', 'left',1), ('l2', 'left',1)]
    other_algs=['GLA', 'FGLA', 'GLADMM', 'INIT']
    algs = GD_algs + ADMM_algs + other_algs
    colors = 5*['mediumturquoise']+(len(GD_algs)-5)*['darkcyan']+len(ADMM_algs)*['orangered']+3*['gold']+['white']
    divs=['sc','stoi','sli_snr']
    n_files=10
    
    #Parse command line arguments
    try:
        opts, args = getopt.getopt(argv, 'hp:m:f:', ['path=', 'metric='])
    except getopt.GetoptError:
        print('plotter_PR.py -p <path> -m <metric> -f <number of files>')
        sys.exit(2)
    for opt, arg in opts:
        if opt =='-h':
            print('plotter_PR.py -p <path> -m <metric> -f <number of files>')
            sys.exit()    
        elif opt in ['-p', '--path']:
            sig_path = arg+'/'
        elif opt in ['-m', '--metric']:
            divs = [arg]
        elif opt in ['-f']:
            n_files = int(arg)
     
    #Compute metrics in divs
    distmat=np.zeros((n_files, len(algs), len(divs)))
    for i in range(n_files):
        original, sr = l.load(sig_path+'%i_ORIGINAL.wav'%i)
        for j in range(len(GD_algs)):
            j_gd = GD_algs[j]
            x_gen, sr = l.load(sig_path+'%i_GD_%s_%s_d%i.wav'%(i, j_gd[0], j_gd[1], j_gd[3]))
            distmat[i,j,:] = measuresDiv(original, x_gen, divs)
        
        for j in range(len(ADMM_algs)):
            j_admm = ADMM_algs[j]
            x_gen, sr = l.load(sig_path+'%i_ADMM_%s_%s_d%i.wav'%(i,j_admm[0], j_admm[1], j_admm[2]))
            distmat[i,j+len(GD_algs),:] = measuresDiv(original, x_gen, divs)
            
        for j in range(len(other_algs)):
            j_other = other_algs[j]
            x_gen, sr = l.load(sig_path+'%i_%s.wav'%(i,j_other))
            distmat[i,j+len(GD_algs)+len(ADMM_algs),:] = measuresDiv(original, x_gen, divs)
            
    plt.close('all')
    distonlymat = distmat[:,:-1,:]
    refmat = distmat[:,-1,:]
    refmat = np.expand_dims(refmat, 1)
    impmat = distonlymat - np.broadcast_to(refmat, distonlymat.shape)
    colors = 5*['mediumturquoise']+(len(GD_algs)-5)*['darkcyan']+len(ADMM_algs)*['orangered']+3*['gold']+['white']
    plt.rcParams.update({'font.weight':'regular','font.size':19})
    
    #Create figure for every metric in divs
    for k in range(len(divs)):
        fig, ax = plt.subplots(figsize=(18,9))
        bplot = ax.boxplot(distmat[:,:,k], patch_artist=True, showfliers=False,
                           showmeans=True, labels=labels, 
                           medianprops={'color':'black'}, 
                           meanprops={'markerfacecolor':'black',
                                      'markeredgecolor':'black',
                                       'marker':'o'})
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        ax.yaxis.grid(True)
        plt.xticks(rotation=30, fontweight='demi')
        plt.yticks(fontweight='demi')
        if divs[k]=='sc':
            tpp = np.max(distmat[:,-1,k])
            btt = np.min(distmat[:,:,k])
            plt.ylim(top=1.05*tpp, bottom = 0.9*btt)
            fig.savefig(sig_path+'%s.pdf'%(divs[k]), bbox_inches='tight', pad_inches=0)
            ax.set_yscale('log')
            plt.ylim(top=1, bottom = 0.9*btt)
            plt.yticks(fontweight='demi')
            plt.grid(True,which="minor", axis='y', linestyle='--')
            fig.savefig(sig_path+'log_%s.pdf'%(divs[k]), bbox_inches='tight', pad_inches=0)
        elif divs[k]=='stoi':
            btt = np.min(distmat[:,-1,k])
            plt.ylim(bottom= btt - 0.01*np.abs(btt))
            fig.savefig(sig_path+'%s.pdf'%(divs[k]), bbox_inches='tight', pad_inches=0)
        elif divs[k]=='sli_snr':
            fig, ax = plt.subplots(figsize=(18,9))
            bplot = ax.boxplot(impmat[:,:,k], patch_artist=True, showfliers=False,
                               showmeans=True, labels=labels[:-1], 
                               medianprops={'color':'black'}, 
                               meanprops={'markerfacecolor':'black',
                                          'markeredgecolor':'black',
                                           'marker':'o'})
            for patch, color in zip(bplot['boxes'], colors[:-1]):
                patch.set_facecolor(color)
            ax.yaxis.grid(True)
            plt.xticks(rotation=30, fontweight='demi')
            plt.yticks(fontweight='demi')
            plt.ylim(bottom= -1)
            fig.savefig(sig_path+'%s.pdf'%(divs[k]), bbox_inches='tight', pad_inches=0)
        else:
            btt = np.min(distmat[:,-1,k])
            plt.ylim(bottom= btt - 0.1*np.abs(btt))
            fig.savefig(sig_path+'%s.pdf'%(divs[k]), bbox_inches='tight', pad_inches=0)

# %%    
if __name__ == '__main__':
    main(sys.argv[1:])