# Phase retrieval with Bregman divergences GitHub repository
Here is the code related to the article entitled [Phase retrieval with Bregman divergences and application to audio signal recovery](https://arxiv.org/abs/2010.00392).
Audio examples of reconstructed signals are available [online](https://magronp.github.io/demos/jstsp21.html).

To reproduce the results from the paper (generate the estimated signals and the corresponding figures), run the ``main.py`` script. Alternatively, a simple demonstration can be run with ``demo.py``.

The following Python librairies are necessary to run the code properly:
- [Librosa](https://librosa.org/)
- [Numpy](https://numpy.org/)
- [Scipy](https://www.scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pystoi](https://pypi.org/project/pystoi/)
- [Tqdm](https://github.com/tqdm/tqdm)

# Demonstration

The ``demo.py`` script allows you to perform phase retrieval (using the exact magnitude spectrogram) and generate the corresponding time signal using any audio file.
To do so, run:

``demo.py -f <filepath> -n <number of iterations> -a <algorithm>``.

where default values are respectively ``datasets/tarantella.wav``, ``1000`` and ``gradient descent``.
Implemented algorithms are ``gradient descent``, ``ADMM``, ``GLA``, ``FGLA``, ``GLADMM``. With ``gradient descent`` and ``ADMM``, the loss is the left Kullback-Leibler divergence (d=1).

# Reproducing the results from the paper

To reproduce the results from the paper, run:

``main.py -s <input SNR in dB> -d <dataset> -n <number of iterations> -f <number of files>``.

If the argument ``-s`` is not specified, the script will perform phase retrieval from exact spectrograms (Section IV-B in the paper).
Conversely, you can specify the input SNR to perform phase retrieval from modified spectrograms (Section IV-C in the paper).
Default values for the other parameters are respectively ``speech``, ``2500``, and ``10``.


### Reference

<details><summary>If you use any of this code for your research, please cite our paper:</summary>
  
```latex
@article{Vial2021jstsp,  
  author={P.-H. Vial and P. Magron and T. Oberlin and C. F{\'e}votte},  
  title={Phase recovery with Bregman divergences and application to audio signal recovery},  
  booktitle={IEEE Journal of Selected Topics in Signal Processing (JSTSP)},  
  year={2021}
}
```
