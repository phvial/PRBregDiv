# Phase retrieval with Bregman divergences GitHub repository
You will find here the code related to the _Phase retrieval with Bregman divergences and application to audio signal recovery_ article. Please cite the [corresponding paper](https://arxiv.org/abs/2010.00392) if you use any piece of this work. Audio examples of reconstructed signals are available [online](https://magronp.github.io/demos/jstsp21.html). Code can be downloaded [here (fix link)](https://).

To run the phase retrieval experiment, generate estimated signals and corresponding figures, please run the ``main.py`` script. A demonstration can be run with ``demo.py``.

The following Python librairies are necessary to run the code properly:
- [Librosa](https://librosa.org/)
- [Numpy](https://numpy.org/)
- [Scipy](https://www.scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pystoi](https://pypi.org/project/pystoi/)
- [Tqdm](https://github.com/tqdm/tqdm)

# Demonstration phase retrieval script
Run ``demo.py`` with optional arguments ``-f <filepath> -n <number of iterations> -a <algorithm>``.
Default values are respectively ``datasets/tarantella.wav``, ``1000`` iterations and ``gradient descent``.

# Phase retrieval with exact spectrograms experiment
Run ``main.py`` with optional arguments ``-d <dataset> -n <number of iterations> -f <number of files>``.
Default values are respectively ``speech`` dataset, ``2500`` iterations and ``10`` sound signals.

# Phase retrieval with modified spectrograms experiment
Run ``main.py`` with argument ``-s <input SNR in dB>`` and optional arguments detailed above.

