# Phase retrieval with Bregman divergences GitHub repository
You will find here the code related to the _Phase retrieval with Bregman divergences and application to audio signal recovery_ article. Please cite the [corresponding paper](https://arxiv.org/abs/2010.00392) if you use any piece of this work. Audio examples of reconstructed signals are available [online](https://magronp.github.io/demos/jstsp21.html). Code can be downloaded [here](https://framadrop.org/lufi/r/FR_RWFDe2r#MhdYAgFUXPHX8OI/z8mHkIGlB/sxZpHWW6td4f4K6wY=).

To run the phase retrieval experiment with exact spectrograms, please run the ``main_PR.py`` script. For the phase retrieval experiment with modified spectrograms, please run ``main_PRMod.py``.
Figures can be generated with ``plotter_PR.py`` and ``plotter_PRMod.py`` scripts.
A demonstration can be run with ``demo.py`` (phase retrieval with exact spectrograms from "speech" corpus, cf. Section 4.2).

The following Python librairies are necessary to run the code properly:
- [Librosa](https://librosa.org/)
- [Numpy](https://numpy.org/)
- [Scipy](https://www.scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pystoi](https://pypi.org/project/pystoi/)
- [Tqdm](https://github.com/tqdm/tqdm)

