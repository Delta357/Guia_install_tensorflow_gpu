# Instalação por comando anaconda
!conda install -c anaconda tensorflow-gpu 

# Instalação por comando pip
!pip install tensorflow-gpu==2.10.0

# Versão tensorflow
import tensorflow as tf
print(tf.__version__)

# Versão das bibliotecas geral
import sys
import nltk
import numpy as np
import seaborn as sns
import matplotlib
import pandas as pd
import sklearn as sk

print(f"Versão biblioteca Python {sys.version}")
print()
print(f"Versão biblioteca Pandas {pd.__version__}")
print()
print(f"Versão biblioteca Scikit-Learn {sk.__version__}")
print()
print(f"Versão biblioteca Numpy {np.__version__}")
print()
print(f"Versão biblioteca Seaborn {sns.__version__}")
print()
print('Versão biblioteca Matplotlib: {}'.format(matplotlib.__version__))
print()
print('Versão biblioteca NLTK: {}'.format(nltk.__version__))

# Tensorflow gpu
import tensorflow as tf
import tensorflow.keras

print(f"Tensorflow versão: {tf.__version__}")
print()
print(f"Keras versão: {tensorflow.keras.__version__}")

print(f"Tensorflow GPU")
print()
print(len(tf.config.list_physical_devices('GPU'))>0)

print(f"Tensorflow GPU")
print("GPU é", "Ativada" if tf.test.is_gpu_available() else "NAO ATIVADA")

# CUDA
! nvcc --version

import torch
import sys
print('__CUDA VERSION', )

from subprocess import call
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')

# Número GPU
import tensorflow as tf
print("Números de gpu ativada:", len(tf.config.list_physical_devices('GPU')))
