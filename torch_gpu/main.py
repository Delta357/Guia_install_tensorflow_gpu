# Guia Torch - GPU

# Comando instalação por anaconda
!conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Comando por instalação por pip
!pip install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Versão Torch
import torch
print(torch.__version__)

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

# Versão driver CUDA
! nvcc --version

import torch
import sys
print('__CUDA VERSION', )

from subprocess import call
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')

# Torch GPU
!nvidia-smi

# Print gpus ativadas
import torch
print("GPU é", "Ativada" if torch.cuda.is_available() else "NAO ATIVADA")

# Números com gpu
import torch
torch.cuda.device_count()

# GPU
print("Nome GPU:", torch.cuda.get_device_name(0))

# Verificando dados gpu com drive cuda

dr = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Drive usando:", dr)
print()

# Info memorias usando driver CUDA
if dr.type == "cuda":
    print("Nome GPU:", torch.cuda.get_device_name(0))
    print("Mémoria usada GPU")
    print('Alocado:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Em cache:', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
 
torch.cuda.max_memory_cached(device=None)
torch.cuda.memory_allocated(device=None)

# Exemplo aplicação torch
torch.rand(10).to(device)

torch.rand(10, device=device)
