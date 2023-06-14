# Guia_install_tensorflow_pytorch_gpu

[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://www.tensorflow.org/install?hl=pt-br)
[![](https://img.shields.io/badge/Keras-red.svg)](https://keras.io/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://developer.nvidia.com/)
[![](https://img.shields.io/badge/PyTorch-red.svg)](https://www.run.ai/guides/gpu-deep-learning/pytorch-gpu)
[![](https://img.shields.io/badge/PyTorch-GPU-blue.svg)](https://www.run.ai/guides/gpu-deep-learning/pytorch-gpu)

## Introdução

A instalação do TensorFlow e do PyTorch é um passo essencial para começar a desenvolver e executar projetos de aprendizado de máquina e deep learning em seu notebook. Essas bibliotecas são amplamente utilizadas pela comunidade de ciência de dados e oferecem uma variedade de recursos poderosos para treinar e implantar modelos de aprendizado de máquina. Neste notebook, vamos fornecer instruções passo a passo sobre como instalar o TensorFlow e o PyTorch no Windows.

Instalação do TensorFlow no Windows:

Aqui estão as etapas para instalar o TensorFlow no Windows:

1. Verifique os requisitos:

Certifique-se de ter o Python instalado em seu sistema. O TensorFlow é compatível com o Python 3.5 a 3.8.
Tenha em mente que o TensorFlow não é oficialmente suportado em versões do Python superiores a 3.8 no Windows.

2. Crie um ambiente virtual (opcional):

É recomendado criar um ambiente virtual para isolar o ambiente de desenvolvimento do TensorFlow. Use o comando abaixo para criar um ambiente virtual chamado "tensorflow-env":

```bash
python -m venv tensorflow-env
```
3.Ative o ambiente virtual (opcional):
Para ativar o ambiente virtual, execute o seguinte comando:

```bash
tensorflow-env\Scripts\activate
```
4. Instale o TensorFlow:

Abra o prompt de comando (CMD) ou o PowerShell.
Execute o seguinte comando para instalar o TensorFlow:

```bash
pip install tensorflow-gpu
```
5.Verifique a instalação:

Para verificar se o TensorFlow foi instalado corretamente, abra o interpretador do Python e importe o TensorFlow:

```bash
import tensorflow as tf
```

#Instalação do PyTorch no Windows

Aqui estão as etapas para instalar o PyTorch no Windows:

1. Verifique os requisitos:

Certifique-se de ter o Python instalado em seu sistema. O PyTorch é compatível com o Python 3.6 ou superior.

2. Crie um ambiente virtual (opcional):

Se preferir, você pode criar um ambiente virtual para o PyTorch da seguinte maneira:

```bash
python -m venv pytorch-env
```
3.Ative o ambiente virtual (opcional):

Para ativar o ambiente virtual, execute o seguinte comando:
```bash
pytorch-env\Scripts\activate
```

4.Instale o PyTorch:

Abra o prompt de comando (CMD) ou o PowerShell.
Execute o seguinte comando para instalar o PyTorch, substituindo <version> pela versão desejada (por exemplo, 1.9.0):

```bash
pip install torch==<version> torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
```
  
5.Verifique a instalação:

Para verificar se o PyTorch foi instalado corretamente, abra o interpretador do Python e importe o PyTorch:
  
```bash
import torch
```
  
# Conclusão
Parabéns! Agora você tem o TensorFlow e o PyTorch instalados no seu notebook Windows. 

Agora você pode explorar e utilizar todas as funcionalidades oferecidas por essas poderosas bibliotecas.

Lembre-se de que o TensorFlow e o Torch são apenas algumas das muitas bibliotecas disponíveis para aprendizado de máquina e deep learning. Conforme você avança em sua jornada nesse campo, pode ser útil explorar outras bibliotecas e frameworks populares, como Keras, PyTorch, scikit-learn, entre outros.

Experimente diferentes algoritmos, arquiteturas de rede e técnicas de pré-processamento de dados para obter resultados mais precisos e eficientes. A documentação oficial dessas bibliotecas é uma ótima fonte de informações e exemplos práticos para ajudá-lo em seu aprendizado contínuo.

Lembre-se também de atualizar regularmente suas bibliotecas para aproveitar as melhorias e correções de bugs mais recentes. Verifique os sites oficiais do TensorFlow e do Torch para obter informações sobre atualizações e novas versões.

Espero que este notebook tenha sido útil para você aprender como instalar o TensorFlow e o Torch em seu ambiente de desenvolvimento. Agora você está pronto para mergulhar no mundo do aprendizado de máquina e deep learning. Boa sorte em seus projetos futuros e aproveite a jornada de descoberta e inovação!

Se você tiver alguma dúvida ou precisar de mais assistência, não hesite em perguntar. Estou aqui para ajudar.
  
Consulte os notebook para mais detalhes.
