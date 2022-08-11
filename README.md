## Introduction
1. FastSpeech2, HiFi-GAN, VITS 오픈 소스를 활용하여 JETS(End-To-End)를 간단 구현하고 한국어 데이터셋(KSS)을 사용해 빠르게 학습합니다.
2. Adversarial Training에서 Discriminator는 VITS에서 사용한 모듈을 그대로 사용합니다.
3. 본 레포지토리에서 HiFi-GAN에서 제안하는 l1 reconstructure loss(only log mel magnitude)를 그대로 사용하면 adversarial loss에서 issue가 발생합니다. 따라서 log stft magnitude와 l1 norm이 같이 계산되는 stft loss로 대체했습니다.
4. 확장성을 위하여 기존 FastSpeech2 구조에서 Decoder 대신 VITS의 Normalizing Flows(CouplingLayer)를 사용하였습니다. 따라서 Posterior Encoder도 같이 사용됩니다. (Quality 향상, Voice Conversion 목적)
5. 기존 Posterior Encoder는 Linear Spectrogram을 입력값으로 사용하지만, 본 레포지토리에서는 Mel Spectrogram을 사용합니다.
6. 기존 오픈소스는 MFA기반 preprocessing을 진행한 상태에서 학습을 진행하지만 본 레포지토리에서는 alignment learning 기반 학습을 진행하고 preprocessing으로 인해 발생할 수 있는 디스크 용량 문제를 방지하기 위해 data_utils.py로부터 학습 데이터가 feeding됩니다.
7. conda 환경으로 진행해도 무방하지만 본 레포지토리에서는 docker 환경만 제공합니다. 기본적으로 ubuntu에 docker, nvidia-docker가 설치되었다고 가정합니다.
8. GPU, CUDA 종류에 따라 Dockerfile 상단 torch image 수정이 필요할 수도 있습니다.
9. preprocessing 단계에서는 학습에 필요한 transcript와 stats 정도만 추출하는 과정만 포함되어 있습니다.
10. 그 외의 다른 preprocessing 과정은 필요하지 않습니다.
11. End-To-End & Adversarial training 기반이기 때문에 우수한 품질의 오디오를 생성하기 위해선 많은 학습을 필요로 합니다.

## Dataset
1. download dataset - https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset
2. `unzip /path/to/the/kss.zip -d /path/to/the/kss`
3. `mkdir /path/to/the/JETS/data/dataset`
4. `mv /path/to/the/kss.zip /path/to/the/JETS/data/dataset`

## Docker build
1. `cd /path/to/the/JETS`
2. `docker build --tag JETS:latest .`

## Training
1. `nvidia-docker run -it --name 'JETS' -v /path/to/JETS:/home/work/JETS --ipc=host --privileged JETS:latest`
2. `cd /home/work/JETS`
5. `ln -s /home/work/JETS/data/dataset/kss`
6. `python preprocess.py ./config/kss/preprocess.yaml`
7. `python train.py -p ./config/kss/preprocess.yaml -s ./config/kss/model.yaml -g ./config/kss/config_v1.json -t ./config/kss/train.yaml`
8. arguments
  * -p : preprocess config path
  * -s : synthesizer config path
  * -g : generator config path
  * -t : train config path
9. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Tensorboard losses
![tensorboard-losses](https://user-images.githubusercontent.com/69423543/184098829-5b010983-0ba7-4548-8c93-65ba7d7a95ed.png)


## Tensorboard Stats
![tensorboard-images](https://user-images.githubusercontent.com/69423543/184098838-96a21194-b503-4e06-bcdf-8ced38dcf2ed.png)


## Reference
1. [JETS: Jointly Training FastSpeech2 and HiFi-GAN for End to End Text to Speech](https://arxiv.org/abs/2203.16852)
2. [Comprehensive-Transformer-TTS](https://github.com/keonlee9420/Comprehensive-Transformer-TTS)
3. [Comprehensive-E2E-TTS](https://github.com/keonlee9420/Comprehensive-E2E-TTS)
4. [FastSpeech2](https://github.com/ming024/FastSpeech2)
5. [HiFi-GAN](https://github.com/jik876/hifi-gan)
6. [VITS](https://github.com/jaywalnut310/vits)
