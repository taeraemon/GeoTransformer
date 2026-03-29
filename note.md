- 가상환경 및 개발환경 셋업

```
pyenv global 3.8.20

python3 -m venv env
source env/bin/activate

export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install --upgrade pip
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install typeguard

python setup.py build develop
```

 

- pre-trained 설치

```
https://github.com/qinzheng93/GeoTransformer/releases
에서 다운받아서
/weights 아래 삽입
```

 

- demo 돌려보기

```
cd experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python demo.py   --src_file=../../data/demo/src.npy   --ref_file=../../data/demo/ref.npy   --gt_file=../../data/demo/gt.npy   --weights=../../weights/geotransformer-3dmatch.pth.tar

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python demo.py   --src_file=../../data/demo/src.npy   --ref_file=../../data/demo/ref.npy   --gt_file=../../data/demo/gt.npy   --weights=../../output/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/epoch-2.pth.tar

CUDA_VISIBLE_DEVICES=0 python demo.py \
  --src_file=../../data/demo/src.npy \
  --ref_file=../../data/demo/ref.npy \
  --weights=../../output/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/epoch-2.pth.tar


RRE(deg): 0.645, RTE(m): 0.025
```

에러나서 [procrustes.py](http://procrustes.py) 수정 (det 계산을 cpu에서 하도록)

 

- 훈련을 위한 데이터셋 다운받기

[https://github.com/prs-eth/OverlapPredator/blob/main/scripts/download_data_weight.sh](https://github.com/prs-eth/OverlapPredator/blob/main/scripts/download_data_weight.sh)

```
wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gseg/Predator/data.zip
unzip data.zip
rm data.zip
```

 

- 훈련을 위해 데이터셋 배치

다운받은 데이터셋 구조

```
/data
    /indoor
        /train    (복사)
        /test     (복사)
    ...
```

배치해야할 구조

```
/data
    /3DMatch
        /metadata     (기존)
        /data         (내가 생성)
            /train    (붙여넣기)
            /test     (붙여넣기)
```

 

- 훈련 이제 돌려보기

```
CUDA_VISIBLE_DEVICES=0 python trainval.py
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 trainval.py
```

loss.py에서 오류나서 [loss.py](http://loss.py) 수정 (inverse를 cpu에서 하도록)