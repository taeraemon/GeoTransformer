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
RRE(deg): 0.645, RTE(m): 0.025
# 해보다가 에러나서 procrustes.py 수정 (det 계산을 cpu에서 하도록)
```

