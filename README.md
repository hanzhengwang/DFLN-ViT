# DFLN-ViT

Code for paper "Spatial-Channel Enhanced Transformer for Visible-Infrared Person Re-Identification"


## Requirments:
pytorch: 1.6.0

torchvision: 0.2.1

numpy: 1.15.0

python: 3.7


## Dataset:
**SYSU-MM01**

**Reg-DB**


## Run:
### SYSU-MM01:
1. prepare training set
```
python pre_process_sysu.py
```
2. train model


To train a model with on SYSU-MM01 with a single GPU device 0, run:
```
python train.py --dataset sysu --lr 0.01 --drop 0.0 --trial 1 --gpu 0 --epochs 60 --w_hc 0.5 --per_img 8 
```

3. evaluate model(single-shot all-search)
```
python test.py --dataset sysu --lr 0.01 --drop 0.0 --trial 1 --gpu 0 --low-dim 512 --w_hc 0.5 --mode all --gall-mode single --model_path 'Your model path'
```

### Reg-DB:
1. train model
```
python train.py --dataset regdb --lr 0.02 --drop 0.0 --trial 1 --gpu 0 --epochs 60 --w_hc 0.5 --per_img 8
```

2. evaluate model
```
python test.py --dataset regdb --lr 0.02 --drop 0.0 --trial 1 --gpu 0 --low-dim 512 --w_hc 0.5 --model_path 'Your model path'
```

## Results:

 SYSU-MM01    [BaiduYun(code:3gfq)](https://pan.baidu.com/s/1xRfq2LpvXHjHQy2NdISWVQ )
 RegDB    [BaiduYun(code:ez32 )](https://pan.baidu.com/s/1opIrI9PflTMk88mPFdBcZw)



**Citation**

Most of the code are borrowed from https://github.com/98zyx/Hetero-center-loss-for-cross-modality-person-re-id. Thanks a lot for the author's contribution.

Please cite the following paper in your publications if it is helpful:
```
@article{zhao2022spatial,
  title={Spatial-Channel Enhanced Transformer for Visible-Infrared Person Re-Identification},
  author={Zhao, Jiaqi and Wang, Hanzheng and Zhou, Yong and Yao, Rui and Chen, Silin and El Saddik, Abdulmotaleb},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}

@article{zhu2020hetero,
  title={Hetero-center loss for cross-modality person re-identification},
  author={Zhu, Yuanxin and Yang, Zhao and Wang, Li and Zhao, Sai and Hu, Xiao and Tao, Dapeng},
  journal={Neurocomputing},
  volume={386},
  pages={97--109},
  year={2020},
  publisher={Elsevier}
}
```




