# nvmed

Environment setup
```
conda create --name=py310  python=3.10
conda activate py310

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install xformers -c xformers

pip install -U pip 
pip install -U pip  lightning rich torch_fidelity torch_ema datasets diffusers hydra-core tensorboard
pip install -U pip 
pip install -U transformers
pip install -U plotly
pip install -U diffusers
pip install -U lightning
pip install -U tensorboard

pip install -U monai[all]
pip install -U einops
pip install -U lmdb
pip install -U mlflow
pip install -U clearml
pip install -U scikit-image
pip install -U pytorch-ignite
pip install -U pandas
pip install -U pynrrd
pip install -U gdown

pip install -U git+https://github.com/Project-MONAI/GenerativeModels.git 
```

Check torch has been compiled with CUDA
```
python -c 'import torch; print(torch.cuda.is_available())'
python -c 'import torch; torch.randn(1, device="cuda:0")'
python -c "import torch; import monai; from monai.config import print_config; print_config()"
```

-----


Change paths of folders in conf/hparams.yaml to the located data
```
train_image2d_folders
val_image2d_folders
test_image2d_folders
```

```
python main.py --config-name hparams
```