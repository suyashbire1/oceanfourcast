# OceanFourcast 
### Can transformer methods be used to create fast emulators for forward and partial derivative computations in ocean modeling?

## Installation
```
git clone git@github.com:suyashbire1/oceanfourcast.git
cd oceanfourcast
conda create --name oceanfourcast
conda activate
pip install -e .
```

## Download data
```
mkdir -p data/processed/
scp name@servername.com:/path/to/file/mitgcm/double_gyre/run3/dynDiag_subset.nc data/processed/. # Sample dataset
scp name@servername.com:/path/to/file/mitgcm/double_gyre/run3/dynDiag.nc data/processed/. # Full dataset
```

## Train
```
python oceanfourcast/train.py --data_file "data/processed/dynDiag_subset.nc" --batch_size 2
```