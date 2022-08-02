import torch
from torch.utils.data import Dataset
import xarray

class OceanDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.ncfile = os.path.join(self.data_dir + "dynDiag.nc")

    def __len__(self):
        with xr.open_Dataset(self.ncfile, decode_times=False) as ds:
            return len(ds.T)

    def __getitem__(self, idx):
        with xr.open_Dataset(self.ncfile, decode_times=False) as ds:
            usurf = ds.UVEL.isel(T=idx, Zmd000015=0).squeeze()
            vsurf = ds.VVEL.isel(T=idx, Zmd000015=0).squeeze()
            wmid = ds.WVEL.isel(T=idx, Zld000015=7).squeeze()
            thetasurf = ds.THETA.isel(T=idx, Zmd000015=0).squeeze()
            Psurf = ds.PHIHYD.isel(T=idx, Zmd000015=0).squeeze()
            Pmid = ds.PHIHYD.isel(T=idx, Zmd000015=7).squeeze()
            data = np.hstack([usurf, vsurf, wmid, thetasurf, Psurf, Pmid])
            T = ds.T.isel(T=idx).values
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            T = self.target_transform(T)
        return data, T
