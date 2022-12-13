from Modeling.DataLoader import REMInTimeDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def get_eigen_rems(data_config,nrems=2000,top_n=1):

    data_config.__number_of_buildings__ = max(16,data_config.__number_of_buildings__)
    data_config.__terrain_size__ = data_config.__terrain_size__ * 2

    dgen = REMInTimeDataset(data_config, nrems)

    intime_dataloader = DataLoader(
        dgen,
        batch_size=25,
        drop_last=True,
        shuffle=True,
        num_workers=28
    )

    rems = []

    for _, rem in tqdm(intime_dataloader):
        rems.append(rem.detach().numpy())

    rems = np.concatenate([i[:, 0, :, :] for i in rems], axis=0)
    out = []
    for i in rems:
        out.append(i[64:, 64:])
        out.append(np.flip(i[64:, :64], axis=1))
        out.append(np.flip(i[:64, 64:], axis=0))
        out.append(np.flip(i[:64, :64], axis=(0, 1)))

    rems = np.stack(out)
    svd_x = rems.reshape(nrems, -1)

    U, S, V = np.linalg.svd(svd_x)

    top_n_components = V[:top_n]

    mean_weight = np.mean(np.dot(svd_x, top_n_components.T), axis=0)
    average_trend = mean_weight @ top_n_components

    data_config.__terrain_size__ = data_config.__terrain_size__ / 2

    return average_trend

def get_mean_rem(data_config,nrems=2000):
    data_config.__number_of_buildings__ = max(16, data_config.__number_of_buildings__)
    data_config.__terrain_size__ = data_config.__terrain_size__ * 2

    dgen = REMInTimeDataset(data_config, nrems, clip=False)

    intime_dataloader = DataLoader(
        dgen,
        batch_size=25,
        drop_last=True,
        shuffle=True,
        num_workers=28
    )

    rems = []

    for _, rem in tqdm(intime_dataloader):
        rems.append(rem.detach().numpy())

    rems = np.concatenate([i[:, 0, :, :] for i in rems], axis=0)
    out = []
    for i in rems:
        out.append(i[64:, 64:])
        out.append(np.flip(i[64:, :64], axis=1))
        out.append(np.flip(i[:64, 64:], axis=0))
        out.append(np.flip(i[:64, :64], axis=(0, 1)))

    rems = np.stack(out)

    return np.mean(rems,axis=0)