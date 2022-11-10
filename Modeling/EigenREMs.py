from Modeling.DataLoader import REMInTimeDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def get_eigen_rems(data_config,nrems=2000,top_n=1):

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

    rems = np.stack(rems)

    svd_x = rems.reshape(nrems, -1)
    mean_x = np.mean(svd_x, axis=0)
    svd_x_centered = svd_x - mean_x

    U, S, V = np.linalg.svd(svd_x_centered)

    top_n_components = V[:top_n].reshape(top_n,rems.shape[1],rems.shape[2])

    return top_n_components