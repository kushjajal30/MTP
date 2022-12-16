import torch
import numpy as np
from REMGeneration import config as data_config
from libpysal.weights import lat2W
from tqdm import tqdm

#Convert sparse scipy matrix to torch sparse tensor
def crs_to_torch_sparse(x):
    #
    # Input:
    # x = crs matrix (scipy sparse matrix)
    # Output:
    # w = weight matrix as toch sparse tensor
    x = x.tocoo()
    values = x.data
    indices = np.vstack((x.row, x.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = x.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense().cuda()

# Create spatial weight matrix
def make_sparse_weight_matrix(h, w, rook=False):
    #
    # Input:
    # h = height
    # w = width
    # rook = use rook weights or not
    # Output:
    # w = weight matrix as toch sparse tensor
    w = lat2W(h, w, rook=rook)
    return crs_to_torch_sparse(w.sparse)

# Local Moran's I
def mi(x, w_sparse):
    #
    # Input:
    # x = input data tensor (flattened or image)
    # w_sparse = spatial weight matrix; torch sparse tensor
    # Output:
    # mi = output data - local Moran's I
    #
    x = x.reshape(-1)
    n = len(x)
    n_1 = n - 1
    z = x - x.mean()
    sx = x.std()
    z /= sx
    den = (z * z).sum()
    zl = torch.mm(w_sparse, z.reshape(-1, 1)).reshape(-1)
    mi = n_1 * z * zl / den
    return mi

mi_weights = make_sparse_weight_matrix(data_config.__terrain_size__,data_config.__terrain_size__)
regularizer_cnn_weights = torch.Tensor([[1,1,1],[1,-8,1],[1,1,1]]).view(1,1,3,3)
regularizer_cnn_weights = regularizer_cnn_weights.to('cuda',dtype=torch.float)

def get_terrain_from_rem(rem,gen,path,updates,lr,lambda_regularizer,lambda_regularizer_cnn,lambda_similarity,lambda_morans_i):
    l2_loss_cal = torch.nn.MSELoss(reduction='none')
    recon_ter = torch.zeros_like(rem, requires_grad=True).to('cuda')
    optim = torch.optim.Adam([recon_ter], lr=lr)
    losses = []
    for iteration in tqdm(range(updates)):

        out_rem = gen(recon_ter)
        cnn_regularizer = torch.mean(
            torch.abs(torch.nn.functional.conv2d(recon_ter, regularizer_cnn_weights, padding='valid')))
        # l1_loss = l1_loss_cal(out_rem,rem)
        l2_loss = (l2_loss_cal(out_rem, rem) * (path == 1)).sum()
        l1_regularizer_term = torch.mean(torch.abs((recon_ter)))
        l2_regularizer_term = torch.mean(recon_ter ** 2)
        if iteration < 10:
            morans_i = 0
        else:
            morans_i = mi(recon_ter, mi_weights).mean()
        loss = lambda_similarity * l2_loss + lambda_regularizer * l1_regularizer_term + lambda_regularizer_cnn * cnn_regularizer + lambda_morans_i * morans_i

        if recon_ter.grad != None:
            recon_ter.grad.zero_()

        losses.append(loss.item())
        loss.backward(retain_graph=True)
        optim.step()

    return recon_ter[0][0].to('cpu').clone().detach(),out_rem[0][0].to('cpu').clone().detach(),losses