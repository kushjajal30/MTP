from Modeling.ForwardMap_Pix2Pix import config
from Modeling.DataLoader import REMInTimeDataset
from Modeling.ForwardMap_Pix2Pix.model import GenUnet, Discriminator
from Modeling.EigenREMs import *
from REMGeneration import config as data_config
from matplotlib import pyplot as plt
import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


def main():

    device = 'cuda'

    if config.__use_pixel_norm__:
        traindataset = REMInTimeDataset(data_config, 1024 , base_rem=get_mean_rem(data_config))
    else:
        traindataset = REMInTimeDataset(data_config,1024)

    train_loader = DataLoader(traindataset, batch_size=config.__bs__, num_workers=config.__dataloader_workers__)

    gen_path = os.path.join(config.__model_path__, 'generator.pth')
    dis_path = os.path.join(config.__model_path__, 'discriminator.pth')

    if config.__retrain__ and os.path.exists(gen_path) and os.path.exists(dis_path):
        gen = torch.load(gen_path).to(device)
        dis = torch.load(dis_path).to(device)

    else:
        if not os.path.exists(config.__model_path__):
            os.mkdir(config.__model_path__)
        gen = GenUnet().to(device)
        dis = Discriminator(2).to(device)

    optimizer_g = optim.Adam(gen.parameters(), lr=config.__gen_lr__, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(dis.parameters(), lr=config.__dis_lr__, betas=(0.5, 0.999))

    bce_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    ones = torch.ones((config.__bs__, 1)).to(device)
    zeros = torch.zeros((config.__bs__, 1)).to(device)

    train_loss = []

    print(config.__gen_lr__, config.__dis_lr__, config.__gen_bce_lambda__, config.__gen_l1_lambda__,
          config.__gen_l2_lambda__)

    for epoch in tqdm(range(config.__epochs__)):

        print(f"Training for epoch:{epoch}")

        train_losses = []

        for i, (ter, rem, ref_rem) in tqdm(enumerate(train_loader), total=int(1024 / config.__bs__)):
            ter = ter.to(device, dtype=torch.float)
            rem = rem.to(device, dtype=torch.float)
            ref_rem = ref_rem.to(device, dtype=torch.float)

            rem_fake = gen(torch.cat([ter, ref_rem], dim=1))

            dis_real_out = dis(torch.cat([ter, rem], dim=1))
            dis_fake_out = dis(torch.cat([ter, rem_fake], dim=1))

            gen_classification_loss = bce_loss(dis_fake_out.view((config.__bs__, -1)), ones)
            gen_l1_loss = l1_loss(rem_fake, rem)
            gen_l2_loss = l2_loss(rem_fake, rem)

            gen_loss = config.__gen_bce_lambda__ * gen_classification_loss + config.__gen_l1_lambda__ * gen_l1_loss + config.__gen_l2_lambda__ * gen_l2_loss
            dis_loss = bce_loss(dis_fake_out.view((config.__bs__, -1)), zeros) + bce_loss(
                dis_real_out.view((config.__bs__, -1)), ones)

            gen.zero_grad()
            dis.zero_grad()

            gen_loss.backward(retain_graph=True)
            dis_loss.backward()
            optimizer_g.step()
            optimizer_d.step()

            train_losses.append((gen_classification_loss, gen_l1_loss, gen_l2_loss, gen_loss, dis_loss))

        print(f"Training Gen Classification Loss:{sum([i[0] for i in train_losses]) / config.__bs__}")
        print(f"Training Gen L1 Loss:{sum([i[1] for i in train_losses]) / config.__bs__}")
        print(f"Training Gen L2 Loss:{sum([i[2] for i in train_losses]) / config.__bs__}")
        print(f"Training Gen Total Loss:{sum([i[3] for i in train_losses]) / config.__bs__}")
        print(f"Training Dis Loss:{sum([i[4] for i in train_losses]) / config.__bs__}")
        train_loss.append(train_losses)

        config.__gen_l1_lambda__ *= 0.95
        config.__gen_l2_lambda__ *= 0.95

        save_rem_real = rem[0][0].to('cpu').clone().detach()
        save_rem_fake = rem_fake[0][0].to('cpu').clone().detach()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        a = axs[0].matshow(save_rem_real, cmap='jet', vmax=1, vmin=0)
        fig.colorbar(a, ax=axs[0])
        a = axs[1].matshow(save_rem_fake, cmap='jet', vmax=1, vmin=0)
        fig.colorbar(a, ax=axs[1])
        plt.show()

        plt.savefig(f"epoch:{epoch}.png")

        torch.save(gen, gen_path)
        torch.save(dis, dis_path)


if __name__ == '__main__':
    main()