from Modeling.ForwardMap_Pix2Pix import config
from Modeling.DataLoader import REMInTimeDataset
from Modeling.ForwardMap_Pix2Pix.model import GenUnet,Discriminator
from REMGeneration import config as data_config
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def main():

    device = 'cuda'

    traindataset = REMInTimeDataset(data_config,1024,True,transforms.ToTensor())
    validdataset = REMInTimeDataset(data_config,64,True,transforms.ToTensor())

    train_loader = DataLoader(traindataset,batch_size=config.__bs__,num_workers=config.__dataloader_workers__)
    valid_loader = DataLoader(validdataset,batch_size=config.__bs__,num_workers=config.__dataloader_workers__)

    if config.__retrain__ and os.path.exists(config.__model_path__):
        gen = os.path.join(config.__model_path__,'generator.pth')
        dis = os.path.join(config.__model_path__,'discriminator.pth')

    else:
        gen = GenUnet()
        dis = Discriminator(2)

    optimizer_g = optim.Adam(gen.parameters(), lr=config.__gen_lr__, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(dis.parameters(), lr=config.__dis_lr__, betas=(0.5, 0.999))

    bce_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()

    ones = torch.ones((config.__bs__,1)).to(device)
    zeros = torch.zeros((config.__bs__,1)).to(device)

    train_loss = []
    valid_loss = []

    for epoch in tqdm(range(config.__epochs__)):

        print(f"Training foir epoch:{epoch}")

        train_losses = []

        for i,(ter,rem) in enumerate(train_loader):

            ter = ter.to(device)
            rem = rem.to(device)

            rem_fake = gen(ter)

            dis_real_out = dis(torch.cat([rem,ter],dim=1))
            dis_fake_out = dis(torch.cat([rem_fake,ter],dim=1))

            gen_classification_loss = bce_loss(dis_fake_out.vies((config.__bs__,-1)),ones)
            gen_l1_loss = l1_loss(rem_fake,rem)

            gen_loss = gen_classification_loss+config.__gen_l1_lambda__*gen_l1_loss
            dis_loss = bce_loss(dis_fake_out.vies((config.__bs__,-1)),zeros) + bce_loss(dis_real_out.vies((config.__bs__,-1)),ones)

            gen.zero_grad()
            dis.zero_grad()

            gen_loss.backward(retain_graph=True)
            optimizer_g.step()
            dis_loss.backward()
            optimizer_d.step()

            train_losses.append((gen_classification_loss,gen_l1_loss,gen_loss,dis_loss))

        print(f"Training Gen Classification Loss:{sum([i[0] for i in train_losses])/config.__bs__}")
        print(f"Training Gen L1 Loss:{sum([i[1] for i in train_losses])/config.__bs__}")
        print(f"Training Gen Total Loss:{sum([i[2] for i in train_losses])/config.__bs__}")
        print(f"Training Dis Loss:{sum([i[3] for i in train_losses])/config.__bs__}")

        print(f"Validating for epoch:{epoch}")

        valid_losses = []

        for i,(ter,rem) in enumerate(valid_loader):

            ter = ter.to(device)
            rem = rem.to(device)

            rem_fake = gen(ter)

            dis_real_out = dis(rem)
            dis_fake_out = dis(rem_fake)

            gen_classification_loss = bce_loss(dis_fake_out.vies((config.__bs__,-1)),ones)
            gen_l1_loss = l1_loss(rem_fake,rem)

            gen_loss = gen_classification_loss+config.__gen_l1_lambda__*gen_l1_loss
            dis_loss = bce_loss(dis_fake_out.vies((config.__bs__,-1)),zeros) + bce_loss(dis_real_out.vies((config.__bs__,-1)),ones)

            valid_losses.append((gen_classification_loss, gen_l1_loss, gen_loss, dis_loss))


        print(f"Validation Gen Classification Loss:{sum([i[0] for i in valid_losses])/config.__bs__}")
        print(f"Validation Gen L1 Loss:{sum([i[1] for i in valid_losses])/config.__bs__}")
        print(f"Validation Gen Total Loss:{sum([i[2] for i in valid_losses])/config.__bs__}")
        print(f"Validation Dis Loss:{sum([i[3] for i in valid_losses])/config.__bs__}")

        train_loss.append(train_losses)
        valid_loss.append(valid_losses)

        save_rem_real = rem[0].to('cpu').clone().detach()
        save_rem_fake = rem_fake[0].to('cpu').clone().detach()

        fig,axs = plt.subplots(1,2,figsize=(10,5))
        axs[0][1].matshow(save_rem_real,cmap='jet',vmax=30,vmin=-100)
        axs[0][2].matshow(save_rem_fake,cmap='jet',vmax=30,vmin=-100)

        plt.savefig(f"epoch:{epoch}.png")

if __name__=='__main__':
    main()