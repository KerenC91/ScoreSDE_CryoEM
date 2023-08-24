import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm_notebook, trange
from ScoreNet import ScoreNet
from scoreSdeTests import diffusion_coeff, marginal_prob_std, loss_fn
import numpy as np

device = 'cpu' #@param ['cuda', 'cpu'] {'type':'string'}

if __name__ == '__main__':
    # Create input data
    #input_data = np.random.normal(mean, sigma, (224, 224))
    sigma = 25.0  # @param {'type':'number'}
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)

    n_epochs = 1  # @param {'type':'integer'}
    ## size of a mini-batch
    batch_size = 32  # @param {'type':'integer'}
    ## learning rate
    lr = 1e-4  # @param {'type':'number'}

    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)  # num_workers=4

    optimizer = Adam(score_model.parameters(), lr=lr)
    tqdm_epoch = trange(n_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            #x = x.to(device) #already on cpu device
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), 'ckpt.pth')