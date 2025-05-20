import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, negative_sampling
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-15
MAX_LOGSTD = 10


class VariationalGraphAutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VariationalGraphAutoEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv( hidden_channels, out_channels, cached=True)
        self.feature_decode = nn.Sequential(
            nn.Linear(out_channels,   hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear( hidden_channels,  hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear( hidden_channels, in_channels),
            # nn.Sigmoid()
            )


    def reparametrize(self, mu, logstd):
        return mu + torch.randn_like(logstd) * torch.exp(logstd)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


    def dot_product_decode_all(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
        return A_pred
    
    def dot_product_docode(self, z, edge_index, sigmoid = True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def recon_loss(self, z, pos_edge_index, neg_edge_index = None):
        pos_loss = -torch.log(
            self.dot_product_docode(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.dot_product_docode(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss


    def kl_loss(self, mu, logstd):
        logstd = logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))


    def forward(self, x, edge_index):
        mu, sigma = self.encode(x, edge_index)
        kl = self.kl_loss(mu, sigma)
        Z = self.reparametrize(mu, sigma)
        A_pred = self.dot_product_decode_all(Z)
        X_pred = self.feature_decode(Z)
        return kl, Z, A_pred, X_pred


# Train function with loss tracking
def train_vae(model, optimizer, x, edge_index, lambd=1e-5):
    model.train()
    optimizer.zero_grad()
    kl, z, A_tilde, X_tilde = model(x, edge_index)
    n = x.shape[0]

    # Reconstruction losses
    loss_A = model.recon_loss(z, edge_index)
    loss_X = ((x - X_tilde)**2).sum()

    # Total loss
    loss = kl / n + loss_A + lambd * loss_X
    loss.backward()
    optimizer.step()
    return loss_A.item(), loss_X.item(), kl.item(), loss.item()

def plot_losses(losses, save_path='loss_plot.png'):
    loss_A, loss_X, kl_loss, total_loss = zip(*losses)

    plt.figure(figsize=(10, 6))
    plt.plot(total_loss, label='Total Loss', color='blue')
    plt.plot(loss_A, label='Reconstruction Loss A', color='green')
    plt.plot(loss_X, label='Reconstruction Loss X', color='orange')
    plt.plot(kl_loss, label='KL Divergence', color='red')
    plt.title('Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


def from_vgae_to_graph(A, X, y,threshold = 0.5):
    mask = torch.eye(A.shape[0], dtype=torch.bool)
    A[mask] = 0 # remove self-loop
    A_sparse =  (A > threshold).type(torch.float32).to_sparse()
    edge_index = A_sparse.indices()
    edge_weight = A[edge_index[0], edge_index[1]]
    G = Data(x = X, y = y, edge_index = edge_index, edge_weight = edge_weight)
    return G


def generate_vae_samples(data, n_samples, in_channels, hidden_channels, out_channels, lr=0.01, epochs=500, lambd=1e-5, threshold=0.95, verbose = True, plot = False):
    # threshold should be pretty large as dot product results in dense graph
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VariationalGraphAutoEncoder(in_channels, hidden_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    losses = []  # To track losses

    # Training loop
    for epoch in range(epochs):
        loss_A, loss_X, kl, loss = train_vae(model, optimizer, x, edge_index, lambd)
        losses.append((loss_A, loss_X, kl, loss))
        if (epoch % 10 == 0 and verbose):
            print(f'Epoch {epoch:03d}, Loss={loss:.4f}, Recon Loss A={loss_A:.4f}, Recon Loss X={loss_X:.4f}, KL Loss={kl:.4f}')
    
    # Plot losses after training
    if plot:
        plot_losses(losses)

    

    samples = {}
    for i in range(n_samples):
        with torch.no_grad():
            kl, _, A_tilde, X_tilde = model(x, edge_index)
        G = from_vgae_to_graph(A_tilde, X_tilde, data.y, threshold)
        samples[i] = G
    return samples
