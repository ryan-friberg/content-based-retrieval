import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torchvision

'''
This file's purpose is to define the learning-based models used for feature extraction. Example
architectures include a vision transformer, and a CNN-based variational autoencoder.

This file may potentially include pre-trained and from-scratch models.
'''

# backward pass is handled by the autograd functionality of pytorch
class FeatureExtractorViT(nn.Module):
    # n_patches is number of patches along a dimension
    def __init__(self, batch_shape, n_patches=6, hidden_size=128, num_blocks=1, num_heads=4, output_feature_size=256):
        super(FeatureExtractorViT, self).__init__()
        assert (batch_shape[1:] == (3,36,36)) # the econding works only for specific image shape
        
        ### assign class variables
        self.shape = batch_shape
        self.n_patches = n_patches

        ### define image resizing based on the image patch size/shape
        self.new_shape = (batch_shape[0], n_patches**2, (batch_shape[1]*batch_shape[2]*batch_shape[3]) // n_patches**2)
        self.p_shape   = (batch_shape[2]/n_patches, batch_shape[3]/n_patches)

        ### define image patching size
        self.p_size  = self.shape[2] // self.n_patches

        ### define the actual transformer model architecture
        self.linear         = nn.Linear(int(batch_shape[0] * self.p_shape[0] * self.p_shape[1]), hidden_size)
        attn_layers         = [EncoderBlock(hidden_size, num_heads) for _ in range(num_blocks)]
        self.encoder_blocks = nn.Sequential(*attn_layers)
        self.output_layer   = nn.Linear(hidden_size, output_feature_size)
        
        ### define positional embedding, mark it untrainable, and add it to the state_dict
        self.pos_embed = nn.Parameter(self.pos_embedding(n_patches ** 2, hidden_size))
        self.pos_embed.requires_grad = False
        self.register_buffer('positional_embedding', self.pos_embed, persistent=False)

    def pos_embedding(self, sequence_len, embed_dim):
        position = torch.arange(0, sequence_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(np.log(10000.0) / embed_dim))
        embeddings = torch.zeros(sequence_len, embed_dim)
        embeddings[:, 0::2] = torch.cos(position * div_term)
        embeddings[:, 1::2] = torch.sin(position * div_term)
        return embeddings

    def forward(self, batch):
        img_patches = batch.unfold(2, self.p_size, self.p_size).unfold(3, self.p_size, self.p_size)
        img_patches = img_patches.reshape(batch.shape[0], batch.shape[1], -1, self.p_size * self.p_size)
        z = self.linear(img_patches)
        pos_embed = self.pos_embed.repeat(batch.shape[0], batch.shape[1], 1, 1)
        z += pos_embed
        z = self.encoder_blocks(z)
        z = self.output_layer(z)
        return z
    

class AttentionEncoder(nn.Module):
    def __init__(self, dim, num_heads=3):
        super(AttentionEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # query, key, value mappings
        self.head_dim = int(dim // num_heads)
        self.q_map = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for q in range(self.num_heads)])
        self.k_map = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for k in range(self.num_heads)])
        self.v_map = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for v in range(self.num_heads)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, linear_embed):
        output = []
        for sequence in linear_embed:
            seq_z = []
            for seq_channel in sequence:
                channel_z = []
                for head in range(self.num_heads):
                    # seq = seq_channel[:, :, head * self.head_dim: (head + 1) * self.head_dim].squeeze(0)
                    seq = seq_channel[:, head * self.head_dim: (head + 1) * self.head_dim]
                    q = self.q_map[head](seq)
                    k = self.k_map[head](seq) 
                    v = self.v_map[head](seq)
                    attention = self.softmax(q @ k.T / np.sqrt(self.head_dim))
                    channel_z.append(attention @ v)
                seq_z.append(torch.hstack(channel_z))
            output.append(torch.hstack(seq_z))
        print(torch.cat([torch.unsqueeze(z, dim=0) for z in output]).shape)
        return torch.cat([torch.unsqueeze(z, dim=0) for z in output])


class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(EncoderBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention = nn.Sequential(nn.LayerNorm(self.hidden_size),
                                       AttentionEncoder(self.hidden_size, self.num_heads))
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.mlp = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(2*self.hidden_size, self.hidden_size))
    
    def forward(self, x):
        z = x + self.attention(x)
        z = self.layer_norm(z)
        z += self.mlp(z)
        return z

if __name__=='__main__':
    from PIL import Image 
    test_file = "../data/datasets/data/0.jpg"
    img = Image.open(test_file)
    img = img.convert('RGB')
    x = torchvision.transforms.ToTensor()(img).unsqueeze(0)
    # x = torchvision.transforms.Grayscale()(x).unsqueeze(0)
    test = FeatureExtractorViT(batch_shape=(1,3,36,36))
    output = test(x)
    print(output.shape)