import torch 
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt 
from random import random 

import torch 
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image

to_tensor = [Resize((144, 144)), ToTensor()]

class Compose(object):
    def __init__(self, transforms): 
        self.transforms = transforms 
        
    def __call__(self, image, target): 
        for t in self.transforms: 
            image = t(image)
        return image, target 


dataset = OxfordIIITPet(root="./", download=True, transforms=Compose(to_tensor))

def show_images(images, num_samples=40, cols=8): 
    plt.figure(figsize=(15,15))
    idx = int(len(dataset)/num_samples)
    for i, img in enumerate(images):
        if i % idx == 0:
            plt.subplot(int(num_samples/cols) + 1, cols, int(i/idx) + 1)
            plt.imshow(to_pil_image(img[0]))
    
# show_images(dataset)

class PatchEmbedding(nn.Module): 
    
    def __init__(self, in_channels=3, patch_size =8, emb_size=128): 
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 =patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size) 
        )
    
    def forward(self, x: Tensor) -> Tensor: 
        x = self.projection(x)
        return x
    
# sample_datapoint = torch.unsqueeze(dataset[0][0], 0)
# print (f'initial shape : {sample_datapoint.shape}')
# embedding = PatchEmbedding()(sample_datapoint)
# print (f'Patches shape : {embedding.shape}')


from einops import rearrange

class Attention(nn.Module): 
    def __init__(self, dim, n_heads, dropout): 
        super().__init__()
        self.n_heads = n_heads 
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim, 
                                                num_heads=n_heads,
                                                dropout=dropout)
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        
    def forward(self, x): 
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.attn(x,x,x)
        return attn_output 
    
# attn = Attention(128, 4, 0)(torch.ones((1,5,128)))
# attn.shape

class PreNorm(nn.Module): 
    def __init__(self, dim, fn): 
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn 
    
    def forward(self, x, **kwargs): 
        return self.fn(self.norm(x), **kwargs)

# norm = PreNorm(128, Attention(dim=128, n_heads=4, dropout=0.))
# norm(torch.ones((1,5,128))).shape


class FeedForward(nn.Sequential): 
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__(
            nn.Linear(dim, hidden_dim), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

# ff = FeedForward(dim=128, hidden_dim=256)
# ff(torch.ones((1,5,128))).shape

class ResidualAdd(nn.Module): 
    def __init__(self, fn): 
        super().__init__()
        self.fn = fn 
    
    def forward(self, x, **kwargs): 
        res = x 
        x = self.fn(x, **kwargs)
        x += res 
        return x 

# residual_attn = ResidualAdd(Attention(dim=128, n_heads=4, dropout=0.))
# residual_attn(torch.ones((1,5,128))).shape


from einops import repeat

class ViT(nn.Module): 
    def __init__(self, ch=3, img_size=144, patch_size=4, emb_dim=32,
                 n_layers=6, out_dim=37, dropout=0.1, heads=2): 
        super(ViT, self).__init__()
        
        # Attributes 
        self.channels = ch
        self.height = img_size 
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers 
        
        # patching 
        self.patch_embedding = PatchEmbedding(in_channels=ch,
                                              patch_size=patch_size, 
                                              emb_size=emb_dim)

        # learnabale parameters 
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, emb_dim))
        self.cls_token= nn.Parameter(torch.rand(1,1, emb_dim))
        
        # Transformer Encoder 
        self.layers = nn.ModuleList([])
        
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads=heads, dropout=dropout))),
                            ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout=dropout))))
            self.layers.append(transformer_block)
            
        # classification head 
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, emb_dim))
    

    def forward(self, img): 
        # get patch embedding vectors 
        x = self.patch_embedding(img)
        b,n,_ = x.shape 
        
        # add cls token to inputs 
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :(n+1)]
        
        # Transformer layers 
        for i in range(self.n_layers): 
            x = self.layers[i](x)
        
        # output based on classifciation token
        return self.head(x[:, 0, :])
            
# model = ViT()
# print (model)
# model(torch.ones((1,3,144,144)))


from torch.utils.data import DataLoader
from torch.utils.data import random_split

train_split = int (0.8* len(dataset))
train, test = random_split(dataset, [train_split, len(dataset)-train_split])

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=32, shuffle=True)


import torch.optim as optim 
import numpy as np 


def get_device(): 
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

device = get_device()
model = ViT().to(device)
optimizer = optim.Adam(model.parameters(), lr =0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100): 
    epoch_losses = []
    model.train()
    
    for step, (inputs, labels) in enumerate(train_loader): 
        inputs, labels = inputs.to(device), labels.to(device)
        
        print (labels)
        optimizer.zero_grad()
        predictions = model(inputs)
        print (labels.shape, predictions.shape)        
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        
        if epoch % 5 == 0:
            print (f'<<< Epoch : {epoch}, loss : {np.nean(epoch_losses)} >>>')
            epoch_losses_test = [] 

            correct = 0 
            for iter, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, labels)
                epoch_losses_test.append(loss.item())
                print (f'<<< Epoch : {epoch}, test loss : {np.mean(epoch_losses_test)}')
                
                if np.argmax(predictions) == labels : 
                    correct +=1 
                print ("Accuracy : {correct/len(labels)}")
                
                    
                