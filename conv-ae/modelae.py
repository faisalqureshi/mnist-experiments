import torch

class AutoEncoder(torch.nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, stride=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(16, 8, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1)
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(8, 16, 3, stride=2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            torch.nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        #print(x.shape)
        
        x = self.decoder(x)
        return x