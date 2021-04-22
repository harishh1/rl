from packages import *


class Std_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        convw, convh =self.conv_out(w, h)
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels= 64, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear( convw * convh * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False


    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def conv_out(self,w, h):
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,5,3),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,5,3),4,2),3,1)
        return convw, convh

