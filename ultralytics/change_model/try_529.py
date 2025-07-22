import torch
import torch.nn as nn

class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k = 1, s = 1, g = 1, act = True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k = 3, s = 1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act = False) if s == 2 else nn.Identity(),  # dw

                                  GhostConv(c_, c2, 1, 1, act = False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act = False),
                                      Conv(c1, c2, 1, 1, act = False)) if s == 2 else nn.Identity()

    def forward(self, x):
        # x1 =self.conv(x)
        # # print('x1', torch.size(x1))
        # x2 = self.shortcut(x)
        # # print('x2', torch.size(x2))
        return self.conv(x) + self.shortcut(x)