from optimizer.sam import SAM
from torchcontrib.optim import SWA
import torch.optim as optim

OPTIM = {'SAM': SAM,
         'SWA': SWA,
         'adam': optim.Adam}
