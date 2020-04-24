import torch
from cmu_dataset import CMUDataset
from trainer import fit
from torch.utils.data import DataLoader
import torch.optim as optim




'''set up data loaders'''

# todo: make it configurable 
trainset = CMUDataset(root = '/cluster/work/riner/users/PLR-2020/lechen/gn_net/gn_net_data',
                      name = 'cmu',
                      image_folder = 'cmu_extended',
                      pair_info_folder = 'correspondence_data',
                      cmu_slice_all = True,
                      cmu_slice = 6,
                      queries_folder = 'query',
                      transform = True
)

batch_size = 2
num_workers = 1
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = None

'''set up the network and training parameters'''
from network.gnnet_model import EmbeddingNet, GNNet
from network.gn_loss import GNLoss

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
print(device)

# set up model
embedding_net = EmbeddingNet()
model = GNNet(embedding_net)
model = model.to(device)
# set up loss 
margin = 1.
loss_fn = GNLoss(margin = 1, lamda = 0.05)
lr = 1e-6
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1) # optional

n_epochs = 40
log_interval = 100
# fit the model
fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)