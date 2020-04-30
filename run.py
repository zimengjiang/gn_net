import torch
import torchsnooper
from cmu_dataset import CMUDataset
from trainer import fit
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_root', type=str, default='/Users/zimengjiang/code/3dv/public_data')
parser.add_argument('--dataset_name', type=str, default='cmu')
parser.add_argument('--dataset_image_folder', type=str, default='images')
parser.add_argument('--pair_info_folder', type=str, default='correspondence_data')
parser.add_argument('--query_folder', type=str, default='query')
parser.add_argument('--all_slice', type=bool, default=True)
parser.add_argument('--slice', type=int, default=6)
parser.add_argument('--transform', type=bool, default=True)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--total_epochs', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
parser.add_argument('--num_workers', '-n', type=int, default=1, help="Number of workers")
parser.add_argument('--lr', type = float, default=1e-6, help="Number of workers")
parser.add_argument('--schedule_lr_frequency', type=int, default=8, help='in number of iterations (0 for no schedule)')
parser.add_argument('--schedule_lr_fraction', type=float, default=0.1)
parser.add_argument('--scale', type=int, default = 4, help="Scaling factor for input image")
parser.add_argument('--save_check_point_root', type=str, default = '/Users/zimengjiang/code/3dv/public_data')

args = parser.parse_args()

# scale the original image by the factor of img_scale
# img_scale = 2

'''set up data loaders'''

# todo: make it configurable 
trainset = CMUDataset(root = args.dataset_root,
                      name = args.dataset_name,
                      image_folder = args.dataset_image_folder,
                      pair_info_folder = args.pair_info_folder,
                      cmu_slice_all = args.all_slice,
                      cmu_slice = args.slice,
                      queries_folder = args.query_folder,
                      transform = args.transform,
                      img_scale = args.scale
)

train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
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
loss_fn = GNLoss(margin = 1, lamda = 0.003, img_scale = args.scale)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = 1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, args.schedule_lr_frequency, gamma=args.schedule_lr_fraction, last_epoch=-1) # optional
n_epochs = args.total_epochs
log_interval = args.log_interval
# fit the model
fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, init=False)