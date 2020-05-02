import torch
from cmu_dataset import CMUDataset
from trainer import fit
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tensorboardX import SummaryWriter
from pathlib import Path
from glob import glob

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_root', type=str, default='/cluster/work/riner/users/PLR-2020/lechen/gn_net/gn_net_data')
parser.add_argument('--dataset_name', type=str, default='cmu')
parser.add_argument('--dataset_image_folder', type=str, default='images')
parser.add_argument('--pair_info_folder', type=str, default='correspondence_data')
parser.add_argument('--query_folder', type=str, default='query')
parser.add_argument('--all_slice', type=bool, default=True)
parser.add_argument('--slice', type=int, default=6)
parser.add_argument('--transform', type=bool, default=True)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--validation_frequency', type=int, default=3)
parser.add_argument('--init', type=bool, default=True, help="Initialize the network weights")
parser.add_argument('--batch_size', '-b', type=int, default=16, help="Batch size")
parser.add_argument('--num_workers', '-n', type=int, default=16, help="Number of workers")
parser.add_argument('--lr', type = float, default=1e-3)
parser.add_argument('--schedule_lr_frequency', type=int, default=6, help='in number of iterations (0 for no schedule)')
parser.add_argument('--schedule_lr_fraction', type=float, default=0.1)
parser.add_argument('--scale', type=int, default = 2, help="Scaling factor for input image")
parser.add_argument('--save_root', type=str, default = '/cluster/work/riner/users/PLR-2020/lechen/gn_net/checkpoint')
parser.add_argument('--gn_loss_lamda', type=float, default=0.6)
parser.add_argument('--weight_decay', type=float, default=0.01)

args = parser.parse_args()

# scale the original image by the factor of img_scale
# img_scale = 2

# num_trainset = 23012
# num_valset = 5754

pair_file_roots1 = Path(args.dataset_root, args.dataset_name, args.pair_info_folder)
# /public_data/cmu/corrrespondence/*.mat
if not args.all_slice:
    suffix1  = 'correspondence_slice{}*.mat'.format(args.slice)
else:
    suffix1 = '*.mat'
pair_files1 = glob(str(Path(pair_file_roots1, suffix1)))
if not len(pair_files1):
    raise Exception('No correspondence file found at {}'.format(pair_file_roots1))

num_dataset = len(pair_files1)
num_valset = round(0.2 * num_dataset)
num_trainset = num_dataset - num_valset
print('\nnum_dataset: {} '.format(num_dataset))
print('num_trainset: {} '.format(num_trainset))
print('num_valset: {} \n'.format(num_valset))

print('Arguments & hyperparams: ')
print(args)

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
print('device: ' + str(device) + '\n')


'''set up data loaders'''
# todo: make it configurable
dataset = CMUDataset(root=args.dataset_root,
                     name=args.dataset_name,
                     image_folder=args.dataset_image_folder,
                     pair_info_folder=args.pair_info_folder,
                     cmu_slice_all=args.all_slice,
                     cmu_slice=args.slice,
                     queries_folder=args.query_folder,
                     transform=args.transform,
                     img_scale=args.scale
                     )

torch.manual_seed(0)

# number of trainset and number of valset should sum up to len(dataset)
trainset, valset = torch.utils.data.random_split(dataset, [num_trainset, num_valset])
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

'''set up the network and training parameters'''
from network.gnnet_model import EmbeddingNet, GNNet
from network.gn_loss import GNLoss

print("****** START ****** \n")

# set up model
embedding_net = EmbeddingNet()
model = GNNet(embedding_net)
model = model.to(device)
# set up loss
margin = 1.
loss_fn = GNLoss(margin = 1, lamda = args.gn_loss_lamda, img_scale = args.scale)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, args.schedule_lr_frequency, gamma=args.schedule_lr_fraction, last_epoch=-1) # optional
n_epochs = args.total_epochs
log_interval = args.log_interval
save_root = args.save_root
validation_frequency = args.validation_frequency
init = args.init
# fit the model
fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, validation_frequency, save_root, init)