import torch
from cmu_dataset import CMUDataset
from robotcar_dataset import RobotcarDataset
from trainer import fit
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tensorboardX import SummaryWriter
from pathlib import Path
from glob import glob
import os
# import wandb
parser = argparse.ArgumentParser()

# dataset arguments
parser.add_argument('--dataset_name', type=str, default='cmu')
parser.add_argument('--dataset_root',
                    type=str,
                    default='/local/home/lixxue/gnnet/gn_net_data_tiny')
parser.add_argument('--save_root',
                    type=str,
                    default='/local/home/lixxue/gnnet/checkpoint')
parser.add_argument('--dataset_image_folder', type=str, default='images')
parser.add_argument('--pair_info_folder', type=str, default='correspondence')
parser.add_argument('--query_folder', type=str, default='query')

# cmu arguments
parser.add_argument('--all_slice', type=bool, default=True)
parser.add_argument('--slice', type=int, default=7)

# robotcar arguments
parser.add_argument('--robotcar_all_weather', type=bool, default=False)
parser.add_argument('--robotcar_weather', type=str, default='sun')

# learning arguments
parser.add_argument('--batch_size',
                    '-b',
                    type=int,
                    default=1,
                    help="Batch size")
parser.add_argument('--num_workers',
                    '-n',
                    type=int,
                    default=16,
                    help="Number of workers")
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--schedule_lr_frequency',
                    type=int,
                    # default=50,
                    default=1,
                    help='in number of iterations (0 for no schedule)')
# parser.add_argument('--schedule_lr_fraction', type=float, default=0.1)
parser.add_argument('--schedule_lr_fraction', type=float, default=0.85)
parser.add_argument('--scale',
                    type=int,
                    default=2,
                    help="Scaling factor for input image")
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--transform', type=bool, default=True)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--total_epochs', type=int, default=50)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--init',
                    type=bool,
                    default=False,
                    help="Initialize the network weights")
parser.add_argument('--resume_checkpoint', type=str, default=None)
# parser.add_argument('--resume_checkpoint', type=str, default='/local/home/lixxue/gnnet/ckpt_test/2_checkpoint.pth.tar')

# loss hyperparameters
parser.add_argument('--gn_loss_lamda', type=float, default=0.003)
parser.add_argument('--contrastive_lamda', type=float, default=1)
parser.add_argument('--num_matches', type=float, default=1024)
parser.add_argument('--margin_pos', type=float, default=0.2)
parser.add_argument('--margin_neg', type=float, default=1)
parser.add_argument('--margin',
                    type=float,
                    default=1,
                    help="triplet loss margin")
parser.add_argument('--e1_lamda', type=float, default=1)
parser.add_argument('--e2_lamda', type=float, default=2/7)

# upsampling
parser.add_argument('--nearest',
                    type=bool,
                    default=True,
                    help="upsampling mode")
parser.add_argument('--bilinear',
                    type=bool,
                    default=False,
                    help="upsampling mode")

# debug arguments
parser.add_argument('--validate',
                    type=bool,
                    default=True,
                    help="validate during training or not")
parser.add_argument('--notes', type=str, default=None)
parser.add_argument('--log_dir', type=str, default='log')


args = parser.parse_args()

# visulaization
# wandb.init(config=args, project="gn_net_workstation")
# wandb.config["more"] = "custom"



# Just for dataset dividing
pair_file_roots1 = Path(args.dataset_root, args.dataset_name, args.pair_info_folder)
if args.dataset_name == 'cmu' and args.all_slice == False:
    suffix1 = 'correspondence_slice{}*.mat'.format(args.slice)
elif args.dataset_name == 'cmu' and args.all_slice == True:
    suffix1 = '*.mat'
elif args.dataset_name == 'robotcar' and args.robotcar_all_weather == True:
    suffix1 = '*.mat'
else:
    suffix1 = 'correspondence_run1_overcast-reference_run2_{}*.mat'.format(args.robotcar_weather)
pair_files1 = glob(str(Path(pair_file_roots1, suffix1)))
if not len(pair_files1):
    raise Exception(
        'No correspondence file found at {}'.format(pair_file_roots1))

# spilt dataset
num_dataset = len(pair_files1)
num_valset = round(0.1 * num_dataset)
num_trainset = num_dataset - num_valset
print('\nnum_dataset: {} '.format(num_dataset))
print('num_trainset: {} '.format(num_trainset))
print('num_valset: {} \n'.format(num_valset))

print('Arguments & hyperparams: ')
print(args)
os.makedirs(args.log_dir, exist_ok=True)
with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
    f.write(str(args))

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
print('device: ' + str(device) + '\n')

'''set up data loaders'''
# todo: make it configurable
if args.dataset_name == 'cmu':
    dataset = CMUDataset(root=args.dataset_root,
                         name=args.dataset_name,
                         image_folder=args.dataset_image_folder,
                         pair_info_folder=args.pair_info_folder,
                         cmu_slice_all=args.all_slice,
                         cmu_slice=args.slice,
                         queries_folder=args.query_folder,
                         transform=args.transform,
                         img_scale=args.scale,
                         num_matches=args.num_matches)
else:
    dataset = RobotcarDataset(root=args.dataset_root,
                              name=args.dataset_name,
                              image_folder=args.dataset_image_folder,
                              pair_info_folder=args.pair_info_folder,
                              queries_folder=args.query_folder,
                              robotcar_weather_all=args.robotcar_all_weather,
                              robotcar_weather=args.robotcar_weather,
                              transform=args.transform,
                              img_scale=args.scale,
                              num_matches=args.num_matches)

torch.manual_seed(0)

# number of trainset and number of valset should sum up to len(dataset)
trainset, valset = torch.utils.data.random_split(dataset,
                                                 [num_trainset, num_valset])
train_loader = DataLoader(trainset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.num_workers)

# modified 5.18 for debugging
# jidegaihuilai
if args.validate:
    val_loader = DataLoader(valset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)
else:
    val_loader = None
'''set up the network and training parameters'''
from network.gnnet_model import EmbeddingNet, GNNet
from network.gn_loss import GNLoss

print("****** START ****** \n")

# set up model
embedding_net = EmbeddingNet(bilinear=args.bilinear, nearest=args.nearest)
model = GNNet(embedding_net)
model = model.to(device)

# set up loss
loss_fn = GNLoss(margin_pos=args.margin_pos, 
                margin_neg=args.margin_neg, 
                margin=args.margin,
                contrastive_lamda=args.contrastive_lamda,
                gn_lamda=args.gn_loss_lamda,
                img_scale=args.scale,
                e1_lamda=args.e1_lamda,
                e2_lamda=args.e2_lamda,
                num_matches=args.num_matches)
optimizer = optim.AdamW(model.parameters(),
                        lr=args.lr,
                        weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      args.schedule_lr_frequency,
                                      gamma=args.schedule_lr_fraction,
                                      last_epoch=-1)  # optional

# if (args.resume_checkpoint):
#     model.load_state_dict(
#         torch.load(args.resume_checkpoint, map_location=torch.device(device)))

if (args.resume_checkpoint):
    checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device(device))
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})" .format(args.resume_checkpoint, checkpoint['epoch']))
else:
    start_epoch = args.start_epoch
    print("Did not use any checkpoint")

start_iteration = (start_epoch)*len(train_loader)
#SummaryWriter encapsulates everything
writer = SummaryWriter(args.log_dir, purge_step=start_iteration)

n_epochs = args.total_epochs
log_interval = args.log_interval
save_root = args.save_root
validation_frequency = args.validation_frequency
init = args.init
# fit the model
fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs,
    cuda, log_interval, validation_frequency, save_root, init, writer, start_epoch)
