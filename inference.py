import torch
import torch.nn as nn
from cmu_dataset import CMUDataset
from torch.utils.data import DataLoader
from network.gnnet_model import EmbeddingNet, GNNet
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--dataset_root', type=str, default='/local/home/lixxue/gnnet/gn_net_data')
parser.add_argument('--dataset_name', type=str, default='cmu')
parser.add_argument('--dataset_image_folder', type=str, default='images')
parser.add_argument('--pair_info_folder', type=str, default='correspondence_data')
parser.add_argument('--query_folder', type=str, default='query')
parser.add_argument('--all_slice', type=bool, default=False)
parser.add_argument('--slice', type=int, default=6)
parser.add_argument('--transform', type=bool, default=True)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--total_epochs', type=int, default=50)
parser.add_argument('--log_interval', type=int, default=160)
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--init', type=bool, default=True, help="Initialize the network weights")
parser.add_argument('--batch_size', '-b', type=int, default=2, help="Batch size")
parser.add_argument('--num_workers', '-n', type=int, default=16, help="Number of workers")
parser.add_argument('--lr', type = float, default=1e-6)
parser.add_argument('--schedule_lr_frequency', type=int, default=5, help='in number of iterations (0 for no schedule)')
parser.add_argument('--schedule_lr_fraction', type=float, default=1)
parser.add_argument('--scale', type=int, default = 2, help="Scaling factor for input image")
parser.add_argument('--save_root', type=str, default = '/local/home/lixxue/gnnet/gn_net_data')
parser.add_argument('--gn_loss_lamda', type=float, default=0.005)
parser.add_argument('--contrastive_lamda', type=float, default=1000)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--num_matches', type=float, default=2000)
parser.add_argument('--resume_checkpoint', type=str, default=None)

args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

model = GNNet(EmbeddingNet())
model.load_state_dict(torch.load("/Users/zimengjiang/code/3dv/ours/S2DHM/checkpoints/gnnet/25_model_best.pth.tar",\
            map_location=torch.device(device)))
dataset = CMUDataset(root=args.dataset_root,
                     name=args.dataset_name,
                     image_folder=args.dataset_image_folder,
                     pair_info_folder=args.pair_info_folder,
                     cmu_slice_all=args.all_slice,
                     cmu_slice=args.slice,
                     queries_folder=args.query_folder,
                     transform=args.transform,
                     img_scale=args.scale,
                     num_matches=args.num_matches
                     )
val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

def test_epoch(val_loader, model, cuda):
    with torch.no_grad():
        model.eval()
        for batch_idx, (img_ab, corres_ab) in enumerate(val_loader):
            corres_ab = corres_ab if len(corres_ab) > 0 else None
            if not type(img_ab) in (tuple, list):
                img_ab = (img_ab,)
            if cuda:
                img_ab = tuple(d.to(device) for d in img_ab)
                if corres_ab is not None:
                    # corres_ab = {key: corres_ab[key].to(device) for key in corres_ab}
                    for c in corres_ab:
                        c = {key: c[key].to(device) for key in c}

            outputs = model(*img_ab)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if corres_ab is not None:
                corres_ab = (corres_ab,)
                loss_inputs += corres_ab

            loss_outputs, contras_loss_outputs, gnloss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            contras_loss = contras_loss_outputs[0] if type(contras_loss_outputs) in (tuple, list) else contras_loss_outputs
            gnloss = gnloss_outputs[0] if type(gnloss_outputs) in (tuple, list) else gnloss_outputs
            val_loss += loss.item()
            val_contras_loss += contras_loss.item()
            val_gnloss += gnloss.item()


            # for metric in metrics:
            #     metric(outputs, target, loss_outputs)

    return val_loss, val_contras_loss, val_gnloss