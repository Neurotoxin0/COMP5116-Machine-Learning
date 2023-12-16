'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''


# Python
import os
import random
import datetime

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
# from torchvision.datasets import CIFAR100, CIFAR10

# Utils
import visdom
from tqdm import tqdm

# Custom
import models.resnet as resnet
import models.lossnet as lossnet
from config import *
from data.sampler import SubsetSequentialSampler
from models.unet.unet_model import UNet
from data.coco_dataloader import COCODataset

# Change the working directory to the location of this script
Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
os.chdir(Path)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\n\nDevice %s: \n\n", device)

# Seed
random.seed("")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

##
# Data
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])


def get_datasets():
    # Create an instance of your COCODataset
    coco_dataset = COCODataset()

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(coco_dataset))
    test_size = len(coco_dataset) - train_size
    train_dataset, test_dataset = random_split(coco_dataset, [train_size, test_size])

    return train_dataset, test_dataset

coco_train, coco_test = get_datasets()
coco_unlabeled = coco_train


##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss



def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)



##
# Train Utils
iters = 0

#
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None, plot_data=None):
    models['backbone'].train()
    models['module'].train()
    global iters


    for data in tqdm(dataloaders['train']):
        inputs = data[0].to(device).permute(0, 3, 1, 2).type(torch.float)
        labels = data[1].to(device).type(torch.long)
        iters += 1

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        # scores, features = models['backbone'](inputs)
        scores, features = models['backbone'](inputs)


        target_loss = criterion(scores, labels)
        target_loss += dice_loss(F.softmax(scores, dim=1).float(),
                          F.one_hot(labels, 90).permute(0, 3, 1, 2).float(),
                          multiclass=True)
        target_loss = torch.mean(torch.mean(target_loss,dim=1), dim=1)


        if epoch > epoch_loss:
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()

        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
        loss            = m_backbone_loss + WEIGHT * m_module_loss

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()

        

        # Visualize
        if (iters % 100 == 0) and (vis != None) and (plot_data != None):
            plot_data['X'].append(iters)
            plot_data['Y'].append([
                m_backbone_loss.item(),
                m_module_loss.item(),
                loss.item()
            ])
            vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': 'Loss over Time',
                    'legend': plot_data['legend'],
                    'xlabel': 'Iterations',
                    'ylabel': 'Loss',
                    'width': 1200,
                    'height': 390,
                },
                win=1
            )
    epoch_msg = f'[Training], target model loss: {m_backbone_loss}, loss module loss: {loss}, total loss: {loss}'
    return epoch_msg

#
def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.to(device).permute(0, 3, 1, 2).type(torch.float)
            labels = labels.to(device).type(torch.long)

            scores, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total/2400

#
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, vis, plot_data):
    print('>> Train a Model.')
    best_acc = 0.
    checkpoint_dir = os.path.join('./coco', 'train', 'weights')
    log_dir = os.path.join('./coco', 'train', 'logs')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        os.makedirs(log_dir)
    
    epoch_msgs = []
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        schedulers['module'].step()

        epoch_msg = train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis, plot_data)
        epoch_msgs.append(f'epoch {epoch}, {epoch_msg}')

        # Save a checkpoint
        if False and epoch % 5 == 4:
            acc = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()
                },
                '%s/active_resnet18_cifar10.pth' % (checkpoint_dir))
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')
    return epoch_msgs

#
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).to(device)

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.to(device).to(device).permute(0, 3, 1, 2).type(torch.float)
            # labels = labels.cuda()

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()


def save_logs(file_name, log_ls):
    log_dir = os.path.join('./coco', 'train', 'logs')

    # Open the file with the unique name in append mode
    with open(f'{log_dir}/{file_name}', 'a') as file:
        # Your for loop
        for line in log_ls:  # Replace 5 with the actual number of iterations in your loop
            # Write each list to the file
            file.write(''.join(line) + '\n')
    

##
# Main
if __name__ == '__main__':
    random_sampling = True
    WEIGHT = 0.5   # 0, 0.5, 1, 1.5, 2
    current_time = datetime.datetime.now()

    # Format the time as a string with underscores
    time_str = current_time.strftime("%Y_%m_%d_%H_%M")
    if random_sampling:
        file_name = f"experiment_random_{WEIGHT}_{time_str}.txt"
    else:
        file_name = f"experiment_learningloss_{WEIGHT}_{time_str}.txt"


    #vis = visdom.Visdom(server='http://localhost', port=8079)
    vis = visdom.Visdom(env=u'main',use_incoming_socket=False)  # have to manually open a new terminal and run python -m visdom.server
    plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}

    for trial in range(TRIALS):
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:ADDENDUM]
        unlabeled_set = indices[ADDENDUM:]
        
        train_loader = DataLoader(coco_train, batch_size=BATCH, 
                                  sampler=SubsetRandomSampler(labeled_set), 
                                  pin_memory=True)
        test_loader  = DataLoader(coco_test, batch_size=BATCH)

        dataloaders  = {'train': train_loader, 'test': test_loader}
        
        # Model
        unet = UNet(n_classes=90, n_channels=3).to(device)
        loss_module = lossnet.LossNet().to(device)
        models      = {'backbone': unet, 'module': loss_module}
        torch.backends.cudnn.benchmark = False

        # Active learning cycles
        for cycle in range(CYCLES):
            # Loss, criterion and scheduler (re)initialization
            criterion      = nn.CrossEntropyLoss(reduction='none') 
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            # Training and test
            cycle_logs = train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, vis, plot_data)
            acc = test(models, dataloaders, mode='test')
            cycle_logs.append(f'cycle {cycle}, [tesing], accuracy: {acc}')

            save_logs(file_name, cycle_logs)
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))

            ##
            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(coco_unlabeled, batch_size=BATCH, 
                                          sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
                                          pin_memory=True)
            
            if not random_sampling:

                # Measure uncertainty of each data points in the subset
                uncertainty = get_uncertainty(models, unlabeled_loader)

                # Index in ascending order
                arg = np.argsort(uncertainty)

                # Update the labeled dataset and the unlabeled dataset, respectively
                labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
                unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]
            else: # TODO: randomly add
                # get random sample 
                # Update the labeled dataset and the unlabeled dataset, respectively
                labeled_set += list(torch.tensor(subset)[-ADDENDUM:].numpy())
                unlabeled_set = list(torch.tensor(subset)[:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

            

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(coco_train, batch_size=BATCH, 
                                              sampler=SubsetRandomSampler(labeled_set), 
                                              pin_memory=True)
        
        # Save a checkpoint
        torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()
                },
                './coco/train/weights/active_unet_coco2017_trial{}.pth'.format(trial))