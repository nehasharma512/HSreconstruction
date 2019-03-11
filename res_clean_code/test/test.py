from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import  os
import time
import scipy.io as sio

from dataset import DatasetFromHdf5
from resblock import resblock,conv_relu_res_relu_block
from utils import AverageMeter,initialize_logger,save_checkpoint,record_loss
from loss import rrmse_loss,rrmse

def main():
    
    cudnn.benchmark = True
    base_path = '/NSL/data/images/HyperspectralImages/ICVL/' 
    # Dataset
    val_data = DatasetFromHdf5(base_path+'/testclean_si50_st80.h5')
    print(len(val_data))

    # Data Loader (Input Pipeline)
    val_loader = DataLoader(dataset=val_data,
                            num_workers=1, 
                            batch_size=1,
                            shuffle=False,
                           pin_memory=True)

    # Model
    model_path = base_path+'hscnn_5layer_dim10_93.pkl'
    result_path = base_path+'/test_results/'
    var_name = 'rad'

    save_point = torch.load(model_path)
    model_param = save_point['state_dict']
    model = resblock(conv_relu_res_relu_block,16,3,31)
    model = nn.DataParallel(model)
    model.load_state_dict(model_param)

    model = model.cuda()
    model.eval()               
   
    model_path = base_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    loss_csv = open(os.path.join(model_path,'loss.csv'), 'w+')
    
    log_dir = os.path.join(model_path,'train.log')
    logger = initialize_logger(log_dir)
    
    test_loss = validate(val_loader, model, rrmse_loss)
    
    
    print ("Test Loss: %.9f " %(test_loss))
        # save loss
    record_loss(loss_csv, test_loss)     
        #logger.info("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f " %(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    
# Training 
def train(train_data_loader, model, criterion, optimizer, iteration, init_lr ,end_epoch):
    losses = AverageMeter()
    for i, (images, labels) in enumerate(train_data_loader):
        labels = labels.cuda()
        images = images.cuda()
        images = Variable(images)
        labels = Variable(labels)    
        
        # Decaying Learning Rate
        
        lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=968000, power=1.5) 
        iteration = iteration + 1
        # Forward + Backward + Optimize       
        output = model(images)
        
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        
        #  record loss
        losses.update(loss.item())
            
    return losses.avg, iteration, lr

# Validate
def validate(val_loader, model, criterion):
    
    
    model.eval()
    losses = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
          input_var = torch.autograd.Variable(input)
          target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)      
        loss = criterion(output, target_var)
        print("calculating loss");
        #  record loss
        losses.update(loss.item())

    return losses.avg

# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

if __name__ == '__main__':
    main()
