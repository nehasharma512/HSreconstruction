from __future__ import division
import torch
import torch.nn as nn

import os
import numpy as np
#from scipy.ndimage import imread
#from PIL import Image
import scipy.misc

from resblock_256 import resblock,conv_relu_res_relu_block
from utils import save_matv73,reconstruction

base_path = '/home/neha512/res_clean_code/test'
model_path = base_path+'/models/res_n14_256_cleanpng.pkl'
img_path = base_path+'/test_imgs/'
result_path = base_path+'/test_results2/'
var_name = 'rad'

save_point = torch.load(model_path)
model_param = save_point['state_dict']
model = resblock(conv_relu_res_relu_block,14,3,31)
model = nn.DataParallel(model)
model.load_state_dict(model_param)

model = model.cuda()
model.eval()



for img_name in sorted(os.listdir(img_path)):
    print (img_name)
    img_path_name = os.path.join(img_path, img_name)
    #rgb = Image.open(img_path_name).convert('RGB')
    rgb = scipy.misc.imread(img_path_name)   
    rgb = rgb/255
    rgb = np.expand_dims(np.transpose(rgb,[2,1,0]), axis=0).copy() 
   
    img_res1 = reconstruction(rgb,model)
    img_res2 = np.flip(reconstruction(np.flip(rgb, 2).copy(),model),1)
    img_res3 = (img_res1+img_res2)/2

    mat_name = img_name[:-10] + '.mat'
    mat_dir= os.path.join(result_path, mat_name)

    save_matv73(mat_dir, var_name,img_res3)
