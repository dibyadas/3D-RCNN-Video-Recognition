
# coding: utf-8

# In[1]:


import torch
import numpy as np
from moviepy.editor import VideoFileClip


# In[2]:


from torchvision.datasets.folder import DatasetFolder
from torchvision.transforms import transforms


# In[3]:


DATA_FOLDER = './data_video'
batch_size = 50

# In[6]:


def ret_frames(vid_name):
    vid = VideoFileClip(vid_name)
    inter = list(vid.iter_frames())
    frames_idx = np.floor(np.linspace(0,len(inter)-1,128)).astype('int')
    return np.array([inter[i].transpose(2,0,1) for i in frames_idx])


# In[7]:


ucf_data = DatasetFolder(DATA_FOLDER,ret_frames,['avi'])

train_dataset_len = len(ucf_data)

UCF_dataloader = torch.utils.data.DataLoader(ucf_data,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=4)


# In[ ]:

