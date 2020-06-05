# code based in part on
# http://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
# and from
# https://github.com/devsisters/neural-combinatorial-rl-tensorflow/blob/master/data_loader.py
import requests
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch
import os
import numpy as np
import re
import zipfile
import itertools
from collections import namedtuple
# from dubins import Dubins
import math
from dubins_path import DubinsPath

#######################################
# Reward Fn
#######################################
def reward(sample_solution, USE_CUDA=False):
    """
    Args:
        List of length sourceL of [batch_size] Tensors
    Returns:
        Tensor of shape [batch_size] containins rewards
    """
    batch_size = sample_solution[0].size(0)
    n = len(sample_solution)
    tour_len = Variable(torch.zeros([batch_size]))
    tour_len_ = Variable(torch.zeros([batch_size]))
    
    if USE_CUDA:
        tour_len = tour_len.cuda()
        tour_len_ = tour_len_.cuda()

    # sample_solution.transpose(1, 2)
    # soln = torch.FloatTensor(sample_solution)
    # print(sample_solution[0].size(0))
    # print(sample_solution[0].size(1))
    # print(sample_solution[0].size(2))
    # for id in range(len(sample_solution)):
    #     num_ten = np.array(sample_solution[id])
    #     print(num_ten)
    
    solution_batch = np.ones((n,batch_size,3))
    for city in range(n):
        solution_batch[city] = np.array(sample_solution[city]) #[batch_size x input_dim]
    # print(solution_batch)

    # path = []
    cost = np.zeros(batch_size)
    # tour_len_ = []
    for i in range(n-1):
        for j in range(batch_size):
            city1 = [solution_batch[i,j,0],solution_batch[i,j,1],solution_batch[i,j,2]*2*math.pi]
            city2 = [solution_batch[i+1,j,0],solution_batch[i+1,j,1],solution_batch[i+1,j,2]*2*math.pi]
            # print(city1, city2)
            dubins = DubinsPath(city1, city2, 0.1)
            dubins.calc_paths()
            path, cost[j] = dubins.get_shortest_path()
    #         # tour_len += torch.norm(path, dim=2)
            tour_len_[j] += cost[j]
            
    for k in range(batch_size):
        city_0 = [solution_batch[0,k,0],solution_batch[0,k,1],solution_batch[0,k,2]*2*math.pi]
        city_n = [solution_batch[n-1,k,0],solution_batch[n-1,k,1],solution_batch[n-1,k,2]*2*math.pi]
        dubins = DubinsPath(city_n, city_0)
        dubins.calc_paths()
        path, cost[k] = dubins.get_shortest_path()
        # tour_len += torch.norm(sample_solution[n-1] - sample_solution[0], dim=1)
        tour_len_[k] += cost[k]

    # tour_dub_len = torch.from_numpy(tour_len_).cuda()
    # print(str("Dubins:"), tour_len_)

    # for i in range(n-1):
    #     tour_len += torch.norm(sample_solution[i] - sample_solution[i+1], dim=1)
        
    
    # tour_len += torch.norm(sample_solution[n-1] - sample_solution[0], dim=1)
    # print(tour_len)

    # For TSP_20 - map to a number between 0 and 1
    # min_len = 3.5
    # max_len = 10.
    # TODO: generalize this for any TSP size
    #tour_len = -0.1538*tour_len + 1.538 
    #tour_len[tour_len < 0.] = 0.

    return tour_len_
    # return tour_len

def reward_dtsp(sample_solution, USE_CUDA=False):
    """
    Args:
        List of length sourceL of [batch_size] Tensors [sourceL x batch_size x input_dim]
    Returns:
        Tensor of shape [batch_size] containins rewards
    """
    # local_planner = Dubins(radius=2, point_separation=.5)
    # batch_size = sample_solution[0].size(0)
    # n = len(sample_solution)
    # tour_len = Variable(torch.zeros([batch_size]))
    
    # if USE_CUDA:
    #     tour_len = tour_len.cuda()

    # for i in range(n-1):
    #     path = local_planner.dubins_path(sample_solution[i], sample_solution[i+1])
    #     # tour_len += torch.norm(sample_solution[i] - sample_solution[i+1], dim=1)
    #     tour_len += torch.norm(path, dim=2)
    
    # path = local_planner.dubins_path(sample_solution[n-1], sample_solution[0])
    # # tour_len += torch.norm(sample_solution[n-1] - sample_solution[0], dim=1)
    # tour_len += torch.norm(path, dim=2)



    batch_size = sample_solution[0].size(0)
    n = len(sample_solution)
    tour_len = Variable(torch.zeros([batch_size]))
    
    if USE_CUDA:
        tour_len = tour_len.cuda()

    # matrix = []
    # for id in range(n):
    #     matrix.append(solution_batch[id], action_id.data, :])

    for id in range(n):
        solution_batch = np.array(sample_solution[id]) #[batch_size x input_dim]
        print(solution_batch)
        for i in range(batch_size):
            dubins = DubinsPath(solution_batch[i], sample_solution[i+1], 2)
            dubins.calc_paths()
            path, cost = dubins.get_shortest_path()
            # tour_len += torch.norm(path, dim=2)
            tour_len+= cost
    
    inital_batch = np.array(sample_solution[0])
    final_batch = np.array(sample_solution[n-1])
    for i in range(batch_size-1):
            dubins = DubinsPath(inital_batch[i], final_batch[i+1], 2)
            dubins.calc_paths()
            path, cost = dubins.get_shortest_path()
            # tour_len += torch.norm(path, dim=2)
            tour_len+= cost
    dubins = DubinsPath(sample_solution[n-1], sample_solution[0], 2)
    dubins.calc_paths()
    path, cost = dubins.get_shortest_path()
    # tour_len += torch.norm(sample_solution[n-1] - sample_solution[0], dim=1)
    tour_len += cost

    # For TSP_20 - map to a number between 0 and 1
    # min_len = 3.5
    # max_len = 10.
    # TODO: generalize this for any TSP size
    #tour_len = -0.1538*tour_len + 1.538 
    #tour_len[tour_len < 0.] = 0.
    return tour_len



#######################################
# Functions for downloading dataset
#######################################
DTSP = namedtuple('DTSP', ['x', 'y', 'name'])

GOOGLE_DRIVE_IDS = {
    'tsp5_train.zip': '0B2fg8yPGn2TCSW1pNTJMXzFPYTg',
    'tsp10_train.zip': '0B2fg8yPGn2TCbHowM0hfOTJCNkU',
    'tsp5-20_train.zip': '0B2fg8yPGn2TCTWNxX21jTDBGeXc',
    'tsp50_train.zip': '0B2fg8yPGn2TCaVQxSl9ab29QajA',
    'tsp20_test.txt': '0B2fg8yPGn2TCdF9TUU5DZVNCNjQ',
    'tsp40_test.txt': '0B2fg8yPGn2TCcjFrYk85SGFVNlU',
    'tsp50_test.txt.zip': '0B2fg8yPGn2TCUVlCQmQtelpZTTQ',
}

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)  
    return True

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk: # filter out keep-alive new chunks
                 f.write(chunk)

def download_google_drive_file(data_dir, task, min_length, max_length):
    paths = {}
    for mode in ['train', 'test']:
        candidates = []
        candidates.append(
            '{}{}_{}'.format(task, max_length, mode))
        candidates.append(
            '{}{}-{}_{}'.format(task, min_length, max_length, mode))

        for key in candidates:
            print(key)
            for search_key in GOOGLE_DRIVE_IDS.keys():
                if search_key.startswith(key):
                    path = os.path.join(data_dir, search_key)
                    print("Download dataset of the paper to {}".format(path))

                    if not os.path.exists(path):
                        download_file_from_google_drive(GOOGLE_DRIVE_IDS[search_key], path)
                    if path.endswith('zip'):
                        with zipfile.ZipFile(path, 'r') as z:
                            z.extractall(data_dir)
                    paths[mode] = path

    return paths

def read_paper_dataset(paths, max_length):
    x, y = [], []
    for path in paths:
        print("Read dataset {} which is used in the paper..".format(path))
        length = max(re.findall('\d+', path))
        with open(path) as f:
            for l in tqdm(f):
                inputs, outputs = l.split(' output ')
                x.append(np.array(inputs.split(), dtype=np.float32).reshape([-1, 3]))
                y.append(np.array(outputs.split(), dtype=np.int32)[:-1]) # skip the last one

    return x, y

def maybe_generate_and_save(self, except_list=[]):
    data = {}

    for name, num in self.data_num.items():
        if name in except_list:
            print("Skip creating {} because of given except_list {}".format(name, except_list))
            continue
        path = self.get_path(name)

        print("Skip creating {} for [{}]".format(path, self.task))
        tmp = np.load(path)
        self.data[name] = DTSP(x=tmp['x'], y=tmp['y'], name=name)

def get_path(self, name):
    return os.path.join(
        self.data_dir, "{}_{}={}.npz".format(
            self.task_name, name, self.data_num[name]))

def read_zip_and_update_data(self, path, name):
    if path.endswith('zip'):
        filenames = zipfile.ZipFile(path).namelist()
        paths = [os.path.join(self.data_dir, filename) for filename in filenames]
    else:
        paths = [path]

    x_list, y_list = read_paper_dataset(paths, self.max_length)

    x = np.zeros([len(x_list), self.max_length, 3], dtype=np.float32)
    y = np.zeros([len(y_list), self.max_length], dtype=np.int32)

    for idx, (nodes, res) in enumerate(tqdm(zip(x_list, y_list))):
        x[idx,:len(nodes)] = nodes
        y[idx,:len(res)] = res

    if self.data is None:
        self.data = {}

    print("Update [{}] data with {} used in the paper".format(name, path))
    self.data[name] = DTSP(x=x, y=y, name=name)


def create_dataset(
    problem_size, 
    data_dir):

    def find_or_return_empty(data_dir, problem_size):
        #train_fname1 = os.path.join(data_dir, 'tsp{}.txt'.format(problem_size))
        val_fname1 = os.path.join(data_dir, 'dtsp{}_test.txt'.format(problem_size))
        #train_fname2 = os.path.join(data_dir, 'tsp-{}.txt'.format(problem_size))
        val_fname2 = os.path.join(data_dir, 'dtsp-{}_test.txt'.format(problem_size))
        
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        else:
    #         if os.path.exists(train_fname1) and os.path.exists(val_fname1):
    #             return train_fname1, val_fname1
    #         if os.path.exists(train_fname2) and os.path.exists(val_fname2):
    #             return train_fname2, val_fname2
    #     return None, None

    # train, val = find_or_return_empty(data_dir, problem_size)
    # if train is None and val is None:
    #     download_google_drive_file(data_dir,
    #         'tsp', '', problem_size) 
    #     train, val = find_or_return_empty(data_dir, problem_size)

    # return train, val
            if os.path.exists(val_fname1):
                return val_fname1
            if os.path.exists(val_fname2):
                return val_fname2
        return None

    val = find_or_return_empty(data_dir, problem_size)
    if val is None:
        download_google_drive_file(data_dir, 'tsp', '', problem_size)
        val = find_or_return_empty(data_dir, problem_size)

    return val


#######################################
# Dataset
#######################################
class DTSPDataset(Dataset):
    
    def __init__(self, dataset_fname=None, train=False, size=50, num_samples=1000000, random_seed=1111):
        super(DTSPDataset, self).__init__()
        #start = torch.FloatTensor([[-1], [-1]]) 
        
        torch.manual_seed(random_seed)

        self.data_set = []
        if not train:
            with open(dataset_fname, 'r') as dset:
                for l in tqdm(dset):
                    inputs, outputs = l.split(' output ')
                    sample = torch.zeros(1, )
                    x_ = np.array(inputs.split(), dtype=np.float32).reshape([-1, 3]).T
                    x = np.array([x_[0]*0.001,x_[1]*0.001,x_[2]])
                    # y.append(np.array(outputs.split(), dtype=np.int32)[:-1]) # skip the last one
                    self.data_set.append(x)
        else:
            # randomly sample points uniformly from [0, 1]
            for l in tqdm(range(num_samples)):
                x = torch.FloatTensor(3, size).uniform_(0, 1)
                #x = torch.cat([start, x], 1)
                self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]
    
if __name__ == '__main__':
    paths = download_google_drive_file('data/tsp', 'tsp', '', '50')
