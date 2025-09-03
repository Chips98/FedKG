

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from math import ceil
from torch.autograd import Variable


import numpy as np




def preprocess_data(file_path, max_len, data_time_str):

    def preprocess_coordinates(lat, lon, max_value=5000):
        # 归一化经纬度
        normalized_lon = (lon + 180) / 360
        normalized_lat = (lat + 90) / 180

        # 映射到整数
        int_lon = int(normalized_lon * max_value)
        int_lat = int(normalized_lat * max_value)

        return int_lat, int_lon

    def encode_time(timestamp, data_time_str):
        # 这里应该是你的时间编码函数的实现
        dt = datetime.strptime(timestamp, data_time_str)
        year = dt.year
        month_of_year = dt.month
        day_of_week = dt.weekday()
        time_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second
        return year, month_of_year, day_of_week, time_of_day



    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()

    # 初始化数据结构
    user_data = {}

    # 解析文件中的每行数据
    for line in lines:
        parts = line.strip().split('\t')
        user_id, poi_id, class_id, lat, lon, timestamp = int(parts[0]), int(parts[1]), int(parts[2]), float(
            parts[4]), float(parts[5]), parts[-1]


        year, month_of_year, day_of_week, time_of_day = encode_time(timestamp, data_time_str)
        # 如果是新用户，初始化数据结构
        if user_id not in user_data:
            user_data[user_id] = {
                'user_id': [],
                'poi_id': [],
                'class_id': [],
                'latitude': [],
                'longitude': [],
                'time': []
            }
        # 将编码后的数据添加到相应用户的数据结构中
        mapped_lat, mapped_lon = preprocess_coordinates(lat, lon)
        user_data[user_id]['user_id'].append(user_id)
        user_data[user_id]['poi_id'].append(poi_id)
        user_data[user_id]['class_id'].append(class_id)
        user_data[user_id]['latitude'].append(mapped_lat)
        user_data[user_id]['longitude'].append(mapped_lon)
        user_data[user_id]['time'].append(month_of_year)


    # 对每个用户的轨迹进行截断或填充
    for user_id, data in user_data.items():
        for key in ['user_id', 'poi_id', 'class_id', 'latitude', 'longitude', 'time']:
            sequence = data[key]
            # 如果序列长度大于max_len，则截断；如果小于max_len，则填充
            if len(sequence) > max_len:
                user_data[user_id][key] = sequence[:max_len]
            else:
                user_data[user_id][key] += [0] * (max_len - len(sequence)) if key != 'time' else [0] * (
                        max_len - len(sequence))

        # 检查CUDA是否可用并设置默认设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将处理后的数据转换为tensor并移动到指定设备
    for user_id in user_data.keys():
        user_data[user_id]['user_id'] = torch.tensor(user_data[user_id]['user_id'], dtype=torch.int64).to(device)
        user_data[user_id]['poi_id'] = torch.tensor(user_data[user_id]['poi_id'], dtype=torch.int64).to(device)
        user_data[user_id]['class_id'] = torch.tensor(user_data[user_id]['class_id'], dtype=torch.int64).to(device)
        user_data[user_id]['latitude'] = torch.tensor(user_data[user_id]['latitude'], dtype=torch.float32).to(device)
        user_data[user_id]['longitude'] = torch.tensor(user_data[user_id]['longitude'], dtype=torch.float32).to(device)
        user_data[user_id]['time'] = torch.tensor(user_data[user_id]['time'], dtype=torch.float32).to(device)

    return user_data


def prepare_data_for_discriminator(pos_samples_poi, pos_samples_time, pos_samples_user, pos_samples_lat, pos_samples_lon,
                                   neg_samples_poi, neg_samples_time, neg_samples_user, neg_samples_lat, neg_samples_lon,
                                   gpu=False):
    """
    接收正样本（真实数据）和负样本（生成数据），为辨别器准备输入和目标数据。
    """
    device = torch.device("cuda" if gpu else "cpu")

    # 确保所有的数据都在同一设备上
    pos_samples_poi = pos_samples_poi.to(device)
    pos_samples_time = pos_samples_time.to(device)
    pos_samples_user = pos_samples_user.to(device)
    pos_samples_lat = pos_samples_lat.to(device)
    pos_samples_lon = pos_samples_lon.to(device)

    neg_samples_poi = neg_samples_poi.to(device)
    neg_samples_time = neg_samples_time.to(device)
    neg_samples_user = neg_samples_user.to(device)
    neg_samples_lat = neg_samples_lat.to(device)
    neg_samples_lon = neg_samples_lon.to(device)

    # 合并正样本的POI、时间、用户、纬度和经度数据
    pos_samples = torch.cat((pos_samples_poi, pos_samples_time, pos_samples_user, pos_samples_lat, pos_samples_lon), dim=1)
    # 合并负样本的POI、时间、用户、纬度和经度数据
    neg_samples = torch.cat((neg_samples_poi, neg_samples_time, neg_samples_user, neg_samples_lat, neg_samples_lon), dim=1)

    # 合并正样本和负样本

    inp = torch.cat((pos_samples, neg_samples), dim=0).type(torch.LongTensor).to(device)
    target = torch.zeros(pos_samples.size(0) + neg_samples.size(0), device=device)
    target[:pos_samples.size(0)] = 1  # 正样本标记为1，负样本标记为0

    # 洗牌
    perm = torch.randperm(inp.size(0))
    inp = inp[perm]
    target = target[perm]

    # 分割POI、时间、用户、纬度和经度数据
    poi_inp = inp[:, :pos_samples_poi.size(1)]
    time_inp = inp[:, pos_samples_poi.size(1):pos_samples_poi.size(1) + pos_samples_time.size(1)]
    user_inp = inp[:, pos_samples_poi.size(1) + pos_samples_time.size(1):pos_samples_poi.size(1) + pos_samples_time.size(1) + pos_samples_user.size(1)]
    lat_inp = inp[:, pos_samples_poi.size(1) + pos_samples_time.size(1) + pos_samples_user.size(1):pos_samples_poi.size(1) + pos_samples_time.size(1) + pos_samples_user.size(1) + pos_samples_lat.size(1) ]  # 假设纬度数据是倒数第二维度
    lon_inp = inp[:, pos_samples_poi.size(1) + pos_samples_time.size(1) + pos_samples_user.size(1) +  pos_samples_lat.size(1) : pos_samples_poi.size(1) + pos_samples_time.size(1) + pos_samples_user.size(1) +  pos_samples_lat.size(1) + pos_samples_lon.size(1)]  # 假设经度数据是最后一个维度


    # 将输入和目标移到正确的设备
    poi_inp = poi_inp.to(device)
    time_inp = time_inp.to(device)
    user_inp = user_inp.to(device)
    lat_inp = lat_inp.to(device)
    lon_inp = lon_inp.to(device)
    target = target.to(device)

    return (poi_inp, time_inp, user_inp, lat_inp, lon_inp), target

def prepare_batch_data_for_generator(samples, start_letter=0, gpu=False):
    """
    Takes samples (a batch) and returns

    Inputs: samples, start_letter, cuda
        - samples: batch_size x seq_len (Tensor with a sample in each row)

    Returns: inp, target
        - inp: batch_size x seq_len (same as target, but with start_letter prepended)
        - target: batch_size x seq_len (Variable same as samples)
    """
    batch_size, seq_len = samples.size()

    inp = torch.zeros(batch_size, seq_len)
    target = samples
    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len-1]

    inp = Variable(inp).type(torch.LongTensor)
    target = Variable(target).type(torch.LongTensor)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target

def sample_from_generator_batch(gen, num_samples, batch_size):
    poi_samples_list = []
    time_samples_list = []
    user_samples_list = []
    lat_samples_list = []  # 经度样本列表
    long_samples_list = [] # 纬度样本列表

    for i in range(int(math.ceil(num_samples / float(batch_size)))):
        poi_samples, time_samples, user_samples, lat_samples, long_samples = gen.sample(batch_size)
        poi_samples_list.append(poi_samples)
        time_samples_list.append(time_samples)
        user_samples_list.append(user_samples)
        lat_samples_list.append(lat_samples)  # 添加经度样本
        long_samples_list.append(long_samples) # 添加纬度样本

    # 分别连接poi, time, user以及经纬度的样本
    combined_poi_samples = torch.cat(poi_samples_list, 0)[:num_samples]
    combined_time_samples = torch.cat(time_samples_list, 0)[:num_samples]
    combined_user_samples = torch.cat(user_samples_list, 0)[:num_samples]
    combined_lat_samples = torch.cat(lat_samples_list, 0)[:num_samples]  # 连接经度样本
    combined_long_samples = torch.cat(long_samples_list, 0)[:num_samples] # 连接纬度样本

    return combined_poi_samples, combined_time_samples, combined_user_samples, combined_lat_samples, combined_long_samples

'''def sample_from_generator_batch(gen, num_samples, batch_size):
    samples = []
    for i in range(int(ceil(num_samples/float(batch_size)))):
        samples.append(gen.sample(batch_size))

    return torch.cat(samples, 0)[:num_samples]'''