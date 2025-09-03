import torch
import pandas as pd


def get_gan_input_file(config):
    dataset = config['dataset']
    # 1
    input_file = f'data/processed_{dataset}.txt'
    out_file_1 = f'data/processed_{dataset}_T.txt'
    POI_dict = load_dataset(input_file, out_file_1)
    # 2
    out_file_2 = f'data/processed_{dataset}_R.trc'
    num_users = config['n_users']
    max_len = config['maxlen']
    txt_2_trc(out_file_1, out_file_2, num_users, max_len)
    return POI_dict, out_file_2



def load_dataset(input_file, out_file):
    grouped_data = {}
    POI_dict = {}

    with open(input_file, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            u, i, v_cat_id, v_cat, lat, lon, time_str, time_UTC = line.rstrip().split('\t')
            u_id = int(u)
            poi_id = int(i)
            lat = float(lat)
            lon = float(lon)
            if u_id not in grouped_data:
                grouped_data[u_id] = []
            grouped_data[u_id].append(poi_id)

            if poi_id not in POI_dict:
                POI_dict[poi_id] = [v_cat_id, v_cat, lat, lon, time_str, time_UTC]

    c_to_r(grouped_data, out_file)
    return POI_dict

def c_to_r(grouped_data, output_file):
    user_poi = grouped_data
    with open(output_file, 'w') as f:
        for user_id, poi_list in user_poi.items():
            poi_list = list(map(str, poi_list))  # 将元素转换为字符串类型
            f.write(' '.join(poi_list) + '\n')

def txt_2_trc(input_file, output_file, num_users, max_len):

    # 读取数据
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 转换为二维张量
    trajs = []
    for line in lines:
        traj = list(map(int, line.strip().split()))
        traj = traj[:max_len] + [0] * (max_len - len(traj[:max_len]))
        trajs.append(traj)
    traj_tensor = torch.LongTensor(trajs[:num_users])

    # 保存到文件
    torch.save(traj_tensor, output_file)

