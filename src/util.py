import copy
import torch
import random
import logging
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import pandas as pd
import os
import csv

def save_time_infors(path, total_time):
    with open(path + "time.txt", "a") as f:
        f.write(f"computation:\n")
        total_all_time = round(total_time, 6)
        f.write(f"total time: {total_all_time}\n")


def save_memory_infors(path, server_send_size, server_recive_size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + "memory.txt", "a") as f:
        f.write(f"communications:\n")
        f.write(f"total_server_send_model_size: {server_send_size} mb\n")
        f.write(f"total_server_recive_model_size: {server_recive_size} mb\n")

# Set up logging
def set_logger(log_path):
    """
    Set up logging to save log information to log_path. Can save terminal output information.
    """
    logger = logging.getLogger()
    logger.handlers.clear()

    logger.setLevel(logging.INFO)
    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)



# Sampling
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():
        user_list = []
        for key in user_train:
            user_list.append(key)
        user = random.choice(user_list)
        # Randomly generate a user_id from user set
        while len(user_train[user]) <= 1: user = random.choice(user_list)
        # When this user's training items <= 1, select another user

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        # Initialize seq, pos, neg
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size, maxlen, n_workers, config):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      config["seed"]
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
# Get global data


def get_global_data(path):
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    user_list = []
    usernum = 0
    itemnum = 0
    f = open(path, 'r', encoding='ISO-8859-1')
    for line in f:
        u, i, v_cat_id, v_cat, lat, lon, time, time_UTC = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        if u in user_list:
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
        else:
            user_list.append(u)
            User[u].append(i)

    f.close()
    user_list.pop(-1)

    for user in user_list:
        nfeedback = len(User[user])

        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    global_data = [user_train, user_valid, user_test, usernum, itemnum, user_list]
    #print(user_list)
    return global_data


def get_model_size(dataset, algo, model, nc):
    path = f"temp/temp_model_{dataset}/{algo}_{nc}_temp.p"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    if hasattr(model, 'state_dict'):
        # Handle single model
        torch.save(model.state_dict(), path)
    else:
        # Handle parameter list
        torch.save(model, path)
    model_size = os.path.getsize(path) / (1024 ** 2)  # Convert to MB
    os.remove(path)  # Delete temporary file
    return model_size


def get_server_data(path):
    User = defaultdict(list)
    user_traj = defaultdict(list)  # unused variable - consider removing
    user_train = {}
    user_valid = {}
    user_test = {}
    user_list = []
    item_list = []  # unused variable - consider removing
    usernum = 0
    itemnum = 0
    count = 0
    f = open(path, 'r', encoding='ISO-8859-1')
    for line in f:
        u, i, v_cat_id, v_cat, lat, lon, time, time_UTC = line.rstrip().split('\t')
        u_id = int(u)
        v_id = int(i)
        if u_id in user_list:
            usernum = max(u_id, usernum)
            itemnum = max(v_id, itemnum)
            User[u_id].append(v_id)
        else:
            user_list.append(u_id)
            User[u_id].append(v_id)
    f.close()  # Close file to prevent resource leak

    for user in User:
        nfeedback = len(User[user])
        count += len(User[user])

        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    server_data = [user_train, user_valid, user_test, usernum, itemnum]
    return server_data, count


# Get client local data, return value count represents the number of samples for this client
def get_local_data(path):
    User = defaultdict(list)
    user_traj = defaultdict(list)  # unused variable - consider removing
    user_train = {}
    user_valid = {}
    user_test = {}
    user_list = []
    item_list = []  # unused variable - consider removing
    usernum = 0
    itemnum = 0
    count = 0
    print(f"path:{path}")
    f = open(path, 'r', encoding='ISO-8859-1')
    for line in f:
        u, i, v_cat_id, v_cat, lat, lon, time, time_UTC = line.rstrip().split('\t')
        u_id = int(u)
        v_id = int(i)
        if u_id in user_list:
            usernum = max(u_id, usernum)
            itemnum = max(v_id, itemnum)
            User[u_id].append(v_id)
        else:
            user_list.append(u_id)
            User[u_id].append(v_id)
    f.close()  # Close file to prevent resource leak

    for user in User:
        nfeedback = len(User[user])
        count += len(User[user])

        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    client_data = [user_train, user_valid, user_test, usernum, itemnum]
    return client_data, count

# Model evaluation
def evaluate(model, dataset, config):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    [train, valid, test, usernum, itemnum, userlist] = copy.deepcopy(dataset)

    NDCG_list = [0, 0, 0]
    HT_list = [0, 0, 0]
    valid_user = 0.0
    temp = [5, 10, 20]

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = userlist
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([config['maxlen']], dtype=np.int32)
        idx = config['maxlen'] - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(config['num_neg']):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1
        for k in temp:
            if rank < k:
                NDCG_list[temp.index(k)] += 1 / np.log2(rank + 2)
                HT_list[temp.index(k)] += 1

    for j in range(3):
        NDCG_list[j] = NDCG_list[j] / valid_user
        HT_list[j] = HT_list[j] / valid_user

    NDCG_list = [round(x, 6) for x in NDCG_list]
    HIT_list = [round(x, 6) for x in HT_list]
    combined_results = NDCG_list + HIT_list

    return combined_results

def test_server(model, test_dataset,config, time=3):
    test_results = []
    metrics = ['NDCG5', 'NDCG10', 'NDCG20', 'HR5', 'HR10', 'HR20']

    # Set different random seeds and test model performance
    for n in range(time):
        # Set random seed
        torch.manual_seed(n)
        np.random.seed(n)
        random.seed(n)

        # Evaluate model
        t_test = evaluate(model, test_dataset, config)

        # Record results
        result_dict = dict(zip(metrics, t_test))
        result_dict['n'] = n + 1
        test_results.append(result_dict)

    # Multiply all metric values by 100 and keep two decimal places

    # Calculate mean and standard deviation of metrics
    avg_results = {metric: sum(result[metric] for result in test_results) / len(test_results) for metric in metrics}
    std_results = {metric: np.std([result[metric] for result in test_results]) for metric in metrics}
    avg_results = {metric: round(avg_results[metric] * 100, 2) for metric in metrics}
    std_results = {metric: round(std_results[metric] * 100, 2) for metric in metrics}
    return avg_results, std_results


def test_model_final(model, test_dataset, save_path, config, time=3, c_id = None):
    test_results = []
    metrics = ['NDCG5', 'NDCG10', 'NDCG20', 'HR5', 'HR10', 'HR20']

    # Set different random seeds and test model performance
    for n in range(time):
        # Set random seed
        torch.manual_seed(n)
        np.random.seed(n)
        random.seed(n)

        # Load model and test data
        model.eval()

        # Evaluate model
        t_test = evaluate(model, test_dataset, config)

        # Record results
        result_dict = dict(zip(metrics, t_test))
        result_dict['n'] = n + 1
        test_results.append(result_dict)

    # Save test results to DataFrame
    df = pd.DataFrame(test_results)

    # Calculate mean and standard deviation for each metric
    mean_row = df[metrics].mean().apply(lambda x: round(x, 2)) * 100
    mean_row['n'] = 'Mean'
    std_row = df[metrics].std().apply(lambda x: round(x, 2)) * 100
    std_row['n'] = 'Std'
    # Add mean and standard deviation rows to DataFrame
    df = pd.concat([df, pd.DataFrame([mean_row]), pd.DataFrame([std_row])], ignore_index=True)

    df.to_csv(save_path, index=False)



def model_equal(model1, model2):
    weights1 = model1.state_dict()  # Get the weight parameters of model 1
    weights2 = model2.state_dict()  # Get the weight parameters of model 2

    if weights1.keys() == weights2.keys():
        for key in weights1.keys():
            if not torch.all(torch.eq(weights1[key], weights2[key])):
                logging.info("The weight parameters of the two models are different")
                break
        else:
            logging.info("The weight parameters of the two models are the same")
    else:
        logging.info("The weight parameters of the two models are different")