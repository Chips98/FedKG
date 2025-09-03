import torch
from torch.autograd import Variable
from math import ceil



def prepare_generator_batch(samples, start_letter=0, device='cuda'):

    batch_size, seq_len = samples.size()

    inp = torch.zeros(batch_size, seq_len, device=device).long()  # Initialize inp directly on the correct device
    target = samples.to(device)  # Ensure target is on the correct device
    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len-1].long()  # Ensure the slice is also long type

    # The Variables are deprecated and no longer needed here
    # The to(device) is already handled above when initializing and copying tensors

    return inp, target


import torch

def prepare_discriminator_data(pos_samples, neg_samples, device='cuda'):
    """
    接收正样本（目标样本）和负样本（生成器样本），准备用于鉴别器的输入数据和目标数据。
    """

    # 将正样本和负样本拼接，并转换为长整型，然后转移到指定的设备上
    inp = torch.cat((pos_samples, neg_samples), 0).long().to(device)
    # 创建一个全为1的目标张量，其长度为正样本和负样本数量之和，然后转移到指定的设备上
    target = torch.ones(pos_samples.size(0) + neg_samples.size(0)).to(device)
    # 将目标张量中对应负样本的部分置为0
    target[pos_samples.size(0):] = 0

    # 打乱数据
    perm = torch.randperm(target.size(0)).to(device)
    target = target[perm]
    inp = inp[perm]

    return inp, target


def batchwise_sample(gen, num_samples, batch_size):
    """
    Sample num_samples samples batch_size samples at a time from gen.
    Does not require gpu since gen.sample() takes care of that.
    """

    samples = []
    for i in range(int(ceil(num_samples/float(batch_size)))):
        samples.append(gen.sample(batch_size))

    return torch.cat(samples, 0)[:num_samples]


def batchwise_oracle_nll(gen, oracle, num_samples, batch_size, max_seq_len, start_letter=0, device='cuda'):
    s = batchwise_sample(gen, num_samples, batch_size)
    oracle_nll = 0
    for i in range(0, num_samples, batch_size):
        inp, target = prepare_generator_batch(s[i:i+batch_size], start_letter, device)
        oracle_loss = oracle.batchNLLLoss(inp, target) / max_seq_len
        oracle_nll += oracle_loss.data.item()

    return oracle_nll/(num_samples/batch_size)
