import time
import logging
from src.util import *
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import csv

class FedKGClient:
    def __init__(self, config, client_id, client_model):
        """
        Client: Train local model using local dataset and upload local model to server
        FedClient class attributes:
            config: parameters
            model: local model
            result: used to save test results of training local model with local dataset
        """
        self.config = config
        self.id = client_id

        self.model = client_model
        self.GAN = None
        self.result = {'NDCG@5': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0, 'HT@5': 0.0, 'HT@10': 0.0, 'HT@20': 0.0}
        dataset = config["dataset"]
        mode = config['data_mode']

        # Set data path according to different configurations
        use_original_data = config.get("use_original_data", "False")

        if use_original_data == "True":
            # Baseline experiment: use original data
            self.merged_path = f'./data/{mode}/{dataset}/client{self.id}_data.txt'
        else:
            # Full training experiment: use complete training data
            self.merged_path = f'./data/{mode}/{dataset}/merged_client{self.id}_data.txt'

        self.ori_path = f'./data/{mode}/{dataset}/client{self.id}_data.txt'
        _, self.count = get_local_data(self.ori_path)

    def save_parameters(self, model):
        teacher_model = copy.deepcopy(self.model)
        student_model = copy.deepcopy(model)
        return teacher_model, student_model

    def get_local_dataset(self):
        pass


    def merge_csv(self, original_csv_path, augmented_csv_path, merged_csv_path):
        with open(merged_csv_path, 'w', newline='', encoding='ISO-8859-1') as merged_file:
            csv_writer = csv.writer(merged_file, delimiter='\t')

            # Write original data
            with open(original_csv_path, 'r', encoding='ISO-8859-1') as original_file:
                csv_reader = csv.reader(original_file, delimiter='\t')
                for row in csv_reader:
                    csv_writer.writerow(row)

            # Write augmented data
            with open(augmented_csv_path, 'r', encoding='ISO-8859-1') as augmented_file:
                csv_reader = csv.reader(augmented_file, delimiter='\t')
                for row in csv_reader:
                    csv_writer.writerow(row)

    def pretrain_GAN(self):
        alpha = self.config["alpha"]
        dataset = self.config["dataset"]
        mode = self.config["data_mode"]
        generator_path = f'./gan/{mode}/{dataset}/client_{self.id}/generator_model.pt'
        os.makedirs(os.path.dirname(generator_path), exist_ok=True)
        gan_epochs_list = [50, 3, 10, 3]
        self.GAN.Train_GAN(self.config, self.id, generator_path, gan_epochs_list)


    def _KD_loss(self, pos_logits, neg_logits, soft_pos_logits, soft_neg_logits, T):

        pred_pos = torch.log_softmax(pos_logits / T, dim=1)
        soft_pos = torch.softmax(soft_pos_logits / T, dim=1)

        pred_neg = torch.log_softmax(neg_logits / T, dim=1)
        soft_neg = torch.softmax(soft_neg_logits / T, dim=1)

        loss_pos = torch.nn.functional.kl_div(pred_pos, soft_pos, reduction='batchmean')
        loss_neg = torch.nn.functional.kl_div(pred_neg, soft_neg, reduction='batchmean')

        return (loss_pos + loss_neg) * (T ** 2) / 2

    def kd_train(self, student_model, teacher_model):
        batch_size = 2 * self.config['client_batch_size']
        student_model.train()
        teacher_model.eval()
        personalized_datasets = get_local_data(self.merged_path)
        [user_train, user_valid, user_test, usernum, itemnum], _ = personalized_datasets
        sampler = WarpSampler(user_train, usernum, itemnum, batch_size, self.config['maxlen'], 1, self.config)
        criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
        optimizer = torch.optim.Adam(student_model.parameters(), lr=self.config['lr'], betas=(0.9, 0.98))
        num_batch = len(user_train) // batch_size

        for epoch in range(self.config['kd_epochs']):
            for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits = student_model(u, seq, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.config['device']), torch.zeros(neg_logits.shape, device=self.config['device'])
                optimizer.zero_grad()
                indices = np.where(pos != 0)

                loss = criterion(pos_logits[indices], pos_labels[indices])
                loss += criterion(neg_logits[indices], neg_labels[indices])

                t_pos_logits, t_neg_logits = teacher_model(u, seq, pos, neg)
                loss_kd = self._KD_loss(pos_logits, neg_logits, t_pos_logits, t_neg_logits, self.config['T'])

                loss = self.config["lamda"] * loss_kd + (1 - self.config["lamda"]) *loss # lamda=3
                loss.backward()
                optimizer.step()
        sampler.close()

        return student_model




    def client_update(self):

        config = self.config
        personalized_datasets = get_local_data(self.merged_path)

        [user_train, user_valid, user_test, usernum, itemnum], _ = personalized_datasets

        local_model = deepcopy(self.model)
        sampler = WarpSampler(user_train, usernum, itemnum, config['client_batch_size'], config['maxlen'], 1, config)

        local_model.train()


        criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
        optimizer = torch.optim.Adam(local_model.parameters(), lr=config['lr'], betas=(0.9, 0.98))
        num_batch = len(user_train) // config['client_batch_size']
        if num_batch == 0:
            num_batch = 1  # Ensure at least one batch for training

        for epoch in range(config['client_epochs']):
            for step in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits = local_model(u, seq, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=config['device']), torch.zeros(neg_logits.shape,device=config['device'])
                optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = criterion(pos_logits[indices], pos_labels[indices])
                loss += criterion(neg_logits[indices], neg_labels[indices])
                for param in local_model.item_emb.parameters(): loss += config['l2_emb'] * torch.norm(param)

                loss.backward()
                optimizer.step()

        self.model = deepcopy(local_model)
        sampler.close()

