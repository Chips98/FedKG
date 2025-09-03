import time
import logging
from src.util import *
from src.models import *
from copy import deepcopy
import matplotlib.pyplot as plt
from src.algorithms.FedKG.client import FedKGClient
from src.algorithms.STGAN.gan_trainer import *
import csv
import os


class FedKGServer:
    """
    Server: Aggregate client parameters, update global model, train global model using public dataset, and pass new global model parameters to clients
    FedServer class attributes:
        config: parameters
        model: global model
        clients: clients
        result: used to save test results of training global model with public dataset
    """
    def __init__(self, config, global_model):
        self.config = config
        self.model = global_model
        self.algorithm = config["algorithm"]

        self.clients = None
        self.result = {'NDCG@5': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0, 'HT@5': 0.0,
                       'HT@10': 0.0, 'HT@20': 0.0}
        self.dataset = config['dataset']

        self.count = {}
        self.selected_clients = None
        self.best_performance = {metric: {'value': float('-inf'), 'round': None} for metric in ['NDCG5', 'NDCG10', 'NDCG20', 'HR5', 'HR10', 'HR20']}
        self.GAN = Gan_trainer(config)
        self.aug_path = None
        self.send_model_size = 0
        self.receive_model_size= 0
        mode = config['data_mode']
        self.gen_path = f'./gan/{mode}/{self.dataset}/global_generator_model.pt'

        self.merged_path = f'./data/{mode}/{self.dataset}/aug_server_data.txt'

    def select_clients(self, ):
        """
        Select clients
        """
        # Select training clients from participating clients according to jr ratio
        num_selected = int(len(self.clients) * self.config['jr'])
        if num_selected == 0:
            num_selected = 1  # Select at least one client
        selected_clients = random.sample(self.clients, num_selected)
        self.selected_clients = selected_clients
        return selected_clients
    
    # Create clients
    def create_client(self):
        clients = []
        for c_id in range(self.config['num_clients']):
            client = FedKGClient(self.config, c_id, client_model=deepcopy(self.model))
            client.GAN = self.GAN
            clients.append(client)
        self.clients = clients
        return clients

    def get_model_size_server(self, model):
        algo = self.config["algorithm"]
        dataset = self.config['dataset']
        nc = self.config['num_clients']
        model_size = get_model_size(dataset, algo, model, nc)
        return model_size


    def send_model(self):
        model_size = 0

        for client in self.clients:
            teacher_model, student_model = client.save_parameters(self.model)
            student_model = client.kd_train(student_model, teacher_model)
            model_size += self.get_model_size_server(student_model)
            for new_param, param in zip(student_model.parameters(), client.model.parameters()):
                param.data = new_param.data.clone()


        self.send_model_size += model_size




    def receive_gan(self, path):
        model_size = 0
        sum_count = 0
        self.uploaded_generators = []
        self.uploaded_weights = []

        for client in self.clients:
            model_size += self.get_model_size_server(client.GAN.generator)
            self.uploaded_generators.append(client.GAN.generator)
            self.uploaded_weights.append(client.count)
            sum_count += client.count

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / sum_count
        self.receive_model_size += model_size

        global_generator = copy.deepcopy(self.uploaded_generators[0])
        for w, client_generator in zip(self.uploaded_weights, self.uploaded_generators):
            for server_param, client_param in zip(global_generator.parameters(), client_generator.parameters()):
                server_param.data += client_param.data.clone() * w
        self.GAN.generator = global_generator
        torch.save(self.GAN.generator.state_dict(), path)



    def receive_model(self):
        sum_count = 0
        model_size = 0
        global_model = self.model
        self.uploaded_models = []
        self.uploaded_weights = []

        for client in self.selected_clients:
            model_size += self.get_model_size_server(client.model)
            updated_local_model = self.norm_train(client.model, global_model)
            self.uploaded_weights.append(client.count)
            sum_count += client.count
            self.uploaded_models.append(updated_local_model)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / sum_count
        self.receive_model_size += model_size



    def aggregate(self):

        for param in self.model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            for server_param, client_param in zip(self.model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w




    def _KD_loss(self, pos_logits, neg_logits, soft_pos_logits, soft_neg_logits, T):

        pred_pos = torch.log_softmax(pos_logits / T, dim=1)
        soft_pos = torch.softmax(soft_pos_logits / T, dim=1)

        pred_neg = torch.log_softmax(neg_logits / T, dim=1)
        soft_neg = torch.softmax(soft_neg_logits / T, dim=1)

        loss_pos = torch.nn.functional.kl_div(pred_pos, soft_pos, reduction='batchmean')
        loss_neg = torch.nn.functional.kl_div(pred_neg, soft_neg, reduction='batchmean')

        return (loss_pos + loss_neg) * (T ** 2) / 2



    def get_server_aug_dataset(self, path):

        global_data = get_server_data(path)
        return global_data


    def norm_train(self, student_model, teacher_model):
        batch_size = 24
        teacher_model.eval()
        student_model.train()
        global_datasets = self.get_server_aug_dataset(self.merged_path)
        [user_train, user_valid, user_test, usernum, itemnum], _ = global_datasets
        sampler = WarpSampler(user_train, usernum, itemnum, batch_size, self.config['maxlen'], 1, self.config)
        criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
        optimizer = torch.optim.Adam(student_model.parameters(), lr=self.config['lr'], betas=(0.9, 0.98))
        num_batch = len(user_train) // batch_size

        for epoch in range(self.config['norm_epochs']):
            for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits = student_model(u, seq, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.config['device']), torch.zeros(
                    neg_logits.shape, device=self.config['device'])
                optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = criterion(pos_logits[indices], pos_labels[indices])
                loss += criterion(neg_logits[indices], neg_labels[indices])

                t_pos_logits, t_neg_logits = teacher_model(u, seq, pos, neg)
                loss_kd = self._KD_loss(pos_logits, neg_logits, t_pos_logits, t_neg_logits, self.config['T'])

                loss = self.config["nor_lamda"] * loss_kd + (1 - self.config["nor_lamda"]) *loss # lamda=3
                loss.backward()
                optimizer.step()
        sampler.close()

        return student_model










