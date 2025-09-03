import sys
from src.algorithms.STGAN.gan_models import *
from src.algorithms.STGAN.helpers import  *
from src.algorithms.STGAN.data_setting import  *
import csv
import torch.optim as optim
import numpy as np

class Gan_trainer:
    def __init__(self, config):
        poi_dict, input_file = get_gan_input_file(config)
        self.poi_dict = poi_dict
        #self.VOCAB_SIZE = config['poi_size'] +1
        self.VOCAB_SIZE = 5136

        self.MAX_SEQ_LEN = 100
        self.START_LETTER = 0
        self.BATCH_SIZE = 128

        self.MLE_TRAIN_EPOCHS = 50  # gen
        self.DIS_TRAIN_EPOCHS = 5  # dis
        self.ADV_TRAIN_EPOCHS = 2  # adv
        self.POS_NEG_SAMPLES = 100

        self.GEN_EMBEDDING_DIM = 32
        self.GEN_HIDDEN_DIM = 32
        self.DIS_EMBEDDING_DIM = 64
        self.DIS_HIDDEN_DIM = 64
        self.device = config['device']


        self.oracle_samples_path = input_file

        self.oracle_samples_path = input_file

        self.oracle_samples = torch.load(self.oracle_samples_path).type(torch.LongTensor)
        print(self.oracle_samples)

        self.generator = Generator(self.GEN_EMBEDDING_DIM, self.GEN_HIDDEN_DIM, self.VOCAB_SIZE, self.MAX_SEQ_LEN, device=self.device)

        self.discriminator = Discriminator(self.DIS_EMBEDDING_DIM, self.DIS_HIDDEN_DIM, self.VOCAB_SIZE, self.MAX_SEQ_LEN, device=self.device)

        self.oracle = self.generator

        self.dataset = config['dataset']

        self.mode = config['data_mode']

        self.gen_path = f"gan/{self.mode}/{self.dataset}/pretrain_generator.pth"

        self.dis_path = f"gan/{self.mode}/{self.dataset}/pretrain_discriminator.pth"

        #self.gan_out_path = f"data/{self.dataset}/gan_out_{self.dataset}_data.txt"
        self.gan_out_path = f"gan_out_{self.dataset}_data.txt"

        self.oracle.to(self.device)
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.oracle_samples = self.oracle_samples.to(self.device)

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=0.01)
        self.dis_optimizer = optim.Adagrad(self.discriminator.parameters())

    def train_generator_MLE(self, gen, gen_opt, oracle, real_data_samples, epochs):
        """
        Max Likelihood Pretraining for the generator
        """
        for epoch in range(epochs):
            print('Generator training epoch %d : ' % (epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0

            for i in range(0, self.POS_NEG_SAMPLES, self.BATCH_SIZE):
                inp, target = prepare_generator_batch(real_data_samples[i:i + self.BATCH_SIZE],
                                                              start_letter=self.START_LETTER,
                                                              device=self.device)

                gen_opt.zero_grad()


                loss = gen.batchNLLLoss(inp, target)
                loss.backward()
                gen_opt.step()

                total_loss += loss.data.item()

                if (i / self.BATCH_SIZE) % ceil(
                        ceil(self.POS_NEG_SAMPLES / float(self.BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            # each loss in a batch is loss per sample
            total_loss = total_loss / ceil(self.POS_NEG_SAMPLES / float(self.BATCH_SIZE)) / self.MAX_SEQ_LEN

            # sample from generator and compute oracle NLL
            oracle_loss = batchwise_oracle_nll(gen, oracle, self.POS_NEG_SAMPLES, self.BATCH_SIZE, self.MAX_SEQ_LEN,
                                                       start_letter=self.START_LETTER, device=self.device)

            print('Generator training  average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))


    def train_generator_PG(self, gen, gen_opt, oracle, dis, num_batches):
        """
        The generator is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """

        for batch in range(num_batches):
            s = gen.sample(self.BATCH_SIZE * 2)  # 64 works best
            inp, target = prepare_generator_batch(s, start_letter=self.START_LETTER, device=self.device)
            rewards = dis.batchClassify(target)

            gen_opt.zero_grad()
            pg_loss = gen.batchPGLoss(inp, target, rewards)
            pg_loss.backward()
            gen_opt.step()
            print(f"PG_LOSS: {pg_loss.item()}\n")

        # sample from generator and compute oracle NLL
        oracle_loss = batchwise_oracle_nll(gen, oracle, self.POS_NEG_SAMPLES, self.BATCH_SIZE, self.MAX_SEQ_LEN,
                                                   start_letter=self.START_LETTER, device=self.device)

        print('Adversarial-Generator: oracle_sample_NLL = %.4f' % oracle_loss)

    def train_discriminator(self, discriminator, dis_opt, real_data_samples, generator, oracle, d_steps, epochs):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
        Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
        """

        # generating a small validation set before training (using oracle and generator)
        pos_val = oracle.sample(100)
        neg_val = generator.sample(100)
        val_inp, val_target = prepare_discriminator_data(pos_val, neg_val, device=self.device)

        for d_step in range(d_steps):
            s = batchwise_sample(generator, self.POS_NEG_SAMPLES, self.BATCH_SIZE)
            dis_inp, dis_target = prepare_discriminator_data(real_data_samples, s, device=self.device)
            for epoch in range(epochs):
                print('Discriminator:  d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
                sys.stdout.flush()
                total_loss = 0
                total_acc = 0

                for i in range(0, 2 * self.POS_NEG_SAMPLES, self.BATCH_SIZE):
                    inp, target = dis_inp[i:i + self.BATCH_SIZE], dis_target[i:i + self.BATCH_SIZE]
                    dis_opt.zero_grad()
                    out = discriminator.batchClassify(inp)
                    loss_fn = nn.BCELoss()
                    loss = loss_fn(out, target)
                    loss.backward()
                    dis_opt.step()

                    total_loss += loss.data.item()
                    total_acc += torch.sum((out > 0.5) == (target > 0.5)).data.item()

                    if (i / self.BATCH_SIZE) % ceil(ceil(2 * self.POS_NEG_SAMPLES / float(
                            self.BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                        print('.', end='')
                        sys.stdout.flush()

                total_loss /= ceil(2 * self.POS_NEG_SAMPLES / float(self.BATCH_SIZE))
                total_acc /= float(2 * self.POS_NEG_SAMPLES)

                val_pred = discriminator.batchClassify(val_inp)
                print('Discriminator:  average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                    total_loss, total_acc, torch.sum((val_pred > 0.5) == (val_target > 0.5)).data.item() / 200.))

    def merge_csv(self, original_csv_path, augmented_csv_path, merged_csv_path):
        with open(merged_csv_path, 'w', newline='', encoding='ISO-8859-1') as merged_file:
            csv_writer = csv.writer(merged_file, delimiter='\t')

            # 写入原始数据
            with open(original_csv_path, 'r', encoding='ISO-8859-1') as original_file:
                csv_reader = csv.reader(original_file, delimiter='\t')
                for row in csv_reader:
                    csv_writer.writerow(row)

            # 写入增强数据
            with open(augmented_csv_path, 'r', encoding='ISO-8859-1') as augmented_file:
                csv_reader = csv.reader(augmented_file, delimiter='\t')
                for row in csv_reader:
                    csv_writer.writerow(row)

    def Train_GAN(self):

        print('Starting Generator MLE Training...')
        self.train_generator_MLE(self.generator, self.gen_optimizer, self.oracle, self.oracle_samples, self.MLE_TRAIN_EPOCHS)

        print('\nStarting Discriminator Training...')
        self.train_discriminator(self.discriminator, self.dis_optimizer, self.oracle_samples, self.generator, self.oracle, 1, self.DIS_TRAIN_EPOCHS)

        print('\nStarting Adversarial Training...')
        oracle_loss = batchwise_oracle_nll(self.generator, self.oracle, self.POS_NEG_SAMPLES, self.BATCH_SIZE, self.MAX_SEQ_LEN,
                                                   start_letter=self.START_LETTER, device=self.device)

        print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)
        for epoch in range(self.ADV_TRAIN_EPOCHS):
            print(f'\n--------\nAdversarial Training EPOCH {epoch + 1} / {self.ADV_TRAIN_EPOCHS}\n--------')
            # TRAIN GENERATOR
            print('\nAdversarial Training Generator : ', end='')
            sys.stdout.flush()
            self.train_generator_PG(self.generator, self.gen_optimizer, self.oracle, self.discriminator, 1)

            # TRAIN DISCRIMINATOR
            print('\nAdversarial Training Discriminator : ')
            self.train_discriminator(self.discriminator, self.dis_optimizer, self.oracle_samples, self.generator,
                                     self.oracle, 5, self.DIS_TRAIN_EPOCHS)

        torch.save(self.generator.state_dict(), self.gen_path)
        torch.save(self.discriminator.state_dict(), self.dis_path)

    def Train_GAN_PretrainOnly(self):
        """
        仅执行GAN预训练，跳过对抗训练阶段
        用于审稿人2的补充实验
        """
        print('Starting Generator MLE Training (Pretrain Only)...')
        self.train_generator_MLE(self.generator, self.gen_optimizer, self.oracle,
                               self.oracle_samples, self.MLE_TRAIN_EPOCHS)

        # 保存仅预训练的生成器
        torch.save(self.generator.state_dict(), self.gen_path)
        print('Pretrain-only GAN training completed.')

        # 跳过判别器训练和对抗训练
        # 不执行以下步骤：
        # - self.train_discriminator(...)
        # - 对抗训练循环

    import numpy as np
    import torch

    # 找到POI_DICT中最接近的POI ID
    def find_nearest_poi_id(self, POI_dict, poi_id):
        all_poi_ids = np.array(list(POI_dict.keys()))
        idx = (np.abs(all_poi_ids - poi_id)).argmin()
        return all_poi_ids[idx]

    # 修改后的get_gan_out函数


    def get_gan_out(self, num_aug):
        POI_dict = self.poi_dict
        gen_path = self.gen_path
        output_file_path = self.gan_out_path
        generator = torch.load(gen_path).to(self.device)
        generated_data = batchwise_sample(generator, num_aug, self.MAX_SEQ_LEN)
        generated_data = generated_data.cpu()  # 将tensor复制到主机内存
        generated_data = np.array(generated_data)

        replaced_poi_counter = 0
        replaced_poi_info = []

        with open(output_file_path, 'w', encoding='utf-8') as f:
            for i in range(generated_data.shape[0]):
                traj = generated_data[i][generated_data[i] != 0]
                for poi_id in traj:
                    if poi_id in POI_dict:
                        info_to_write = POI_dict[poi_id]
                    else:
                        # POI ID不在POI_dict中，找到最近的POI ID
                        replaced_poi_counter += 1
                        nearest_poi_id = self.find_nearest_poi_id(POI_dict, poi_id)
                        info_to_write = POI_dict[nearest_poi_id]
                        replaced_poi_info.append((poi_id, nearest_poi_id))

                    # 保存POI的附属信息
                    f.write(f'{i + 101}\t{poi_id}\t{info_to_write[0]}\t{info_to_write[1]}\t{info_to_write[2]}\t{info_to_write[3]}\t{info_to_write[4]}\t{info_to_write[5]}\n')

        # 输出替换的POI个数及信息
        print(f'Replaced POI Count: {replaced_poi_counter}')
        for original_poi, replaced_poi in replaced_poi_info:
            print(f'Replaced POI ID {original_poi} with {replaced_poi}')

        #return replaced_poi_info

    def get_fl_gan_out(self, gen_path, output_file_path, num_aug):
        POI_dict = self.poi_dict

        self.generator.load_state_dict(torch.load(gen_path))

        generated_data = batchwise_sample(self.generator, num_aug, self.MAX_SEQ_LEN)
        generated_data = generated_data.cpu()  # 将tensor复制到主机内存
        generated_data = np.array(generated_data)

        replaced_poi_counter = 0
        replaced_poi_info = []

        with open(output_file_path, 'w', encoding='utf-8') as f:
            for i in range(generated_data.shape[0]):
                traj = generated_data[i][generated_data[i] != 0]
                for poi_id in traj:
                    if poi_id in POI_dict:
                        info_to_write = POI_dict[poi_id]
                    else:
                        # POI ID不在POI_dict中，找到最近的POI ID
                        replaced_poi_counter += 1
                        nearest_poi_id = self.find_nearest_poi_id(POI_dict, poi_id)
                        info_to_write = POI_dict[nearest_poi_id]
                        replaced_poi_info.append((poi_id, nearest_poi_id))

                    # 保存POI的附属信息
                    f.write(f'{i + 101}\t{poi_id}\t{info_to_write[0]}\t{info_to_write[1]}\t{info_to_write[2]}\t{info_to_write[3]}\t{info_to_write[4]}\t{info_to_write[5]}\n')

        # 输出替换的POI个数及信息
        #print(f'Replaced POI Count: {replaced_poi_counter}')
        #for original_poi, replaced_poi in replaced_poi_info:
            #print(f'Replaced POI ID {original_poi} with {replaced_poi}')

    def merge_csv(self, original_csv_path, augmented_csv_path, merged_csv_path):
        """合并原始数据和增强数据"""
        import csv
        import os

        # 确保输出目录存在
        os.makedirs(os.path.dirname(merged_csv_path), exist_ok=True)

        with open(merged_csv_path, 'w', newline='', encoding='ISO-8859-1') as merged_file:
            csv_writer = csv.writer(merged_file, delimiter='\t')

            # 写入原始数据
            if os.path.exists(original_csv_path):
                with open(original_csv_path, 'r', encoding='ISO-8859-1') as original_file:
                    csv_reader = csv.reader(original_file, delimiter='\t')
                    for row in csv_reader:
                        csv_writer.writerow(row)

            # 写入增强数据
            if os.path.exists(augmented_csv_path):
                with open(augmented_csv_path, 'r', encoding='ISO-8859-1') as augmented_file:
                    csv_reader = csv.reader(augmented_file, delimiter='\t')
                    for row in csv_reader:
                        csv_writer.writerow(row)
