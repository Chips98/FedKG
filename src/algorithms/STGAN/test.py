
import torch
from src.util import *

from src.algorithms.FedKG_GAN.utils import *
from src.algorithms.FedKG_GAN.gan_models import *


class Gan_trainer:
    def __init__(self, args):
        self.user_data = None
        self.num_users = args.num_users
        self.max_len = args.max_len
        self.batch_size = args.batch_size

        self.embedding_dim_gen = args.embedding_dim_gen
        self.embedding_dim_dis = args.embedding_dim_dis

        self.hidden_dim_gen = args.hidden_dim_gen
        self.hidden_dim_dis = args.hidden_dim_dis

        self.poi_size = args.poi_size
        self.time_size = args.time_size

        self.mle_train_epochs = args.mle_train_epochs
        self.dis_train_epochs = args.dis_train_epochs
        self.adv_train_epochs = args.adv_train_epochs
        self.pg_train_epochs = args.pg_train_epochs

        self.cuda = args.cuda

        self.file_path = args.file_path
        self.data_time_str = args.data_time_str

        self.generator = Generator(self.embedding_dim_gen, self.hidden_dim_gen, self.poi_size, self.time_size, self.max_len, gpu=self.cuda)
        self.discriminator = Discriminator(self.embedding_dim_dis, self.hidden_dim_dis, self.poi_size, self.time_size, self.max_len, gpu=self.cuda)

        if self.cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=0.01)
        self.dis_optimizer = optim.Adagrad(self.discriminator.parameters())

    def train_generator_MLE(self, poi_samples, time_samples, lat_samples, lon_samples, epochs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(device)  # Move the generator to the correct device

        """
        Max Likelihood Pretraining for the generator
        """
        for epoch in range(epochs):
            print('Generator training epoch %d : ' % (epoch + 1), end='')
            total_loss = 0

            for i in range(0, self.num_users, self.batch_size):
                poi_input, poi_target = prepare_batch_data_for_generator(poi_samples[i:i + self.batch_size], start_letter=0, gpu=device)
                time_input, time_target = prepare_batch_data_for_generator(time_samples[i:i + self.batch_size], start_letter=0, gpu=device)
                lat_input, lat_target = prepare_batch_data_for_generator(lat_samples[i:i + self.batch_size], start_letter=0, gpu=device)
                lon_input, lon_target = prepare_batch_data_for_generator(lon_samples[i:i + self.batch_size], start_letter=0, gpu=device)

                poi_input, poi_target = poi_input.to(device), poi_target.to(device)
                time_input, time_target = time_input.to(device), time_target.to(device)
                lat_input, lat_target = lat_input.to(device), lat_target.to(device)
                lon_input, lon_target = lon_input.to(device), lon_target.to(device)

                self.gen_optimizer.zero_grad()

                loss = self.generator.batchNLLLoss(poi_input, poi_target, time_input, time_target, lat_input, lat_target, lon_input, lon_target)

                loss.backward()

                self.gen_optimizer.step()

                total_loss += loss.data.item()

            # each loss in a batch is loss per sample
            avg_loss = total_loss / ceil(self.num_users / float(self.batch_size)) / max_len
            print(f'Generator training  average_train_loss: {avg_loss}.\n')


    def train_discriminator(self, real_data_samples_poi, real_data_samples_time, real_data_samples_lat, real_data_samples_lon, generator, dis_epochs):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
        Samples are drawn d_steps times, and the discriminator is trained for dis_epochs epochs.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sample from generator
        neg_samples_poi, neg_samples_time, neg_samples_lat, neg_samples_lon = sample_from_generator_batch(generator, self.num_users, self.batch_size)
        neg_samples_poi, neg_samples_time, neg_samples_lat, neg_samples_lon = neg_samples_poi.to(device), neg_samples_time.to(device), neg_samples_lat.to(device), neg_samples_lon.to(device)

        # Prepare discriminator data
        dis_inp, dis_target = prepare_data_for_discriminator(real_data_samples_poi, real_data_samples_time,real_data_samples_lat, real_data_samples_lon,
                                                             neg_samples_poi, neg_samples_time, neg_samples_lat, neg_samples_lon,
                                                             gpu=torch.cuda.is_available())

        for epoch in range(dis_epochs):
            total_loss = 0
            total_acc = 0
            for i in range(0, dis_inp[0].size(0), self.batch_size):
                inp_poi, inp_time , inp_lat, inp_lon= dis_inp[0][i:i + self.batch_size], dis_inp[1][i:i + self.batch_size], dis_inp[2][i:i + self.batch_size], dis_inp[3][i:i + self.batch_size]
                target = dis_target[i:i + self.batch_size].to(device)
                self.dis_optimizer.zero_grad()
                out = self.discriminator.batchClassify(inp_poi, inp_time , inp_lat, inp_lon)
                loss_fn = nn.BCELoss().to(device)
                loss = loss_fn(out, target.float())
                loss.backward()
                self.dis_optimizer.step()
                total_loss += loss.item()
                total_acc += ((out > 0.5) == target).float().sum().item()
            total_loss /= (dis_inp[0].size(0) / self.batch_size)
            total_acc /= dis_inp[0].size(0)

            print(f'Discriminator epoch {epoch}: average_loss = {total_loss:.4f}, train_acc = {total_acc:.4f}')

    def train_generator_PG(self, dis, pg_epochs, start_poi, start_time, ):
        """
        The generator is trained using policy gradients, using the reward signal from the discriminator.
        Training is done for num_batches batches.
        """

        for epoch in range(pg_epochs):

            # Sample sequences from the generator (both POI and Time)
            poi_samples, time_samples = self.generator.sample(self.batch_size * 2, start_poi=start_poi,
                                                                  start_time=start_time)

            # Ensure that we have 2-dimensional data for discriminator
            if poi_samples.dim() == 1:
                poi_samples = poi_samples.view(self.batch_size * 2, -1)
            if time_samples.dim() == 1:
                time_samples = time_samples.view(self.batch_size * 2, -1)

            # Prepare batch for the discriminator
            rewards = dis.batchClassify(poi_samples, time_samples)

            # Reshape rewards to match the batch size of the generator's output
            rewards = rewards.view(self.batch_size * 2, -1).mean(dim=1)

            # Normalize rewards
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # Ensure the rewards are correctly reshaped
            assert rewards.dim() == 1, "Rewards should be 1-dimensional"
            assert rewards.size(0) == self.batch_size * 2, "Rewards size should match the number of samples"

            # Get generator's loss
            self.gen_optimizer.zero_grad()
            pg_loss = self.generator.batchPGLoss(poi_samples, time_samples, rewards)
            pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.gen_optimizer.step()
            print(f'pg_loss: {pg_loss.item()}.\n')



    def preprocess_and_load_data(self):
        # Data preprocessing and loading implementation
        return preprocess_data(self.file_path, self.max_len, self.data_time_str)

    def save_model(self, model, filepath):
        torch.save(model.state_dict(), filepath)

    def load_model(self, model, filepath):
        model.load_state_dict(torch.load(filepath))
        model.eval()  # Set the model to evaluation mode

    def Train_GAN(self):
        # Main training loop
        self.user_data = self.preprocess_and_load_data()

        poi_id_tensors = [user_info['poi_id'].clone().detach() for user_info in self.user_data.values()]
        poi_samples = torch.stack(poi_id_tensors)

        time_tensors = [user_info['time'].clone().detach() for user_info in self.user_data.values()]
        time_samples = torch.stack(time_tensors)

        lat_tensors = [user_info['latitude'].clone().detach() for user_info in self.user_data.values()]
        lat_samples = torch.stack(lat_tensors)

        lon_tensors = [user_info['longitude'].clone().detach() for user_info in self.user_data.values()]
        lon_samples = torch.stack(lon_tensors)

        print('Starting Generator MLE Training...')
        self.train_generator_MLE(poi_samples, time_samples, self.mle_train_epochs)

        print('\nStarting Discriminator Training...')
        self.train_discriminator(poi_samples, time_samples, self.generator, self.dis_train_epochs)

        print('\nStarting Adversarial Training...')
        for epoch in range(self.adv_train_epochs):
            print(f'\n--------\nAdversarial Training EPOCH {epoch + 1} / {self.adv_train_epochs}\n--------')
            print('\nAdversarial Training Generator : \n', end='')
            self.train_generator_PG(self.discriminator, self.pg_train_epochs, 0, 0)

            print('\nAdversarial Training Discriminator : ')
            self.train_discriminator(poi_samples, time_samples, self.generator, 3)

            # 保存模型
            generator_path = 'generator_model.pt'
            discriminator_path = 'discriminator_model.pt'
            self.save_model(self.generator, generator_path)
            self.save_model(self.discriminator, discriminator_path)

    def find_insertion_index(self, time_sequence):
        # 这个函数需要根据您的具体逻辑来实现，比如找到最大时间间隔的位置
        # 下面是一个示例逻辑
        max_gap = 0
        max_gap_index = 0
        for i in range(len(time_sequence) - 1):
            gap = time_sequence[i + 1] - time_sequence[i]
            if gap > max_gap:
                max_gap = gap
                max_gap_index = i + 1
        return max_gap_index

    def save_augmented_data(self, augmented_data, filename, class_name='class_name'):
        with open(filename, 'w', encoding='ISO-8859-1') as f:
            for user_id, poi_id, time in augmented_data:
                print(f"generate for user_id [{user_id}]\n")
                # Assuming lat, lon, timezone offset, and time_UTC are constants for this example
                # They could be replaced with actual variables if they are different for each entry
                lat = "40.79889202318283"
                lon = "-74.11812337576457"
                timezone_offset = "-240"
                time_UTC = "Wek Mon day 00:00:00 +0000 0000"
                line = f"{user_id}\t{poi_id}\t{time}\t{class_name}\t{lat}\t{lon}\t{timezone_offset}\t{time_UTC}\n"
                f.write(line)

    def generate_aug_data(self, mode, num_augmented_users=100, num_augmented_points=100):
        user_data = self.user_data
        generator_path = 'generator_model.pt'
        self.load_model(self.generator, generator_path)

        if mode == 'new_users':
            # 生成新用户的轨迹数据
            augmented_data = []
            for user_id in range(101, 101 + num_augmented_users):
                # 假设generator可以一次性生成一整条轨迹
                poi_sequence, time_sequence = self.generator.generate(1, num_elements=100)

                # 将张量移动到CPU并转化为列表
                poi_list = poi_sequence.squeeze().tolist() if poi_sequence.numel() > 1 else [poi_sequence.item()]
                time_list = time_sequence.squeeze().tolist() if time_sequence.numel() > 1 else [time_sequence.item()]

                # 这里依次取出poi_list和time_list中的对应的元素
                for poi, time in zip(poi_list, time_list):
                    # 由于poi和time是从张量中提取的，它们可能是张量，需要转换为标准Python数据类型
                    poi_value = poi.item() if isinstance(poi, torch.Tensor) else poi
                    time_value = time.item() if isinstance(time, torch.Tensor) else time

                    augmented_data.append((user_id, poi_value, time_value))
            # 保存一次整批数据，而不是每个用户重复保存
            self.save_augmented_data(augmented_data, 'augmented_data_new_users.txt')

        elif mode == 'trajectory_completion':
            # 对现有用户数据进行轨迹补全
            for user_id, user_info in user_data.items():
                original_poi_sequence = user_info['poi_id']
                original_time_sequence = user_info['time']
                # 假设有一个函数可以确定插入新轨迹点的位置
                insertion_index = self.find_insertion_index(original_time_sequence)
                # 假设generator可以生成指定数量的轨迹点
                new_poi_sequence, new_time_sequence = self.generator.generate(user_id, num_augmented_points)
                # 插入新生成的轨迹
                completed_poi_sequence = torch.cat((original_poi_sequence[:insertion_index], new_poi_sequence,
                                                    original_poi_sequence[insertion_index:]))
                completed_time_sequence = torch.cat((original_time_sequence[:insertion_index], new_time_sequence,
                                                     original_time_sequence[insertion_index:]))

                with open(f'augmented_data_user_{user_id}.txt', 'w') as f:
                    for poi, time in zip(completed_poi_sequence, completed_time_sequence):
                        f.write(f'{user_id} {poi} {time}\n')

