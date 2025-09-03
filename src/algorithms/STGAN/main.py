from gan_trainer import Gan_trainer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train FCL-GAN model.")

    # Set up the arguments
    parser.add_argument('--num_users', type=int, default=100, help='Number of users.')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum sequence length.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--embedding_dim_gen', type=int, default=32, help='Generator embedding dimension.')
    parser.add_argument('--embedding_dim_dis', type=int, default=64, help='Discriminator embedding dimension.')
    parser.add_argument('--hidden_dim_gen', type=int, default=32, help='Generator hidden dimension.')
    parser.add_argument('--hidden_dim_dis', type=int, default=64, help='Discriminator hidden dimension.')
    parser.add_argument('--poi_size', type=int, default=10000, help='POI size.')
    parser.add_argument('--time_size', type=int, default=13, help='Time size.')
    parser.add_argument('--mle_train_epochs', type=int, default=100, help='MLE training epochs for generator.')
    parser.add_argument('--dis_train_epochs', type=int, default=20, help='Training epochs for discriminator.')
    parser.add_argument('--adv_train_epochs', type=int, default=5, help='Adversarial training epochs.')
    parser.add_argument('--pg_train_epochs', type=int, default=10, help='MLE training epochs for generator.')
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA if available.')
    parser.add_argument('--file_path', type=str, default='processed_nyc.txt', help='Path to the data file.')
    parser.add_argument('--data_time_str', type=str, default='%a %b %d %H:%M:%S %z %Y', help='Data time format string.')

    return parser.parse_args()


def main():
    args = parse_args()

    # Create FCL_GAN instance with parsed args
    gan = Gan_trainer(args)

    # Start the training process
    gan.Train_GAN()
    gan.generate_aug_data("new_users")



if __name__ == '__main__':
    main()
