import os
import argparse
from solver import Solver

def main(config):
    # path for models
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    # import data loader
    if config.dataset == 'mtat':
        from data_loader.mtat_loader import get_audio_loader
    elif config.dataset == 'msd':
        from data_loader.msd_loader import get_audio_loader
    elif config.dataset == 'jamendo':
        from data_loader.jamendo_loader import get_audio_loader
    elif config.dataset == 'MER31K':
        from data_loader.MER31K_loader import get_audio_loader
    elif config.dataset == 'labelstudio':
        from data_loader.labelstudio_loader import get_audio_loader

    # audio length
    if config.model_type == 'fcn' or config.model_type == 'crnn':
        config.input_length = 29 * 16000
    elif config.model_type == 'musicnn':
        config.input_length = 3 * 16000
    elif config.model_type in ['sample', 'se', 'short', 'short_res']:
        config.input_length = 59049
    elif config.model_type == 'hcnn':
        config.input_length = 80000
    elif config.model_type == 'attention':
        config.input_length = 15 * 16000


    # get data loder
    train_loader = get_audio_loader(config.data_path,
                                    config.batch_size,
									split='TRAIN',
                                    input_length=config.input_length,
                                    num_workers=config.num_workers)

    print("main")

    solver = Solver(train_loader, config)
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='labelstudio', choices=['mtat', 'msd', 'jamendo', 'MER31K', 'labelstudio'])
    parser.add_argument('--model_type', type=str, default='short', choices=['fcn', 'musicnn', 'crnn', 'sample', 'se', 'short', 'short_res', 'attention', 'hcnn'])
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4) # 1e-4
    parser.add_argument('--use_tensorboard', type=int, default=1)
    parser.add_argument('--model_save_path', type=str, default='./../models/labelstudio/first')
    parser.add_argument('--model_load_path', type=str, default='.')#/best_model.pth
    parser.add_argument('--data_path', type=str, default='./labelstudio')
    parser.add_argument('--log_step', type=int, default=20)

    config = parser.parse_args()

    print(config)
    main(config)