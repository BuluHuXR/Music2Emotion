from data_loader.labelstudio_loader import get_audio_loader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import model as Model
import numpy as np
import argparse
import torch
import os

def main(config):
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

    train_loader = get_audio_loader(config.data_path,
                                    config.batch_size,
                                    split='TRAIN',
                                    input_length=config.input_length,
                                    num_workers=config.num_workers)

    valid_loader = get_audio_loader(config.data_path,
                                    config.batch_size,
                                    split='VALID',
                                    input_length=config.input_length,
                                    num_workers=config.num_workers)

    def fine_tune_model(pretrained_model, num_classes, device='cuda'):
        model = Model.ShortChunkCNN(n_class=num_classes)

        checkpoint = torch.load(pretrained_model, map_location=torch.device('cpu'))
        checkpoint = {k: v for k, v in checkpoint.items() if 'dense2' not in k}
        model.load_state_dict(checkpoint, strict=False)

        for name, param in model.named_parameters():
            param.requires_grad = False

        model.dense2 = nn.Linear(512, num_classes)
        model.dense2.requires_grad = True

        model.to(device)

        return model

    def train(model, dataloader, criterion, optimizer, device):
        model.train()
        total_loss = 0

        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)


    def validate(model, dataloader, criterion, device):
        model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            all_labels.append(labels.detach().cpu().numpy())
            all_preds.append(outputs.detach().cpu().numpy())

        total_loss /= len(dataloader)
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        fpr, tpr, _ = roc_curve(all_labels.ravel(), all_preds.ravel())
        roc_auc = auc(fpr, tpr)

        return total_loss, roc_auc

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = fine_tune_model(os.path.join(config.pretrained_model_path, "best_model.pth"), num_classes=5, device=device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.dense2.parameters(), lr=config.lr)

    train_losses = []
    valid_losses = []
    for epoch in range(config.n_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        valid_loss, roc_auc = validate(model, valid_loader, criterion, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'Epoch {epoch + 1}/{config.n_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Validation ROC AUC: {roc_auc:.4f}')

    torch.save(model.state_dict(), os.path.join(config.finetune_model_path, "finetuned_model.pth"))

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join(config.finetune_model_path, "loss_curve.png"))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='labelstudio', choices=['mtat', 'msd', 'jamendo', 'MER31K', 'labelstudio'])
    parser.add_argument('--model_type', type=str, default='short_res', choices=['fcn', 'musicnn', 'crnn', 'sample', 'se', 'short', 'short_res', 'attention', 'hcnn'])
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--use_tensorboard', type=int, default=1)
    parser.add_argument('--model_save_path', type=str, default='./../models/labelstudio/finetune')
    parser.add_argument('--data_path', type=str, default='./labelstudio_3.6')
    parser.add_argument('--log_step', type=int, default=20)
    parser.add_argument('--pretrained_model_path', type=str, default='./../models/')
    parser.add_argument('--finetune_model_path', type=str, default='./../models/labelstudio/finetune')

    config = parser.parse_args()

    print(config)
    main(config)
