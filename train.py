import torch
import numpy as np
import os
import re
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score
from utils import set_seed, save_model, load_model
from data import get_dataloader
from model.model import Net
from config import config
# loss/acc可视化
import matplotlib.pyplot as plt


class Main():
    def __init__(self, seed: int = 114514) -> None:
        set_seed(seed)
        self.seed = seed
        self._load_model_()
        self._load_data_()
        self._initialize_net_()
        self._set_loss_function_()
        self._set_optimizer_()
        self._load_state_dict_to_net_()
    
    def _load_model_(self):
        if not config.combined:
            model_pattern = re.compile('(.*?)_.*?')
            config.model = '/'.join(['pretrain-model', re.findall(model_pattern, config.data_name)[0] + '_pretrained.pth'])
            config.data_path = './data/'
        else:
            config.data_path = './combined_data/'

    def _load_data_(self):
        print(f'<===== Loading data from {config.data_name} =====>')
        self.train_loader, self.test_loader = get_dataloader(train_path=config.data_path + config.data_name + '/train.tsv',
                                                test_path=config.data_path + config.data_name + '/test.tsv')
        print("<===== Loaded =====>")
    
    def _initialize_net_(self):
        self.model = Net(4, 2, use_spectic_conv1d=True, use_spectic_transformer=True)
        self.model.to(config.device)
        print('Created model on', config.device)
    
    def _set_loss_function_(self):
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def _set_optimizer_(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def _load_state_dict_to_net_(self):
        if not config.combined:
            print('Loading state_dict from file')
            state_dict = load_model(config.model, map_location=config.device)
            print(self.model.load_state_dict(state_dict['model_state_dict']))
            print(self.criterion.load_state_dict(state_dict['criterion_state_dict']))
            print(self.optimizer.load_state_dict(state_dict['optimizer_state_dict']))
    
    def train_test(self):


        self.train_loss_table = []
        self.train_acc_table = []
        self.test_loss_table = []
        self.test_acc_table = []

        print('Start training')
        max_acc = 0
        max_f1 = 0

        for epoch in range(config.epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            total_step = len(self.train_loader)
            for i, train_data in enumerate(self.train_loader):
                inputs, labels = train_data

                self.optimizer.zero_grad()
                outputs, _ = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

                train_accuracy = train_correct / train_total

                if (i == total_step-1):
                    train_loss /= len(train_data)
                    self.train_loss_table.append(train_loss)
                    self.train_acc_table.append(train_accuracy)
                    print(f'Epoch [{epoch+1}/{config.epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}({train_correct}/{train_total})')

            # valiation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            self.outputs_all = np.empty((0, 2))
            self.labels_all = np.empty(0)
            self.predicted_all = np.empty(0)


            with torch.no_grad():
                for i, test_data in enumerate(self.test_loader):
                    inputs, labels = test_data

                    self.labels_all = np.concatenate((self.labels_all, labels.cpu().numpy()))


                    outputs, y = self.model(inputs)


                    self.outputs_all = np.concatenate((self.outputs_all, outputs.cpu().numpy()))

                    _, predicted = torch.max(outputs, 1)

                    self.predicted_all = np.concatenate((self.predicted_all, predicted.cpu().numpy()))

                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            val_accuracy = val_correct / val_total
            f1 = f1_score(self.labels_all, self.predicted_all)                # F1-Score

            os.makedirs("models") if not os.path.exists('models') else None

            if val_accuracy > max_acc or f1 > max_f1:
                state_dict = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'criterion_state_dict': self.criterion.state_dict()
                    }
                if val_accuracy > max_acc:
                    print('Found optimized model on accuracy. Saving state_dict to file.')
                    
                    save_model(state_dict, f'./models/{config.data_name}_ACC_best.pt')

                if f1 > max_f1:
                    print('Found optimized model on F1-score. Saving state_dict to file.')
                    save_model(state_dict, f'./models/{config.data_name}_F1_best.pt')

                sn = recall_score(self.labels_all, self.predicted_all)            # SN
                cm = confusion_matrix(self.labels_all, self.predicted_all)
                tn, fp, fn, tp = cm.ravel()
                sp = tn / (tn + fp)                                     # SP
                precision = precision_score(self.labels_all, self.predicted_all)  # Precision
                mcc = matthews_corrcoef(self.labels_all, self.predicted_all)      # MCC
                outputs_pos_all = self.outputs_all[:, 1]
                auc = roc_auc_score(self.labels_all, outputs_pos_all)        # AUC
                
                max_acc = max(max_acc, val_accuracy)
                max_f1 = max(max_f1, f1)

            avg_val_loss = val_loss / len(test_data)

            self.test_loss_table.append(avg_val_loss)
            self.test_acc_table.append(val_accuracy)
            print(f'Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}({val_correct}/{val_total}), Max Accuracy: {max_acc:.4f}\n')
            print(f'ACC={max_acc:.4f}, SN={sn:.4f}, SP={sp:.4f}, AUC={auc:.4f}, MCC={mcc:.4f}, F1={f1:.4f}')


        print('Finished training. Best performance:')
        print(f'ACC={max_acc:.4f}, SN={sn:.4f}, SP={sp:.4f}, AUC={auc:.4f}, MCC={mcc:.4f}, F1={f1:.4f}')
        print(f'Seed: {self.seed}')
        with open('results.csv', 'a+') as f:
            f.write(f'{config.data_name}, {config.model}, {max_acc:.4f}, {sn:.4f}, {sp:.4f}, {auc:.4f}, {mcc:.4f}, {precision:.4f}, {f1:.4f}\n')

        if config.combined and not os.path.exists(config.model):
            print('Saving state_dict to file')
            save_model({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'criterion_state_dict': self.criterion.state_dict()
            }, config.model)
        
    def draw_acc_loss_line(self):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(list(range(1, config.epochs+1)), self.train_loss_table, label='Train', color='blue')
        plt.plot(list(range(1, config.epochs+1)), self.test_loss_table, label='Test', color='orange')
        plt.legend()
        plt.title('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(list(range(1, config.epochs+1)), self.train_acc_table, label='Train', color='blue')
        plt.plot(list(range(1, config.epochs+1)), self.test_acc_table, label='Test', color='orange')
        plt.ylim(0, 1)
        plt.legend()
        plt.title('Accuracy')

        plt.suptitle(config.data_name)
        plt.show()


if __name__ == "__main__":
    main = Main(seed=114514)
    main.train_test()
    main.draw_acc_loss_line()

    
