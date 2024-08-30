import torch
import numpy as np
import os
import re
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score
from utils import set_seed, save_model, load_model
from data import get_dataloader
from model import Net
from config import config
import matplotlib.pyplot as plt

class Main():
    def __init__(self, seed: int = 114514) -> None:
        # Set the random seed for reproducibility
        set_seed(seed)
        self.seed = seed
        
        # Load model configuration and paths
        self._load_model_()
        
        # Load the training and testing data
        self._load_data_()
        
        # Initialize the neural network model
        self._initialize_net_()
        
        # Set the loss function (CrossEntropyLoss)
        self._set_loss_function_()
        
        # Set the optimizer (Adam)
        self._set_optimizer_()
        
        # Load the pretrained model weights if available
        self._load_state_dict_to_net_()
    
    def _load_model_(self):
        # Determine the model file path based on whether combined mode is enabled
        if not config.combined:
            model_pattern = re.compile('(.*?)_.*?')
            config.model = '/'.join(['pretrain-model', re.findall(model_pattern, config.data_name)[0] + '_pretrained.pth'])
            config.data_path = './data/'
        else:
            config.data_path = './combined_data/'

    def _load_data_(self):
        # Load the data from specified paths
        print(f'<===== Loading data from {config.data_name} =====>')
        self.train_loader, self.test_loader = get_dataloader(train_path=config.data_path + config.data_name + '/train.tsv',
                                                test_path=config.data_path + config.data_name + '/test.tsv')
        print("<===== Loaded =====>")
    
    def _initialize_net_(self):
        # Initialize the model with specific parameters
        self.model = Net(4, 2, use_spectic_conv1d=True, use_spectic_transformer=True)
        self.model.to(config.device)  # Move model to specified device (CPU or GPU)
        print('Created model on', config.device)
    
    def _set_loss_function_(self):
        # Set the loss function to CrossEntropyLoss
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def _set_optimizer_(self):
        # Set the optimizer to Adam with a learning rate specified in config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.optimizer.param_groups[0]['capturable'] = True

    def _load_state_dict_to_net_(self):
        # Load state dictionaries for model, criterion, and optimizer from a file if not in combined mode
        if not config.combined:
            print('Loading state_dict from file')
            state_dict = load_model(config.model, map_location=config.device)
            print(self.model.load_state_dict(state_dict['model_state_dict']))
            print(self.criterion.load_state_dict(state_dict['criterion_state_dict']))
            print(self.optimizer.load_state_dict(state_dict['optimizer_state_dict']))
    
    def train_test(self):
        # Initialize tables to store loss and accuracy for training and testing
        self.train_loss_table = []
        self.train_acc_table = []
        self.test_loss_table = []
        self.test_acc_table = []

        print('Start training')
        max_acc = 0  # Variable to keep track of maximum accuracy
        max_f1 = 0   # Variable to keep track of maximum F1-score

        # Training loop for specified number of epochs
        for epoch in range(config.epochs):
            self.model.train()  # Set model to training mode
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            total_step = len(self.train_loader)
            
            # Iterate over the training data loader
            for i, train_data in enumerate(self.train_loader):
                inputs, labels = train_data

                self.optimizer.zero_grad()  # Zero the parameter gradients
                outputs, _ = self.model(inputs)  # Forward pass
                _, predicted = torch.max(outputs, 1)  # Get predictions
                loss = self.criterion(outputs, labels)  # Calculate loss

                loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights

                # Accumulate loss and correct predictions for accuracy calculation
                train_loss += loss.item()
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

                train_accuracy = train_correct / train_total

                # Record training loss and accuracy at the end of each epoch
                if (i == total_step-1):
                    train_loss /= len(train_data)
                    self.train_loss_table.append(train_loss)
                    self.train_acc_table.append(train_accuracy)
                    print(f'Epoch [{epoch+1}/{config.epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}({train_correct}/{train_total})')

            # Validation phase
            self.model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            # Initialize arrays to store all outputs, labels, and predictions for metrics calculation
            self.outputs_all = np.empty((0, 2))
            self.labels_all = np.empty(0)
            self.predicted_all = np.empty(0)

            # Iterate over the testing data loader
            with torch.no_grad():
                for i, test_data in enumerate(self.test_loader):
                    inputs, labels = test_data

                    self.labels_all = np.concatenate((self.labels_all, labels.cpu().numpy()))  # Store true labels

                    outputs, y = self.model(inputs)  # Forward pass

                    self.outputs_all = np.concatenate((self.outputs_all, outputs.cpu().numpy()))  # Store outputs

                    _, predicted = torch.max(outputs, 1)  # Get predictions

                    self.predicted_all = np.concatenate((self.predicted_all, predicted.cpu().numpy()))  # Store predictions

                    loss = self.criterion(outputs, labels)  # Calculate loss

                    # Accumulate loss and correct predictions for accuracy calculation
                    val_loss += loss.item()
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            # Calculate validation accuracy and F1-score
            val_accuracy = val_correct / val_total
            f1 = f1_score(self.labels_all, self.predicted_all)  # F1-Score

            # Create directory for saving models if it doesn't exist
            os.makedirs("models") if not os.path.exists('models') else None

            # Save the best model based on accuracy or F1-score
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

                # Calculate additional metrics for model evaluation
                sn = recall_score(self.labels_all, self.predicted_all)  # Sensitivity (Recall)
                cm = confusion_matrix(self.labels_all, self.predicted_all)  # Confusion matrix
                tn, fp, fn, tp = cm.ravel()
                sp = tn / (tn + fp)  # Specificity
                precision = precision_score(self.labels_all, self.predicted_all)  # Precision
                mcc = matthews_corrcoef(self.labels_all, self.predicted_all)  # MCC
                outputs_pos_all = self.outputs_all[:, 1]
                auc = roc_auc_score(self.labels_all, outputs_pos_all)  # AUC
                
                # Update maximum accuracy and F1-score
                max_acc = max(max_acc, val_accuracy)
                max_f1 = max(max_f1, f1)

            # Calculate average validation loss
            avg_val_loss = val_loss / len(test_data)

            # Record validation loss and accuracy
            self.test_loss_table.append(avg_val_loss)
            self.test_acc_table.append(val_accuracy)
            print(f'Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}({val_correct}/{val_total}), Max Accuracy: {max_acc:.4f}\n')
            print(f'ACC={max_acc:.4f}, SN={sn:.4f}, SP={sp:.4f}, AUC={auc:.4f}, MCC={mcc:.4f}, F1={f1:.4f}')


        # Training completed, print the best performance metrics
        print('Finished training. Best performance:')
        print(f'ACC={max_acc:.4f}, SN={sn:.4f}, SP={sp:.4f}, AUC={auc:.4f}, MCC={mcc:.4f}, F1={f1:.4f}')
        print(f'Seed: {self.seed}')
        
        # Save the results to a CSV file
        with open('results.csv', 'a+') as f:
            f.write(f'{config.data_name}, {config.model}, {max_acc:.4f}, {sn:.4f}, {sp:.4f}, {auc:.4f}, {mcc:.4f}, {precision:.4f}, {f1:.4f}\n')

        # Save the state dictionary if combined mode is enabled and model file doesn't exist
        if config.combined and not os.path.exists(config.model):
            print('Saving state_dict to file')
            save_model({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'criterion_state_dict': self.criterion.state_dict()
            }, config.model)

    def draw_acc_loss_line(self):
        # Draw and save accuracy and loss line plots
        plt.figure()
        plt.plot(self.train_acc_table, 'ro-', label='Train accuracy')
        plt.plot(self.test_acc_table, 'bs-', label='Val accuracy')
        plt.legend()
        plt.savefig(config.data_name + '_accuracy.png')
        plt.show()

        plt.figure()
        plt.plot(self.train_loss_table, 'ro-', label='Train loss')
        plt.plot(self.test_loss_table, 'bs-', label='Val loss')
        plt.legend()
        plt.savefig(config.data_name + '_loss.png')
        plt.show()


if __name__ == "__main__":
    main = Main(seed=114514)
    main.train_test()
    if config.is_drawing_plot:
        main.draw_acc_loss_line()

    
