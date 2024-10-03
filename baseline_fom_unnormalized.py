import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import LambdaLR
import learn2learn as l2l

# set the random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

batch_size = 64
input_dim = 2  # For example, (x1, x2) as inputs would require 2 dim
hidden_dim = 64
basis_function_dim = 32
output_dim = 1  # For example, z as output
learning_rate = 1e-7
num_adapt_epochs = 5
num_train_epochs = 1000
few_shot_k = 5
l1_lambda = 0.7
l2_lambda = 0.1
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class FOMDataset(Dataset):
    def __init__(self, csv_file, train, few_k_shot=5):
        # Load the CSV file
        self.data = pd.read_csv(csv_file)
        self.few_k_shot = few_k_shot

        # Extract features (thickness, wavelength) and target (fom)
        self.X = self.data[['thickness', 'wavelength']].values
        self.y = self.data['fom'].values

        # divide X and y into sets of 9, each set representing a type of sensor
        self.X = self.X.reshape(-1, 9, 2)
        self.y = self.y.reshape(-1, 9)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=5)

        if train:
            self.train_x = X_train[:, : self.few_k_shot, :]
            self.train_y = y_train[:, : self.few_k_shot]
            self.test_x = X_train[:, self.few_k_shot:, :]
            self.test_y = y_train[:, self.few_k_shot:]
        else:
            self.train_x = X_test[:, : self.few_k_shot, :]
            self.train_y = y_test[:, : self.few_k_shot]
            self.test_x = X_test[:, self.few_k_shot :, :]
            self.test_y = y_test[:, self.few_k_shot :]
        
        print(f"Train = {train}. Data loaded. Train X: {self.train_x.shape}, Train Y: {self.train_y.shape}, Test X: {self.test_x.shape}, Test Y: {self.test_y.shape}")

    def __len__(self):
        # Return the total number of samples
        return self.train_x.shape[0]

    def __getitem__(self, idx):
        # Get the input features and target for a given index
        context_x = torch.tensor(self.train_x[idx], dtype=torch.float32)
        context_y = torch.tensor(self.train_y[idx], dtype=torch.float32)
        target_x = torch.tensor(self.test_x[idx], dtype=torch.float32)
        target_y = torch.tensor(self.test_y[idx], dtype=torch.float32)
        return context_x, context_y, target_x, target_y
    
    def scale_inverse(self, y):
        return self.scaler_y.inverse_transform(y)
    
    def scale_inverse_x(self, x):
        return self.scaler_x.inverse_transform(x)

fom_dataset_train = FOMDataset(csv_file="fom.csv", train=True, few_k_shot=few_shot_k)
fom_dataset_test = FOMDataset(csv_file="fom.csv", train=False, few_k_shot=few_shot_k)

fom_dataloader_train = DataLoader(
    fom_dataset_train, batch_size=batch_size, shuffle=True, num_workers=16
)
fom_dataloader_test = DataLoader(
    fom_dataset_test, batch_size=batch_size, shuffle=False, num_workers=16
)

for t in fom_dataloader_train:
  print(f"Train x, train y: {t[0].shape, t[1].shape}")
  break

for t in fom_dataloader_test:
  print(f"Test x, test y: {t[0].shape, t[1].shape}")
  break

class BasisFunctionLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BasisFunctionLearner, self).__init__()
        # Define a fully connected network with ReLU activations
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        return self.fc6(x)  # This represents the basis functions, Φ(x)

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.ff_layer_0 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.ff_layer_1 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        dotp = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attention_weights = F.softmax(dotp, dim=-1)
        weighted_sum = torch.matmul(attention_weights, value)
        x = weighted_sum + query  # Adding the query for residual connection
        x = F.layer_norm(x, x.size()[1:])

        dense_out_0 = F.relu(self.ff_layer_0(x))
        x = x + self.ff_layer_1(dense_out_0)
        x = F.layer_norm(x, x.size()[1:])

        return x

class WeightsGenerator(nn.Module):
    def __init__(self, basis_function_dim, hidden_dim=512, attention_layers=8):
        super(WeightsGenerator, self).__init__()
        self.top_attention_layer = AttentionLayer(basis_function_dim, hidden_dim)
        self.attention_layers = nn.ModuleList([AttentionLayer(hidden_dim, hidden_dim) for _ in range(attention_layers)])

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

        self.final_dense = nn.Linear(hidden_dim, basis_function_dim)

    def forward(self, inputs):
        x = inputs
        x = self.top_attention_layer(x)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        final_weights = self.final_dense(x)
        return final_weights

class FewShotRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, basis_function_dim, output_dim):
        super(FewShotRegressionModel, self).__init__()
        self.basis_learner = BasisFunctionLearner(input_dim, hidden_dim, basis_function_dim)
        self.weights_generator = WeightsGenerator(basis_function_dim)

    def forward(self, x):
        # Step 1: Generate basis functions Φ(x)
        basis_functions = self.basis_learner(x)

        # Step 2: Generate weights w
        weights = self.weights_generator(basis_functions)

        final_pred = torch.diag(torch.matmul(basis_functions, weights.T))
        # print(f"Basis Functions: {basis_functions}, Weights: {weights}, Final Prediction: {final_pred}")

        # Step 3: Compute final prediction z = Φ(x) * w
        return weights, final_pred

model = FewShotRegressionModel(
    input_dim, hidden_dim, basis_function_dim, output_dim
).to(device)

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_trainable_params}")

maml = l2l.algorithms.MAML(model, lr=learning_rate, first_order=False, allow_unused=True)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
total_steps = len(fom_dataloader_train) * num_train_epochs
warmup_steps = total_steps // 10

def lr_lambda(current_step: int):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(
        0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
    )
scheduler = LambdaLR(optimizer, lr_lambda)

torch.autograd.set_detect_anomaly(True)
losses = []
val_losses = []

for epoch in range(num_train_epochs):
    for idx, (context_x, context_y, target_x, target_y) in enumerate(fom_dataloader_train):
        meta_train_loss = 0
        learn_loss = 0
        context_x, context_y, target_x, target_y = context_x.to(device), context_y.to(device), target_x.to(device), target_y.to(device)
        effective_batch_size = context_x.size(0)
        for i in range(effective_batch_size):
            learner = maml.clone(first_order=True)
            x_support, y_support = context_x[i], context_y[i]
            x_query, y_query = target_x[i], target_y[i]
            learn_loss = 0
            for _ in range(num_adapt_epochs):
                wts, predictions = learner(x_support)
                loss = criterion(predictions, y_support)
                learn_loss += loss.detach().cpu().item()
                learner.adapt(loss)
            wts, predictions = learner(x_query)
            loss = criterion(predictions, y_query)
            meta_train_loss += loss
            print(meta_train_loss)

        meta_train_loss /= effective_batch_size
        print("average meta train loss: ", meta_train_loss)
        learn_loss /= num_adapt_epochs*effective_batch_size
        losses.append(learn_loss)
        val_losses.append(meta_train_loss.item())

        meta_train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Meta Train Loss: {sum(losses)/len(losses):.4f}, Val Loss: {sum(val_losses)/len(val_losses):.4f}")

    if epoch % 50 == 0:
        torch.save(model.state_dict(), f'baseline_fom_wts.pt')

# plot the losses in a graph and save as a image
plt.plot(losses)
plt.plot(val_losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend(['Train Loss', 'Test Loss'])
plt.savefig(f'baseline_fom.png')


test_losses = []
for idx, (context_x, context_y, target_x, target_y) in enumerate(fom_dataloader_test):
    meta_test_loss = 0
    context_x, context_y, target_x, target_y = context_x.to(device), context_y.to(device), target_x.to(device), target_y.to(device)
    effective_batch_size = context_x.size(0)

    for i in range(effective_batch_size):
        learner = maml.clone(first_order = True)
        x_support, y_support = context_x[i], context_y[i]
        x_query, y_query = target_x[i], target_y[i]
        print(f"Support X: {x_support.shape}, Support Y: {y_support.shape}")
        for i in range(num_adapt_epochs):
            wts, predictions = learner(x_support)
            print(predictions, y_support)
            loss = criterion(predictions, y_support)
            print(loss)
            learner.adapt(loss)
        wts, predictions = learner(x_query)
        loss = criterion(predictions, y_query)
        # print few test input, predicted output and actual output
        for i in range(2):
            print(f"Input: {target_x[i]} Predicted Output: {fom_dataset_test.scale_inverse(predictions.reshape(-1, 1).detach().cpu().numpy())[i]}, Actual Output: {fom_dataset_test.scale_inverse(target_y.reshape(-1, 1).detach().cpu().numpy())[i]}")
        meta_test_loss += loss
    test_losses.append(loss.detach().cpu().item())
    if idx % 10 == 0:
        print(f"Step: {idx}, Test Loss: {sum(test_losses)/len(test_losses):.4f}")


print(f'Average Test Loss: {sum(test_losses)/len(test_losses):.4f}')

