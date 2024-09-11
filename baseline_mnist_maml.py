# %%
import numpy as np
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

# %%
# set the random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

batch_size = 128
num_shot = 50  # Example values, can be adjusted
num_val = 50
input_dim = 2  # For example, (x1, x2) as inputs would require 2 dim
hidden_dim = 64
basis_function_dim = 32
output_dim = 1  # For example, z as output
learning_rate = 2e-5
num_epochs = 5
l1_lambda = 0.5
l2_lambda = 0.05
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# %%
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

        # Step 3: Compute final prediction z = Φ(x) * w
        return weights, torch.diag(torch.matmul(basis_functions, weights.T))


# %%
class MnistDataset(Dataset):
    def __init__(self, num_shot, num_val, train=True, bmaml=False):
        self.num_shot = num_shot
        self.bmaml = bmaml
        self.dim_input = 2
        self.dim_output = 1

        # Load data from the pickle file
        with open('mnist_data/mnist.pkl', 'rb') as f:
            data = pickle.load(f)
            print(data['meta_train_x'].shape, data['meta_train_y'].shape, data['meta_test_x'].shape, data['meta_test_y'].shape)

        # Select train or test data
        if train:
            data_str = 'train'
        else:
            data_str = 'test'
            
        # Create dataset with train_x, train_y, test_x, test_y
        self.train_x = data['meta_' + data_str + '_x'][:, :num_shot, :]
        self.train_y = data['meta_' + data_str + '_y'][:, :num_shot, :]
        self.test_x = data['meta_' + data_str + '_x'][:, num_shot:num_shot+num_val, :]
        self.test_y = data['meta_' + data_str + '_y'][:, num_shot:num_shot+num_val, :]

        print('Data loaded: train_x', self.train_x.shape, 'test_x', self.test_x.shape, 
              'train_y', self.train_y.shape, 'test_y', self.test_y.shape)

    def __len__(self):
        # Return the number of examples
        return self.train_x.shape[0]

    def __getitem__(self, indx):
        # Fetch context and target data based on index
        context_x = torch.tensor(self.train_x[indx], dtype=torch.float32)
        context_y = torch.tensor(self.train_y[indx], dtype=torch.float32)
        target_x = torch.tensor(self.test_x[indx], dtype=torch.float32)
        target_y = torch.tensor(self.test_y[indx], dtype=torch.float32)

        if self.bmaml:
            # If bmaml is True, concatenate context and target data for leader_x and leader_y
            leader_x = torch.cat([context_x, target_x], dim=0)
            leader_y = torch.cat([context_y, target_y], dim=0)
            return context_x, context_y, leader_x, leader_y, target_x, target_y

        return context_x, context_y, target_x, target_y

def create_dataloader(num_shot, num_val, batch_size, train=True, bmaml=False):
    dataset = MnistDataset(num_shot=num_shot, num_val=num_val, train=train, bmaml=bmaml)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def custom_loss_function(predictions, targets, weights, l1_lambda=0.001, l2_lambda=0.0001):
    mse_loss = nn.MSELoss()(predictions, targets)
    l1_loss = l1_lambda * torch.norm(weights, p=1)
    l2_loss = l2_lambda * torch.norm(weights, p=2)
    return mse_loss + l1_loss + l2_loss

# Create dataloaders
train_loader = create_dataloader(num_shot, num_val, batch_size, train=True)
test_loader = create_dataloader(num_shot, 784-num_val, batch_size, train=False)

# %%
for t in train_loader:
  print(f"Context x, context y, target x, target y: {t[0].shape, t[1].shape, t[2].shape, t[3].shape}")
  break

for t in test_loader:
  print(f"Context x, context y, target x, target y: {t[0].shape, t[1].shape, t[2].shape, t[3].shape}")
  break

# %%
model = FewShotRegressionModel(input_dim, hidden_dim, basis_function_dim, output_dim).to(device)
maml = l2l.algorithms.MAML(model, lr=learning_rate, first_order=False, allow_unused=True)
opt = optim.Adam(maml.parameters(), lr=learning_rate)

# %%
for idx, (context_x, context_y, target_x, target_y) in enumerate(train_loader):
    meta_train_loss = 0.0
    context_x, context_y, target_x, target_y = context_x.to(device), context_y.to(device), target_x.to(device), target_y.to(device)
    effective_batch_size = context_x.size(0)
    for i in range(effective_batch_size):
        learner = maml.clone(first_order=True)
        x_support, y_support = context_x[i], context_y[i]
        x_query, y_query = target_x[i], target_y[i]
        y_support = y_support.view(-1)
        y_query = y_query.view(-1)
        for _ in range(num_epochs):
            wts, predictions = learner(x_support)
            loss = custom_loss_function(predictions, y_support, wts)
            learner.adapt(loss)
        wts, predictions = learner(x_query)
        loss = custom_loss_function(predictions, y_query, wts)
        meta_train_loss += loss

    meta_train_loss /= effective_batch_size
    if idx % 10 == 0:
        print(f"Iteration: {idx+1}, Meta train loss: {meta_train_loss}")
    if idx % 50 == 0:
        torch.save(model.state_dict(), f'mnist_model_weights_{idx}.pth')
    
    opt.zero_grad()
    meta_train_loss.backward()
    opt.step()
