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

# set the random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

batch_size = 64
input_dim = 2  # For example, (x1, x2) as inputs would require 2 dim
hidden_dim = 64
basis_function_dim = 32
output_dim = 1  # For example, z as output
learning_rate = 2e-5
num_epochs = 1000
l1_lambda = 0.5
l2_lambda = 0.05
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class FOMDataset(Dataset):
    def __init__(self, csv_file, train, few_k_shot=5):
        # Load the CSV file
        self.data = pd.read_csv(csv_file)
        self.few_k_shot = few_k_shot

        # Extract features (thickness, wavelength) and target (fom)
        X = self.data[['thickness', 'wavelength']].values
        y = self.data['fom'].values

        # divide X and y into sets of 9, each set representing a type of sensor
        X = X.reshape(-1, 9, 2)
        y = y.reshape(-1, 9)

        if train:
            self.X = X[:, : self.few_k_shot, :].reshape(-1, 2)
            self.y = y[:, : self.few_k_shot].reshape(-1)
        else:
            self.X = X[:, self.few_k_shot:, :].reshape(-1, 2)
            self.y = y[:, self.few_k_shot:].reshape(-1)
        
        self.scaler_X = StandardScaler()
        self.X = self.scaler_X.fit_transform(self.X)

        self.scaler_y = StandardScaler()
        self.y = self.scaler_y.fit_transform(self.y.reshape(-1, 1)).reshape(-1)
    def __len__(self):
        # Return the total number of samples
        return len(self.X)

    def __getitem__(self, idx):
        # Get the input features and target for a given index
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)
        return x, y
    
    def scale_inverse(self, y):
        return self.scaler_y.inverse_transform(y)

fom_dataset_train = FOMDataset(csv_file="fom.csv", train=True)
fom_dataset_test = FOMDataset(csv_file="fom.csv", train=False)

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

        # Step 3: Compute final prediction z = Φ(x) * w
        return weights, torch.diag(torch.matmul(basis_functions, weights.T))

def custom_loss_function(predictions, targets, weights, l1_lambda=0.001, l2_lambda=0.0001):
    mse_loss = nn.MSELoss()(predictions, targets)
    l1_loss = l1_lambda * torch.norm(weights, p=1)
    l2_loss = l2_lambda * torch.norm(weights, p=2)
    return mse_loss + l1_loss + l2_loss

# Lambda function for linear schedule
def lr_lambda(current_step: int):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(
        0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
    )

model = FewShotRegressionModel(
    input_dim, hidden_dim, basis_function_dim, output_dim
).to(device)

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_trainable_params}")

model = FewShotRegressionModel(input_dim, hidden_dim, basis_function_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
total_steps = len(fom_dataloader_train) * num_epochs
warmup_steps = total_steps // 10
scheduler = LambdaLR(optimizer, lr_lambda)

torch.autograd.set_detect_anomaly(True)
losses = []
test_losses = []
for epoch in trange(num_epochs):
    model.train()

    for batch_x_train, batch_y_train in fom_dataloader_train:
        # move to device
        batch_x_train = batch_x_train.to(device, torch.float32).view(-1, input_dim)
        batch_y_train = batch_y_train.to(device, torch.float32).view(-1)
        
        batch_size = batch_x_train.shape[0]

        loss = 0
        wts, predictions = model(batch_x_train)

        loss += custom_loss_function(predictions, batch_y_train, wts, l1_lambda, l2_lambda)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        losses.append(loss.item())
        temp_test_losses = []
        for batch_x_test, batch_y_test in fom_dataloader_test:
            batch_x_test = batch_x_test.to(device, torch.float32).view(-1, input_dim)
            batch_y_test = batch_y_test.to(device, torch.float32).view(-1)

            wts, predictions = model(batch_x_test)
            loss = custom_loss_function(predictions, batch_y_test, wts, l1_lambda, l2_lambda)
            temp_test_losses.append(loss.item())
        test_losses.append(sum(temp_test_losses)/len(temp_test_losses))
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

# plot the losses in a graph and save as a image
plt.plot(losses)
plt.plot(test_losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend(['Train Loss', 'Test Loss'])
plt.savefig(f'baseline_fom_{epoch+1}.png')

model.eval()
with torch.no_grad():
    test_losses = []
    for batch_x_test, batch_y_test in fom_dataloader_test:
        batch_x_test = batch_x_test.to(device, torch.float32).view(-1, input_dim)
        batch_y_test = batch_y_test.to(device, torch.float32).view(-1)

        wts, predictions = model(batch_x_test)
        loss = custom_loss_function(predictions, batch_y_test, wts, l1_lambda, l2_lambda)
        test_losses.append(loss.item())
    print(f'Test Loss: {sum(test_losses)/len(test_losses):.4f}')

    # print few test input, predicted output and actual output after inverse scaling
    for i in range(5):
        # print(f"Test Input: {batch_x_test[i].cpu().numpy()}, Predicted Output: {predictions[i].cpu().numpy()}, Actual Output: {batch_y_test[i].cpu().numpy()}")
        print(f"Predicted Output (inverse scaled): {fom_dataset_test.scale_inverse(predictions.reshape(-1, 1).cpu().numpy())[i]}, Actual Output (inverse scaled): {fom_dataset_test.scale_inverse(batch_y_test.reshape(-1, 1).cpu().numpy())[i]}")