import torch
import torch.nn as nn
import torch.optim as optim


def tensor_k_func(output,  l):
    diff = output[:, 1:] - output[:, :-1]
    sum_k = l * 0.5 * torch.sum(diff**2, dim=-1)
    k_padded = torch.cat([torch.zeros(output.size(0), 1), k], dim=1)
    return k_padded

def tensor_v_func(output, l, elt_v_func):
    potential = elt_v_func(output)
    sum_v = torch.sum(potential, dim=-1) / l
    return sum_v

def flatten_cells(cell, grid_len):
    return cell[:, 0] * grid_len + cell[:, 1]

def mod_2pi(x):
    return x - 2 * torch.pi * torch.floor(x / (2 * torch.pi))

class RecurrentModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, v_func, k_func, l, hbar=1):
        super(RecurrentModel, self).__init__()
        self.l = l
        self.v_func = v_func
        self.k_func = k_func
        self.hbar = hbar
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        rnn_out, hidden = self.rnn(x, hidden)
        output = torch.floor(self.fc(rnn_out)).long()
        start_idx = flatten_cells(output[:, 0])
        end_idx = flatten_cells(output[:, -1])
        action = self.k_func(output) - self.v_func(output)
        action_sort = torch.zeros((input_size ** 2, input_size ** 2), action, accumulate=True)
        return mod_2pi(action_sort), hidden

def run_multiple_passes(model, input_seq, num_passes):
    cumulative_output = torch.zeros_like(input_seq)
    hidden = None

    for _ in range(num_passes):
        output, hidden = model(input_seq, hidden)
        cumulative_output += output
    
    return cumulative_output

def custom_loss_function(T_sol, T_out_sum):
    loss = torch.norm(T_sol - T_out_sum)
    return loss

def train_model(model, data_loader, num_passes, optimizer, T_sol, num_epochs=10):
    for epoch in range(num_epochs):
        for input_seq in data_loader:
            optimizer.zero_grad()
            cumulative_output = run_multiple_passes(model, input_seq, num_passes)
            loss = custom_loss_function(T_sol, cumulative_output)
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
