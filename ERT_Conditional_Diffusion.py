import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import Generate_ERT_utils as ert_utils
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from scipy.stats import wasserstein_distance
import imageio.v2 as imageio
import io
import pandas as pd
import matplotlib.cm as cm

#%% Fix random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
import random
random.seed(42)

#%% Define functions
def transform_to_unconstrained(x, a, b):
    """
    Given x in [a, b] (as tensor or np.array), transforms it to the unconstrained real line using the logit transform.
    For x in [a, b], first compute x_norm = (x - a)/(b - a) to scale to [0,1],
    then apply the logit: u = log(x_norm/(1 - x_norm)).
    """
    eps = 1e-6  # To prevent division by zero / log(0)
    if isinstance(x, torch.Tensor):
        x_norm = (x - a) / (b - a)
        x_norm = torch.clamp(x_norm, min=eps, max=1 - eps)
        return torch.log(x_norm / (1 - x_norm))
    else:
        x_norm = (x - a) / (b - a)
        x_norm = np.clip(x_norm, eps, 1 - eps)
        return np.log(x_norm / (1 - x_norm))

def inverse_transform(u, a, b):
    """
    Inverse transform: 
      Given u ∈ ℝ, apply the sigmoid to map back to (0,1) and then scale:
         x = a + (b - a) * sigmoid(u)
    """
    if isinstance(u, torch.Tensor):
        x_norm = torch.sigmoid(u)
        return a + (b - a) * x_norm
    else:
        x_norm = 1 / (1 + np.exp(-u))
        return a + (b - a) * x_norm

class DiffusionDataset(Dataset):
    def __init__(self, sim_param, ert_sim):
        """
        sim_param: numpy array of shape (N, 29, 1) or (N, 29)
        ert_sim: numpy array of shape (N, 4693, 14) where 14 represents surveys and 4693 observations.
                 We will transpose ert_sim to (N, 14, 4693) so that the 14 surveys are channels for Conv.
        """
        # Process simulation parameters:
        if sim_param.ndim == 3 and sim_param.shape[2] == 1:
            raw_params = np.squeeze(sim_param, axis=2)
        else:
            raw_params = sim_param.copy()
        # Convert raw parameters to torch and reparameterize (map [a, b] into the unconstrained space)
        self.params = transform_to_unconstrained(torch.from_numpy(raw_params).float(), a, b)
        
        # Process ERT simulation data:
        # Raw shape assumed: (N, 4693, 14) -> we transpose to (N, 14, 4693)
        self.conditions = torch.from_numpy(np.transpose(ert_sim, (0, 2, 1))).float()
        
    def __len__(self):
        return self.params.shape[0]
    
    def __getitem__(self, idx):
        return self.params[idx], self.conditions[idx]

def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000.0) / (half_dim - 1)
    exponents = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb_t = timesteps.float().unsqueeze(1) * exponents.unsqueeze(0)
    emb = torch.cat([torch.sin(emb_t), torch.cos(emb_t)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(timesteps.size(0), 1, device=timesteps.device)], dim=1)
    return emb

def get_diffusion_schedule(T, beta_start=1e-4, beta_end=0.02, device='cpu'):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar

def q_sample(x0, t, noise, alpha_bar):
    sqrt_alpha_bar = torch.sqrt(alpha_bar[t]).unsqueeze(1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t]).unsqueeze(1)
    return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

# Reverse Sampling with Temperature (optional)
@torch.no_grad()
def sample_model(model, condition, T, betas, alphas, alpha_bar, param_dim, device, num_steps=None, temperature=1.0):
    if num_steps is None:
        num_steps = T
    B = condition.size(0)
    x = torch.randn(B, param_dim, device=device)
    for t_ in reversed(range(num_steps)):
        t_tensor = torch.full((B,), t_, device=device, dtype=torch.long)
        pred_noise = model(x, t_tensor, condition)
        alpha_t = alphas[t_]
        alpha_bar_t = alpha_bar[t_]
        coef = (1 - alpha_t) / (math.sqrt(1 - alpha_bar_t) + 1e-8)
        x = (1.0 / math.sqrt(alpha_t)) * (x - coef * pred_noise)
        if t_ > 0:
            z = torch.randn_like(x)
            sigma_t = math.sqrt(betas[t_])
            x = x + sigma_t * temperature * z
    return x

# Conditional Diffusion Model (using a 1D condition encoder)
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, param_dim, hidden_dim=128):
        """
        Predicts the noise injected into the simulation parameters (x) conditioned on:
         - t: the diffusion timestep.
         - condition: ERT simulation data (with shape (B, 14, 4693))
        """
        super(ConditionalDiffusionModel, self).__init__()
        self.param_dim = param_dim
        
        # Condition encoder: a 1D CNN for the condition (14 channels, 4693 length)
        self.condition_encoder = nn.Sequential(
            nn.Conv1d(in_channels=14, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Output becomes (B, 64, 1)
            nn.Flatten(),             # Now (B, 64)
            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )
        # Time embedding layer
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Main MLP: combines the noisy simulation parameters, time embedding, and condition embedding.
        self.mlp = nn.Sequential(
            nn.Linear(param_dim + 2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim)
        )
        
    def forward(self, x, t, condition):
        # x: (B, param_dim) (noisy target in unconstrained/u space)
        # t: (B,) timesteps
        # condition: (B, 14, 4693)
        t_emb = get_timestep_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)            # (B, hidden_dim)
        cond_emb = self.condition_encoder(condition)  # (B, hidden_dim)
        h = torch.cat([x, t_emb, cond_emb], dim=1)  # (B, param_dim + 2*hidden_dim)
        out = self.mlp(h)  # predicted noise (in unconstrained space)
        return out

def mode_kde_calculation(array):

    data_min = np.min(array)
    data_max = np.max(array)
    x_range = np.linspace(data_min, data_max, 1000) 

    kde = stats.gaussian_kde(array)
    
    # Evaluate the KDE on the common x_range
    kde_values = kde(x_range)
    
    # Find the peak (mode)
    max_index = np.argmax(kde_values)
    point_mode = x_range[max_index]
            
    return point_mode

def check_param_bounds(param, limits):
    """
    Check if parameter sets are within bounds and return only valid sets.
    
    Args:
        param: numpy array of shape (n_samples, n_params)
        limits: numpy array of shape (n_params, 2) containing [min, max] for each parameter
        
    Returns:
        valid_params: numpy array containing only parameter sets where all parameters are within bounds
    """
    valid_params = []
    for batch_idx in range(param.shape[0]):
        sample_params = param[batch_idx]
        valid = True
        
        # Check if any parameter is outside its bounds
        for param_idx, (min_val, max_val) in enumerate(limits):
            param_val = sample_params[param_idx]
            if param_val < min_val or param_val > max_val:
                valid = False
                print(f"Sample {batch_idx} Parameter {param_idx}: {param_val:.4f} (out of bounds [{min_val:.4f}, {max_val:.4f}])")
                break
                
        # If all parameters are within bounds, add to valid_params
        if valid:
            valid_params.append(sample_params)
            
    # Only stack if we have valid parameters
    if valid_params:
        valid_params = np.stack(valid_params)
        #print(f"Found {len(valid_params)} valid parameter sets out of {param.shape[0]} total sets")
        return valid_params
    else:
        #print("No valid parameter sets found")
        return None

#%% Data loading and Conditional Model Training

# Main training and sampling script
ert_sim_file = 'sim_ert_sobol_5000.npy'
sim_param_file = 'sim_param_sobol_5000.npy'
# Load the data.
sim_param = np.load(sim_param_file)  # Shape: (5076, 29, 1)
ert_sim = np.load(ert_sim_file)        # Shape: (5076, 4693, 14)

#%% 
a, b = 0.0, 1.0

# Apply min-max normalization to ensure data is in [a, b].
param_scaler = MinMaxScaler(feature_range=(a, b))
sim_param_2d = sim_param.reshape(sim_param.shape[0], -1)
sim_param_scaled = param_scaler.fit_transform(sim_param_2d)
sim_param_scaled = sim_param_scaled.reshape(sim_param.shape)

for i in range(sim_param_scaled.shape[1]):
    print (f"Parameter {i}:")
    print (f"Min: {np.min(sim_param_scaled[:,i,:])}")
    print (f"Max: {np.max(sim_param_scaled[:,i,:])}")

if sim_param_scaled.ndim == 3 and sim_param_scaled.shape[2] == 1:
    raw_params = np.squeeze(sim_param_scaled, axis=2)
else:
    raw_params = sim_param.copy()

unc_params = transform_to_unconstrained(raw_params, a, b)
for i in range(unc_params.shape[1]):
    plt.figure(figsize=(6, 4), dpi=250)
    plt.hist(sim_param_scaled[:,i,:], bins=100, density=True, alpha=0.5, color='red', label='Constrained')
    plt.figure(figsize=(6, 4), dpi=250)
    plt.hist(unc_params[:,i], bins=100, density=True, alpha=0.5, color='blue', label='Unconstrained')
    plt.legend()

# Similarly, standardize the ERT simulation data.
ert_scaler = MinMaxScaler(feature_range=(0, 1))
ert_shape = ert_sim.shape  # (5076, 4693, 14)
ert_sim_2d = ert_sim.reshape(ert_shape[0], -1)
ert_sim_scaled = ert_scaler.fit_transform(ert_sim_2d)
ert_sim_scaled = ert_sim_scaled.reshape(ert_shape)

norm_ert_sim_min = np.min(ert_sim_scaled)
norm_ert_sim_max = np.max(ert_sim_scaled)
print(f"Normalized ERT simulation data range: [{norm_ert_sim_min:.4f}, {norm_ert_sim_max:.4f}]")


# Create the diffusion dataset using the reparameterized simulation targets.
dataset = DiffusionDataset(sim_param_scaled, ert_sim_scaled)
N = len(dataset)
train_size = int(0.8 * N)
val_size = int(0.1 * N)
test_size = N - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print("Using device:", device)

# Determine parameter dimension.
param_dim = dataset.params.shape[1] 
model = ConditionalDiffusionModel(param_dim=param_dim, hidden_dim=128).to(device)

# Diffusion schedule.
T = 500
betas, alphas, alpha_bar = get_diffusion_schedule(T, beta_start=1e-4, beta_end=0.02, device=device)
alpha_bar = alpha_bar.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
train_loss_history = []
val_loss_history = []

num_epochs = 500
best_val_loss = float('inf')
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for (x0, cond) in train_loader:
        x0 = x0.to(device)
        cond = cond.to(device)
        B = x0.size(0)
        t = torch.randint(0, T, (B,), device=device).long()
        noise = torch.randn_like(x0)
        x_noisy = q_sample(x0, t, noise, alpha_bar)
        pred_noise = model(x_noisy, t, cond)
        loss = criterion(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * B

    epoch_loss = running_loss / len(train_dataset)

    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for (x0_val, cond_val) in val_loader:
            x0_val = x0_val.to(device)
            cond_val = cond_val.to(device)
            B_val = x0_val.size(0)
            t_val = torch.randint(0, T, (B_val,), device=device).long()
            noise_val = torch.randn_like(x0_val)
            x_noisy_val = q_sample(x0_val, t_val, noise_val, alpha_bar)
            pred_noise_val = model(x_noisy_val, t_val, cond_val)
            loss_val = criterion(pred_noise_val, noise_val)
            val_loss_total += loss_val.item() * B_val

    val_loss = val_loss_total / len(val_dataset)
    train_loss_history.append(epoch_loss)
    val_loss_history.append(val_loss)

    improved = val_loss < best_val_loss
    if improved:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "train_history": train_loss_history,
            "val_history": val_loss_history,
            "param_dim": param_dim
        }, best_model_path)
        print(f"[Epoch {epoch+1}] Validation improved -> saved best model ({best_val_loss:.6f})")

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f} | Best: {best_val_loss:.6f}")


plt.figure(figsize=(8, 4),dpi=250)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

#%% Sampling: generate simulation parameters from a given condition.

def load_best_model(path, model, optimizer=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f'Loaded best model from epoch {ckpt["epoch"]} with val loss {ckpt["best_val_loss"]:.6f}')
    return ckpt

ckpt = load_best_model("checkpoints/best_model.pt", model, optimizer, map_location=device)
model.eval()

test_batch = next(iter(test_loader))
x0_test, ert_test = test_batch
ert_test = ert_test.to(device)

param_limits = ert_utils.ParameterLimits()
limits = param_limits.plims
limits = param_limits.plims


ert_condition = ert_test
ert_true_param = x0_test
print (ert_condition.shape)
print (ert_true_param.shape)

uncertainty_samples = 50  # you can increase this to get more samples

params_realizations = []
param_realizations_norm = []
for realization in range(uncertainty_samples):
    generated_params_unconstrained = sample_model(model, ert_condition, T, betas, alphas, alpha_bar, param_dim, device)
    param_realizations_norm.append(generated_params_unconstrained.cpu().numpy())
    # Convert the unconstrained variables back to the original parameter space:
    generated_params = inverse_transform(generated_params_unconstrained, a, b)
    # If necessary, convert to numpy:
    generated_params_np = generated_params.cpu().numpy()
    generated_params_np = param_scaler.inverse_transform(generated_params_np)
    valid_params = check_param_bounds(generated_params_np, limits)
    if valid_params is not None:
        params_realizations.append(valid_params)
    else:
        print(f"Skipping realization {realization} due to invalid parameter sets")
params_realizations = np.stack(params_realizations) #Original scale parameters
params_realizations_norm = np.stack(param_realizations_norm) #Unconstrained parameters

ert_true_param_inv = inverse_transform(ert_true_param.cpu().numpy(), a, b) #Inverse sigmoid
ert_true_param_inv = param_scaler.inverse_transform(ert_true_param_inv)

ert_sim_test = ert_test.cpu().numpy().swapaxes(1, 2)
ert_sim_test_inv = ert_scaler.inverse_transform(ert_sim_test.reshape(ert_sim_test.shape[0], -1))
ert_sim_inv = ert_sim_test_inv.reshape(ert_sim_test.shape)

print (ert_true_param_inv.shape)
print (ert_sim_inv.shape)


plt.figure(figsize=(4, 4),dpi=250)
plt.imshow(ert_sim_inv[0,:,:], aspect='auto',origin='lower',cmap='jet')
cbar = plt.colorbar()
cbar.set_label('Transfer resistivity [Ω]')
plt.xlabel('ERT surveys')
plt.ylabel('ERT measurements')
plt.title('True ERT data')
plt.show()


print (params_realizations.shape)
print (ert_true_param_inv.shape)
print (ert_sim_inv.shape)


#%% Perform simulations for the generated parameters
import os
from datetime import datetime
import pickle
import time
import signal
from pathlib import Path
import json
from collections import Counter

class TimeoutException(Exception):
    pass

def load_simulation_data(base_folder,number_of_simulations):
    base_path = Path(base_folder)
    all_data = []
    all_parameters = []
    missing_folders = []
    discarded_indices = []
    shapes_info = []  # To store shapes of discarded simulations
    shapes_data = []
    shapes_params = []
    for i in range(number_of_simulations):
        sim_folder = base_path / f"simulation_{i:04d}"
        try:
            data = np.load(sim_folder / "data.npy")
            params = np.load(sim_folder / "parameters.npy")
            shapes_data.append(data.shape)
            shapes_params.append(params.shape)
        except FileNotFoundError:
            missing_folders.append(i)
            continue
    
    most_common_data_shape = Counter(shapes_data).most_common(1)[0][0]
    most_common_param_shape = Counter(shapes_params).most_common(1)[0][0]
    
    print(f"Most common data shape: {most_common_data_shape}")
    print(f"Most common parameter shape: {most_common_param_shape}")
    
    valid_simulations = 0
    
    for i in range(number_of_simulations):
        if i in missing_folders:
            continue
        sim_folder = base_path / f"simulation_{i:04d}"
        try:
            data = np.load(sim_folder / "data.npy")
            params = np.load(sim_folder / "parameters.npy")
            if data.shape == most_common_data_shape and params.shape == most_common_param_shape:
                all_data.append(data)
                all_parameters.append(params)
                valid_simulations += 1
            else:
                discarded_indices.append(i)
                shapes_info.append((i, data.shape, params.shape))
                
        except FileNotFoundError:
            missing_folders.append(i)
            continue
    all_data = np.array(all_data)
    all_parameters = np.array(all_parameters)
    
    print(f"\nProcessing complete:")
    print(f"Valid simulations: {valid_simulations}")
    print(f"Discarded simulations: {len(discarded_indices)}")
    print(f"Missing folders: {len(missing_folders)}")
    print(f"\nFinal data shape: {all_data.shape}")
    print(f"Final parameters shape: {all_parameters.shape}")
    
    print("\nDiscarded simulations (index, data shape, params shape):")
    for info in shapes_info:
        print(f"simulation_{info[0]:04d}: data{info[1]}, params{info[2]}")
    
    print("\nMissing folders:")
    for idx in missing_folders:
        print(f"simulation_{idx:04d}")
    
    return all_data, all_parameters

def timeout_handler(signum, frame):
    raise TimeoutException("Simulation timed out")

class SimulationManager:
    def __init__(self, max_simulation_time=3600):  # default 1 hour timeout
        self.max_simulation_time = max_simulation_time
        self.base_output_dir = self._create_output_directory()
        self.failed_simulations = []
        
    def _create_output_directory(self):
        """Create a new directory for this batch of simulations"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = Path(f'simulation_results_{timestamp}')
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir
    
    def save_simulation_result(self, sim_idx, params, data):
        """Save individual simulation results"""
        sim_dir = self.base_output_dir / f'simulation_{sim_idx:04d}'
        sim_dir.mkdir(exist_ok=True)
        
        # Save parameters
        np.save(sim_dir / 'parameters.npy', params)
        # Save data
        np.save(sim_dir / 'data.npy', data)
        # Save metadata
        metadata = {
            'simulation_index': sim_idx,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'status': 'completed'
        }
        with open(sim_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
    
    def save_failed_simulation(self, sim_idx, params, error_msg):
        """Record failed simulation details"""
        self.failed_simulations.append({
            'simulation_index': sim_idx,
            'parameters': params.tolist(),
            'error': error_msg,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        })
        
        # Save failed simulations log
        with open(self.base_output_dir / 'failed_simulations.json', 'w') as f:
            json.dump(self.failed_simulations, f, indent=2)



pflotran_sim = ert_utils.PflotranSimulator('1002023c.in', 'conditional_diffusion_constrained')
ert_handler = ert_utils.ERTDataHandler()
forward_runner = ert_utils.ForwardModelRunner(
    pflotran_sim, 
    None,
    ert_handler,
    pflotran_path="/Users/hern856/pflotran/src/pflotran/pflotran"
)

sim_manager = SimulationManager(max_simulation_time=3600)
signal.signal(signal.SIGALRM, timeout_handler)
start_time = time.time()

#%% Test the simulation manager
conditional_idx = 0
#conditional_idx = 5

conditional_param_sample = ert_true_param_inv[conditional_idx, :].reshape(1, -1).squeeze()
print (conditional_param_sample.shape)

conditional_ert_sample = ert_sim_inv[conditional_idx, :, :]   
print (conditional_ert_sample.shape)

plt.figure(figsize=(4, 4),dpi=250)
plt.imshow(conditional_ert_sample, aspect='auto',origin='lower',cmap='jet')
cbar = plt.colorbar()
cbar.set_label('Transfer resistance [Ω]') 
plt.xlabel('ERT surveys')
plt.ylabel('ERT measurements')
plt.title('Conditional ERT data')
plt.show()


n_samples = 2
param_names = ert_utils.ParameterNames()
names = param_names.names
#sample_idx = np.random.choice(params_realizations.shape[1], n_samples, replace=False)
sample_idx = np.array([0,5])
for param in range(29):
    plt.figure(figsize=(6, 4),dpi=250)
    for i in range(len(sample_idx)):
        param_dist = params_realizations[:, sample_idx[i] , param]
        param_true = ert_true_param_inv[sample_idx[i], param]
        mode_param = np.mean(param_dist)
        confidence_interval = np.percentile(param_dist, [2.5, 97.5])
        true_val_in_interval = confidence_interval[0] <= param_true <= confidence_interval[1]
        #print(f"Parameter {param}: True={param_true:.4f}, Mode={mode_param:.4f}, 95% CI={confidence_interval}, True in CI={true_val_in_interval}")
        if sample_idx[i] == 0:
            case = 1
        else:
            case = 2
        plt.hist(param_dist, bins=22, density=True, alpha=0.3, color=f'C{i}', label=f'ERT Case: {case}')
        sns.kdeplot(data=param_dist,color=f'C{i}', fill=False, alpha=0.7)
        #sns.histplot(data=param_dist, bins=22, kde=True, stat='density', color=f'C{i}',edgecolor='none', alpha=0.2, label=f'ERT Condition: {sample_idx[i]}')
        plt.axvline(param_true,linewidth = 1.2, linestyle='--',color=f'C{i}', label=f'True parameter')
        #plt.axvline(param_mode,linewidth = 1, linestyle='--', color='purple', label=f'Mode')
        #plt.axvline(limits[param][0], color=f'C{sample_idx}', linestyle='--', label='Parameter bounds', alpha=0.5)
        #plt.axvline(limits[param][1], color=f'C{sample_idx}', linestyle='--', alpha=0.5)
        #plt.axvline(confidence_interval[0], linestyle='--',color = 'green', label='95 CI', alpha=0.5)
        #plt.axvline(confidence_interval[1], linestyle='--',color = 'green', alpha=0.5)
        plt.xlabel(f"{names[param]}")
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=n_samples)
        #plt.savefig(f"{results_dir}/examples_{conditional_idx}_conditional_param_{param}.png", bbox_inches='tight')
        #plt.show()


#%% Run simulations for the generated parameters

results_dir = f'/Users/hern856/Codes/Pflotran_Sim/Transformer Surrogate/Inversion_results/Conditional_Sample_{conditional_idx}'
results_dir = Path(results_dir)
if not results_dir.exists():
    results_dir.mkdir(parents=True)
# Load the simulation data

pflotran_sim = ert_utils.PflotranSimulator('1002023c.in', 'conditional_diffusion_constrained')
ert_handler = ert_utils.ERTDataHandler()
forward_runner = ert_utils.ForwardModelRunner(
    pflotran_sim, 
    None,
    ert_handler,
    pflotran_path="/Users/hern856/pflotran/src/pflotran/pflotran"
)

sim_manager = SimulationManager(max_simulation_time=3600)
completed_sims = 0

cond_paramts = params_realizations[:,conditional_idx,:]

for (i, params) in enumerate(cond_paramts):
 
    print(f"\nStarting simulation {i+1}/{uncertainty_samples}")
    simulation_start = time.time()
    
    try:
        # Set timeout for this simulation
        signal.alarm(sim_manager.max_simulation_time)
        
        # Run simulation
        sim_data = forward_runner.run_simulations_with_params_single(params, i)
        sim_data = np.vstack(sim_data)
        
        # Clear timeout
        signal.alarm(0)
        
        # Save results
        sim_manager.save_simulation_result(i, params, sim_data)
        completed_sims += 1
        
        # Calculate and display progress
        elapsed_time = time.time() - start_time
        avg_time_per_sim = elapsed_time / (i + 1)
        estimated_remaining = avg_time_per_sim * (uncertainty_samples - (i + 1))
        
        print(f"Simulation {i+1} completed successfully")
        print(f"Time taken: {time.time() - simulation_start:.2f} seconds")
        print(f"Estimated time remaining: {estimated_remaining/3600:.2f} hours")
        print(f"Progress: {completed_sims}/{uncertainty_samples} ({completed_sims/uncertainty_samples*100:.1f}%)")
        
    except TimeoutException:
        print(f"Simulation {i+1} timed out after {sim_manager.max_simulation_time} seconds")
        sim_manager.save_failed_simulation(i, params, "Timeout")
        continue
        
    except Exception as e:
        print(f"Error in simulation {i+1}: {str(e)}")
        sim_manager.save_failed_simulation(i, params, str(e))
        continue

#%% Load and visualize the simulation results

folder_path = "/Users/hern856/Codes/Pflotran_Sim/Transformer Surrogate/simulation_results_20250731_110916_Conditional_idx_0"
sim_ert, sim_param = load_simulation_data(folder_path,uncertainty_samples)

print("Data shape:", sim_ert.shape)
print("Parameters shape:", sim_param.shape)


ert_surveys = 14
etr_measurements = int(sim_ert.shape[1]/ert_surveys)
total_sim = sim_ert.shape[0]
print ('Total number of simulations:', total_sim)
print ('Number of ERT surveys:', ert_surveys)
print ('Number of ERT measurements:', etr_measurements)

sim_data_param = np.zeros((total_sim,etr_measurements,ert_surveys))
sim_param_data = np.zeros((total_sim, sim_param.shape[1], 1))


sim_data = np.zeros((total_sim,etr_measurements,ert_surveys))
param_data = np.zeros((total_sim, sim_param.shape[1], 1))

for sim_idx in range(total_sim):
    sim_ert_instance = sim_ert[sim_idx,:].squeeze()
    new_arrange = []
    for time_step in range(ert_surveys):
        time_start = time_step * etr_measurements
        time_end = time_start + etr_measurements  
        measurement_data = sim_ert_instance[time_start:time_end]  
        new_arrange.append(measurement_data)
    sim_out = np.array(new_arrange).T
    sim_inp = sim_param[sim_idx,:].reshape(-1, 1)

    sim_data[sim_idx,:,:] = sim_out
    param_data[sim_idx,:] = sim_inp

print ('Simulated ERT data shape:', sim_data.shape)
print ('Simulated parameter data shape:', param_data.shape)

for sim_idx in range(total_sim):
    plt.figure(figsize=(6, 5), dpi=250)
    plt.imshow(sim_data[sim_idx,:,:], aspect='auto',origin='lower',cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('Transfer resistance [Ω]') 
    plt.xlabel('ERT surveys')
    plt.ylabel('ERT measurements')
    #plt.title('Simulated ERT data')
    plt.show()


ensemble_mode = np.zeros((4693, 14))
# Determine the global min and max values
data_min = np.min(sim_data)
data_max = np.max(sim_data)
x_range = np.linspace(data_min, data_max, 5000) 

for i in range(sim_data.shape[1]):
    for j in range(sim_data.shape[2]):
        if (i * sim_data.shape[2] + j) % 100 == 0:
            print(f"Processing point ({i}, {j})")
        point = sim_data[:, i, j]
        kde = stats.gaussian_kde(point)
        kde_values = kde(x_range)
        max_index = np.argmax(kde_values)
        point_mode = x_range[max_index]
        ensemble_mode[i, j] = point_mode



#%% SWWE per realization
def WSSE_metric(A,B,predictions, observations):
    sd = A*np.abs(observations)+B
    WSE = (predictions - observations)**2/(sd)**2

    WSSE = np.average(WSE)
    print (f'WSSE: {WSSE}')
    return WSSE, WSE

A_sd = 0.1
B_sd = 0.01
WSSE_sim = []
for sim in range(sim_data.shape[0]):
    WSSE_time = []
    for es in range(ert_surveys):
        WSSE, _ = WSSE_metric(A_sd, B_sd, sim_data[sim][:, es], conditional_ert_sample[:, es])
        WSSE_time.append(WSSE)
    WSSE_sim.append(WSSE_time)
WSSE_sim = np.array(WSSE_sim)  # Shape: (total_sim, ert_surveys)
print (WSSE_sim.shape)

wsse_total = WSSE_sim.sum(axis=1)
sorted_indices = np.argsort(wsse_total)
best_n = 3  # Number of best simulations to highlight

plt.figure(figsize=(8, 5.5), dpi=250)
for sim in range(WSSE_sim.shape[0]):
    plt.plot(WSSE_sim[sim, :], color='gray', alpha=0.7, linewidth=1)
colors = cm.viridis(np.linspace(0, 1, best_n))
for i, idx in enumerate(sorted_indices[:best_n]):
    plt.plot(
        WSSE_sim[idx, :],
        color=colors[i],
        linewidth=1.5,
        #label=f'Sim {idx}: WSSE={wsse_total[idx]:.2f}'
        label=f'Sim {idx}'
    )
plt.xlabel('ERT Survey')
plt.yscale('log')
plt.ylabel('WSSE')
#plt.title('WSSE per Simulation')
plt.xticks(ticks=np.arange(14), labels=np.arange(1, 15))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=best_n, frameon=True)
plt.tight_layout()
#plt.grid(True, which='major', linestyle='--', linewidth=0.5)
plt.show()


universal_fontsize = 12
plt.rcParams.update({'font.size': universal_fontsize})

plt.figure(figsize=(6.5, 6), dpi=250)
im1 = plt.imshow(conditional_ert_sample, aspect='auto', origin='lower', cmap='jet')
plt.title("Conditional ERT", fontsize=universal_fontsize)
plt.xlabel("ERT Surveys", fontsize=universal_fontsize)
plt.ylabel("ERT Measurements", fontsize=universal_fontsize)
plt.tick_params(axis='both', which='major', labelsize=universal_fontsize)
plt.xticks(ticks=np.arange(14), labels=np.arange(1, 15))
cbar1 = plt.colorbar(im1, shrink=1.0, aspect=20)
cbar1.set_label('Transfer resistance [Ω]', fontsize=universal_fontsize)
cbar1.ax.tick_params(labelsize=universal_fontsize)


fig, axs = plt.subplots(nrows=best_n, ncols=3, figsize=(18, 16), dpi=250)
for row, idx in enumerate(sorted_indices[:best_n]):
    # ERT image
    im = axs[row, 0].imshow(sim_data[idx, :, :], aspect='auto', origin='lower', cmap='jet')
    #axs[row, 0].set_title(f'Sim {idx}: WSSE={wsse_total[idx]:.2f}', fontsize=universal_fontsize)
    axs[row, 0].set_title(f'Sim {idx}', fontsize=universal_fontsize)
    axs[row, 0].set_xlabel("ERT Surveys", fontsize=universal_fontsize)
    axs[row, 0].set_ylabel("ERT Measurements", fontsize=universal_fontsize)
    axs[row, 0].tick_params(axis='both', which='major', labelsize=universal_fontsize)
    axs[row, 0].set_xticks(np.arange(14))
    axs[row, 0].set_xticklabels(np.arange(1, 15))
    cbar = fig.colorbar(im, ax=axs[row, 0], fraction=0.046, pad=0.04)
    cbar.set_label('Transfer resistance [Ω]', fontsize=universal_fontsize)
    cbar.ax.tick_params(labelsize=universal_fontsize)

    # Scatter plot
    min_val = min(np.min(sim_data[idx, :, :]), np.min(conditional_ert_sample))
    max_val = max(np.max(sim_data[idx, :, :]), np.max(conditional_ert_sample))
    axs[row, 1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1, label='Identity Line')
    axs[row, 1].scatter(sim_data[idx, :, :].flatten(), conditional_ert_sample.flatten(), color='black', s=5)
    axs[row, 1].set_xlabel('Inverted ERT [Ω]', fontsize=universal_fontsize)
    axs[row, 1].set_ylabel('Conditional ERT [Ω]', fontsize=universal_fontsize)
    axs[row, 1].legend(fontsize=universal_fontsize)
    axs[row, 1].set_aspect('equal', adjustable='box')
    axs[row, 1].grid(True, which='both', linestyle='--', linewidth=0.3)
    # Distribution comparison (KDE)
    sns.kdeplot(sim_data[idx, :, :].flatten(), color='C1', label='Inverted ERT', ax=axs[row, 2])
    sns.kdeplot(conditional_ert_sample.flatten(), color='C0', label='Conditional ERT', ax=axs[row, 2])
    axs[row, 2].set_xlabel('Transfer resistance [Ω]', fontsize=universal_fontsize)
    axs[row, 2].set_ylabel('Density', fontsize=universal_fontsize)
    axs[row, 2].legend(fontsize=universal_fontsize)
    w_distance = wasserstein_distance(sim_data[idx, :, :].flatten(), conditional_ert_sample.flatten())
    axs[row, 2].set_title(f'Wasserstein Distance: {w_distance:.4f}', fontsize=universal_fontsize)
plt.tight_layout()
plt.show()

#%%

ensemble_mean = np.mean(sim_data, axis=0)  # Shape: (H, W)
ensemble_std = np.std(sim_data, axis=0)      # Shape: (H, W)
ensemble_variance = np.var(sim_data, axis=0)  # Shape: (H, W)
P25 = np.percentile(sim_data, 25, axis=0)  # Shape: (H, W)
P50 = np.percentile(sim_data, 50, axis=0)  # Shape: (H, W)
P75 = np.percentile(sim_data, 75, axis=0)  # Shape: (H, W)

ensemble_mean_abs = np.abs(ensemble_mean)
coefficient_of_variation = ensemble_std / (ensemble_mean_abs + 1e-8)  # Avoid division by zero
standard_difference = ensemble_std / (ensemble_mean_abs + 1e-8)  # Relative standard deviation

diff_map_mean = conditional_ert_sample - ensemble_mean 
diff_map_mode = conditional_ert_sample - ensemble_mode   # Difference map


percentage_error_mean = (np.abs(ensemble_mean - conditional_ert_sample) / np.abs(conditional_ert_sample)) * 100
percentage_error_mode = (np.abs(ensemble_mode - conditional_ert_sample) / np.abs(conditional_ert_sample)) * 100
avg_percentage_error_mean = np.mean(percentage_error_mean)
avg_percentage_error_mode = np.mean(percentage_error_mode)


plt.figure(figsize=(6, 4), dpi=250)
sns.kdeplot(ensemble_mean.flatten(), label='Ensemble Mean')
sns.kdeplot(conditional_ert_sample.flatten(), label='Conditional ERT')
sns.kdeplot(ensemble_mode.flatten(), label='Ensemble Mode')
plt.xlabel('Transfer resistance [Ω]')
plt.ylabel('Density')
plt.legend()
plt.title('Distribution of MSE between Ensemble and Conditional ERT')
#plt.savefig(f"{results_dir}/ert_cond_{conditional_idx}_distributions.png", bbox_inches='tight')
plt.show()

w_distance_mean = wasserstein_distance(ensemble_mean.flatten(), conditional_ert_sample.flatten())
w_distance_mode = wasserstein_distance(ensemble_mode.flatten(), conditional_ert_sample.flatten())

print(f"Wasserstein Distance (Ensemble Mean): {w_distance_mean:.4f}")
print(f"Wasserstein Distance (Ensemble Mode): {w_distance_mode:.4f}")

plt.figure(figsize=(6, 4), dpi=250)
min_val = np.min([ensemble_mean.min(), conditional_ert_sample.min()])
max_val = np.max([ensemble_mean.max(), conditional_ert_sample.max()])
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)
plt.scatter(ensemble_mean.flatten(), conditional_ert_sample.flatten(), color='black', s=10)
plt.xlabel('Ensemble Mean')
plt.ylabel('Conditional ERT')
plt.title('Ensemble Mean vs. Conditional ERT')  
#plt.savefig(f"{results_dir}/ert_cond_{conditional_idx}_Scatter_Mean.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 4), dpi=250)
min_val = np.min([ensemble_mode.min(), conditional_ert_sample.min()])
max_val = np.max([ensemble_mode.max(), conditional_ert_sample.max()])
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)
plt.scatter(ensemble_mode.flatten(), conditional_ert_sample.flatten(), color='black', s=10)
plt.xlabel('Ensemble Mode')
plt.ylabel('Conditional ERT')
plt.title('Ensemble Modes vs. Conditional ERT')
#plt.savefig(f"{results_dir}/ert_cond_{conditional_idx}_Scatter_Mode.png", bbox_inches='tight')
plt.show()

v_min = np.min([ensemble_mean.min(), conditional_ert_sample.min()])
v_max = np.max([ensemble_mean.max(), conditional_ert_sample.max()])

mse_distribution = []
for sim_idx in range(sim_data.shape[0]):
    mse = mean_squared_error(conditional_ert_sample.flatten(), sim_data[sim_idx].flatten())
    mse_distribution.append(mse)
    
plt.figure(figsize=(6, 4), dpi=250)
sns.kdeplot(mse_distribution)
plt.axvline(0, linewidth=1, linestyle='--', color='black')
plt.xlabel('MSE')
plt.ylabel('Density')
plt.title('Distribution of MSE between ERT Simulations and Conditional ERT')
#plt.savefig(f"{results_dir}/ert_cond_{conditional_idx}_MSE_distribution.png", bbox_inches='tight')
plt.show()

mse_mode = mean_squared_error(conditional_ert_sample.flatten(), ensemble_mode.flatten())
mse_mean = mean_squared_error(conditional_ert_sample.flatten(), ensemble_mean.flatten())

rmse_mode = np.sqrt(mse_mode)
rmse_mean = np.sqrt(mse_mean)
print ('RMSE Mean:', rmse_mean)
print ('MSE Mean:', mse_mean)

print ('RMSE Mode:', rmse_mode)
print ('MSE Mode:', mse_mode)


font_size = 12
plt.rcParams.update({'font.size': font_size})
fig, axs = plt.subplots(3, 3, figsize=(24, 21), dpi=250)
v_min = min(ensemble_mean.min(), conditional_ert_sample.min(), ensemble_mode.min())
v_max = max(ensemble_mean.max(), conditional_ert_sample.max(), ensemble_mode.max())
im1 = axs[0, 0].imshow(conditional_ert_sample, aspect='auto', origin='lower', cmap='jet', vmin=v_min, vmax=v_max)
axs[0, 0].set_title("Conditional ERT", fontsize=16)
axs[0, 0].set_xlabel("ERT Surveys", fontsize=16)
axs[0, 0].set_ylabel("ERT Measurements", fontsize=16)
axs[0, 0].tick_params(axis='both', which='major', labelsize=16)
cbar1 = plt.colorbar(im1, ax=axs[0, 0], shrink=1.0, aspect=20)
cbar1.set_label('Transfer resistance [Ω]', fontsize=16)
cbar1.ax.tick_params(labelsize=16)
im2 = axs[0, 1].imshow(ensemble_mean, aspect='auto', origin='lower', cmap='jet', vmin=v_min, vmax=v_max)
axs[0, 1].set_title("Ensemble Mean", fontsize=16)
axs[0, 1].set_xlabel("ERT Surveys", fontsize=16)
axs[0, 1].set_ylabel("ERT Measurements", fontsize=16)
axs[0, 1].tick_params(axis='both', which='major', labelsize=16)
cbar2 = plt.colorbar(im2, ax=axs[0, 1], shrink=1.0, aspect=20)
cbar2.set_label('Transfer resistance [Ω]', fontsize=16)
cbar2.ax.tick_params(labelsize=16)
im3 = axs[0, 2].imshow(ensemble_mode, aspect='auto', origin='lower', cmap='jet', vmin=v_min, vmax=v_max)
axs[0, 2].set_title("Ensemble Mode", fontsize=16)
axs[0, 2].set_xlabel("ERT Surveys", fontsize=16)
axs[0, 2].set_ylabel("ERT Measurements", fontsize=16)
axs[0, 2].tick_params(axis='both', which='major', labelsize=16)
cbar3 = plt.colorbar(im3, ax=axs[0, 2], shrink=1.0, aspect=20)
cbar3.set_label('Transfer resistance [Ω]', fontsize=16)
cbar3.ax.tick_params(labelsize=16)
sns.kdeplot(diff_map_mode.flatten(), color='blue', label='Ensemble Mode', ax=axs[1, 0])
sns.kdeplot(diff_map_mean.flatten(), color='red', label='Ensemble Mean', ax=axs[1, 0])
axs[1, 0].set_ylabel('Density', fontsize=16)
axs[1, 0].set_xlabel('Difference (Conditional - Ensemble Mean)', fontsize=16)
axs[1, 0].set_title('Difference Distribution', fontsize=16)
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].tick_params(axis='both', which='major', labelsize=16)
axs[1, 0].legend(fontsize=16)
vmax_mean = np.max(np.abs(diff_map_mean))
im4 = axs[1, 1].imshow(diff_map_mean, aspect='auto', origin='lower', cmap='seismic', vmin=-vmax_mean, vmax=vmax_mean)
axs[1, 1].set_title("Difference (Mean)", fontsize=16)
axs[1, 1].set_xlabel("ERT Surveys", fontsize=16)
axs[1, 1].set_ylabel("ERT Measurements", fontsize=16)
axs[1, 1].tick_params(axis='both', which='major', labelsize=16)
cbar4 = plt.colorbar(im4, ax=axs[1, 1], shrink=1.0, aspect=20)
cbar4.set_label('Difference (Conditional - Ensemble Mean)', fontsize=16)
cbar4.ax.tick_params(labelsize=16)
vmax_mode = np.max(np.abs(diff_map_mode))
im5 = axs[1, 2].imshow(diff_map_mode, aspect='auto', origin='lower', cmap='seismic', vmin=-vmax_mode, vmax=vmax_mode)
axs[1, 2].set_title("Difference (Mode)", fontsize=16)
axs[1, 2].set_xlabel("ERT Surveys", fontsize=16)
axs[1, 2].set_ylabel("ERT Measurements", fontsize=16)
axs[1, 2].tick_params(axis='both', which='major', labelsize=16)
cbar5 = plt.colorbar(im5, ax=axs[1, 2], shrink=1.0, aspect=20)
cbar5.set_label('Difference (Conditional - Ensemble Mode)', fontsize=16)
cbar5.ax.tick_params(labelsize=16)
v_min_quantiles = min(P25.min(), P50.min(), P75.min())
v_max_quantiles = max(P25.max(), P50.max(), P75.max())
im6 = axs[2, 0].imshow(P25, aspect='auto', origin='lower', cmap='jet', vmin=v_min_quantiles, vmax=v_max_quantiles)
axs[2, 0].set_title('25th Percentile (P25)', fontsize=16)
axs[2, 0].set_xlabel('ERT Surveys', fontsize=16)
axs[2, 0].set_ylabel('ERT Measurements', fontsize=16)
axs[2, 0].tick_params(axis='both', which='major', labelsize=16)
cbar6 = plt.colorbar(im6, ax=axs[2, 0], shrink=1.0, aspect=20)
cbar6.set_label('Transfer resistance [Ω]', fontsize=16)
cbar6.ax.tick_params(labelsize=16)
im7 = axs[2, 1].imshow(P50, aspect='auto', origin='lower', cmap='jet', vmin=v_min_quantiles, vmax=v_max_quantiles)
axs[2, 1].set_title('50th Percentile (P50 - Median)', fontsize=16)
axs[2, 1].set_xlabel('ERT Surveys', fontsize=16)
axs[2, 1].set_ylabel('ERT Measurements', fontsize=16)
axs[2, 1].tick_params(axis='both', which='major', labelsize=14)
cbar7 = plt.colorbar(im7, ax=axs[2, 1], shrink=1.0, aspect=20)
cbar7.set_label('Transfer resistance [Ω]', fontsize=14)
cbar7.ax.tick_params(labelsize=16)
im8 = axs[2, 2].imshow(P75, aspect='auto', origin='lower', cmap='jet', vmin=v_min_quantiles, vmax=v_max_quantiles)
axs[2, 2].set_title('75th Percentile (P75)', fontsize=16)
axs[2, 2].set_xlabel('ERT Surveys', fontsize=16)
axs[2, 2].set_ylabel('ERT Measurements', fontsize=16)
axs[2, 2].tick_params(axis='both', which='major', labelsize=16)
cbar8 = plt.colorbar(im8, ax=axs[2, 2], shrink=1.0, aspect=20)
cbar8.set_label('Transfer resistance [Ω]', fontsize=16)
cbar8.ax.tick_params(labelsize=16)
plt.tight_layout()
#plt.savefig(f"{results_dir}/ert_cond_{conditional_idx}_reorganized_results.png", bbox_inches='tight', dpi=250)
plt.show()

#%% Uncertainty evaluation
model.eval()
uncertainty_samples = 50  
pred_params_batches = []  
true_params_batches = []

for batch_idx, (x0_test, cond_test) in enumerate(test_loader):
    cond_test = cond_test.to(device)
    
    # Store true parameters
    ert_true_param_inv = inverse_transform(x0_test.cpu().numpy(), a, b)
    ert_true_param_inv = param_scaler.inverse_transform(ert_true_param_inv)
    true_params_batches.append(ert_true_param_inv)
    
    # Generate multiple samples for this batch
    batch_samples = []
    for sample_idx in range(uncertainty_samples):
        pred_params_unconstrained = sample_model(model, cond_test, T, betas, alphas, alpha_bar, param_dim, device)
        # Convert to original parameter space
        pred_params = inverse_transform(pred_params_unconstrained, a, b)
        pred_params_np = pred_params.cpu().numpy()
        pred_params_np = param_scaler.inverse_transform(pred_params_np)
        
        # Check bounds and store valid parameters
        valid_pred_params = check_param_bounds(pred_params_np, limits)
        if valid_pred_params is not None:
            batch_samples.append(valid_pred_params)
        else:
            print(f"Batch {batch_idx}, Sample {sample_idx}: No valid parameters")
    
    if batch_samples:
        # Stack all valid samples for this batch
        batch_samples = np.stack(batch_samples)  # Shape: (n_valid_samples, batch_size, param_dim)
        pred_params_batches.append(batch_samples)

# Concatenate all batches
if pred_params_batches:
    pred_params_all = np.concatenate(pred_params_batches, axis=1)  # Shape: (n_samples, total_test_size, param_dim)
    true_params_all = np.concatenate(true_params_batches, axis=0)  # Shape: (total_test_size, param_dim)
    
    print("Generated parameters shape:", pred_params_all.shape)
    print("True parameters shape:", true_params_all.shape)
else:
    print("No valid parameter sets generated")

#np.save("Uncertainty_params.npy", pred_params_all)
#np.save("true_params.npy", true_params_all)

generated_params_np = np.load("Uncertainty_params.npy")
true_params_np = np.load("true_params.npy")

print (generated_params_np.shape)
print (true_params_np.shape)


def avg_prop_indicator_function(avg_proportion, prob_array):
    ind_avg_proportion = []
    for i in range(prob_array.shape[0]):
        if avg_proportion[i] >= prob_array[i]:
            ind_avg_proportion.append(1)
        else:
            ind_avg_proportion.append(0)
    return np.array(ind_avg_proportion)

def accuracy_score(a_p,prob_array):
    Accuracy_int = np.trapz(a_p,prob_array,dx=0.0001)
    return Accuracy_int

def preccision_score(Accuracy_unc,avg_proportion,prob_array,a_p):
    if Accuracy_unc == 0:
        return 0.0
    else: 
        precc_array = a_p*(avg_proportion-prob_array)
        Preccision_int = np.trapz(precc_array,prob_array,dx=0.01)
        Presicion_unc = 1-2*Preccision_int #By definition precision is defined when we have accuracy
    return Presicion_unc

def goodness_score(a_p,avg_proportion,prob_array):
    good_int = (3*a_p - 2) * (avg_proportion - prob_array)
    Goodness_int = np.trapz(good_int, prob_array,dx=0.001)
    Goodness_metric = 1 - Goodness_int
    return Goodness_metric


param_dist = generated_params_np[:, : , :]
param_true = true_params_np[:, :]

prob_array = np.linspace(0.01, 0.99, 30)
avg_proportion = np.zeros(len(prob_array))
for prob in enumerate(prob_array):
    p_low=(1-prob[1])/2
    p_upp=(1+prob[1])/2
    low_bound = np.percentile(generated_params_np, p_low*100,axis=0)
    upp_bound = np.percentile(generated_params_np, p_upp*100,axis=0)

    true_data_obs = true_params_np
    indicator_matrix = (low_bound < true_data_obs) & (true_data_obs <= upp_bound)
    indicator_matrix = indicator_matrix.astype(int)
    avg_proportion[prob[0]] = np.mean(indicator_matrix)

a_p = avg_prop_indicator_function(avg_proportion, prob_array)
Accuracy_unc = accuracy_score(a_p,prob_array)
Precision_unc = preccision_score(Accuracy_unc,avg_proportion,prob_array,a_p)
Goodness_metric = goodness_score(a_p,avg_proportion,prob_array)

plt.figure(figsize=(6, 4), dpi=250)
plt.plot(prob_array, avg_proportion, color='black', linewidth=1)
plt.scatter(prob_array, avg_proportion, color='black', s=10)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1)
plt.xlabel('Probability interval, p')
plt.ylabel(r'$\overline{\xi(p)}$')
plt.title(f'Conditional Diffusion Model, Goodness Metric: {Goodness_metric:.2f}')
# Add textbox with metrics
textstr = '\n'.join((
    f'Accuracy: {Accuracy_unc:.2f}',
    f'Precision: {Precision_unc:.2f}',
    f'Goodness: {Goodness_metric:.2f}'))

props_metrics = dict(facecolor='none',edgecolor='none')
props_desc = dict(facecolor='none', edgecolor='none')

plt.annotate('', xy=(0.25, 0.75), xytext=(0.5, 0.50),  # Shortened upper left arrow
            xycoords='axes fraction', 
            arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.2, 
                        width=0.5, headwidth=3, headlength=4))

plt.annotate('', xy=(0.75, 0.25), xytext=(0.5,0.50),  # Shortened lower right arrow
            xycoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.2,
                        width=0.5, headwidth=3, headlength=4))

# Description boxes
plt.text(0.02, 0.95, "Accurate but not precise", transform=plt.gca().transAxes, 
        fontsize=8, bbox=props_desc)
plt.text(0.98, 0.03, "Not accurate not precise", transform=plt.gca().transAxes,
        fontsize=8, bbox=props_desc, 
        horizontalalignment='right')
plt.text(0.5, 0.32, "Accurate and precise", transform=plt.gca().transAxes,
    fontsize=8, bbox=props_desc,
    horizontalalignment='center', rotation=36)

metrics_box = f'Accuracy: {Accuracy_unc:.2f}   Precision: {Precision_unc:.2f}   Goodness: {Goodness_metric:.2f}'
plt.annotate(metrics_box, 
            xy=(0.5, -0.2), 
            xycoords='axes fraction',
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
            ha='center',
            va='center',
            fontsize=10)
#plt.savefig(f"General_accuracy_plot.png", bbox_inches='tight')
plt.show()


#%% Uncertainty evaluation for each parameter
param_accuracy = []
param_precision = []
param_goodness = []

for param_idx in range(29): 
    param_dist = generated_params_np[:, : , param_idx]
    param_true = true_params_np[:, param_idx]

    avg_proportion = np.zeros(len(prob_array))
    for prob in enumerate(prob_array):
        p_low=(1-prob[1])/2
        p_upp=(1+prob[1])/2
        low_bound = np.percentile(param_dist, p_low*100,axis=0)
        upp_bound = np.percentile(param_dist, p_upp*100,axis=0)

        true_data_obs = param_true
        indicator_matrix = (low_bound < true_data_obs) & (true_data_obs <= upp_bound)
        indicator_matrix = indicator_matrix.astype(int)
        avg_proportion[prob[0]] = np.mean(indicator_matrix)

    a_p = avg_prop_indicator_function(avg_proportion, prob_array)
    Accuracy_unc = accuracy_score(a_p,prob_array)
    Precision_unc = preccision_score(Accuracy_unc,avg_proportion,prob_array,a_p)
    Goodness_metric = goodness_score(a_p,avg_proportion,prob_array)

    param_accuracy.append(Accuracy_unc)
    param_precision.append(Precision_unc)
    param_goodness.append(Goodness_metric)
    print(f"Parameter {names[param_idx]}: Accuracy={Accuracy_unc:.4f}, Precision={Precision_unc:.4f}, Goodness={Goodness_metric:.4f}")
    
    plt.figure(figsize=(6, 4), dpi=250)
    plt.plot(prob_array, avg_proportion, color='black', linewidth=1)
    plt.scatter(prob_array, avg_proportion, color='black', s=10)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1)
    plt.xlabel('Probability interval, p')
    plt.ylabel(r'$\overline{\xi(p)}$')
    plt.title(f'{names[param_idx]}')
    
    # Add textbox with metrics
    textstr = '\n'.join((
        f'Accuracy: {Accuracy_unc:.2f}',
        f'Precision: {Precision_unc:.2f}',
        f'Goodness: {Goodness_metric:.2f}'))
    
    props_metrics = dict(facecolor='none',edgecolor='none')
    props_desc = dict(facecolor='none', edgecolor='none')

    plt.annotate('', xy=(0.25, 0.75), xytext=(0.5, 0.50),  # Shortened upper left arrow
                xycoords='axes fraction', 
                arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.2, 
                            width=0.5, headwidth=3, headlength=4))

    plt.annotate('', xy=(0.75, 0.25), xytext=(0.5,0.50),  # Shortened lower right arrow
                xycoords='axes fraction',
                arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.2,
                            width=0.5, headwidth=3, headlength=4))
    
    # Description boxes
    plt.text(0.02, 0.95, "Accurate but not precise", transform=plt.gca().transAxes, 
            fontsize=8, bbox=props_desc)
    plt.text(0.98, 0.03, "Not accurate not precise", transform=plt.gca().transAxes,
            fontsize=8, bbox=props_desc, 
            horizontalalignment='right')
    plt.text(0.5, 0.32, "Accurate and precise", transform=plt.gca().transAxes,
        fontsize=8, bbox=props_desc,
       horizontalalignment='center', rotation=36)
    
    metrics_box = f'Accuracy: {Accuracy_unc:.2f}   Precision: {Precision_unc:.2f}   Goodness: {Goodness_metric:.2f}'
    plt.annotate(metrics_box, 
                xy=(0.5, -0.2), 
                xycoords='axes fraction',
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
                ha='center',
                va='center',
                fontsize=10)
    #plt.savefig(f"Accuracy_plot_{param_idx}.png", bbox_inches='tight')
    plt.show()

param_accuracy = np.array(param_accuracy)
param_precision = np.array(param_precision)
param_goodness = np.array(param_goodness)   

#Save as a dataframe

df = pd.DataFrame({
    'Parameter': names,
    'Accuracy': param_accuracy,
    'Precision': param_precision,
    'Goodness': param_goodness
})
df.to_csv('Parameter_uncertainty_metrics.csv', index=False)
