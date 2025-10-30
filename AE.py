import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import time
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataloader import CMUMotionDataset
from visualization import *
from dataloader import CMUMotionDataset, recover_global_motion 

class MotionAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for Motion Data as described in the paper
    "Learning Motion Manifolds with Convolutional Autoencoders"
    
    """
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def __init__(self, input_dim=63, latent_dim=128):
        super(MotionAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ----------------
        # Encoder (Conv1d)
        # T -> T/2 -> T/4
        # ----------------
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=7, padding=3),  # keep T
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),  # T/2
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2),  # T/4
            nn.ReLU(inplace=True),

            nn.Conv1d(256, latent_dim, kernel_size=3, padding=1)      # latent feature channels
        )

        # --------------------
        # Decoder (ConvTranspose1d)
        # T/4 -> T/2 -> T
        # --------------------
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1),  # x2 in T
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # x2 in T
            nn.ReLU(inplace=True),

            nn.Conv1d(128, input_dim, kernel_size=7, padding=3)                # back to C_in
        )

        self._init_weights()

    
        # Encoder network with 3 convolutional layers
        # TODO: Complete the encoder architecture.
        # self.encoder = None
        
        # The decoder needs to upsample to restore the original dimensions
        # TODO: Complete the decoder architecture.
        # self.decoder = None
    
    def encode(self, x):
        """Project onto the manifold (Φ operation)"""
        return self.encoder(x)
    
    def decode(self, z):
        """Inverse projection from the manifold (Φ† operation)"""
        return self.decoder(z)
    
    def forward(self, x, corrupt_input=False, corruption_prob=0.1):
        """Forward pass with optional denoising"""
        if corrupt_input and self.training:
            # random feature dropout (denoising)
            mask = torch.bernoulli(torch.ones_like(x) * (1 - corruption_prob))
            x_in = x * mask
        else:
            x_in = x

        z = self.encode(x_in)
        x_hat = self.decode(z)
        # Ensure output has the same temporal length as input

        if x_hat.size(-1) > x.size(-1):
            x_hat = x_hat[..., :x.size(-1)]
        elif x_hat.size(-1) < x.size(-1):
            pad = x_hat.new_zeros(x_hat.size(0), x_hat.size(1), x.size(-1) - x_hat.size(-1))
            x_hat = torch.cat([x_hat, pad], dim=-1)

        return x_hat, z

class MotionManifoldTrainer:
    """Trainer for the Motion Manifold Convolutional Autoencoder"""
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
        epochs: int = 25,
        fine_tune_epochs: int = 25,
        learning_rate: float = 1e-3,
        fine_tune_lr: float = 5e-4,
        sparsity_weight: float = 1e-3,
        window_size: int = 160,
        val_split: float = 0.1,
        device: str = None
    ):

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir if cache_dir else os.path.join(data_dir, "cache")
        self.batch_size = batch_size
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.learning_rate = learning_rate
        self.fine_tune_lr = fine_tune_lr
        self.sparsity_weight = sparsity_weight
        self.window_size = window_size
        self.val_split = val_split
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        
        # Set device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        # Load dataset
        self._load_dataset()
        
        # Initialize model
        self._init_model()

    def _prep_batch(self, batch):
            """
            Build the model input [B, C, T] from a dataloader batch.
            Channels = flattened normalized local positions (+ optional velocities).
            """
            # positions_normalized: [B, T, J, 3]
            pos = batch["positions_normalized"].to(self.device).float()
            B, T, J, _ = pos.shape

            # flatten joints: [B, T, J*3]
            x_pos = pos.view(B, T, J * 3)

            # concatenate extra channels if present: trans_vel_xz [B, T, 2], rot_vel_y [B, T] -> [B, T, 1]
            feats = [x_pos]
            if ("trans_vel_xz" in batch) and ("rot_vel_y" in batch):
                tv = batch["trans_vel_xz"].to(self.device).float()          # [B, T, 2]
                ry = batch["rot_vel_y"].to(self.device).float().unsqueeze(-1)  # [B, T, 1]
                feats += [tv, ry]

            # [B, T, C] -> [B, C, T]
            x_btC = torch.cat(feats, dim=-1)              # [B, T, C]
            x_bCt = x_btC.permute(0, 2, 1).contiguous()   # [B, C, T]

            # pad/truncate channels to match the model's first conv in_channels
            in_ch = next(self.model.encoder.parameters()).shape[1]
            cur_ch = x_bCt.size(1)
            if cur_ch < in_ch:
                pad = x_bCt.new_zeros(B, in_ch - cur_ch, T)
                x_bCt = torch.cat([x_bCt, pad], dim=1)
            elif cur_ch > in_ch:
                x_bCt = x_bCt[:, :in_ch, :]

            return x_bCt
        
    def _load_dataset(self):
        """Load the CMU Motion dataset and create training/validation splits"""
        # Create dataset
        #from dataloader2 import CMUMotionDataset
        
        self.dataset = CMUMotionDataset(
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            frame_rate=30,
            window_size=self.window_size,
            overlap=0.5,
            include_velocity=True,
            include_foot_contact=True
        )
        
        # Split into training and validation sets
        val_size = int(self.val_split * len(self.dataset))
        train_size = len(self.dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Dataset loaded with {len(self.dataset)} windows from {len(self.dataset.motion_data)} files")
        print(f"Training samples: {train_size}, Validation samples: {val_size}")
        
        # Get mean and std for normalization
        self.mean_pose = torch.tensor(self.dataset.get_mean_pose(), device=self.device, dtype=torch.float32)
        self.std = torch.tensor(self.dataset.get_std(), device=self.device, dtype=torch.float32)
        self.joint_names = self.dataset.get_joint_names()
        self.joint_parents = self.dataset.get_joint_parents()
        
    def _init_model(self):
        """Initialize the motion autoencoder model"""
        # Get a sample to determine dimensions
        sample = self.dataset[0]
        
        # Get the flattened positions with velocities for the paper's approach
        # We'll use positions_flat which has global transforms removed
        if "positions_flat" in sample:
            positions_flat = sample["positions_flat"]
            
            # Check if we need to add velocities to the input 
            # The paper mentions including rotational velocity around Y and translational velocity in XZ
            if "trans_vel_xz" in sample and "rot_vel_y" in sample:
                # Get velocity data
                trans_vel_xz = sample["trans_vel_xz"]
                rot_vel_y = sample["rot_vel_y"]
                
                # Create input with features as separate channels (matches paper description)
                input_dim = positions_flat.shape[1] + trans_vel_xz.shape[1] + 1  # positions + trans_vel_xz + rot_vel_y
                print(f"Input includes positions ({positions_flat.shape[1]} dims) and velocities ({trans_vel_xz.shape[1] + 1} dims)")
            else:
                # Just use positions if velocities aren't available
                input_dim = positions_flat.shape[1]
                print(f"Input only includes positions ({input_dim} dims)")
        else:
            # Fallback to original positions if flattened positions aren't available
            positions = sample["positions"]
            # Calculate input dimension from sample
            # For the paper's approach, we need to flatten joints and dimensions
            # positions is [time, joints, 3], we need to get joints*3
            input_dim = positions.shape[1] * positions.shape[2]
            print(f"Using fallback input dimension: {input_dim}")
        
        # Create model
        self.model = MotionAutoencoder(input_dim=input_dim).to(self.device)
        print(f"Created model with input dimension: {input_dim}")
        
    def train(self):
        """Train the motion autoencoder in two phases: initial training and fine-tuning"""
        # TODO: Implement the training phases for the motion autoencoder.
        # There are mutiple training phases described in the paper, you can implement them in this function. You can implement the training phases as separate functions if you prefer or you can use several parameters to combine them to one function.

        # ---- Phase 1: denoising pretrain ----
        stats_phase1 = self._train_phase(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            corruption_prob=0.15,
            sparsity_weight=self.sparsity_weight,
            phase_name="denoise_pretrain"
        )

        # ---- Phase 2: clean fine-tune ----
        stats_phase2 = self._train_phase(
            epochs=self.fine_tune_epochs,
            learning_rate=self.fine_tune_lr,
            corruption_prob=0.0,
            sparsity_weight=self.sparsity_weight * 0.5,
            phase_name="finetune"
        )
        
        # Combine statistics for plotting
        all_stats = {
            "denoise_pretrain": stats_phase1,
            "finetune": stats_phase2
        }
        
        # Save training statistics
        with open(os.path.join(self.output_dir, "training_stats.json"), "w") as f:
            json.dump(all_stats, f, indent=2)
            
        # Save final model
        self._save_model()
            
        # Save normalization parameters
        self._save_normalization_params()
        
        # Plot training curves
        self._plot_training_curves(all_stats)
        
        return all_stats

    # This is a sample of what you can use in the training phase. You are not required to follow it as long as you can provide the training statistics we required.
    def _train_phase(self, epochs, learning_rate, corruption_prob, sparsity_weight, phase_name):
        """Train the model for a specific phase (initial training or fine-tuning)"""
        print(f"\n===== {phase_name.capitalize()} Training Phase =====")
        
        # Define optimizer and loss function
        opt = optim.AdamW(self.model.parameters(), lr=learning_rate)
        recon_loss = nn.MSELoss()
        
        # Training stats
        stats = {
            # Statistics to track. You can add more if needed.
            "train_loss": [],
            "val_loss": []
        }
        
        # Track training checkpoints
        best_val_loss = float("inf")
        
        # Train for specified epochs
        for epoch in range(epochs):
            # Training
            self.model.train()
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            running, n = 0.0, 0
            for batch in progress_bar:
                # TODO: Implement the training loop for the motion autoencoder.
                x = self._prep_batch(batch)  # [B,C,T]
                x_hat, z = self.model(x, corrupt_input=(corruption_prob > 0.0),
                                      corruption_prob=corruption_prob)

                loss_rec = recon_loss(x_hat, x)
                loss_sparse = torch.mean(torch.abs(z))
                loss = loss_rec + sparsity_weight * loss_sparse

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                running += loss.item()
                n += 1
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "rec": f"{loss_rec.item():.4f}",
                    "l1(z)": f"{loss_sparse.item():.4f}"
                })
            train_loss = running / max(1, n)

            # ---------- Val ----------
            self.model.eval()
            v_running = 0.0
            v_n = 0
            with torch.no_grad():
                vbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for batch in vbar:
                    x = self._prep_batch(batch)
                    x_hat, _ = self.model(x, corrupt_input=False)
                    vloss = recon_loss(x_hat, x).item()
                    v_running += vloss
                    v_n += 1
                    vbar.set_postfix({"val_loss": f"{vloss:.4f}"})
            val_loss = v_running / max(1, v_n)

            stats["train_loss"].append(train_loss)
            stats["val_loss"].append(val_loss)

            # Save best checkpoint for this phase
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, end=f'_valloss_{val_loss:.6f}', phase_name=phase_name)
                print(f"  Saved checkpoint with val_loss: {val_loss:.6f}")
        
        return stats
    
    def _save_checkpoint(self, epoch, end, phase_name):
        """Save a model checkpoint"""
        checkpoint_path = os.path.join(
            self.output_dir, "checkpoints", f"{phase_name}_epoch_{epoch+1}{end}.pt"
        )
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
        }, checkpoint_path)
    
    def _save_model(self):
        """Save the trained model"""
        model_path = os.path.join(self.output_dir, "models", "motion_autoencoder.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def _save_normalization_params(self):
        """Save normalization parameters for inference if needed"""
        norm_data = {
            "mean_pose": self.mean_pose.cpu().numpy(),
            "std": self.std.cpu().numpy(),
            "joint_names": self.joint_names,
            "joint_parents": self.joint_parents
        }
        
        np.save(os.path.join(self.output_dir, "normalization.npy"), norm_data)
        print(f"Normalization parameters saved to {self.output_dir}/normalization.npy")
    
    def _plot_training_curves(self, stats):
        """Plot training curves for one or more training phases"""
        if not isinstance(stats[list(stats.keys())[0]], dict):
            stats = {"train": stats}
            
        n_p = len(list(stats.keys()))
        plt.figure(figsize=(12, 4 * n_p))
        # Multiple training phases
        for i, (phase_name, phase_stats) in enumerate(stats.items()):
            plt.subplot(n_p, 1, i+1)
            for key, values in phase_stats.items():
                plt.plot(values, label=key)
            plt.title(f"{phase_name.capitalize()} Training Phase")
            plt.xlabel("Epoch")
            plt.ylabel("Statistics")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "training_curves.png"))
        plt.close()
        
        print(f"Training curves saved to {self.output_dir}/plots/training_curves.png")


class MotionManifoldSynthesizer:
    """Synthesizer for generating, fixing, and analyzing motion using the learned manifold"""
    def __init__(
        self,
        model_path: str,
        dataset: CMUMotionDataset,
        device: str = None
    ):
        # Set device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        # Load normalization parameters
        self._load_normalization(dataset)
        
        # Load model
        self._load_model(model_path)
    
    def _load_normalization(self, dataset: CMUMotionDataset):
        """Load normalization parameters from dataset"""
        self.mean_pose = torch.tensor(dataset.mean_pose, device=self.device, dtype=torch.float32)
        self.std = torch.tensor(dataset.std, device=self.device, dtype=torch.float32)
        self.joint_names = dataset.joint_names
        self.joint_parents = dataset.joint_parents
    
    def _load_model(self, model_path):
        """Load trained model"""
        if os.path.exists(model_path):
            # Determine input dimension from the model's saved state
            model_state = torch.load(model_path, map_location=self.device)
            
            # Try to infer input dimension from the first layer weights
            first_layer_weight = None
            for key in model_state.keys():
                if 'encoder.0.weight' in key:
                    first_layer_weight = model_state[key]
                    break
            
            if first_layer_weight is not None:
                input_dim = first_layer_weight.shape[1]
                print(f"Inferred input dimension {input_dim} from model weights")
            else:
                # Fallback to calculating from mean_pose if we can't find the weights
                input_dim = self.mean_pose.shape[0] * self.mean_pose.shape[1]
                print(f"Using fallback input dimension: {input_dim}")
                
            # Create model
            self.model = MotionAutoencoder(input_dim=input_dim).to(self.device)
            
            # Load weights
            self.model.load_state_dict(model_state)
            
            # Set to evaluation mode
            self.model.eval()
            
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
    
    def fix_corrupted_motion(self, motion, corruption_type='zero', corruption_params=None):
        """
        Fix corrupted motion by projecting onto the manifold and recovering global motion
        
        Args:
            motion: tensor of shape [batch_size, time_steps, joints, dims]
            corruption_type: Type of corruption to apply ('zero', 'noise', or 'missing')
            corruption_params: Parameters for corruption
                    
        Returns:
            Tuple of (corrupted_motion, fixed_motion)
        """
        # positions = motion['positions'].to(self.device)
        # # Store original shape
        # original_shape = positions.shape
        # batch_size, time_steps, joints, dims = original_shape
        
        # # Apply corruption if not already corrupted
        # if corruption_params is not None:
        #     corrupted_motion = self._apply_corruption(positions, corruption_type, corruption_params)
        # else:
        #     corrupted_motion = positions.clone()
            
        # # TODO: Fix the corrupted motion by your model.
        # # HINT: You need to normalize the corrupted motion and then project it onto the manifold using the model. Then unnormalize the fixed motion.
        # # HINT: If you like, you can recover global motion by calling recover_global_motion in dataloader.py
        # fixed_motion = None
        
        # # Return corrupted motion and fixed motion with global transform applied
        # return corrupted_motion, fixed_motion

        from dataloader import recover_global_motion  # local import to avoid circulars

        # ---- Fetch & shape to [B,T,J,3] ----
        positions = motion['positions'].to(self.device)  # [T,J,3] expected
        if positions.dim() == 3:
            positions = positions.unsqueeze(0)           # [1,T,J,3]
        B, T, J, _ = positions.shape

        # ---- Corrupt (in local space) ----
        if corruption_params is not None:
            corrupted_local = self._apply_corruption(positions, corruption_type, corruption_params)
        else:
            corrupted_local = positions.clone()

        # ---- Normalize positions (dataset-wide mean/std on local) ----
        # mean_pose/std: [J,3]  -> broadcast to [B,T,J,3]
        mean = self.mean_pose
        std  = self.std
        Xpos = (corrupted_local - mean) / (std + 1e-8)   # [B,T,J,3]

        # ---- Build model input [B,C,T] ----
        # Base channels: J*3 (flattened positions)
        X_list = [Xpos.view(B, T, J * 3)]

        # If the model was trained with extra channels, append them to match input_dim
        in_ch = next(self.model.encoder.parameters()).shape[1]  # encoder first conv in_channels

        # Append velocities if present and needed
        extra_added = 0
        if 'trans_vel_xz' in motion and 'rot_vel_y' in motion:
            tv = motion['trans_vel_xz'].to(self.device)  # [T,2]
            ry = motion['rot_vel_y'].to(self.device)     # [T]
            if tv.dim() == 2:
                tv = tv.unsqueeze(0)                     # [1,T,2]
            if ry.dim() == 1:
                ry = ry.unsqueeze(0)                     # [1,T]
            X_list.append(tv)                             # [B,T,2]
            X_list.append(ry.unsqueeze(-1))               # [B,T,1]
            extra_added = 3

        X_btC = torch.cat(X_list, dim=-1)                 # [B,T,J*3 (+ 2 + 1 if present)]
        X = X_btC.permute(0, 2, 1).contiguous()           # [B,C,T]

        # If still under the model's expected channels, pad zeros (robustness)
        if X.size(1) < in_ch:
            padC = in_ch - X.size(1)
            X = torch.cat([X, X.new_zeros(B, padC, T)], dim=1)
        elif X.size(1) > in_ch:
            # If too many channels, truncate to the model input
            X = X[:, :in_ch, :]

        # ---- Manifold projection (denoising via autoencoder) ----
        with torch.no_grad():
            x_hat, _ = self.model(X, corrupt_input=False)          # [B,C,T]

        # ---- Extract reconstructed positions channels & unnormalize back to local space ----
        pos_ch = J * 3
        Xhat_pos = x_hat[:, :pos_ch, :].permute(0, 2, 1).contiguous().view(B, T, J, 3)  # [B,T,J,3] (normalized)
        fixed_local = Xhat_pos * (std + 1e-8) + mean

        # ---- If velocities exist, recover global for BOTH corrupted & fixed ----
        has_vel = ('trans_vel_xz' in motion) and ('rot_vel_y' in motion)
        if has_vel:
            trans_vel_xz = motion['trans_vel_xz'].to(self.device)
            rot_vel_y    = motion['rot_vel_y'].to(self.device)
            if trans_vel_xz.dim() == 2:
                trans_vel_xz = trans_vel_xz.unsqueeze(0)  # [B,T,2]
            if rot_vel_y.dim() == 1:
                rot_vel_y = rot_vel_y.unsqueeze(0)        # [B,T]

            corrupted_global = recover_global_motion(corrupted_local, trans_vel_xz, rot_vel_y, frame_rate=30.0)
            fixed_global     = recover_global_motion(fixed_local,     trans_vel_xz, rot_vel_y, frame_rate=30.0)
            return corrupted_global, fixed_global

        # Otherwise, return local motions
        return corrupted_local, fixed_local
    
    def _apply_corruption(self, motion, corruption_type, params):
        """Apply corruption to motion data"""
        corrupted = motion.clone()
        
        if corruption_type == 'zero':
            # Randomly set values to zero
            prob = params.get('prob', 0.5)
            mask = torch.bernoulli(torch.ones_like(corrupted) * (1 - prob))
            corrupted = corrupted * mask
            
        elif corruption_type == 'noise':
            # Add Gaussian noise
            noise_scale = params.get('scale', 0.1)
            noise = torch.randn_like(corrupted) * noise_scale
            corrupted = corrupted + noise
            
        elif corruption_type == 'missing':
            # Set specific joint to zero
            joint_idx = params.get('joint_idx', 0)
            corrupted[:, :, joint_idx, :] = 0.0
            
        return corrupted
    
    def interpolate_motions(self, motion1, motion2, t):
        """
        Interpolate between two motions on the manifold, handling global transforms
        
        Args:
            motion1: tensor of shape [batch_size, time_steps, joints, dims]
            motion2: tensor of shape [batch_size, time_steps, joints, dims]
            t: Interpolation parameter (0 to 1)
                    
        Returns:
            Interpolated motion as tensor of shape [batch_size, time_steps, joints, dims]
        """
        # TODO: Implement motion interpolation on the manifold.
        # HINT: You can use the model to project the motions onto the manifold and then interpolate in the latent space.
        # HINT: To simplify implementation, you could only implement the version where both motioins are local motions (without global transforms).
        # ---- Fetch & prepare shapes ----
        P1 = motion1["positions"].to(self.device)
        P2 = motion2["positions"].to(self.device)
        if P1.dim() == 3: P1 = P1.unsqueeze(0)  # [1,T,J,3]
        if P2.dim() == 3: P2 = P2.unsqueeze(0)

        B1, T1, J, _ = P1.shape
        B2, T2, J2, _ = P2.shape
        assert B1 == B2 == 1, "This helper assumes batch size 1 for interpolation."
        assert J == J2, "Joint counts must match."

        # Align time (use common prefix length)
        T = min(T1, T2)
        P1 = P1[:, :T]
        P2 = P2[:, :T]

        # ---- Normalize local positions ----
        mean, std = self.mean_pose, self.std  # [J,3]
        X1_pos = (P1 - mean) / (std + 1e-8)   # [B,T,J,3]
        X2_pos = (P2 - mean) / (std + 1e-8)

        # ---- Build model input [B,C,T] to match training channels ----
        def build_input(motion, Xpos):
            X_list = [Xpos.view(Xpos.size(0), Xpos.size(1), J * 3)]  # [B,T,J*3]
            if ("trans_vel_xz" in motion) and ("rot_vel_y" in motion):
                tv = motion["trans_vel_xz"].to(self.device)  # [T,2] or [B,T,2]
                ry = motion["rot_vel_y"].to(self.device)     # [T]   or [B,T]
                if tv.dim() == 2: tv = tv.unsqueeze(0)
                if ry.dim() == 1: ry = ry.unsqueeze(0)
                X_list.append(tv)                 # [B,T,2]
                X_list.append(ry.unsqueeze(-1))   # [B,T,1]
            X_btC = torch.cat(X_list, dim=-1)     # [B,T,C]
            X = X_btC.permute(0, 2, 1).contiguous()  # [B,C,T]
            # pad/truncate channels to match model input
            in_ch = next(self.model.encoder.parameters()).shape[1]
            if X.size(1) < in_ch:
                X = torch.cat([X, X.new_zeros(X.size(0), in_ch - X.size(1), X.size(2))], dim=1)
            elif X.size(1) > in_ch:
                X = X[:, :in_ch, :]
            return X

        X1 = build_input(motion1, X1_pos)
        X2 = build_input(motion2, X2_pos)

        # ---- Encode, interpolate in latent, decode ----
        with torch.no_grad():
            z1 = self.model.encode(X1)          # [B, D, T/4] (temporal latent)
            z2 = self.model.encode(X2)
            zt = (1.0 - t) * z1 + t * z2
            xhat = self.model.decode(zt)        # [B, C, ~T]

            # Ensure temporal length matches input T
            if xhat.size(-1) > T:
                xhat = xhat[..., :T]
            elif xhat.size(-1) < T:
                xhat = torch.cat([xhat, xhat.new_zeros(xhat.size(0), xhat.size(1), T - xhat.size(-1))], dim=-1)

        # ---- Extract position channels and unnormalize to local space ----
        pos_ch = J * 3
        Xhat_pos = xhat[:, :pos_ch, :].permute(0, 2, 1).contiguous().view(1, T, J, 3)  # [B,T,J,3]
        P_interp = Xhat_pos * (std + 1e-8) + mean                                      # local (global-removed)

        return P_interp
        #pass
    
    # You can add more functions for Extra Credit.
    
def main():
    """Example usage of the motion manifold training"""
    
    # # Training parameters
    # data_dir = "path/to/cmu-mocap"
    # output_dir = "./output/ae"
    
    # trainer = MotionManifoldTrainer(
    #     data_dir=data_dir,
    #     output_dir=output_dir,
    #     batch_size=32,
    #     epochs=25,              # Initial training epochs
    #     fine_tune_epochs=25,    # Fine-tuning epochs
    #     learning_rate=0.001,    # Initial learning rate
    #     fine_tune_lr=0.001,     # Fine-tuning learning rate
    #     sparsity_weight=0.01,   # Sparsity constraint weight
    #     window_size=160,        # Window size (as in paper)
    #     val_split=0.1           # Validation split
    # )
    
    # # Train the model
    # trainer.train()
    
    # For inference, you can load the dataset and model and use the synthesizer for different tasks. 
    # You can also use the visualization functions to visualize the results following examples in dataloader.py.

    import argparse, os, torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="path/to/cmu-mocap",
                        help="Root folder that contains BVH files (e.g., ./cmu-mocap)")
    parser.add_argument("--out_dir", type=str, default="./output/ae",
                        help="Where to save models/plots/videos")
    parser.add_argument("--epochs", type=int, default=10, help="Pretrain epochs")
    parser.add_argument("--finetune_epochs", type=int, default=10, help="Finetune epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Pretrain learning rate")
    parser.add_argument("--lr_ft", type=float, default=5e-4, help="Finetune learning rate")
    parser.add_argument("--sparsity", type=float, default=1e-3, help="L1 weight on latent")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--window", type=int, default=160)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

    # ---- Train ----
    trainer = MotionManifoldTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        fine_tune_epochs=args.finetune_epochs,
        learning_rate=args.lr,
        fine_tune_lr=args.lr_ft,
        sparsity_weight=args.sparsity,
        window_size=args.window,
        val_split=args.val_split
    )
    trainer.train()

    # ---- Inference demos ----
    # Build synthesizer with the just-trained model
    model_path = os.path.join(output_dir, "models", "motion_autoencoder.pt")
    synth = MotionManifoldSynthesizer(model_path=model_path, dataset=trainer.dataset)

    # 1) Interpolation between two windows
    ds = trainer.dataset
    if len(ds) >= 2:
        m1 = ds[0]
        m2 = ds[1]
        P_interp = synth.interpolate_motions(m1, m2, t=0.5)  # [1,T,J,3] local
        # Save as video
        interp_vid = os.path.join(output_dir, "interp.mp4")
        # visualization.py expected signature:
        # visualize_interpolation(P1, P2, Pmid, joint_parents, out_path)
        visualize_interpolation(
            m1["positions"].unsqueeze(0).cpu(),
            m2["positions"].unsqueeze(0).cpu(),
            P_interp.cpu(),
            ds.joint_parents,
            interp_vid
        )
        print(f"Saved interpolation demo to {interp_vid}")

    # 2) Corruption → fix demo on a single window
    if len(ds) >= 1:
        sample = ds[0]
        corr, fixed = synth.fix_corrupted_motion(
            sample,
            corruption_type="zero",
            corruption_params={"prob": 0.5}
        )
        fix_vid = os.path.join(output_dir, "fix_compare.mp4")
        # visualization.py expected signature:
        # visualize_motion_comparison(motion_a, motion_b, joint_parents, out_path)
        visualize_motion_comparison(
            corr.cpu(), fixed.cpu(), ds.joint_parents, fix_vid
        )
        print(f"Saved corruption-fix demo to {fix_vid}")


if __name__ == "__main__":
    main()
    
    