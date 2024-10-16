import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import (
    Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, ColorJitter
)
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration
)
from sentence_transformers import SentenceTransformer
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# Define Model Components
# ===========================

# TabTransformer Model
class TabTransformer(nn.Module):
    def __init__(
        self,
        num_categories,
        continuous_dim,
        text_embed_dim,
        embed_dim=128,
        num_heads=8,
        num_blocks=4,
        dropout=0.1,
        temperature=0.07
    ):
        super(TabTransformer, self).__init__()
        self.embed_dim = embed_dim

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num, embed_dim) for num in num_categories
        ])

        # Transformer blocks for processing categorical embeddings
        self.transformer_blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                embed_dim,
                num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_blocks)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Projection layer for continuous features
        self.cont_proj = nn.Sequential(
            nn.Linear(continuous_dim, embed_dim) if continuous_dim > 0 else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Fully connected layers for generating embeddings
        fc_input_size = embed_dim * (
            int(continuous_dim > 0) + int(len(num_categories) > 0)
        ) + text_embed_dim

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.info_nce_loss = InfoNCELoss(temperature=temperature)

    def forward(self, categorical_data, continuous_data, text_embedding, labels=None):
        batch_size = text_embedding.size(0)

        # Process categorical data
        if categorical_data.size(1) > 0:
            cat_embeds = [
                embed(categorical_data[:, i]) for i, embed in enumerate(self.embeddings[:categorical_data.size(1)])
            ]
            cat_embeds = torch.stack(cat_embeds, dim=1)
            cat_embeds = self.transformer_blocks(cat_embeds)
            cat_embeds = self.layer_norm(cat_embeds)
            cat_embeds = cat_embeds.mean(dim=1)
        else:
            cat_embeds = torch.zeros(batch_size, 0).to(text_embedding.device)

        # Process continuous data
        if continuous_data.size(1) > 0:
            continuous_embedding = self.cont_proj(continuous_data)
        else:
            continuous_embedding = torch.zeros(batch_size, self.embed_dim).to(text_embedding.device)

        # Combine embeddings
        embeddings_to_concat = [text_embedding]
        if continuous_embedding.size(1) > 0:
            embeddings_to_concat.append(continuous_embedding)
        if cat_embeds.size(1) > 0:
            embeddings_to_concat.append(cat_embeds)

        combined = torch.cat(embeddings_to_concat, dim=-1)
        embedding = self.fc(combined)  # Output embedding

        if labels is not None:
            # Filter out invalid labels (-1)
            valid_indices = labels != -1
            if valid_indices.any():
                valid_labels = labels[valid_indices]
                valid_text_embedding = text_embedding[valid_indices]
                valid_cat_embeds = cat_embeds[valid_indices]
                info_nce_loss = self.info_nce_loss(valid_text_embedding, valid_cat_embeds, valid_labels)
            else:
                info_nce_loss = torch.tensor(0.0, device=text_embedding.device)
        else:
            info_nce_loss = torch.tensor(0.0, device=text_embedding.device)

        return embedding, info_nce_loss

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, embedding_dim=768):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.feature_projection = nn.Linear(128, embedding_dim)  # Adjust dimensions as needed

    def forward(self, features_1, features_2, labels):
        if features_1.size(0) == 0 or features_2.size(0) == 0:
            return torch.tensor(0.0, device=features_1.device)

        # Project features_2 to match features_1 dimension
        features_2 = self.feature_projection(features_2)

        # Normalize the features
        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)

        # Compute logits
        logits = torch.matmul(features_1, features_2.T) / self.temperature

        # Create labels for contrastive loss
        labels = torch.arange(features_1.size(0)).to(features_1.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


# Manifold Autoencoder
class ManifoldAutoencoder(nn.Module):
    def __init__(self, input_dim=512, latent_dim=512):
        super(ManifoldAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, latent_dim),
        )
        self.log_var_layer = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        mu = self.encoder(x)
        log_var = self.log_var_layer(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)

        # KL-divergence regularization
        kl_loss = (
            -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        ) / x.size(0)

        return x_recon, z, kl_loss

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

# VQ-VAE Components
class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn(self.conv5(x))
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, 512, 3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.tanh(self.conv5(x))
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=2048, embedding_dim=512, commitment_cost=0.25, chunk_size=1024):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.chunk_size = chunk_size  # Chunk size to handle large tensors

        # Embedding codebook
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        # z is of shape [batch_size, D, H, W]
        z = z.permute(0, 2, 3, 1).contiguous()  # Shape: [batch_size, H, W, D]
        flat_z = z.view(-1, self.embedding_dim)  # Flatten z to [batch_size * H * W, embedding_dim]

        # Compute distances to each embedding
        encoding_indices = []
        num_chunks = (flat_z.size(0) + self.chunk_size - 1) // self.chunk_size

        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, flat_z.size(0))
            chunk = flat_z[start:end]  # Chunk of flat_z to avoid OOM

            # Squared L2 distance
            distances = torch.sum((chunk.unsqueeze(1) - self.embeddings.weight.unsqueeze(0)) ** 2, dim=2)
            indices = torch.argmin(distances, dim=1)
            encoding_indices.append(indices)

        encoding_indices = torch.cat(encoding_indices, dim=0)  # Concatenate all chunks' indices

        # Quantize and reshape the embeddings back
        quantized = torch.index_select(self.embeddings.weight, dim=0, index=encoding_indices)
        quantized = quantized.view(z.shape)  # Shape: [batch_size, H, W, D]

        # Compute VQ Losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())

        # Combine quantization loss
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        # Permute quantized back to [batch_size, D, H, W]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized, vq_loss

# Conditional Diffusion Model
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, in_channels, latent_dim, timesteps=1000):
        super(ConditionalDiffusionModel, self).__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Define a more robust U-Net architecture
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels + latent_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x_noisy, latents, t):
        # Ensure latents have the correct spatial dimensions
        latents = F.interpolate(latents, size=x_noisy.shape[2:], mode='nearest')

        # Concatenate x_noisy and latents along the channel dimension
        x = torch.cat([x_noisy, latents], dim=1)

        # U-Net forward pass
        d1 = self.down1(x)
        d2 = self.down2(d1)
        m = self.middle(d2)
        u1 = self.up1(m + d2)  # Skip connection
        u2 = self.up2(u1 + d1)  # Skip connection

        return u2


# TokenImageCNN
class TokenImageCNN(nn.Module):
    def __init__(
        self, embedding_dim, output_dim, h_prime, w_prime, c_prime, num_heads=4
    ):
        super(TokenImageCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.h_prime = h_prime
        self.w_prime = w_prime
        self.c_prime = c_prime
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=0.5)

        # Learnable projection to reshape token embeddings into image-like structure
        self.projection = nn.Linear(embedding_dim, h_prime * w_prime * c_prime)

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=c_prime, out_channels=64, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * (h_prime // 8) * (w_prime // 8), 1024)
        self.fc2 = nn.Linear(1024, output_dim)

    def forward(self, token_embeddings):
        # Project token embeddings into image-like structure
        projected_embeddings = self.projection(token_embeddings)
        projected_embeddings = projected_embeddings.view(
            -1, self.c_prime, self.h_prime, self.w_prime
        )

        # Apply convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(projected_embeddings))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Flatten and apply fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# VQ-VAE with Diffusion Model
class VQVAEWithDiffusion(nn.Module):
    def __init__(
        self,
        in_channels,
        num_embeddings,
        commitment_cost,
        latent_dim,
        device,
        clip_model,
        clip_processor,
        token_image_cnn,
        manifold_autoencoder,
        tab_transformer,
        num_classes  # Add num_classes as a parameter
    ):
        super(VQVAEWithDiffusion, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(in_channels=512)
        self.vq_layer = VectorQuantizer(
            num_embeddings, 512, commitment_cost
        )
        self.diffusion_model = ConditionalDiffusionModel(in_channels, latent_dim)
        self.device = device
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.token_image_cnn = token_image_cnn
        self.manifold_autoencoder = manifold_autoencoder
        self.tab_transformer = tab_transformer

        # Projection layers for image and text embeddings
        self.image_projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(self.token_image_cnn.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.difference_projection = nn.Linear(512, latent_dim)

        # Final fully connected layers for classification/regression
        self.final_fc = nn.Sequential(
            nn.Linear(256 + 256 + self.tab_transformer.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),  # Use num_classes here
        )

    def forward(
        self,
        x,
        x_noisy=None,
        timesteps=None,
        text_embeddings=None,
        tabular_data=None,
        labels=None,
        visualization_mode=False,
        train_vqvae_only=False
    ):
        x = x.to(self.device)
        batch_size = x.size(0)

        # Step 1: Encode the input images
        z = self.encoder(x)

        # Step 2: Vector Quantization
        quantized, vq_loss = self.vq_layer(z)

        if train_vqvae_only:
            # Only perform reconstruction using decoder
            x_recon = self.decoder(quantized)
            return x_recon, vq_loss

        if not train_vqvae_only and (x_noisy is None or timesteps is None):
            raise ValueError("x_noisy and timesteps must be provided when not in train_vqvae_only mode.")

        # Proceed with the rest of the model
        # Process text embeddings using TokenImageCNN
        text_embeddings = text_embeddings.to(self.device)
        text_images = self.token_image_cnn(text_embeddings)

        # CLIP Feature Extraction for images
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(pixel_values=x)

        # Project features to common dimension
        image_features_proj = self.image_projection(image_features)
        text_images_proj = self.text_projection(text_images)

        # Concatenate the projected features
        combined_image_text_features = torch.cat((image_features_proj, text_images_proj), dim=1)

        # Calculate difference from mean
        combined_features_avg = combined_image_text_features.mean(dim=0)
        difference = combined_image_text_features - combined_features_avg.unsqueeze(0)
        projected_difference = self.difference_projection(difference)

        # Pass through manifold autoencoder
        aggregated_difference, _, kl_loss = self.manifold_autoencoder(projected_difference)

        # Process tabular data through TabTransformer
        if tabular_data is not None:
            tabular_embedding, info_nce_loss = self.tab_transformer(
                categorical_data=tabular_data['categorical'],
                continuous_data=tabular_data['continuous'],
                text_embedding=text_embeddings,
                labels=labels
            )
        else:
            tabular_embedding = torch.zeros(batch_size, self.tab_transformer.embed_dim).to(self.device)
            info_nce_loss = torch.tensor(0.0).to(self.device)

        # Reshape aggregated_difference to match quantized dimensions
        _, _, H, W = quantized.shape  # Get spatial dimensions
        aggregated_difference = aggregated_difference.view(batch_size, -1, 1, 1).expand(-1, -1, H, W)

        # Combine the quantized result with the reshaped difference
        combined_input = quantized + aggregated_difference

        # Reverse Diffusion (Reconstruction)
        x_recon = self.diffusion_model(x_noisy, combined_input, timesteps)

        # Compute reconstructed images x0_hat
        x0_hat = self.p_sample(x_noisy, timesteps, x_recon)

        # Clamp x0_hat to [-1, 1]
        x0_hat = torch.clamp(x0_hat, -1, 1)

        # Combine features for final classification
        combined_features = torch.cat([image_features_proj, text_images_proj, tabular_embedding], dim=1)
        final_output = self.final_fc(combined_features)

        return x_recon, x0_hat, vq_loss, kl_loss, info_nce_loss, final_output

    def q_sample(self, x_start, t, noise):
        betas = torch.linspace(0.00005, 0.02, self.diffusion_model.timesteps).to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-10)  # Prevent zeros
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x_noisy, t, predicted_noise):
        betas = torch.linspace(0.0001, 0.02, self.diffusion_model.timesteps).to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-10)  # Prevent zeros
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        sqrt_recip_alphas_cumprod_t = sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)

        x0_hat = sqrt_recip_alphas_cumprod_t * x_noisy - sqrt_recipm1_alphas_cumprod_t * predicted_noise
        x0_hat = torch.clamp(x0_hat, -1, 1)
        return x0_hat

# ===========================
# Training and Evaluation Utilities
# ===========================

def get_kl_loss_weight(epoch, max_epochs, max_weight=1.0):
    return min(max_weight, (epoch / max_epochs) ** 0.5 * max_weight)

def compute_diffusion_loss(x_recon, noise):
    mse_loss = F.mse_loss(x_recon, noise)
    return mse_loss

def compute_clip_loss(x_recon, captions, model, clip_model, clip_processor, device, temperature=0.5):
    if x_recon.size(0) != len(captions):
        raise ValueError("Batch size of x_recon and captions must match.")

    batch_size = x_recon.size(0)

    # Extract features
    image_features = clip_model.get_image_features(pixel_values=x_recon)
    projected_image_features = model.image_projection(image_features)

    text_inputs = clip_processor(
        text=captions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(device)
    text_features = clip_model.get_text_features(**text_inputs)
    projected_text_features = model.text_projection(text_features)

    # Normalize features
    image_embeddings = F.normalize(projected_image_features, p=2, dim=1)
    text_embeddings = F.normalize(projected_text_features, p=2, dim=1)

    # Compute logits
    logits_per_image = torch.matmul(image_embeddings, text_embeddings.T) / temperature
    logits_per_text = logits_per_image.T

    # Labels
    labels = torch.arange(batch_size).to(device)

    # Cross-entropy loss
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    clip_loss = (loss_i2t + loss_t2i) / 2

    return clip_loss


def calculate_metrics(images, x_recon):
    # Clamp the images to the valid range [-1, 1]
    images = torch.clamp(images, -1, 1)
    x_recon = torch.clamp(x_recon, -1, 1)

    # Map images to [0, 1]
    images_np = (images.cpu().numpy() + 1) / 2
    x_recon_np = (x_recon.cpu().detach().numpy() + 1) / 2

    # Compute PSNR and SSIM
    psnr_value = 0.0
    ssim_value = 0.0
    for i in range(images_np.shape[0]):
        psnr_value += psnr(images_np[i], x_recon_np[i], data_range=1)
        ssim_value += ssim(
            images_np[i].transpose(1, 2, 0),
            x_recon_np[i].transpose(1, 2, 0),
            multichannel=True,
            data_range=1,
            win_size=3,
        )
    psnr_value /= images_np.shape[0]
    ssim_value /= images_np.shape[0]

    return psnr_value, ssim_value

def visualize_reconstructions(original_images, reconstructed_images, epoch, save_path, num_images=5):
    # Convert images to the range [0, 1] for visualization
    images_np = (original_images.cpu().numpy() + 1) / 2
    x0_hat_np = (reconstructed_images.cpu().detach().numpy() + 1) / 2

    # Plot and save images
    for i in range(min(num_images, images_np.shape[0])):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(np.transpose(images_np[i], (1, 2, 0)))
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(np.transpose(x0_hat_np[i], (1, 2, 0)))
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'reconstruction_epoch_{epoch}_{i}.png'))
        plt.close(fig)

# Exponential Moving Average (EMA) with dynamic decay adjustment
class EMA:
    def __init__(self, model, decay_start=0.999, decay_end=0.9999):
        self.model = model
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.shadow = {}
        self.backup = {}
        self.current_decay = decay_start

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    continue
                new_average = (1.0 - self.current_decay) * param.data + self.current_decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def adjust_decay(self, epoch, max_epochs):
        self.current_decay = self.decay_start + (self.decay_end - self.decay_start) * (epoch / max_epochs)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    continue
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.backup:
                    continue
                param.data = self.backup[name]
        self.backup = {}

def combined_lr_scheduler(optimizer, warmup_epochs, total_epochs, min_lr):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)  # Linear warmup
        else:
            progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return max(min_lr / optimizer.defaults['lr'], cosine_decay)
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

import base64
from io import BytesIO

# ===========================
# Dataset Preparation
# ===========================
# Dataset Preparation
class CombinedDataset(Dataset):
    def __init__(self, parquet_file, transform=None):
        # Load the data from the Parquet file
        self.data = pd.read_parquet(parquet_file)
        self.transform = transform

        # Initialize label encoders for categorical fields if needed
        self.label_encoders = {}
        self.categorical_columns = ['set', 'lang', 'rarity', 'artist', 'frame', 'border_color']  # Add more if needed
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            self.label_encoders[col] = le

        # Standardize continuous data if needed
        self.continuous_columns = ['cmc', 'power', 'toughness', 'loyalty']  # Add more if needed
        self.continuous_data = self.data[self.continuous_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        scaler = StandardScaler()
        self.continuous_data = scaler.fit_transform(self.continuous_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the row data for the given index
        row = self.data.iloc[idx]

        # Load image from base64 encoded string
        img_data = row['image_data']
        try:
            image = Image.open(BytesIO(base64.b64decode(img_data))).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image for card {row['id']}: {e}")
            image = Image.new('RGB', (224, 224))  # Return a blank image if missing

        if self.transform:
            image = self.transform(image)

        # Get categorical data
        categorical_data = torch.tensor([row[col] for col in self.categorical_columns], dtype=torch.long)

        # Get continuous data
        continuous_data = torch.tensor(self.continuous_data[idx], dtype=torch.float32)

        # Get text description (you can customize this as needed)
        description = (
            f"Name: {row['name']}\n"
            f"Set: {row['set_name']}\n"
            f"Mana Cost: {row['mana_cost']}\n"
            f"Type Line: {row['type_line']}\n"
            f"Oracle Text: {row['oracle_text']}\n"
            f"Power: {row['power']}\n"
            f"Toughness: {row['toughness']}\n"
            f"Rarity: {row['rarity']}\n"
            f"Artist: {row['artist']}\n"
        )

        # Get the label if applicable (e.g., based on card type or set)
        label = row['rarity']  # This is just an example; adjust based on your use case

        return {
            'image': image,
            'description': description,
            'categorical': categorical_data,
            'continuous': continuous_data,
            'label': torch.tensor(label, dtype=torch.long),
            'id': row['id'],
            'oracle_id': row['oracle_id'],
            'set': row['set'],
            'lang': row['lang'],
            'collector_number': row['collector_number'],
            'flavor_text': row['flavor_text'],
            'prices': row['prices'],
            # Add more fields as needed
        }


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized tensors by padding.
    """
    # Separate the batch into individual components (images, descriptions, etc.)
    images = [item['image'] for item in batch]
    descriptions = [item['description'] for item in batch]
    categorical_data = [item['categorical'] for item in batch]
    continuous_data = [item['continuous'] for item in batch]
    labels = [item['label'] for item in batch]

    # Pad categorical data to match the maximum length in the batch
    max_cat_len = max([cat.size(0) for cat in categorical_data])
    padded_categorical = [
        torch.cat([cat, torch.zeros(max_cat_len - cat.size(0), dtype=torch.long)]) for cat in categorical_data
    ]

    # Pad continuous data to match the maximum length in the batch
    max_cont_len = max([cont.size(0) for cont in continuous_data])
    padded_continuous = [
        torch.cat([cont, torch.zeros(max_cont_len - cont.size(0))]) for cont in continuous_data
    ]

    # Stack the padded data and images into tensors
    images = torch.stack(images)
    padded_categorical = torch.stack(padded_categorical)
    padded_continuous = torch.stack(padded_continuous)
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        'image': images,
        'description': descriptions,  # Keep descriptions as a list of strings
        'categorical': padded_categorical,
        'continuous': padded_continuous,
        'label': labels
    }

def plot_losses(
    train_avg_losses_per_epoch: list,
    val_avg_losses_per_epoch: list,
    train_recon_losses_per_epoch: list,
    val_recon_losses_per_epoch: list,
    train_vq_losses_per_epoch: list,
    val_vq_losses_per_epoch: list,
    train_clip_losses_per_epoch: list,
    val_clip_losses_per_epoch: list,
    train_kl_losses_per_epoch: list,
    val_kl_losses_per_epoch: list,
    epoch: int,
    save_path: str
) -> None:
    # Create a figure for the plot
    plt.figure(figsize=(12, 6))

    # Create a range of epochs for x-axis
    epochs = range(1, epoch + 1)

    # Plot the losses
    plt.plot(epochs, train_avg_losses_per_epoch, label='Train Total Loss', linewidth=2)
    plt.plot(epochs, val_avg_losses_per_epoch, label='Validation Total Loss', linewidth=2)
    plt.plot(epochs, train_recon_losses_per_epoch, label='Train Reconstruction Loss', linewidth=2)
    plt.plot(epochs, val_recon_losses_per_epoch, label='Validation Reconstruction Loss', linewidth=2)
    plt.plot(epochs, train_vq_losses_per_epoch, label='Train VQ Loss', linewidth=2)
    plt.plot(epochs, val_vq_losses_per_epoch, label='Validation VQ Loss', linewidth=2)
    plt.plot(epochs, train_clip_losses_per_epoch, label='Train CLIP Loss', linewidth=2)
    plt.plot(epochs, val_clip_losses_per_epoch, label='Validation CLIP Loss', linewidth=2)
    plt.plot(epochs, train_kl_losses_per_epoch, label='Train KL Loss', linewidth=2)
    plt.plot(epochs, val_kl_losses_per_epoch, label='Validation KL Loss', linewidth=2)

    # Label the axes
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Add a legend to differentiate the losses
    plt.legend()

    # Set a title
    plt.title(f'Losses during Training and Validation - Epoch {epoch}')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot to the specified path
    plt.savefig(os.path.join(save_path, f"loss_plot_epoch_{epoch}.png"))

    # Close the plot
    plt.close()

def calculate_mean_std(dataset):
    means = torch.zeros(3)
    stds = torch.zeros(3)
    
    for data in dataset:
        image = data['image']  # This is a PIL image
        image_tensor = ToTensor()(image)  # Convert PIL image to tensor
        for i in range(3):  # Loop over the 3 color channels
            means[i] += image_tensor[i, :, :].mean()
            stds[i] += image_tensor[i, :, :].std()
    
    means /= len(dataset)
    stds /= len(dataset)
    
    return means, stds

# ===========================
# Main Training Loop
# ===========================

def main():
    # Hyperparameters
    num_epochs = 20
    max_epochs = num_epochs
    batch_size = 16
    num_workers = 4  # Adjusted for better performance
    latent_dim = 512
    save_path = "results"
    os.makedirs(save_path, exist_ok=True)

    warmup_epochs = 0  # Added warmup epochs for learning rate scheduler

    # Define the number of epochs for each training phase
    vqvae_epochs = 0       # Number of epochs to train VQ-VAE only
    diffusion_epochs = 0   # Number of epochs to train diffusion model only

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parquet_file = "/home/plunder/data/mtgdata.parquet"

    # Create the dataset without the transform first
    dataset_without_transform = CombinedDataset(parquet_file)
    print(dataset_without_transform[0])
    
    # Calculate mean and std for your dataset
    mean, std = calculate_mean_std(dataset_without_transform)
    print(f"Mean: {mean}")
    print(f"Std: {std}")

    # Define the transform using the calculated mean and std
    transform = Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    # Create the dataset with the transform
    dataset = CombinedDataset(parquet_file, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn)

    # Set the number of classes based on rarity, set, and language
    num_classes = (
        len(dataset.label_encoders['rarity'].classes_) *
        len(dataset.label_encoders['set'].classes_) *
        len(dataset.label_encoders['lang'].classes_)
    )

    # Load Models and Processors
    clip_model_name = 'openai/clip-vit-base-patch32'
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_processor.feature_extractor.do_rescale = False

    # Freeze CLIP model parameters
    for param in clip_model.parameters():
        param.requires_grad = False

    sentence_transformer_model_name = 'sentence-transformers/bert-base-nli-mean-tokens'
    sentence_transformer_model = SentenceTransformer(
        sentence_transformer_model_name).to(device)

    blip_model_name = 'Salesforce/blip-image-captioning-base'
    blip_processor = BlipProcessor.from_pretrained(blip_model_name)
    blip_model = BlipForConditionalGeneration.from_pretrained(
        blip_model_name).to(device)
    blip_processor.image_processor.do_rescale = False

    # Initialize Components
    token_image_cnn = TokenImageCNN(
        embedding_dim=768,  # Assuming text embeddings are of size 768
        output_dim=512,
        h_prime=14,  # Adjust based on desired output size
        w_prime=14,
        c_prime=64,
    ).to(device)

    manifold_autoencoder = ManifoldAutoencoder(
        input_dim=512, latent_dim=latent_dim,
    ).to(device)

    # Initialize the TabTransformer
    tab_transformer = TabTransformer(
        num_categories=[len(le.classes_) for le in dataset.label_encoders.values()],
        continuous_dim=dataset.continuous_data.shape[1],
        text_embed_dim=768,
        embed_dim=128,
        num_heads=8,
        num_blocks=4,
        dropout=0.1,
        temperature=0.07
    ).to(device)

    model = VQVAEWithDiffusion(
        in_channels=3,
        num_embeddings=2048,  # Increase number of embeddings
        commitment_cost=0.4,  # Adjust commitment cost
        latent_dim=latent_dim,
        device=device,
        clip_model=clip_model,
        clip_processor=clip_processor,
        token_image_cnn=token_image_cnn,
        manifold_autoencoder=manifold_autoencoder,
        tab_transformer=tab_transformer,
        num_classes=num_classes
    ).to(device)

    # Optimizers
    # VQ-VAE parameters
    vqvae_parameters = (
        list(model.encoder.parameters()) +
        list(model.decoder.parameters()) +
        list(model.vq_layer.parameters())
    )
    vqvae_optimizer = AdamW(vqvae_parameters, lr=2e-4, weight_decay=1e-4)

    # Diffusion model parameters
    diffusion_parameters = list(model.diffusion_model.parameters())
    diffusion_optimizer = AdamW(diffusion_parameters, lr=2e-4, weight_decay=1e-4)

    # Other model parameters
    other_parameters = (
        list(model.token_image_cnn.parameters()) +
        list(model.image_projection.parameters()) +
        list(model.text_projection.parameters()) +
        list(model.difference_projection.parameters()) +
        list(model.final_fc.parameters()) +
        list(model.manifold_autoencoder.parameters()) +
        list(model.tab_transformer.parameters())
    )
    other_optimizer = AdamW(other_parameters, lr=2e-4, weight_decay=1e-4)

    # Learning Rate Scheduler (Cosine Annealing with Warmup)
    vqvae_lr_scheduler = combined_lr_scheduler(
        vqvae_optimizer, warmup_epochs, num_epochs, min_lr=1e-6)
    diffusion_lr_scheduler = combined_lr_scheduler(
        diffusion_optimizer, warmup_epochs, num_epochs, min_lr=1e-6)
    other_lr_scheduler = combined_lr_scheduler(
        other_optimizer, warmup_epochs, num_epochs, min_lr=1e-6)

    # EMA Initialization
    ema = EMA(model, decay_start=0.999, decay_end=0.9999)
    ema.register()

    # Initialize lists to store losses per epoch
    train_avg_losses_per_epoch = []
    val_avg_losses_per_epoch = []
    train_recon_losses_per_epoch = []
    val_recon_losses_per_epoch = []
    train_vq_losses_per_epoch = []
    val_vq_losses_per_epoch = []
    train_clip_losses_per_epoch = []
    val_clip_losses_per_epoch = []
    train_kl_losses_per_epoch = []
    val_kl_losses_per_epoch = []
    train_info_nce_losses_per_epoch = []
    val_info_nce_losses_per_epoch = []

    # Training Loop
    for epoch in range(1, num_epochs + 1):
        # Adjust loss weights and training modes based on epoch
        if epoch <= vqvae_epochs:
            train_vqvae_only = True
            training_diffusion_only = False
            # Set loss weights
            classification_loss_weight = 0.0  # No classification during VQ-VAE training
            diffusion_loss_weight = 0.0       # No diffusion loss during VQ-VAE training
            vq_loss_weight = 1.0
            clip_loss_weight = 0.0
            kl_weight = 0.0
            info_nce_loss_weight = 0.0
            # Freeze other model parameters
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze VQ-VAE parameters
            for param in vqvae_parameters:
                param.requires_grad = True
        elif epoch <= vqvae_epochs + diffusion_epochs:
            train_vqvae_only = False
            training_diffusion_only = True
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze diffusion model parameters
            for param in diffusion_parameters:
                param.requires_grad = True
            # Set loss weights
            diffusion_loss_weight = 1.0
            vq_loss_weight = 0.0
            clip_loss_weight = 0.0
            kl_weight = 0.0
            info_nce_loss_weight = 0.0
            classification_loss_weight = 0.0  # No classification during diffusion-only training
        else:
            train_vqvae_only = False
            training_diffusion_only = False
            # Unfreeze all parameters
            for param in model.parameters():
                param.requires_grad = True
            # Set loss weights, gradually increasing
            max_clip_loss_weight = 0.1
            max_kl_weight = 0.05
            max_info_nce_loss_weight = 1.0
            # Compute relative epoch (how many epochs into full model training)
            rel_epoch = epoch - vqvae_epochs - diffusion_epochs
            # Gradually increase the weights
            clip_loss_weight = min(max_clip_loss_weight, rel_epoch * 0.02)
            kl_weight = min(max_kl_weight, rel_epoch * 0.01)
            info_nce_loss_weight = min(max_info_nce_loss_weight, rel_epoch * 0.2)
            diffusion_loss_weight = 1.0
            vq_loss_weight = 0.25
            classification_loss_weight = 1.0  # Enable classification during full model training

        # Training loop
        model.train()
        total_loss = 0.0

        # Initialize loss trackers
        total_diffusion_loss = 0.0
        total_vq_loss = 0.0
        total_clip_loss = 0.0
        total_kl_loss = 0.0
        total_info_nce_loss = 0.0

        # Initialize variables to track accuracy
        total_correct_train = 0
        total_train_samples = 0

        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch}/{num_epochs}")

        for batch in progress_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            descriptions = batch['description']
            categorical_data = batch['categorical'].to(device)
            continuous_data = batch['continuous'].to(device)

            # Zero gradients
            if train_vqvae_only:
                vqvae_optimizer.zero_grad()
            elif training_diffusion_only:
                diffusion_optimizer.zero_grad()
            else:
                vqvae_optimizer.zero_grad()
                diffusion_optimizer.zero_grad()
                other_optimizer.zero_grad()

            # Forward pass
            if train_vqvae_only:
                x_recon, vq_loss = model(images, train_vqvae_only=True)
                diffusion_loss = torch.tensor(0.0, device=device)
                clip_loss = torch.tensor(0.0, device=device)
                kl_loss = torch.tensor(0.0, device=device)
                info_nce_loss = torch.tensor(0.0, device=device)
                classification_loss = torch.tensor(0.0, device=device)
                x0_hat = x_recon  # For metrics calculation
            elif training_diffusion_only:
                # Generate noise and x_noisy
                noise = torch.randn_like(images).to(device)
                timesteps = torch.randint(0, model.diffusion_model.timesteps, (images.size(0),), device=device).long()
                x_noisy = model.q_sample(images, timesteps, noise)

                # Use random latents
                latents = torch.randn(images.size(0), latent_dim, images.size(2) // 16, images.size(3) // 16).to(device)

                # Forward pass through diffusion model
                x_recon = model.diffusion_model(x_noisy, latents, timesteps)
                diffusion_loss = compute_diffusion_loss(x_recon, noise)
                vq_loss = torch.tensor(0.0, device=device)
                clip_loss = torch.tensor(0.0, device=device)
                kl_loss = torch.tensor(0.0, device=device)
                info_nce_loss = torch.tensor(0.0, device=device)
                classification_loss = torch.tensor(0.0, device=device)
                x0_hat = model.p_sample(x_noisy, timesteps, x_recon)
            else:
                # Proceed with all data
                text_embeddings = [sentence_transformer_model.encode(desc, convert_to_tensor=True).to(device) for desc in descriptions]
                text_embeddings = torch.stack(text_embeddings).to(device)

                tabular_data = {
                    'categorical': categorical_data,
                    'continuous': continuous_data
                }

                # Generate noise and x_noisy
                noise = torch.randn_like(images).to(device)
                timesteps = torch.randint(0, model.diffusion_model.timesteps, (images.size(0),), device=device).long()
                x_noisy = model.q_sample(images, timesteps, noise)

                # Forward pass
                x_recon, x0_hat, vq_loss, kl_loss, info_nce_loss, final_output = model(
                    images,
                    x_noisy=x_noisy,
                    timesteps=timesteps,
                    text_embeddings=text_embeddings,
                    tabular_data=tabular_data,
                    labels=labels,
                    visualization_mode=False,
                )

                # Filter out invalid labels (-1)
                valid_indices = labels != -1
                valid_labels = labels[valid_indices]
                valid_final_output = final_output[valid_indices]

                if valid_labels.size(0) > 0:
                    # Compute classification loss
                    classification_loss = F.cross_entropy(valid_final_output, valid_labels)

                    # Compute accuracy
                    _, predicted = torch.max(valid_final_output, 1)
                    correct_predictions = (predicted == valid_labels).sum().item()
                    total_correct_train += correct_predictions
                    total_train_samples += valid_labels.size(0)
                else:
                    classification_loss = torch.tensor(0.0, device=device)

                # Compute other losses
                diffusion_loss = compute_diffusion_loss(x_recon, noise)
                clip_loss = compute_clip_loss(x0_hat, descriptions, model, clip_model, clip_processor, device)
                kl_weight_current = get_kl_loss_weight(epoch, max_epochs)

            # Total loss
            total_loss_value = (
                diffusion_loss_weight * diffusion_loss +
                vq_loss_weight * vq_loss +
                clip_loss_weight * clip_loss +
                kl_weight * kl_loss +
                info_nce_loss_weight * info_nce_loss +
                classification_loss_weight * classification_loss
            )

            # Backward pass and optimization
            total_loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.25)

            if train_vqvae_only:
                vqvae_optimizer.step()
            elif training_diffusion_only:
                diffusion_optimizer.step()
            else:
                vqvae_optimizer.step()
                diffusion_optimizer.step()
                other_optimizer.step()

            ema.update()

            # Metrics calculation
            psnr_value, ssim_value = calculate_metrics(images, x0_hat)

            total_loss += total_loss_value.item()
            total_diffusion_loss += diffusion_loss.item()
            total_vq_loss += vq_loss.item()
            total_clip_loss += clip_loss.item()
            total_kl_loss += kl_loss.item()
            total_info_nce_loss += info_nce_loss.item()

            progress_bar.set_postfix({
                "Loss": total_loss_value.item(),
                "PSNR": psnr_value,
                "SSIM": ssim_value,
                "Class Loss": classification_loss.item(),
                "Diff Loss": diffusion_loss.item(),
                "VQ Loss": vq_loss.item(),
                "Clip Loss": clip_loss.item(),
                "KL Loss": kl_loss.item(),
                "InfoNCE Loss": info_nce_loss.item(),
            })

        # Compute average losses
        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_diffusion_loss = total_diffusion_loss / len(train_dataloader)
        avg_train_vq_loss = total_vq_loss / len(train_dataloader)
        avg_train_clip_loss = total_clip_loss / len(train_dataloader)
        avg_train_kl_loss = total_kl_loss / len(train_dataloader)
        avg_train_info_nce_loss = total_info_nce_loss / len(train_dataloader)

        # Learning Rate Scheduler Step
        if train_vqvae_only:
            vqvae_lr_scheduler.step()
        elif training_diffusion_only:
            diffusion_lr_scheduler.step()
        else:
            vqvae_lr_scheduler.step()
            diffusion_lr_scheduler.step()
            other_lr_scheduler.step()

        # Validation loop
        model.eval()
        ema.apply_shadow()
        val_loss = 0.0

        # Initialize loss trackers for validation
        val_total_diffusion_loss = 0.0
        val_total_vq_loss = 0.0
        val_total_clip_loss = 0.0
        val_total_kl_loss = 0.0
        val_total_info_nce_loss = 0.0

        # Initialize variables to track validation accuracy
        total_correct_val = 0
        total_val_samples = 0

        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, desc=f"Validation Epoch {epoch}/{num_epochs}")
            for batch in progress_bar:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                descriptions = batch['description']
                categorical_data = batch['categorical'].to(device)
                continuous_data = batch['continuous'].to(device)

                if train_vqvae_only:
                    x_recon, vq_loss = model(images, train_vqvae_only=True)
                    diffusion_loss = torch.tensor(0.0, device=device)
                    clip_loss = torch.tensor(0.0, device=device)
                    kl_loss = torch.tensor(0.0, device=device)
                    info_nce_loss = torch.tensor(0.0, device=device)
                    classification_loss = torch.tensor(0.0, device=device)
                    x0_hat = x_recon  # For metrics calculation
                elif training_diffusion_only:
                    # Generate noise and x_noisy
                    noise = torch.randn_like(images).to(device)
                    timesteps = torch.randint(0, model.diffusion_model.timesteps, (images.size(0),), device=device).long()
                    x_noisy = model.q_sample(images, timesteps, noise)

                    # Use random latents
                    latents = torch.randn(images.size(0), latent_dim, images.size(2) // 16, images.size(3) // 16).to(device)

                    # Forward pass through diffusion model
                    x_recon = model.diffusion_model(x_noisy, latents, timesteps)
                    diffusion_loss = compute_diffusion_loss(x_recon, noise)
                    vq_loss = torch.tensor(0.0, device=device)
                    clip_loss = torch.tensor(0.0, device=device)
                    kl_loss = torch.tensor(0.0, device=device)
                    info_nce_loss = torch.tensor(0.0, device=device)
                    classification_loss = torch.tensor(0.0, device=device)
                    x0_hat = model.p_sample(x_noisy, timesteps, x_recon)
                else:
                    # Proceed with all data
                    text_embeddings = [sentence_transformer_model.encode(desc, convert_to_tensor=True).to(device) for desc in descriptions]
                    text_embeddings = torch.stack(text_embeddings).to(device)

                    tabular_data = {
                        'categorical': categorical_data,
                        'continuous': continuous_data
                    }

                    # Generate noise and x_noisy
                    noise = torch.randn_like(images).to(device)
                    timesteps = torch.randint(0, model.diffusion_model.timesteps, (images.size(0),), device=device).long()
                    x_noisy = model.q_sample(images, timesteps, noise)

                    # Forward pass
                    x_recon, x0_hat, vq_loss, kl_loss, info_nce_loss, final_output = model(
                        images,
                        x_noisy=x_noisy,
                        timesteps=timesteps,
                        text_embeddings=text_embeddings,
                        tabular_data=tabular_data,
                        labels=labels,
                        visualization_mode=False,
                    )

                    # Filter out invalid labels (-1)
                    valid_indices = labels != -1
                    valid_labels = labels[valid_indices]
                    valid_final_output = final_output[valid_indices]

                    if valid_labels.size(0) > 0:
                        # Compute classification loss
                        classification_loss = F.cross_entropy(valid_final_output, valid_labels)

                        # Compute accuracy
                        _, predicted = torch.max(valid_final_output, 1)
                        correct_predictions = (predicted == valid_labels).sum().item()
                        total_correct_val += correct_predictions
                        total_val_samples += valid_labels.size(0)
                    else:
                        classification_loss = torch.tensor(0.0, device=device)

                    # Compute other losses
                    diffusion_loss = compute_diffusion_loss(x_recon, noise)
                    clip_loss = compute_clip_loss(x0_hat, descriptions, model, clip_model, clip_processor, device)
                    kl_weight_current = get_kl_loss_weight(epoch, max_epochs)

                # Total loss
                total_loss_value = (
                    diffusion_loss_weight * diffusion_loss +
                    vq_loss_weight * vq_loss +
                    clip_loss_weight * clip_loss +
                    kl_weight_current * kl_loss +  # Use kl_weight_current here
                    info_nce_loss_weight * info_nce_loss +
                    classification_loss_weight * classification_loss
)
                val_loss += total_loss_value.item()
                val_total_diffusion_loss += diffusion_loss.item()
                val_total_vq_loss += vq_loss.item()
                val_total_clip_loss += clip_loss.item()
                val_total_kl_loss += kl_loss.item()
                val_total_info_nce_loss += info_nce_loss.item()

                # Reconstruct images for metrics
                psnr_value, ssim_value = calculate_metrics(images, x0_hat)

                progress_bar.set_postfix({
                    "Loss": total_loss_value.item(),
                    "PSNR": psnr_value,
                    "SSIM": ssim_value,
                    "Class Loss": classification_loss.item(),
                    "Diff Loss": diffusion_loss.item(),
                    "VQ Loss": vq_loss.item(),
                    "Clip Loss": clip_loss.item(),
                    "KL Loss": kl_loss.item(),
                    "InfoNCE Loss": info_nce_loss.item(),
                })

        ema.restore()

        # Compute average validation losses
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_diffusion_loss = val_total_diffusion_loss / len(val_dataloader)
        avg_val_vq_loss = val_total_vq_loss / len(val_dataloader)
        avg_val_clip_loss = val_total_clip_loss / len(val_dataloader)
        avg_val_kl_loss = val_total_kl_loss / len(val_dataloader)
        avg_val_info_nce_loss = val_total_info_nce_loss / len(val_dataloader)

        # Append losses for plotting
        train_avg_losses_per_epoch.append(avg_train_loss)
        val_avg_losses_per_epoch.append(avg_val_loss)
        train_recon_losses_per_epoch.append(avg_train_diffusion_loss)
        val_recon_losses_per_epoch.append(avg_val_diffusion_loss)
        train_vq_losses_per_epoch.append(avg_train_vq_loss)
        val_vq_losses_per_epoch.append(avg_val_vq_loss)
        train_clip_losses_per_epoch.append(avg_train_clip_loss)
        val_clip_losses_per_epoch.append(avg_val_clip_loss)
        train_kl_losses_per_epoch.append(avg_train_kl_loss)
        val_kl_losses_per_epoch.append(avg_val_kl_loss)
        train_info_nce_losses_per_epoch.append(avg_train_info_nce_loss)
        val_info_nce_losses_per_epoch.append(avg_val_info_nce_loss)

        # Save model checkpoints
        torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch}.pth'))

        # Print epoch summary
        print(f'Epoch {epoch}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Val Loss: {avg_val_loss:.6f}')

        # Calculate overall accuracy for the epoch
        train_accuracy = total_correct_train / total_train_samples if total_train_samples > 0 else 0
        val_accuracy = total_correct_val / total_val_samples if total_val_samples > 0 else 0

        # Print the accuracy at the end of the epoch
        print(f"Epoch {epoch}: Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Plot losses
        plot_losses(
            train_avg_losses_per_epoch,
            val_avg_losses_per_epoch,
            train_recon_losses_per_epoch,
            val_recon_losses_per_epoch,
            train_vq_losses_per_epoch,
            val_vq_losses_per_epoch,
            train_clip_losses_per_epoch,
            val_clip_losses_per_epoch,
            train_kl_losses_per_epoch,
            val_kl_losses_per_epoch,
            epoch,
            save_path
        )

        # Visualize reconstructions
        with torch.no_grad():
            model.eval()
            try:
                sample_batch = next(iter(val_dataloader))
            except StopIteration:
                print("Validation dataloader is empty.")
                continue

            sample_images = sample_batch['image'].to(device)
            sample_descriptions = sample_batch['description']
            sample_categorical = sample_batch['categorical'].to(device)
            sample_continuous = sample_batch['continuous'].to(device)
            sample_labels = sample_batch['label'].to(device)

            if train_vqvae_only:
                # Visualize reconstructions from VQ-VAE
                x_recon, _ = model(
                    sample_images,
                    train_vqvae_only=True
                )
                visualize_reconstructions(sample_images, x_recon, epoch, save_path, num_images=5)

            elif training_diffusion_only:
                # Visualize reconstructions from the diffusion model
                latents = torch.randn(sample_images.size(0), latent_dim, 1, 1).to(device)
                latents = latents.expand(-1, -1, sample_images.size(2) // 16, sample_images.size(3) // 16)

                # Simulate forward diffusion (adding noise)
                noise = torch.randn_like(sample_images).to(device)
                timesteps = torch.randint(
                    0, model.diffusion_model.timesteps, (sample_images.size(0),), device=device
                ).long()
                x_noisy = model.q_sample(sample_images, timesteps, noise)

                # Forward pass through the diffusion model
                x_recon = model.diffusion_model(x_noisy, latents, timesteps)
                x0_hat = model.p_sample(x_noisy, timesteps, x_recon)

                visualize_reconstructions(sample_images, x0_hat, epoch, save_path, num_images=5)

            else:
                # Visualize reconstructions using the full model
                # Encode text descriptions
                sample_text_embeddings = [
                    sentence_transformer_model.encode(desc, convert_to_tensor=True).to(device)
                    for desc in sample_descriptions
                ]
                sample_text_embeddings = torch.stack(sample_text_embeddings).to(device)

                sample_tabular_data = {
                    'categorical': sample_categorical,
                    'continuous': sample_continuous
                }

                # Generate noise and x_noisy
                noise = torch.randn_like(sample_images).to(device)
                timesteps = torch.randint(
                    0, model.diffusion_model.timesteps, (sample_images.size(0),), device=device
                ).long()
                x_noisy = model.q_sample(sample_images, timesteps, noise)

                # Forward pass through the full model
                x_recon, x0_hat, _, _, _, _ = model(
                    sample_images,
                    x_noisy=x_noisy,
                    timesteps=timesteps,
                    text_embeddings=sample_text_embeddings,
                    tabular_data=sample_tabular_data,
                    labels=sample_labels,
                    visualization_mode=False,
                )
                visualize_reconstructions(sample_images, x0_hat, epoch, save_path, num_images=5)

if __name__ == "__main__":
    main()
