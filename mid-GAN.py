import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import pickle
from typing import List, Tuple, Dict
import librosa
import soundfile as sf

class CarnaticMusicDataset(Dataset):
    """Dataset class for Carnatic music sequences"""
    
    def __init__(self, sequences: List[np.ndarray], sequence_length: int = 128):
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.processed_sequences = self._process_sequences()
    
    def _process_sequences(self) -> List[torch.Tensor]:
        processed = []
        for seq in self.sequences:
            # Normalize sequence to [-1, 1] range
            normalized = 2 * (seq - seq.min()) / (seq.max() - seq.min()) - 1
            
            # Create overlapping windows
            for i in range(0, len(normalized) - self.sequence_length, self.sequence_length // 2):
                window = normalized[i:i + self.sequence_length]
                if len(window) == self.sequence_length:
                    processed.append(torch.FloatTensor(window).unsqueeze(0))  # Add channel dimension
        
        return processed
    
    def __len__(self):
        return len(self.processed_sequences)
    
    def __getitem__(self, idx):
        return self.processed_sequences[idx]

class CarnaticGenerator(nn.Module):
    """Generator network for Carnatic music"""
    
    def __init__(self, noise_dim: int = 100, sequence_length: int = 128, 
                 hidden_dims: List[int] = [256, 512, 1024]):
        super(CarnaticGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.sequence_length = sequence_length
        
        # Calculate the initial size after first linear layer
        self.init_size = sequence_length // 8  # We'll upsample 3 times (2^3 = 8)
        
        # Initial linear layer
        self.fc = nn.Linear(noise_dim, hidden_dims[0] * self.init_size)
        
        # Convolutional transpose layers for upsampling
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose1d(hidden_dims[0], hidden_dims[1], 4, 2, 1),  # 2x upsample
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(True),
            
            nn.ConvTranspose1d(hidden_dims[1], hidden_dims[2], 4, 2, 1),  # 2x upsample
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(True),
            
            nn.ConvTranspose1d(hidden_dims[2], hidden_dims[1], 4, 2, 1),  # 2x upsample
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(True),
            
            # Final layer to get single channel output
            nn.Conv1d(hidden_dims[1], 1, 7, 1, 3),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
    def forward(self, noise):
        batch_size = noise.size(0)
        
        # Linear transformation
        x = self.fc(noise)
        x = x.view(batch_size, -1, self.init_size)
        
        # Convolutional upsampling
        x = self.conv_transpose(x)
        
        # Ensure exact sequence length
        if x.size(2) != self.sequence_length:
            x = nn.functional.interpolate(x, size=self.sequence_length, mode='linear', align_corners=False)
        
        return x

class CarnaticDiscriminator(nn.Module):
    """Discriminator network for Carnatic music"""
    
    def __init__(self, sequence_length: int = 128, hidden_dims: List[int] = [64, 128, 256, 512]):
        super(CarnaticDiscriminator, self).__init__()
        
        # Convolutional layers for feature extraction
        layers = []
        in_channels = 1
        
        for i, out_channels in enumerate(hidden_dims):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            if i > 0:  # No batch norm for first layer
                layers.insert(-1, nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate the size after conv layers
        conv_output_size = sequence_length // (2 ** len(hidden_dims))
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1] * conv_output_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

class CarnaticMusicGAN:
    """Complete GAN system for generating Carnatic music"""
    
    def __init__(self, noise_dim: int = 100, sequence_length: int = 128, 
                 learning_rate: float = 0.0002, device: str = 'cpu'):
        self.device = device
        self.noise_dim = noise_dim
        self.sequence_length = sequence_length
        
        # Initialize networks
        self.generator = CarnaticGenerator(noise_dim, sequence_length).to(device)
        self.discriminator = CarnaticDiscriminator(sequence_length).to(device)
        
        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.losses_G = []
        self.losses_D = []
    
    def train_step(self, real_data):
        batch_size = real_data.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # Train Discriminator
        self.optimizer_D.zero_grad()
        
        # Real data
        real_output = self.discriminator(real_data)
        real_loss = self.criterion(real_output, real_labels)
        
        # Fake data
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        fake_data = self.generator(noise)
        fake_output = self.discriminator(fake_data.detach())
        fake_loss = self.criterion(fake_output, fake_labels)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.optimizer_D.step()
        
        # Train Generator
        self.optimizer_G.zero_grad()
        
        fake_output = self.discriminator(fake_data)
        g_loss = self.criterion(fake_output, real_labels)  # Generator wants discriminator to think fake is real
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return d_loss.item(), g_loss.item()
    
    def train(self, dataloader, epochs: int = 100, print_every: int = 10):
        """Train the GAN"""
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0
            
            for batch_idx, real_data in enumerate(dataloader):
                real_data = real_data.to(self.device)
                
                d_loss, g_loss = self.train_step(real_data)
                
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
            
            # Average losses
            avg_d_loss = epoch_d_loss / len(dataloader)
            avg_g_loss = epoch_g_loss / len(dataloader)
            
            self.losses_D.append(avg_d_loss)
            self.losses_G.append(avg_g_loss)
            
            if (epoch + 1) % print_every == 0:
                print(f'Epoch [{epoch+1}/{epochs}], D_Loss: {avg_d_loss:.4f}, G_Loss: {avg_g_loss:.4f}')
    
    def generate_music(self, num_samples: int = 5, temperature: float = 1.0):
        """Generate new Carnatic music sequences"""
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.noise_dim).to(self.device) * temperature
            generated = self.generator(noise)
            
        return generated.cpu().numpy()
    
    def plot_training_history(self):
        """Plot training losses"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.losses_D, label='Discriminator Loss')
        plt.plot(self.losses_G, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.losses_D, label='Discriminator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Discriminator Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'losses_G': self.losses_G,
            'losses_D': self.losses_D,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.losses_G = checkpoint['losses_G']
        self.losses_D = checkpoint['losses_D']
        
        print(f"Model loaded from {path}")

class CarnaticMusicPreprocessor:
    """Utility class for preprocessing Carnatic music data"""
    
    @staticmethod
    def extract_features_from_audio(file_path: str, sr: int = 22050) -> np.ndarray:
        """Extract musical features from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=sr)
            
            # Extract features commonly used in Indian classical music analysis
            # Pitch (fundamental frequency) - very important for Carnatic music
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            pitch_track = []
            
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                pitch_track.append(pitch if pitch > 0 else 0)
            
            pitch_track = np.array(pitch_track)
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            # MFCCs (timbral features)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=0)
            
            # Chroma features (pitch class profiles)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=0)
            
            # Combine features
            features = np.concatenate([
                pitch_track[:len(spectral_centroid)],  # Match lengths
                spectral_centroid,
                mfcc_mean[:len(spectral_centroid)],
                np.repeat(chroma_mean, len(spectral_centroid) // len(chroma_mean) + 1)[:len(spectral_centroid)]
            ])
            
            return features
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return np.array([])
    
    @staticmethod
    def sequences_to_audio(sequences: np.ndarray, sr: int = 22050, 
                          duration_per_sequence: float = 2.0) -> np.ndarray:
        """Convert generated sequences back to audio (simplified approach)"""
        audio_data = []
        
        for seq in sequences:
            # Simple approach: treat sequence as amplitude modulation of sine waves
            t = np.linspace(0, duration_per_sequence, int(sr * duration_per_sequence))
            
            # Use sequence values to modulate frequency
            freq_base = 220  # A3 as base frequency
            freq_modulation = seq.flatten()[:len(t)] if len(seq.flatten()) >= len(t) else np.tile(seq.flatten(), len(t) // len(seq.flatten()) + 1)[:len(t)]
            
            # Generate audio
            frequencies = freq_base * (1 + 0.5 * freq_modulation)
            audio = 0.3 * np.sin(2 * np.pi * frequencies * t)
            
            audio_data.extend(audio)
        
        return np.array(audio_data)

# Example usage and training script
def example_usage():
    """Example of how to use the Carnatic Music GAN"""
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dummy data for demonstration (replace with real Carnatic music data)
    print("Creating dummy dataset...")
    dummy_sequences = []
    for i in range(100):  # 100 sample sequences
        # Generate sequences that mimic some musical patterns
        sequence = np.sin(np.linspace(0, 4*np.pi, 128)) + 0.5 * np.sin(np.linspace(0, 8*np.pi, 128))
        sequence += 0.1 * np.random.randn(128)  # Add some noise
        dummy_sequences.append(sequence)
    
    # Create dataset and dataloader
    dataset = CarnaticMusicDataset(dummy_sequences, sequence_length=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize GAN
    print("Initializing GAN...")
    gan = CarnaticMusicGAN(
        noise_dim=100,
        sequence_length=128,
        learning_rate=0.0002,
        device=device
    )
    
    # Train the model
    print("Starting training...")
    gan.train(dataloader, epochs=50, print_every=10)
    
    # Generate new music
    print("Generating new Carnatic music sequences...")
    generated_music = gan.generate_music(num_samples=5)
    
    print(f"Generated {generated_music.shape[0]} sequences of length {generated_music.shape[2]}")
    
    # Plot training history
    gan.plot_training_history()
    
    # Save the model
    gan.save_model('carnatic_music_gan.pth')
    
    return gan, generated_music

if __name__ == "__main__":
    # Run example
    trained_gan, generated_sequences = example_usage()
    
    print("\nTo use with real Carnatic music data:")
    print("1. Collect Carnatic music audio files")
    print("2. Use CarnaticMusicPreprocessor.extract_features_from_audio() to process them")
    print("3. Create sequences from the extracted features")
    print("4. Train the GAN with your real data")
    print("5. Generate new Carnatic music sequences")
