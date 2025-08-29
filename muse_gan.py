import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import pretty_midi
import librosa
import soundfile as sf
from typing import List, Tuple, Dict, Optional
import os
import pickle
from scipy.io.wavfile import write
import requests
import zipfile
import io

class MIDIDataset(Dataset):
    """Dataset for MIDI-based music data"""
    
    def __init__(self, midi_files: List[str], n_tracks: int = 4, 
                 n_bars: int = 4, n_steps_per_bar: int = 16, n_pitches: int = 84):
        self.midi_files = midi_files
        self.n_tracks = n_tracks
        self.n_bars = n_bars  
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        self.sequences = self.process_midi_files()
    
    def process_midi_files(self) -> List[np.ndarray]:
        """Convert MIDI files to pianoroll representation"""
        sequences = []
        
        for midi_file in self.midi_files:
            try:
                # Load MIDI file
                pm = pretty_midi.PrettyMIDI(midi_file)
                
                # Convert to pianoroll
                pianoroll = self.midi_to_pianoroll(pm)
                
                if pianoroll is not None:
                    # Split into segments
                    segments = self.split_into_segments(pianoroll)
                    sequences.extend(segments)
                    
            except Exception as e:
                print(f"Error processing {midi_file}: {e}")
        
        print(f"Processed {len(sequences)} musical segments")
        return sequences
    
    def midi_to_pianoroll(self, pm: pretty_midi.PrettyMIDI) -> Optional[np.ndarray]:
        """Convert PrettyMIDI to pianoroll representation"""
        if len(pm.instruments) == 0:
            return None
        
        # Get pianoroll with 16th note resolution (4 beats per bar, 4 steps per beat)
        fs = 4  # 4 steps per beat
        pianoroll = pm.get_piano_roll(fs=fs)
        
        # Transpose to fit our pitch range (C1 to B7)
        if pianoroll.shape[0] < self.n_pitches:
            # Pad with zeros if not enough pitches
            padded = np.zeros((self.n_pitches, pianoroll.shape[1]))
            padded[:pianoroll.shape[0], :] = pianoroll
            pianoroll = padded
        else:
            # Take middle section if too many pitches
            start_pitch = (pianoroll.shape[0] - self.n_pitches) // 2
            pianoroll = pianoroll[start_pitch:start_pitch + self.n_pitches, :]
        
        # Binarize (velocity > 0)
        pianoroll = (pianoroll > 0).astype(np.float32)
        
        return pianoroll
    
    def split_into_segments(self, pianoroll: np.ndarray) -> List[np.ndarray]:
        """Split pianoroll into fixed-size segments"""
        segments = []
        total_steps = self.n_bars * self.n_steps_per_bar
        
        # Split into segments of n_bars
        for i in range(0, pianoroll.shape[1] - total_steps + 1, total_steps // 2):
            segment = pianoroll[:, i:i + total_steps]
            
            if segment.shape[1] == total_steps and np.sum(segment) > 0:  # Has some notes
                # Reshape to (n_pitches, n_bars, n_steps_per_bar)
                segment = segment.reshape(self.n_pitches, self.n_bars, self.n_steps_per_bar)
                segments.append(segment)
        
        return segments
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx])

class MuseGANGenerator(nn.Module):
    """MuseGAN Generator for multi-track music generation"""
    
    def __init__(self, z_dim: int = 32, n_tracks: int = 4, n_bars: int = 4,
                 n_steps_per_bar: int = 16, n_pitches: int = 84):
        super(MuseGANGenerator, self).__init__()
        
        self.z_dim = z_dim
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        
        # Shared temporal structure generator
        self.temporal_generator = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, n_bars * 64),
            nn.BatchNorm1d(n_bars * 64),
            nn.ReLU(True)
        )
        
        # Track-specific generators
        self.track_generators = nn.ModuleList([
            self._build_track_generator() for _ in range(n_tracks)
        ])
        
        # Bar generators
        self.bar_generators = nn.ModuleList([
            self._build_bar_generator() for _ in range(n_bars)
        ])
        
    def _build_track_generator(self):
        """Build generator for individual track"""
        return nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )
    
    def _build_bar_generator(self):
        """Build generator for individual bar"""
        return nn.Sequential(
            nn.ConvTranspose2d(64 + 512, 512, (2, 1), (2, 1)),  # (2, 16) -> (4, 16)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, (2, 1), (2, 1)),  # (4, 16) -> (8, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, (3, 1), (1, 1), (1, 0)),  # (8, 16) -> (10, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, (1, 1), (1, 1)),  # (10, 16) -> (10, 16)
            nn.Sigmoid()
        )
    
    def forward(self, z_temporal, z_tracks):
        """
        z_temporal: (batch_size, z_dim) - temporal structure
        z_tracks: (batch_size, n_tracks, z_dim) - track-specific noise
        """
        batch_size = z_temporal.size(0)
        
        # Generate temporal structure
        temporal_features = self.temporal_generator(z_temporal)  # (batch, n_bars * 64)
        temporal_features = temporal_features.view(batch_size, self.n_bars, 64)
        
        # Generate track-specific features
        track_features = []
        for track_idx in range(self.n_tracks):
            track_z = z_tracks[:, track_idx, :]  # (batch, z_dim)
            track_feat = self.track_generators[track_idx](track_z)  # (batch, 512)
            track_features.append(track_feat)
        
        # Generate music for each track
        generated_tracks = []
        
        for track_idx in range(self.n_tracks):
            track_bars = []
            
            for bar_idx in range(self.n_bars):
                # Combine temporal and track features
                temporal_bar = temporal_features[:, bar_idx, :].unsqueeze(2).unsqueeze(3)  # (batch, 64, 1, 1)
                track_feat = track_features[track_idx].unsqueeze(2).unsqueeze(3)  # (batch, 512, 1, 1)
                
                # Expand to match bar generator input
                temporal_bar = temporal_bar.expand(-1, -1, 2, self.n_steps_per_bar)  # (batch, 64, 2, 16)
                track_feat = track_feat.expand(-1, -1, 2, self.n_steps_per_bar)  # (batch, 512, 2, 16)
                
                # Concatenate features
                combined = torch.cat([temporal_bar, track_feat], dim=1)  # (batch, 576, 2, 16)
                
                # Generate bar
                bar = self.bar_generators[bar_idx](combined)  # (batch, 1, 10, 16)
                
                # Pad or crop to match n_pitches
                if bar.size(2) < self.n_pitches:
                    padding = torch.zeros(batch_size, 1, self.n_pitches - bar.size(2), self.n_steps_per_bar)
                    if bar.is_cuda:
                        padding = padding.cuda()
                    bar = torch.cat([bar, padding], dim=2)
                elif bar.size(2) > self.n_pitches:
                    bar = bar[:, :, :self.n_pitches, :]
                
                track_bars.append(bar.squeeze(1))  # Remove channel dim
            
            # Combine bars for this track
            track_output = torch.stack(track_bars, dim=2)  # (batch, n_pitches, n_bars, n_steps_per_bar)
            generated_tracks.append(track_output)
        
        # Stack tracks
        output = torch.stack(generated_tracks, dim=1)  # (batch, n_tracks, n_pitches, n_bars, n_steps_per_bar)
        
        return output

class MuseGANDiscriminator(nn.Module):
    """MuseGAN Discriminator for multi-track music"""
    
    def __init__(self, n_tracks: int = 4, n_bars: int = 4,
                 n_steps_per_bar: int = 16, n_pitches: int = 84):
        super(MuseGANDiscriminator, self).__init__()
        
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        
        # Track discriminators
        self.track_discriminators = nn.ModuleList([
            self._build_track_discriminator() for _ in range(n_tracks)
        ])
        
        # Bar discriminator
        self.bar_discriminator = self._build_bar_discriminator()
        
        # Overall discriminator
        self.overall_discriminator = nn.Sequential(
            nn.Linear(n_tracks + n_bars, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def _build_track_discriminator(self):
        """Build discriminator for individual track"""
        return nn.Sequential(
            nn.Conv2d(self.n_bars, 128, (1, 12), (1, 1)),  # (n_bars, n_pitches, n_steps_per_bar) -> (128, ?, ?)
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def _build_bar_discriminator(self):
        """Build discriminator for individual bar"""
        return nn.Sequential(
            nn.Conv2d(self.n_tracks, 128, (1, 12), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: (batch, n_tracks, n_pitches, n_bars, n_steps_per_bar)
        """
        batch_size = x.size(0)
        
        # Track-level discrimination
        track_scores = []
        for track_idx in range(self.n_tracks):
            # Get track data: (batch, n_pitches, n_bars, n_steps_per_bar)
            track_data = x[:, track_idx, :, :, :]
            # Reshape for conv2d: (batch, n_bars, n_pitches, n_steps_per_bar)
            track_data = track_data.permute(0, 2, 1, 3)
            track_score = self.track_discriminators[track_idx](track_data)
            track_scores.append(track_score)
        
        # Bar-level discrimination
        bar_scores = []
        for bar_idx in range(self.n_bars):
            # Get bar data: (batch, n_tracks, n_pitches, n_steps_per_bar)
            bar_data = x[:, :, :, bar_idx, :]
            bar_score = self.bar_discriminator(bar_data)
            bar_scores.append(bar_score)
        
        # Combine all scores
        all_scores = torch.cat(track_scores + bar_scores, dim=1)  # (batch, n_tracks + n_bars)
        
        # Overall discrimination
        overall_score = self.overall_discriminator(all_scores)
        
        return overall_score

class MusicDataLoader:
    """Load music data from various sources"""
    
    @staticmethod
    def download_lmd_matched_subset():
        """Download Lakh MIDI Dataset matched subset (small sample)"""
        print("Downloading sample MIDI files...")
        
        # Create directory
        os.makedirs("midi_data", exist_ok=True)
        
        # Sample MIDI URLs (public domain classical music)
        midi_urls = [
            "https://www.mfiles.co.uk/downloads/edvard-grieg-in-the-hall-of-the-mountain-king.mid",
            "https://www.mfiles.co.uk/downloads/johann-pachelbel-canon-in-d-major.mid",
            "https://www.mfiles.co.uk/downloads/wolfgang-amadeus-mozart-eine-kleine-nachtmusik-1st-movement.mid",
            "https://www.mfiles.co.uk/downloads/ludwig-van-beethoven-symphony-no-5-1st-movement.mid",
        ]
        
        downloaded_files = []
        
        for i, url in enumerate(midi_urls):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    filename = f"midi_data/classical_{i+1}.mid"
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    downloaded_files.append(filename)
                    print(f"Downloaded: {filename}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")
        
        return downloaded_files
    
    @staticmethod
    def create_synthetic_midi(num_files: int = 10):
        """Create synthetic MIDI files for testing"""
        print(f"Creating {num_files} synthetic MIDI files...")
        
        os.makedirs("midi_data", exist_ok=True)
        created_files = []
        
        for i in range(num_files):
            # Create a simple MIDI file
            pm = pretty_midi.PrettyMIDI()
            
            # Create instrument
            instrument = pretty_midi.Instrument(program=1)  # Piano
            
            # Generate simple melody
            notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
            start_time = 0
            
            for j in range(16):  # 16 notes
                note_num = notes[j % len(notes)] + (12 * (j // 8))  # Change octave every 8 notes
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=note_num,
                    start=start_time,
                    end=start_time + 0.5
                )
                instrument.notes.append(note)
                start_time += 0.5
            
            pm.instruments.append(instrument)
            
            # Save MIDI file
            filename = f"midi_data/synthetic_{i+1:03d}.mid"
            pm.write(filename)
            created_files.append(filename)
        
        print(f"Created {len(created_files)} synthetic MIDI files")
        return created_files

class MuseGANTrainer:
    """Main training class for MuseGAN"""
    
    def __init__(self, n_tracks: int = 4, n_bars: int = 4, 
                 n_steps_per_bar: int = 16, n_pitches: int = 84,
                 z_dim: int = 32, device: str = 'cpu'):
        
        self.device = device
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        self.z_dim = z_dim
        
        # Initialize models
        self.generator = MuseGANGenerator(
            z_dim=z_dim,
            n_tracks=n_tracks,
            n_bars=n_bars,
            n_steps_per_bar=n_steps_per_bar,
            n_pitches=n_pitches
        ).to(device)
        
        self.discriminator = MuseGANDiscriminator(
            n_tracks=n_tracks,
            n_bars=n_bars,
            n_steps_per_bar=n_steps_per_bar,
            n_pitches=n_pitches
        ).to(device)
        
        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss
        self.criterion = nn.BCELoss()
        
        # Training history
        self.losses_G = []
        self.losses_D = []
    
    def train_step(self, real_data):
        """Single training step"""
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
        z_temporal = torch.randn(batch_size, self.z_dim).to(self.device)
        z_tracks = torch.randn(batch_size, self.n_tracks, self.z_dim).to(self.device)
        
        fake_data = self.generator(z_temporal, z_tracks)
        fake_output = self.discriminator(fake_data.detach())
        fake_loss = self.criterion(fake_output, fake_labels)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.optimizer_D.step()
        
        # Train Generator
        self.optimizer_G.zero_grad()
        
        fake_output = self.discriminator(fake_data)
        g_loss = self.criterion(fake_output, real_labels)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return d_loss.item(), g_loss.item()
    
    def train(self, dataloader, epochs: int = 100, print_every: int = 10):
        """Train MuseGAN"""
        print(f"Starting MuseGAN training for {epochs} epochs...")
        
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
                
                # Generate sample
                self.generate_sample_midi(f"sample_epoch_{epoch+1}.mid")
    
    def generate_music(self, num_samples: int = 1, temperature: float = 1.0):
        """Generate new music"""
        self.generator.eval()
        
        with torch.no_grad():
            z_temporal = torch.randn(num_samples, self.z_dim).to(self.device) * temperature
            z_tracks = torch.randn(num_samples, self.n_tracks, self.z_dim).to(self.device) * temperature
            
            generated = self.generator(z_temporal, z_tracks)
        
        self.generator.train()
        return generated.cpu().numpy()
    
    def generate_sample_midi(self, filename: str):
        """Generate and save a sample MIDI file"""
        generated = self.generate_music(1)
        self.pianoroll_to_midi(generated[0], filename)
        print(f"Generated sample: {filename}")
    
    def pianoroll_to_midi(self, pianoroll, filename: str):
        """Convert pianoroll to MIDI file"""
        # pianoroll shape: (n_tracks, n_pitches, n_bars, n_steps_per_bar)
        pm = pretty_midi.PrettyMIDI()
        
        # Time resolution
        beat_resolution = 0.25  # 16th notes
        
        for track_idx in range(self.n_tracks):
            instrument = pretty_midi.Instrument(program=track_idx)
            
            track_data = pianoroll[track_idx]  # (n_pitches, n_bars, n_steps_per_bar)
            
            for bar_idx in range(self.n_bars):
                for step_idx in range(self.n_steps_per_bar):
                    for pitch_idx in range(self.n_pitches):
                        if track_data[pitch_idx, bar_idx, step_idx] > 0.5:  # Note is active
                            pitch = pitch_idx + 24  # Start from C2
                            start_time = (bar_idx * self.n_steps_per_bar + step_idx) * beat_resolution
                            end_time = start_time + beat_resolution
                            
                            note = pretty_midi.Note(
                                velocity=80,
                                pitch=pitch,
                                start=start_time,
                                end=end_time
                            )
                            instrument.notes.append(note)
            
            if len(instrument.notes) > 0:  # Only add non-empty tracks
                pm.instruments.append(instrument)
        
        pm.write(filename)
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'losses_G': self.losses_G,
            'losses_D': self.losses_D,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def plot_training_history(self):
        """Plot training losses"""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.losses_D, label='Discriminator Loss', alpha=0.7)
        plt.plot(self.losses_G, label='Generator Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        if len(self.losses_G) > 10:
            # Moving average for smoother visualization
            window = 5
            smooth_d = np.convolve(self.losses_D, np.ones(window)/window, mode='valid')
            smooth_g = np.convolve(self.losses_G, np.ones(window)/window, mode='valid')
            plt.plot(smooth_d, label='Discriminator (smoothed)', alpha=0.7)
            plt.plot(smooth_g, label='Generator (smoothed)', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Loss (Smoothed)')
            plt.title('Smoothed Training Losses')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

def quick_start_musegan():
    """Quick start function to train MuseGAN"""
    print("=== MuseGAN Music Generator - Quick Start ===\n")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get data
    print("\n1. Loading music data...")
    
    # Option 1: Download classical MIDI files
    midi_files = MusicDataLoader.download_lmd_matched_subset()
    
    # Option 2: Create synthetic data if download fails
    if len(midi_files) == 0:
        print("Download failed, creating synthetic MIDI data...")
        midi_files = MusicDataLoader.create_synthetic_midi(20)
    
    print(f"Loaded {len(midi_files)} MIDI files")
    
    # Create dataset
    print("\n2. Processing MIDI data...")
    dataset = MIDIDataset(midi_files, n_tracks=4, n_bars=2, n_steps_per_bar=16, n_pitches=84)
    
    if len(dataset) == 0:
        print("No valid musical segments found. Creating more synthetic data...")
        midi_files = MusicDataLoader.create_synthetic_midi(50)
        dataset = MIDIDataset(midi_files, n_tracks=4, n_bars=2, n_steps_per_bar=16, n_pitches=84)
    
    print(f"Dataset contains {len(dataset)} musical segments")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Initialize trainer
    print("\n3. Initializing MuseGAN...")
    trainer = MuseGANTrainer(
        n_tracks=4,
        n_bars=2,
        n_steps_per_bar=16,
        n_pitches=84,
        z_dim=32,
        device=device
    )
    
    # Train
    print("\n4. Training MuseGAN...")
    trainer.train(dataloader, epochs=50, print_every=10)
    
    # Generate music
    print("\n5. Generating new music...")
    generated_music = trainer.generate_music(num_samples=3)
    
    print(f"Generated music shape: {generated_music.shape}")
    
    # Save generated music as MIDI files
    for i, music in enumerate(generated_music):
        filename = f"generated_music_{i+1}.mid"
        trainer.pianoroll_to_midi(music, filename)
        print(f"Saved: {filename}")
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save model
    trainer.save_model("musegan_model.pth")
    
    print("\n✅ Training complete!")
    print("Generated MIDI files are ready to play!")
    
    return trainer

if __name__ == "__main__":
    # Run the complete MuseGAN training pipeline
    trained_model = quick_start_musegan()
    
    print("\nTo generate more music:")
    print("music = trained_model.generate_music(num_samples=5)")
    print("\nTo convert to audio:")
    print("Use a DAW or online MIDI player to play the generated .mid files")
