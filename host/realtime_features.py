#!/usr/bin/env python3
"""
Real-time Feature Extraction for EEG Data
Implements Task 18 from tasks_final.yml

Features:
- Sliding window with 50% overlap (64 sample step)
- Band-pass filtering (0.5-45 Hz) 
- Power spectral density features
- Artifact rejection (eye blinks, muscle artifacts)
- Feature normalization and scaling
"""

import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt, welch
from collections import deque
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class FilteredEEGData:
    """Container for filtered EEG data with metadata"""
    channels: np.ndarray  # Shape: (n_samples, n_channels)
    timestamps: List[float]
    sample_rate: float
    filter_applied: str
    quality_score: float

@dataclass
class EEGFeatures:
    """Container for extracted EEG features"""
    power_features: np.ndarray  # Power spectral density features
    time_features: np.ndarray   # Time domain features
    frequency_bands: Dict[str, np.ndarray]  # Power in frequency bands
    artifact_flags: List[bool]  # Artifact detection flags
    quality_score: float
    extraction_time: float  # Processing time in milliseconds

class RealTimeEEGProcessor:
    """
    Real-time EEG feature extraction and preprocessing
    Optimized for low-latency processing in biometric authentication
    """
    
    def __init__(self, 
                 sample_rate: float = 100.0,
                 n_channels: int = 12,
                 window_size: int = 128,
                 step_size: int = 64,
                 filter_order: int = 4):
        
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.window_size = window_size
        self.step_size = step_size
        self.filter_order = filter_order
        
        # Frequency bands for analysis
        self.frequency_bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 45.0)
        }
        
        # Data buffers
        self.data_buffer = deque(maxlen=window_size * 3)  # Triple buffer for safety
        self.timestamp_buffer = deque(maxlen=window_size * 3)
        
        # Filter coefficients (pre-computed for efficiency)
        self._precompute_filters()
        
        # Artifact detection parameters
        self.artifact_threshold_std = 3.0  # Standard deviations for outlier detection
        self.muscle_artifact_threshold = 100.0  # Threshold for muscle artifacts
        self.eye_blink_threshold = 150.0  # Threshold for eye blink artifacts
        
        # Feature normalization parameters
        self.feature_mean = None
        self.feature_std = None
        self.normalization_window = 100  # Number of windows to use for normalization
        self.feature_history = deque(maxlen=self.normalization_window)
        
        # Performance tracking
        self.processing_times = deque(maxlen=50)
        
        # Logger
        self.logger = logging.getLogger('RealTimeEEGProcessor')
        
        self.logger.info(f"Initialized EEG processor: {sample_rate}Hz, "
                        f"{n_channels} channels, window={window_size}, step={step_size}")
    
    def _precompute_filters(self):
        """Pre-compute filter coefficients for efficiency"""
        nyquist = self.sample_rate / 2.0
        
        # Band-pass filter (0.5-45 Hz)
        low_cutoff = 0.5 / nyquist
        high_cutoff = 45.0 / nyquist
        
        # Ensure cutoff frequencies are valid
        low_cutoff = max(low_cutoff, 0.01)
        high_cutoff = min(high_cutoff, 0.99)
        
        self.bandpass_b, self.bandpass_a = butter(
            self.filter_order, [low_cutoff, high_cutoff], btype='band'
        )
        
        # Notch filter for 50/60 Hz line noise
        notch_freq = 50.0  # Can be changed to 60.0 for US
        quality_factor = 30.0
        notch_w0 = notch_freq / nyquist
        
        self.notch_b, self.notch_a = signal.iirnotch(notch_w0, quality_factor)
        
        self.logger.debug("Filter coefficients computed")
    
    def add_sample(self, eeg_channels: List[float], timestamp: float):
        """
        Add a new EEG sample to the processing buffer
        
        Args:
            eeg_channels: List of 12 EEG channel values
            timestamp: Timestamp of the sample
        """
        if len(eeg_channels) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {len(eeg_channels)}")
        
        # Add to buffers
        self.data_buffer.append(np.array(eeg_channels))
        self.timestamp_buffer.append(timestamp)
    
    def is_ready_for_processing(self) -> bool:
        """Check if enough data is available for feature extraction"""
        return len(self.data_buffer) >= self.window_size
    
    def extract_features(self) -> Optional[EEGFeatures]:
        """
        Extract features from current window with 50% overlap
        
        Returns:
            EEGFeatures object or None if insufficient data
        """
        if not self.is_ready_for_processing():
            return None
        
        start_time = time.time()
        
        try:
            # Get current window of data
            window_data = np.array(list(self.data_buffer)[-self.window_size:])
            window_timestamps = list(self.timestamp_buffer)[-self.window_size:]
            
            # Apply filtering
            filtered_data = self._apply_filters(window_data)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(filtered_data)
            
            # Detect artifacts
            artifact_flags = self._detect_artifacts(filtered_data)
            
            # Extract power spectral density features
            power_features = self._extract_psd_features(filtered_data)
            
            # Extract time domain features
            time_features = self._extract_time_features(filtered_data)
            
            # Extract frequency band powers
            band_powers = self._extract_band_powers(filtered_data)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.processing_times.append(processing_time)
            
            # Create features object
            features = EEGFeatures(
                power_features=power_features,
                time_features=time_features,
                frequency_bands=band_powers,
                artifact_flags=artifact_flags,
                quality_score=quality_score,
                extraction_time=processing_time
            )
            
            # Update normalization parameters
            self._update_normalization(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in feature extraction: {e}")
            return None
    
    def _apply_filters(self, data: np.ndarray) -> np.ndarray:
        """
        Apply band-pass and notch filters to EEG data
        
        Args:
            data: EEG data (samples x channels)
            
        Returns:
            Filtered EEG data
        """
        filtered_data = data.copy()
        
        # Apply filters to each channel
        for ch in range(self.n_channels):
            channel_data = filtered_data[:, ch]
            
            # Band-pass filter (0.5-45 Hz)
            try:
                channel_data = filtfilt(self.bandpass_b, self.bandpass_a, channel_data)
            except Exception as e:
                self.logger.warning(f"Band-pass filter failed for channel {ch}: {e}")
            
            # Notch filter (50/60 Hz)
            try:
                channel_data = filtfilt(self.notch_b, self.notch_a, channel_data)
            except Exception as e:
                self.logger.warning(f"Notch filter failed for channel {ch}: {e}")
            
            filtered_data[:, ch] = channel_data
        
        return filtered_data
    
    def _calculate_quality_score(self, data: np.ndarray) -> float:
        """
        Calculate signal quality score based on noise and artifacts
        
        Args:
            data: Filtered EEG data
            
        Returns:
            Quality score (0.0 to 1.0, higher is better)
        """
        try:
            # Calculate signal-to-noise ratio estimates
            signal_power = np.mean(np.var(data, axis=0))
            
            # Check for excessive noise (high frequency content)
            high_freq_power = 0.0
            for ch in range(self.n_channels):
                freqs, psd = welch(data[:, ch], fs=self.sample_rate, nperseg=min(64, len(data)))
                high_freq_indices = freqs > 30  # Above 30 Hz
                high_freq_power += np.mean(psd[high_freq_indices])
            
            high_freq_power /= self.n_channels
            
            # Calculate quality score (inverse relationship with noise)
            if signal_power > 0:
                snr_estimate = signal_power / (high_freq_power + 1e-10)
                quality_score = min(1.0, max(0.0, np.log10(snr_estimate + 1) / 3.0))
            else:
                quality_score = 0.0
            
            return quality_score
            
        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return 0.5  # Default moderate quality
    
    def _detect_artifacts(self, data: np.ndarray) -> List[bool]:
        """
        Detect artifacts in EEG data using multiple methods
        
        Args:
            data: Filtered EEG data
            
        Returns:
            List of artifact flags for each channel
        """
        artifact_flags = []
        
        for ch in range(self.n_channels):
            channel_data = data[:, ch]
            is_artifact = False
            
            # Method 1: Amplitude-based detection
            channel_std = np.std(channel_data)
            channel_max = np.max(np.abs(channel_data))
            
            if channel_max > self.artifact_threshold_std * channel_std:
                is_artifact = True
            
            # Method 2: Muscle artifact detection (high frequency power)
            try:
                freqs, psd = welch(channel_data, fs=self.sample_rate, nperseg=min(32, len(channel_data)))
                muscle_band_power = np.mean(psd[(freqs >= 20) & (freqs <= 45)])
                
                if muscle_band_power > self.muscle_artifact_threshold:
                    is_artifact = True
            except:
                pass
            
            # Method 3: Eye blink detection (low frequency, high amplitude)
            try:
                low_freq_power = np.mean(psd[(freqs >= 0.5) & (freqs <= 4)])
                if low_freq_power > self.eye_blink_threshold and channel_max > 100:
                    is_artifact = True
            except:
                pass
            
            artifact_flags.append(is_artifact)
        
        return artifact_flags
    
    def _extract_psd_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract power spectral density features
        
        Args:
            data: Filtered EEG data
            
        Returns:
            PSD feature vector
        """
        psd_features = []
        
        for ch in range(self.n_channels):
            channel_data = data[:, ch]
            
            try:
                # Calculate power spectral density
                freqs, psd = welch(
                    channel_data, 
                    fs=self.sample_rate, 
                    nperseg=min(64, len(channel_data)),
                    noverlap=min(32, len(channel_data)//2)
                )
                
                # Extract features in relevant frequency range (0.5-45 Hz)
                freq_mask = (freqs >= 0.5) & (freqs <= 45)
                relevant_psd = psd[freq_mask]
                
                # Features: mean power, peak frequency, spectral centroid
                mean_power = np.mean(relevant_psd)
                peak_freq_idx = np.argmax(relevant_psd)
                peak_freq = freqs[freq_mask][peak_freq_idx]
                spectral_centroid = np.sum(freqs[freq_mask] * relevant_psd) / np.sum(relevant_psd)
                
                channel_features = [mean_power, peak_freq, spectral_centroid]
                psd_features.extend(channel_features)
                
            except Exception as e:
                self.logger.warning(f"PSD extraction failed for channel {ch}: {e}")
                # Use default values
                psd_features.extend([0.0, 10.0, 10.0])
        
        return np.array(psd_features)
    
    def _extract_time_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract time domain features
        
        Args:
            data: Filtered EEG data
            
        Returns:
            Time domain feature vector
        """
        time_features = []
        
        for ch in range(self.n_channels):
            channel_data = data[:, ch]
            
            # Statistical features
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            skewness = self._calculate_skewness(channel_data)
            kurtosis = self._calculate_kurtosis(channel_data)
            
            # Activity features
            zero_crossings = self._count_zero_crossings(channel_data)
            hjorth_activity, hjorth_mobility, hjorth_complexity = self._hjorth_parameters(channel_data)
            
            channel_features = [
                mean_val, std_val, skewness, kurtosis,
                zero_crossings, hjorth_activity, hjorth_mobility, hjorth_complexity
            ]
            time_features.extend(channel_features)
        
        return np.array(time_features)
    
    def _extract_band_powers(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract power in specific frequency bands
        
        Args:
            data: Filtered EEG data
            
        Returns:
            Dictionary of band powers for each channel
        """
        band_powers = {band: [] for band in self.frequency_bands.keys()}
        
        for ch in range(self.n_channels):
            channel_data = data[:, ch]
            
            try:
                freqs, psd = welch(channel_data, fs=self.sample_rate, nperseg=min(64, len(channel_data)))
                
                for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    band_power = np.mean(psd[band_mask]) if np.any(band_mask) else 0.0
                    band_powers[band_name].append(band_power)
                    
            except Exception as e:
                self.logger.warning(f"Band power extraction failed for channel {ch}: {e}")
                for band_name in self.frequency_bands.keys():
                    band_powers[band_name].append(0.0)
        
        # Convert to numpy arrays
        for band_name in band_powers.keys():
            band_powers[band_name] = np.array(band_powers[band_name])
        
        return band_powers
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the data"""
        try:
            from scipy.stats import skew
            return float(skew(data))
        except:
            # Manual calculation if scipy.stats not available
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0.0
            return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the data"""
        try:
            from scipy.stats import kurtosis
            return float(kurtosis(data))
        except:
            # Manual calculation if scipy.stats not available
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0.0
            return np.mean(((data - mean_val) / std_val) ** 4) - 3.0
    
    def _count_zero_crossings(self, data: np.ndarray) -> float:
        """Count zero crossings in the data"""
        zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
        return float(zero_crossings) / len(data)  # Normalize by length
    
    def _hjorth_parameters(self, data: np.ndarray) -> Tuple[float, float, float]:
        """Calculate Hjorth parameters (activity, mobility, complexity)"""
        try:
            # First derivative
            d1 = np.diff(data)
            # Second derivative
            d2 = np.diff(d1)
            
            # Variances
            var_data = np.var(data)
            var_d1 = np.var(d1)
            var_d2 = np.var(d2)
            
            # Hjorth parameters
            activity = var_data
            mobility = np.sqrt(var_d1 / var_data) if var_data > 0 else 0.0
            complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 > 0 and mobility > 0 else 0.0
            
            return float(activity), float(mobility), float(complexity)
            
        except Exception as e:
            self.logger.warning(f"Hjorth parameters calculation failed: {e}")
            return 0.0, 0.0, 0.0
    
    def _update_normalization(self, features: EEGFeatures):
        """Update feature normalization parameters"""
        # Combine all features into a single vector
        all_features = np.concatenate([
            features.power_features,
            features.time_features,
            np.concatenate([band_power for band_power in features.frequency_bands.values()])
        ])
        
        self.feature_history.append(all_features)
        
        # Update normalization parameters if we have enough history
        if len(self.feature_history) >= 10:
            feature_matrix = np.array(list(self.feature_history))
            self.feature_mean = np.mean(feature_matrix, axis=0)
            self.feature_std = np.std(feature_matrix, axis=0)
            self.feature_std[self.feature_std == 0] = 1.0  # Avoid division by zero
    
    def normalize_features(self, features: EEGFeatures) -> np.ndarray:
        """
        Normalize features using running statistics
        
        Args:
            features: EEGFeatures object
            
        Returns:
            Normalized feature vector
        """
        # Combine all features
        all_features = np.concatenate([
            features.power_features,
            features.time_features,
            np.concatenate([band_power for band_power in features.frequency_bands.values()])
        ])
        
        # Normalize if parameters are available
        if self.feature_mean is not None and self.feature_std is not None:
            normalized_features = (all_features - self.feature_mean) / self.feature_std
        else:
            # Use z-score normalization with current data
            mean_val = np.mean(all_features)
            std_val = np.std(all_features)
            if std_val > 0:
                normalized_features = (all_features - mean_val) / std_val
            else:
                normalized_features = all_features
        
        return normalized_features
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the processor"""
        if not self.processing_times:
            return {'avg_processing_time': 0.0, 'max_processing_time': 0.0}
        
        return {
            'avg_processing_time': np.mean(list(self.processing_times)),
            'max_processing_time': np.max(list(self.processing_times)),
            'min_processing_time': np.min(list(self.processing_times)),
            'samples_processed': len(self.processing_times),
            'buffer_size': len(self.data_buffer)
        }
    
    def reset(self):
        """Reset the processor state"""
        self.data_buffer.clear()
        self.timestamp_buffer.clear()
        self.feature_history.clear()
        self.processing_times.clear()
        self.feature_mean = None
        self.feature_std = None
        self.logger.info("Processor state reset")

def demo_real_time_processing():
    """Demo function to test real-time feature extraction"""
    print("=== Real-Time EEG Feature Extraction Demo ===")
    
    # Initialize processor
    processor = RealTimeEEGProcessor(
        sample_rate=100.0,
        n_channels=12,
        window_size=128,
        step_size=64
    )
    
    # Simulate real-time data
    print("Simulating real-time EEG data...")
    
    for i in range(200):  # Simulate 200 samples
        # Generate synthetic EEG data (12 channels)
        timestamp = time.time()
        
        # Create realistic EEG-like signals with different frequencies
        t = i / 100.0  # Time in seconds
        eeg_channels = []
        
        for ch in range(12):
            # Mix of different frequency components
            signal_val = (
                2.0 * np.sin(2 * np.pi * 10 * t + ch * 0.5) +  # Alpha wave
                1.5 * np.sin(2 * np.pi * 4 * t + ch * 0.3) +   # Theta wave
                0.8 * np.sin(2 * np.pi * 20 * t + ch * 0.7) +  # Beta wave
                0.3 * np.random.randn()  # Noise
            )
            eeg_channels.append(signal_val)
        
        # Add sample to processor
        processor.add_sample(eeg_channels, timestamp)
        
        # Try to extract features
        if processor.is_ready_for_processing():
            features = processor.extract_features()
            
            if features is not None:
                print(f"Sample {i}: Quality={features.quality_score:.2f}, "
                      f"Processing={features.extraction_time:.1f}ms, "
                      f"Artifacts={sum(features.artifact_flags)}")
                
                # Print band powers for first channel
                band_summary = {band: powers[0] for band, powers in features.frequency_bands.items()}
                print(f"  Band powers (Ch1): {band_summary}")
        
        # Small delay to simulate real-time
        time.sleep(0.01)
    
    # Print final performance metrics
    metrics = processor.get_performance_metrics()
    print(f"\n=== Performance Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demo
    demo_real_time_processing() 