import torch
import torchaudio
try:
    import librosa # pip install librosa
except ImportError:
    librosa = None
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from typing import Dict, Optional
import os
from numba import jit
try:
    from formants_tracker import track_formants_pipeline
except ImportError:
    from .formants_tracker import track_formants_pipeline
    
@jit(nopython=True) 
def _get_top_positions(array_y, n_positions):
    """Get indices of top n_positions values (NumPy implementation)"""
    order = array_y.argsort()
    ranks = order.argsort() #ascending
    top_indexes = np.zeros((n_positions,), dtype=np.int16)
    #print(array_y)
    i = int(n_positions - 1)

    while(i >= 0):
        itemindices = np.where(ranks==(len(array_y)-1-i))
        for itemindex in itemindices:
            if(itemindex.size):
                #print(i, array_y[itemindex], itemindex)
                top_indexes[i] = itemindex[0]
            else:   #for when positions are more than array size
                itemindices2 = np.where(ranks==len(array_y)-1-i+len(array_y) )
                for itemindex2 in itemindices2:
                    #print(i, array_y[itemindex2], itemindex2)
                    top_indexes[i] = itemindex2[0]
            i -= 1

    return top_indexes


@jit(nopython=True)
def _create_hamming_window(length):
    """Create Hamming window using numba for speed"""
    window = np.zeros(length)
    for i in range(length):
        window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / (length - 1))
    return window



@jit(nopython=True)
def _compute_energy_batch(frames):
    """Compute RMS energy for all frames using numba"""
    num_frames, frame_length = frames.shape
    energy = np.zeros(num_frames)
    for i in range(num_frames):
        sum_sq = 0.0
        for j in range(frame_length):
            sum_sq += frames[i, j] ** 2
        energy[i] = np.sqrt(sum_sq / frame_length)
    return energy

def librosa_audio_loader(audio_path: str, sample_rate: int = 16000, mono=True) -> torch.Tensor:
    """librosa audio loader that handles different audio formats - uses librosa since torchaudio isn't working as expected"""
    if librosa is None:
        raise ImportError("librosa is not installed. Install it with 'pip install librosa'")
    wav, sr = librosa.load(audio_path, sr=sample_rate, mono=mono) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
    wav = torch.FloatTensor(wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]

    assert wav.dim() == 2, f"Expected 2D tensor, got {wav.dim()}D tensor for audio: {audio_path}"
    if mono:
        assert wav.shape[0] == 1, f"Expected channel number of 1, got {wav.shape[0]} for audio: {audio_path}"

    return wav


class ProsodicFeatureExtractor:
    """
    Optimized prosodic feature extractor with CUDA support for batch processing.
    Initialize once, then process multiple audio segments efficiently.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 frame_ms: int = 50,
                 hop_ms: int = 16,
                 emphasize_ratio: float = 0.7, 
                 output_normalize: bool = True):
        """
        Initialize the prosodic feature extractor.
        
        Args:
            sample_rate: Target sample rate in Hz
            frame_ms: Frame length in milliseconds
            hop_ms: Hop length in milliseconds
            emphasize_ratio: Pre-emphasis filter coefficient
            output_normalize: Whether to normalize output features
        """
        
        print(f"ProsodicFeatureExtractor initialized")
        
        # Audio parameters
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.frame_length = int(sample_rate * frame_ms / 1000)
        self.hop_length = int(sample_rate * hop_ms / 1000)
        self.emphasize_ratio = emphasize_ratio
        self.max_freq = 6000  # Maximum frequency for formant analysis
        self.power_scale = 10000 # to scale power values to a reasonable range to avoid overflow in int16
        self.smoothing_level = 'medium'  # Default smoothing level for formants, options: ['light', 'medium', 'heavy']

        self.output_normalize = output_normalize
        self.formant_ranges = [
            (80, 400),    # F0 range
            (200, 1200),  # F1 range  
            (800, 3500),  # F2 range
        ]
        # if output_normalize is True:
        self.normal_f0width_abs_max = 50  # will be used for global normalization
        self.normal_power_abs_max = 3 # will be used for global normalization (after sqrt)
        self.normal_snr_abs_max = 50
        
        # Audio storage
        self.audio_tensor = None
        self.audio_path = None
        
        # Pre-compute filter banks and windows for reuse
        self._precompute_filterbanks()
        
        self.resampler = torchaudio.transforms.Resample(
            self.sample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
    
    def _precompute_filterbanks(self):
        """Pre-compute filter banks that don't change between calls"""
        f0_min, f0_max = 30, self.max_freq
        num_filt = 256
        self.NFFT = num_filt * 32
        
        # Pre-compute mel filter banks
        nfilt = num_filt
        low_freq_mel = (2595 * np.log10(1 + (f0_min) / 700))
        high_freq_mel = (2595 * np.log10(1 + (f0_max) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        hz_points = (700 * (10**(mel_points / 2595) - 1))
        bin_indices = np.floor((self.NFFT + 1) * hz_points / self.sample_rate)
        
        n_overlap = int(np.floor(self.NFFT / 2 + 1))
        fbank = np.zeros((nfilt, n_overlap))
        
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin_indices[m - 1])
            f_m = int(bin_indices[m])
            f_m_plus = int(bin_indices[m + 1])
            
            for k in range(f_m_minus, f_m):
                if k < n_overlap:
                    fbank[m - 1, k] = (k - bin_indices[m - 1]) / (bin_indices[m] - bin_indices[m - 1])
            for k in range(f_m, f_m_plus):
                if k < n_overlap:
                    fbank[m - 1, k] = (bin_indices[m + 1] - k) / (bin_indices[m + 1] - bin_indices[m])
        
        # Store pre-computed values
        self.fbank = fbank
        self.hz_bins_mid = hz_points[1:num_filt+1]
        
        # Pre-compute Hamming window
        self.hamming_window = _create_hamming_window(self.frame_length)
    
    @staticmethod
    def _rms_normalize(wav):
        """Normalize audio by RMS"""
        rms = torch.sqrt(torch.mean(wav ** 2))
        if rms > 0:
            wav = wav / rms
        return wav
    
    def _hz_to_mel(self, hz):
        """Convert Hz to mel scale."""
        return 2595 * np.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel):
        """Convert mel scale to Hz."""
        return 700 * (np.power(10, mel / 2595) - 1)
    
    def _extract_formant_descriptors(self, fft_x, fft_y, formants=2, f_min=30, f_max=4000):
        """
        Extract formant descriptors using peaks and valleys approach.
        Based on Extract_formant_descriptors from FormantsExtract.py
        
        Returns:
            tuple: (frequencies, powers, widths, dissonances) for each formant
        """
        freq_idx, power_idx, width_idx, dissonance_idx = 0, 1, 2, 3
        len_of_x = len(fft_x)
        len_of_y = len(fft_y)
        
        #for 4 features
        returno = np.zeros((formants*4,), dtype=np.uint64)

        if(len_of_x!=len_of_y) or (len_of_x<=3):
            #print("Empty Frame")
            return returno, 0

        peak_indices = argrelextrema(fft_y, np.greater, mode='wrap')
        valley_indices = argrelextrema(fft_y, np.less, mode='wrap')
        peak_indices = peak_indices[0]
        peak_fft_x, peak_fft_y = fft_x[peak_indices], fft_y[peak_indices]
        valley_fft_x, valley_fft_y = fft_x[valley_indices], fft_y[valley_indices]

        
        len_of_peaks = len(peak_indices)
        if(len_of_peaks < 1) or (len(valley_indices) < 1):
            #print("Silence")
            return returno, 0


        ground_level = 0
        if (len(valley_fft_y) > 1):
            ground_level = np.max(valley_fft_y)  #range(valleys_y)/2
        if(ground_level<0.000001):
            #Silence
            return returno, ground_level
        
        #add extra valleys at start and end - optimized concatenation
        x_parts = []
        y_parts = []

        if(peak_fft_x[0] < valley_fft_x[0]):
            x_parts.append([f_min/2])
            y_parts.append([ground_level/8])

        x_parts.append(valley_fft_x)
        y_parts.append(valley_fft_y)

        if(peak_fft_x[-1] > valley_fft_x[-1]):
            x_parts.append([f_max+f_min])
            y_parts.append([ground_level/8])

        valley_fft_x = np.concatenate(x_parts)
        valley_fft_y = np.concatenate(y_parts)

        top_peaks_n = formants*2
        #make sure fft has enought points
        
        if(len(peak_fft_y)<(formants+1)):
            return returno, ground_level
        if(len(peak_fft_y)<(top_peaks_n-1)):
            top_peaks_n = len(peak_fft_y) - 1

        tp_indexes = _get_top_positions(peak_fft_y, top_peaks_n) #descending
        dissonance_peak = np.zeros(top_peaks_n)
        biggest_peak_y = peak_fft_y[tp_indexes[0]]
        
        formants_detected = 0

        #calc width and dissonance
        for i in range(0, top_peaks_n):
            
            if(dissonance_peak[i]==0) and (peak_fft_y[tp_indexes[i]] > (biggest_peak_y/16))  and (peak_fft_x[tp_indexes[i]] >= f_min) and (peak_fft_x[tp_indexes[i]] <= f_max) and (formants_detected < formants):
                next_valley = np.min(np.where(valley_fft_x > peak_fft_x[tp_indexes[i]]))
                next_valley_x = valley_fft_x[next_valley]
                next_valley_y = valley_fft_y[next_valley]

                this_peak_gnd_thresh = peak_fft_y[tp_indexes[i]]/4

                
                while(next_valley_y > this_peak_gnd_thresh) and (len(np.where(valley_fft_x > next_valley_x)[0])>0):
                    valley_next_peak_ind = np.where(peak_fft_x > next_valley_x)
                    if(len(valley_next_peak_ind[0])>0):
                        valley_next_peak = np.min(valley_next_peak_ind)
                        if(peak_fft_y[tp_indexes[i]] > peak_fft_y[valley_next_peak]):
                            next_valley = np.min(np.where(valley_fft_x > next_valley_x))
                            next_valley_x = valley_fft_x[next_valley]
                            next_valley_y = valley_fft_y[next_valley]
                        else:
                            break
                    else:
                        break
                    
                    
                            
                prev_valley = np.max(np.where(valley_fft_x < peak_fft_x[tp_indexes[i]]))
                prev_valley_x = valley_fft_x[prev_valley]
                prev_valley_y = valley_fft_y[prev_valley]

                while(prev_valley_y > this_peak_gnd_thresh) and (len(np.where(valley_fft_x < prev_valley_x)[0])>0):
                    valleys_prev_peak_ind = np.where(peak_fft_x < prev_valley)
                    if(len(valleys_prev_peak_ind[0])>0):
                        valley_prev_peak = np.max(valleys_prev_peak_ind)
                        if(peak_fft_y[tp_indexes[i]] > peak_fft_y[valley_prev_peak]):
                            prev_valley = np.max(np.where(valley_fft_x < prev_valley_x))
                            prev_valley_x = valley_fft_x[prev_valley]
                            prev_valley_y = valley_fft_y[prev_valley]
                        else:
                            break
                    else:
                        break


                dissonance_peak[i] = 1
                this_dissonane = 0
                # dissonance calculation: measures ration of the main peak power to the other peaks in the same valley
                for k in range(0, top_peaks_n):
                    if(peak_fft_x[tp_indexes[k]] < next_valley_x) and (peak_fft_x[tp_indexes[k]] > prev_valley_x) and k!=i:
                        dissonance_peak[k] = 1
                        if(np.abs(peak_fft_x[tp_indexes[k]] - peak_fft_x[tp_indexes[i]]) > (peak_fft_x[tp_indexes[i]]/50)):
                            this_dissonane += peak_fft_y[tp_indexes[k]]
                        else:
                            peak_fft_x[tp_indexes[i]] = (peak_fft_x[tp_indexes[i]]+peak_fft_x[tp_indexes[k]])/2
                            peak_fft_y[tp_indexes[i]] = (peak_fft_y[tp_indexes[i]]+peak_fft_y[tp_indexes[k]])/2
                

                this_dissonane = this_dissonane/peak_fft_y[tp_indexes[i]]
                this_width = np.log(next_valley_x)-np.log(prev_valley_x)
                

                returno[freq_idx + (formants_detected*4)] = peak_fft_x[tp_indexes[i]]
                returno[power_idx + (formants_detected*4)] = peak_fft_y[tp_indexes[i]]*self.power_scale # because it's int dtype
                returno[width_idx + (formants_detected*4)] = this_width*10
                returno[dissonance_idx + (formants_detected*4)] = this_dissonane*100
                
                
                formants_detected += 1

                
        return returno, ground_level


    def load_audio(self, audio_path: str, resample: bool = False, loader="torchaudio") -> torch.Tensor:
        """
        Load audio file and store as tensor.
        
        Args:
            audio_path: Path to audio file
            resample: Whether to resample the audio to the target sample rate
            loader: Audio loading library to use ("librosa" or "torchaudio")
        Returns:
            Loaded audio tensor
        """
        if loader=="librosa":
            # Load audio
            waveform = librosa_audio_loader(audio_path, sample_rate=self.sample_rate, mono=True)
        elif loader=="torchaudio":
            # Load audio
            waveform, original_sr = torchaudio.load(audio_path, normalize=True)
            
            # Resample if necessary
            if original_sr != self.sample_rate and not resample:
                raise ValueError(
                    f"Audio sample rate {original_sr} does not match expected {self.sample_rate}. "
                )
                
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr,
                new_freq=self.sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
            waveform = resampler(waveform) # resample all for consistency
        else:
            raise ValueError(f"Unsupported audio loader: {loader}")

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        
        self.audio_tensor = waveform[0]
        self.audio_path = audio_path
        
        return self.audio_tensor

    def get_dummy_features(self, duration_ms: int) -> Dict[str, torch.Tensor]:
        self.audio_tensor = torch.zeros(int(self.sample_rate * duration_ms / 1000))
        
        return self.extract(0, duration_ms)
    
    def get_frames_count(self, signal_length: int) -> int:
        """
        Estimate maximum number of frames for a given waveform length in samples.
        """
        if signal_length <= 0:
            return 0
        # Frame segmentation parameters
        frame_length = self.frame_length
        frame_step = self.hop_length
        
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
        
        if num_frames < 1:
            num_frames =  1
        return num_frames

    def organize_formants(self, formants_accum, ground_accum, energy_values):
        '''
        Given formants_accum, ground_accum and energy_values, return a dictionary with nicely-sorted formants and their features.
        The main purpose is to extract the top formants and their features in a structured way.
        Use the formants_accum matrix to sweep through the frames again and match formants along the time axis.
        '''
        
        num_frames = formants_accum.shape[0]
        ground_max = np.max(ground_accum)

        nz_grounds = ground_accum[np.where(ground_accum > 0)[0]]
        ground_thresh = np.mean(nz_grounds) if len(nz_grounds) > 0 else 0
        
        if (ground_thresh > 0):
            while (ground_max/ground_thresh < 20) or (ground_thresh > 0.0001):
                ground_thresh *= 0.99
                #print(f"Adjusting ground threshold to {ground_thresh}")
        if ground_thresh == 0:
            ground_thresh = 0.00005

        non_zero_frames = np.where(ground_accum > ground_thresh)[0]
        all_energy_mean = np.mean(energy_values) if len(non_zero_frames) > 0 else 0
        nz_energy_mean = np.mean(energy_values[non_zero_frames]) if len(non_zero_frames) > 0 else 0

        
        #print(f"Ground level threshold: {ground_thresh}, All energy mean: {all_energy_mean}, Non-zero energy mean: {nz_energy_mean}", "ground_max", ground_max)

        energy_thresh = nz_energy_mean * 0.1
        if energy_thresh < ground_thresh: energy_thresh = ground_thresh * 2
        
        #print(f"Energy threshold: {energy_thresh}, Ground level threshold: {ground_thresh}", "non-zero ratio", len(non_zero_frames)/num_frames)

        voicing_now = False
        arranged_features = {"f0": np.zeros(num_frames), "f0_width": np.zeros(num_frames), "f1": np.zeros(num_frames), "f2": np.zeros(num_frames), "power": np.zeros(num_frames), "snr": np.zeros(num_frames)}
        segment_start = 0
        segment_end = 0
        for frame_index in range(num_frames):
            if ground_accum[frame_index] > ground_thresh and energy_values[frame_index] > energy_thresh:
                if not voicing_now:
                    voicing_now = True
                    segment_start = frame_index
            else:
                if voicing_now:
                    voicing_now = False
                    segment_end = frame_index
                    if segment_end - segment_start > 1:
                        # Collect formants for this segment
                        segment_formants = formants_accum[segment_start:segment_end]
                        segment_energy = energy_values[segment_start:segment_end]
                        segment_formants = track_formants_pipeline(segment_formants, formants_out=3,
                            smoothing_level=self.smoothing_level,
                            formant_ranges=self.formant_ranges
                            )

                        arranged_features["power"][segment_start:segment_end] = segment_formants[:, 1]/self.power_scale
                        arranged_features["f0"][segment_start:segment_end] = segment_formants[:, 0]
                        arranged_features["f0_width"][segment_start:segment_end] = segment_formants[:, 2]
                        arranged_features["f1"][segment_start:segment_end] = segment_formants[:, 4] # i*4 + 0  = 4
                        arranged_features["f2"][segment_start:segment_end] = segment_formants[:, 8] # i*4 + 0 = 8
                        all_formant_powers = segment_formants[:, 1] + segment_formants[:, 5] + segment_formants[:, 9]
                        arranged_features["snr"][segment_start:segment_end] = np.where(segment_energy == 0, 0, all_formant_powers / segment_energy)/self.power_scale  # Avoid division by zero

        return arranged_features

    def extract(self, start_ms: int = 0, end_ms: int = -1, audio_tensor: torch.Tensor = None, input_normalize: bool = True) -> Dict[str, torch.Tensor]:
        """Optimized feature extraction using vectorized operations and pre-computed values.
        Args:
            start_ms (int): Start time in milliseconds. Use 0 (default) to indicate the absolute start of the audio.
            end_ms (int): End time in milliseconds. Use -1 (default) to indicate the absolute end of the audio.
            audio_tensor (torch.Tensor, optional): Audio tensor to process. It overrides/skip `.load_audio()` routine. If `None`, uses the internally stored audio tensor loaded by  `.load_audio()`.
            input_normalize (bool): Whether to amplitude normalize input audio using RMS normalization `_rms_normalize(...)`

        Returns:
            Dict[str, torch.Tensor]: Extracted features. Total 6 features with keys: 'f0', 'f0_width', 'f1', 'f2', 'power', and 'snr'. Each feature is a 1D tensor of the same length of frames.
        """

        if (audio_tensor is None): 
            audio_tensor = self.audio_tensor
        if audio_tensor is None:
            raise ValueError("No audio loaded. Call load_audio() first.")
        
        if (start_ms==0) and (end_ms==-1):
            # Use full audio length if no specific segment is provided
            audio_segment = audio_tensor
        else:
            # Get audio segment
            start_sample = max(0, int(start_ms * self.sample_rate / 1000)) if start_ms >= 0 else 0
            if (not (end_ms > start_ms)) or (end_ms <= -1):
                end_sample = len(audio_tensor)
            else:
                end_sample = min(len(audio_tensor), int(end_ms * self.sample_rate / 1000))
            
            audio_segment = audio_tensor[start_sample:end_sample]

        # remove channel dim:
        if audio_segment.dim() > 1:
            audio_segment = audio_segment.mean(dim=0, keepdim=True)
            audio_segment = audio_segment.squeeze(0)

        if (input_normalize):
            audio_segment = self._rms_normalize(audio_segment).numpy()
        else:
            audio_segment = audio_segment.numpy()
        
        
        if len(audio_segment) == 0:
            empty = torch.zeros(0)
            return {
                'f0': empty, 'f0_width': empty, 'f1': empty, 
                'f2': empty, 'power': empty, 'snr': empty
            }
        
        # Apply pre-emphasis - vectorized approach
        signal_to_plot = np.empty_like(audio_segment)
        signal_to_plot[0] = audio_segment[0]
        signal_to_plot[1:] = audio_segment[1:] - self.emphasize_ratio * audio_segment[:-1]
        
        # Frame segmentation parameters
        frame_length = self.frame_length
        frame_step = self.hop_length
        signal_length = len(signal_to_plot)
        
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
        
        if num_frames < 1:
            empty = torch.zeros(0)
            return {
                'f0': empty, 'f0_width': empty, 'f1': empty, 
                'f2': empty, 'power': empty, 'snr': empty
            }
        
        # Pad signal
        pad_signal_length = num_frames * frame_step + frame_length
        pad_signal = np.pad(signal_to_plot, (0, pad_signal_length - signal_length), mode='constant')
        
        # Create frame indices - vectorized
        indices = np.arange(0, frame_length)[np.newaxis, :] + np.arange(0, num_frames * frame_step, frame_step)[:, np.newaxis]
        frames = pad_signal[indices]

        
        # Apply Hamming window to all frames at once
        frames *= self.hamming_window[np.newaxis, :]
        
        # Batch FFT computation
        mag_frames = np.abs(np.fft.rfft(frames, self.NFFT))
        pow_frames = (1.0 / self.NFFT) * (mag_frames ** 2)
        
        # Batch filter bank application
        filter_banks = np.dot(pow_frames, self.fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        
        # Batch energy calculation using numba
        energy_accum = _compute_energy_batch(frames)
        
        # Initialize storage
        formants_mem = 7
        formants_accum = np.zeros((num_frames, formants_mem, 4), dtype=np.float64)
        ground_accum = np.zeros(num_frames, dtype=np.float64)
        
        # Process frames - this loop is hard to vectorize due to peak finding complexity
        for frame_index in range(num_frames):
            peak_indexes = argrelextrema(filter_banks[frame_index], np.greater, mode='wrap')[0]
            
            if len(peak_indexes) == 0:
                continue
                
            peak_fft_x = self.hz_bins_mid[peak_indexes]
            peak_fft_y = filter_banks[frame_index][peak_indexes]
            
            # Extract formant descriptors
            formant_data, ground_level = self._extract_formant_descriptors(
                peak_fft_x, peak_fft_y, formants=formants_mem, f_min=30, f_max=self.max_freq
            )
            ground_accum[frame_index] = ground_level
            
            if len(formant_data) >= formants_mem * 4:
                # Sort formants by frequency and assign appropriately
                formant_freqs = []
                formant_powers = []
                formant_widths = []
                formant_dissonances = []
                
                for i in range(formants_mem):
                    freq = formant_data[i * 4 + 0]
                    power = formant_data[i * 4 + 1]
                    width = formant_data[i * 4 + 2]
                    dissonance = formant_data[i * 4 + 3]
                    
                    if freq > 0:
                        formant_freqs.append(freq)
                        formant_powers.append(power)
                        formant_widths.append(width)
                        formant_dissonances.append(dissonance)
                
                if len(formant_freqs) > 0:
                    # Sort by frequency
                    sorted_indices = np.argsort(formant_freqs)
                    
                    for i, idx in enumerate(sorted_indices):
                        freq = formant_freqs[idx]
                        power = formant_powers[idx]
                        width = formant_widths[idx]
                        dissonance = formant_dissonances[idx]
                        if freq > 0:
                            formants_accum[frame_index, i, 0] = freq
                            formants_accum[frame_index, i, 1] = power
                            formants_accum[frame_index, i, 2] = width
                            formants_accum[frame_index, i, 3] = dissonance
        
        # Organize formants
        arranged_features = self.organize_formants(formants_accum, ground_accum, energy_accum)
        
        # Vectorized normalization
        if self.output_normalize:
            # F0: Only normalize non-zero values
            f0_mask = arranged_features['f0'] > 0
            arranged_features['f0'] = np.where(
                f0_mask,
                (arranged_features['f0'] - self.formant_ranges[0][0]) / (self.formant_ranges[0][1] - self.formant_ranges[0][0]),
                0
            )
            
            # F0-width: Only normalize non-zero values  
            f0_width_mask = arranged_features['f0_width'] > 0
            arranged_features['f0_width'] = np.where(
                f0_width_mask,
                np.clip(arranged_features['f0_width'] / self.normal_f0width_abs_max, 0, 1.0),
                0
            )

            # F1: Only normalize non-zero values
            f1_mask = arranged_features['f1'] > 0
            arranged_features['f1'] = np.where(
                f1_mask,
                (arranged_features['f1'] - self.formant_ranges[1][0]) / (self.formant_ranges[1][1] - self.formant_ranges[1][0]),
                0
            )
            
            # F2: Only normalize non-zero values
            f2_mask = arranged_features['f2'] > 0
            arranged_features['f2'] = np.where(
                f2_mask,
                (arranged_features['f2'] - self.formant_ranges[2][0]) / (self.formant_ranges[2][1] - self.formant_ranges[2][0]),
                0
            )
            
            # Power and SNR: Normalize all values
            arranged_features['power'] = np.clip(
                np.sqrt(np.sqrt(arranged_features['power'])) / self.normal_power_abs_max, 0, 1.0
            )
            arranged_features['snr'] = np.clip(
                arranged_features['snr'] / self.normal_snr_abs_max, 0, 1.0
            )
            
            # Handle NaNs vectorized
            for key in arranged_features:
                arranged_features[key] = np.nan_to_num(arranged_features[key], nan=0.0)

        # Assertions remain the same
        assert len(arranged_features['f0']) == num_frames, "Feature length mismatch"
        assert len(arranged_features['f0_width']) == num_frames, "Feature length mismatch"
        assert len(arranged_features['f1']) == num_frames, "Feature length mismatch"
        assert len(arranged_features['f2']) == num_frames, "Feature length mismatch"
        assert len(arranged_features['power']) == num_frames, "Feature length mismatch"
        assert len(arranged_features['snr']) == num_frames, "Feature length mismatch"
        
        return arranged_features

    
    
    def plot_features(self, features: Dict[str, torch.Tensor], title: str = "Prosodic Features",
                     save_path: Optional[str] = None) -> None:
        """Plot all features."""

        
        
        fig, axes = plt.subplots(6, 1, figsize=(12, 16))
        fig.suptitle(title, fontsize=16)
        
        # Create timestamps based on hop length
        timestamps = np.arange(len(features['f0'])) * self.hop_ms / 1000.0
        
        # F0
        f0_plot = features['f0']
        f0_plot[f0_plot == 0] = np.nan
        axes[0].plot(timestamps, f0_plot, 'b-', linewidth=2, label='F0')
        axes[0].set_ylabel('F0 ' + "(norm)" if self.output_normalize else "(Hz)")
        if self.output_normalize: axes[0].set_ylim(0, 1.1)
        else: axes[0].set_ylim(70, 410)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_title('Fundamental Frequency (F0)')
        
        # F0-width
        f0_width_plot = features['f0_width']
        f0_width_plot[f0_width_plot == 0] = np.nan
        axes[1].plot(timestamps, f0_width_plot, 'g-', linewidth=2, label='F0-width')
        axes[1].set_ylabel('F0-width ' + "(norm)" if self.output_normalize else "(Hz)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_title('F0 Bandwidth')
        
        # Formants
        f1_plot = features['f1']
        f2_plot = features['f2']
        f1_plot[f1_plot == 0] = np.nan
        f2_plot[f2_plot == 0] = np.nan
        
        axes[2].plot(timestamps, f1_plot, 'r-', linewidth=2, label='F1', marker='o', markersize=3)
        axes[2].plot(timestamps, f2_plot, 'm-', linewidth=2, label='F2', marker='s', markersize=3)
        axes[2].set_ylabel('Frequency ' + "(norm)" if self.output_normalize else "(Hz)")
        if self.output_normalize: axes[2].set_ylim(0, 1.1)
        else: axes[2].set_ylim(0, self.max_freq)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        axes[2].set_title('Formants (Peaks & Valleys Method)')
        
        # Energy
        axes[3].plot(timestamps, features['power'], 'orange', linewidth=2, label='Energy')
        axes[3].set_ylabel('RMS Energy' + "(norm)" if self.output_normalize else "" )
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        axes[3].set_title('Signal Energy')
        
        # SNR
        snr_plot = features['snr']
        snr_plot[snr_plot == 0] = np.nan
        axes[4].plot(timestamps, snr_plot, 'c-', linewidth=2, label='SNR')
        axes[4].set_ylabel('SNR' + "(norm)" if self.output_normalize else "")
        axes[4].grid(True, alpha=0.3)
        axes[4].legend()
        axes[4].set_title('Signal-to-Noise Ratio')
        
        # All normalized
        def normalize_feature(feat_array, name):
            if self.output_normalize:
                return feat_array.copy()
            else:
                if name in ['f0', 'f0_width', 'f1', 'f2', 'snr']:
                    feat_array[feat_array == 0] = np.nan
                if np.nanmax(feat_array) > np.nanmin(feat_array):
                    return (feat_array - np.nanmin(feat_array)) / (np.nanmax(feat_array) - np.nanmin(feat_array))
                return feat_array
        
        axes[5].plot(timestamps, normalize_feature(features['f0'], 'f0'), 'b-', linewidth=1, label='F0', alpha=0.8)
        axes[5].plot(timestamps, normalize_feature(features['f0_width'], 'f0_width'), 'g-', linewidth=1, label='F0-width', alpha=0.8)
        axes[5].plot(timestamps, normalize_feature(features['f1'], 'f1'), 'r-', linewidth=1, label='F1', alpha=0.8)
        axes[5].plot(timestamps, normalize_feature(features['f2'], 'f2'), 'm-', linewidth=1, label='F2', alpha=0.8)
        axes[5].plot(timestamps, normalize_feature(features['power'], 'power'), 'orange', linewidth=1, label='Power', alpha=0.8)
        axes[5].plot(timestamps, normalize_feature(features['snr'], 'snr'), 'c-', linewidth=1, label='SNR', alpha=0.8)
        
        axes[5].set_xlabel('Time (s)')
        axes[5].set_ylabel('Normalized')
        axes[5].set_ylim(-0.1, 1.1)
        axes[5].grid(True, alpha=0.3)
        axes[5].legend(loc='upper right')
        axes[5].set_title('All Features (Normalized)')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
            

# Example usage
if __name__ == "__main__":
    print("Initializing Peaks & Valleys Prosodic Feature Extractor...")
    extractor = ProsodicFeatureExtractor(
        sample_rate=16000,
        frame_ms=50,
        hop_ms=15,
        output_normalize=True, # Set to True for normalized output (i.e., [0, 1] range for all features)
    )
    
    audio_path = "audio_samples/109867__timkahn__butterfly.wav"
    audio_tensor = extractor.load_audio(audio_path, loader="torchaudio")
    
    segments = [(0, 2000)] # List of (start_ms, end_ms) tuples for segments to process
    
    for i, (start_ms, end_ms) in enumerate(segments):
        print(f"\nProcessing segment {i+1}: {start_ms}-{end_ms}ms")
        
        features = extractor.extract(start_ms, end_ms, input_normalize=True)
        
        print(f"Features extracted for segment {i+1}: Total frames {features["f0"].shape[0]}")
        # explain features
        print ("Preview first and last 5 frames of features:")
        for key, value in features.items():
            print(f"{np.round(value[:5], 3)}...{np.round(value[-5:], 3)}, {key}")


        plot_save_path = f"audio_samples/outputs/plots/peaks_valleys_formants_{os.path.basename(audio_path)}_{i+1}.png"


        extractor.plot_features(
            features, 
            title=f"Peaks & Valleys Formant Tracking: {start_ms}-{end_ms}ms",
            save_path=plot_save_path
        )

    # print timining information
    dummy_duration = 1.0  # seconds
    dummy_features = extractor.get_dummy_features(duration_ms=dummy_duration * 1000)
    print(f"Frames per second: {dummy_features['f0'].shape[0]} frames == estimated: {extractor.get_frames_count(dummy_duration*16000)}")

    print("Done")