
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter



def track_formants____no_smoothing(segment_formants, formants_out=3): # old version, not used anymore
    '''
    This function aligns the formants frame-by-frame, interpolates missing formants 
    for up to at max 2 frames, avoids duplicates, jumps, skips in formants.
    Now utilizes formant_tracks for continuity analysis.
    
    Args:
        segment_formants: numpy array of shape (num_frames, max_formants, 4)
                        where 4 features are [freq, power, width, dissonance]
        formants_out: number of output formants to track (default 3 for F0, F1, F2)
    
    Returns:
        numpy array of shape (num_frames, formants_out * 4) with aligned formants
    '''
    num_frames = segment_formants.shape[0]
    max_formants = segment_formants.shape[1]
    return_array = np.zeros((num_frames, formants_out * 4), dtype=np.float64)
    
    if num_frames == 0:
        return return_array
    
    formant_ranges = [
            (80, 400),    # F0 range
            (200, 1200),  # F1 range  
            (800, 3500),  # F2 range
        ]
    
    # Track formant continuity across frames - now actually used!
    formant_tracks = [[] for _ in range(formants_out)]  # Store full history of each track
    missing_count = [0 for _ in range(formants_out)]
    last_valid = [None for _ in range(formants_out)]
    
    # Parameters for track analysis
    power_scale = 10000  # Scale factor for power values
    track_history_length = 5  # How many recent frames to consider for trend analysis
    
    for frame_idx in range(num_frames):
        frame_formants = segment_formants[frame_idx]  # shape: (max_formants, 4)
        
        # Extract valid formants (non-zero frequency)
        valid_formants = []
        for f_idx in range(max_formants):
            freq = frame_formants[f_idx, 0]
            if freq > 0:
                valid_formants.append({
                    'freq': freq,
                    'power': frame_formants[f_idx, 1],
                    'width': frame_formants[f_idx, 2], 
                    'dissonance': frame_formants[f_idx, 3],
                    'index': f_idx,
                    'frame': frame_idx
                })
        
        # Sort by frequency
        valid_formants.sort(key=lambda x: x['freq'])
        
        # Assign formants to tracks based on frequency ranges and continuity
        frame_assignments = [None for _ in range(formants_out)]
        
        for target_idx in range(formants_out):
            f_min, f_max = formant_ranges[target_idx]
            best_candidate = None
            best_score = float('inf')
            
            # Find candidates in frequency range
            candidates = [f for f in valid_formants if f_min <= f['freq'] <= f_max]
            
            if candidates:
                for candidate in candidates:
                    score = 0
                    
                    # Continuity scoring using track history
                    if len(formant_tracks[target_idx]) > 0:
                        # Get recent track history for trend analysis
                        recent_track = formant_tracks[target_idx][-track_history_length:]
                        
                        # Frequency continuity - penalize jumps
                        last_freq = recent_track[-1]['freq']
                        freq_diff = abs(candidate['freq'] - last_freq)
                        score += freq_diff / 100.0
                        
                        # Trend analysis - predict expected frequency based on recent trend
                        if len(recent_track) >= 3:
                            # Calculate frequency trend (simple linear regression)
                            recent_freqs = [item['freq'] for item in recent_track]
                            trend_slope = (recent_freqs[-1] - recent_freqs[0]) / len(recent_freqs)
                            predicted_freq = recent_freqs[-1] + trend_slope
                            
                            # Penalize candidates that deviate from predicted trend
                            trend_deviation = abs(candidate['freq'] - predicted_freq)
                            score += trend_deviation / 50.0
                        
                        # Power continuity
                        last_power = recent_track[-1]['power']
                        power_diff = abs(candidate['power'] - last_power)
                        score += power_diff / power_scale
                        
                        # Stability bonus - reward consistent formants
                        if len(recent_track) >= 2:
                            recent_freq_variance = np.var([item['freq'] for item in recent_track])
                            if recent_freq_variance < 100:  # Stable track
                                stability_bonus = 1.0 / (1.0 + recent_freq_variance / 10.0)
                                score -= stability_bonus  # Lower score is better
                    
                    # Base preference for higher power formants
                    score += (power_scale - candidate['power']) / power_scale
                    
                    # Penalize formants that would create unrealistic jumps
                    if last_valid[target_idx] is not None:
                        freq_jump = abs(candidate['freq'] - last_valid[target_idx]['freq'])
                        # Severe penalty for jumps > 20% of center frequency
                        center_freq = (f_min + f_max) / 2
                        if freq_jump > center_freq * 0.2:
                            score += 5.0  # Heavy penalty
                    
                    if score < best_score:
                        best_score = score
                        best_candidate = candidate
                
                if best_candidate:
                    frame_assignments[target_idx] = best_candidate
                    # Remove from available candidates to avoid duplicates
                    valid_formants.remove(best_candidate)
        
        # Handle assignments, track updates, and interpolation
        for target_idx in range(formants_out):
            base_idx = target_idx * 4
            
            if frame_assignments[target_idx] is not None:
                # Valid formant found
                formant = frame_assignments[target_idx]
                return_array[frame_idx, base_idx] = formant['freq']
                return_array[frame_idx, base_idx + 1] = formant['power']
                return_array[frame_idx, base_idx + 2] = formant['width']
                return_array[frame_idx, base_idx + 3] = formant['dissonance']
                
                # Update tracking data structures
                formant_tracks[target_idx].append(formant.copy())
                last_valid[target_idx] = formant
                missing_count[target_idx] = 0
                
                # Limit track history to prevent memory growth
                if len(formant_tracks[target_idx]) > track_history_length * 2:
                    formant_tracks[target_idx] = formant_tracks[target_idx][-track_history_length:]
                
            else:
                # No formant found - check if we can interpolate
                missing_count[target_idx] += 1
                
                if missing_count[target_idx] <= 2 and len(formant_tracks[target_idx]) > 0:
                    # Interpolation using track history

                    # Look ahead to find next valid formant
                    next_formant = None
                    look_ahead = min(3, num_frames - frame_idx - 1)
                    
                    for ahead_idx in range(1, look_ahead + 1):
                        if frame_idx + ahead_idx < num_frames:
                            future_frame = segment_formants[frame_idx + ahead_idx]
                            f_min, f_max = formant_ranges[target_idx]
                            
                            for f_idx in range(max_formants):
                                freq = future_frame[f_idx, 0]
                                if f_min <= freq <= f_max:
                                    # Additional validation using track history
                                    if len(formant_tracks[target_idx]) > 0:
                                        last_freq = formant_tracks[target_idx][-1]['freq']
                                        if abs(freq - last_freq) < (f_max - f_min) * 0.3:  # Reasonable jump
                                            next_formant = {
                                                'freq': freq,
                                                'power': future_frame[f_idx, 1],
                                                'width': future_frame[f_idx, 2],
                                                'dissonance': future_frame[f_idx, 3]
                                            }
                                            break
                            if next_formant:
                                break
                    
                    if next_formant:
                        # Smart interpolation using track trend
                        last_formant = formant_tracks[target_idx][-1]
                        alpha = 1.0 / (missing_count[target_idx] + 1)
                        
                        # If we have trend data, use it for better interpolation
                        if len(formant_tracks[target_idx]) >= 2:
                            # Linear interpolation with trend consideration
                            recent_freqs = [item['freq'] for item in formant_tracks[target_idx][-3:]]
                            trend = (recent_freqs[-1] - recent_freqs[0]) / len(recent_freqs) if len(recent_freqs) > 1 else 0
                            
                            # Interpolate with trend compensation
                            base_interp_freq = last_formant['freq'] * (1 - alpha) + next_formant['freq'] * alpha
                            trend_adjustment = trend * missing_count[target_idx] * 0.5  # Damped trend
                            interp_freq = base_interp_freq + trend_adjustment
                        else:
                            interp_freq = last_formant['freq'] * (1 - alpha) + next_formant['freq'] * alpha
                        
                        interp_power = last_formant['power'] * (1 - alpha) + next_formant['power'] * alpha
                        interp_width = last_formant['width'] * (1 - alpha) + next_formant['width'] * alpha
                        interp_dissonance = last_formant['dissonance'] * (1 - alpha) + next_formant['dissonance'] * alpha
                        
                        return_array[frame_idx, base_idx] = interp_freq
                        return_array[frame_idx, base_idx + 1] = interp_power
                        return_array[frame_idx, base_idx + 2] = interp_width
                        return_array[frame_idx, base_idx + 3] = interp_dissonance
                        
                        # Add interpolated point to track history
                        interp_formant = {
                            'freq': interp_freq,
                            'power': interp_power,
                            'width': interp_width,
                            'dissonance': interp_dissonance,
                            'frame': frame_idx,
                            'interpolated': True
                        }
                        formant_tracks[target_idx].append(interp_formant)
                        
                    else:
                        # Use track-based decay prediction
                        if len(formant_tracks[target_idx]) > 0:
                            last_formant = formant_tracks[target_idx][-1]
                            decay = 0.9 ** missing_count[target_idx]
                            
                            # Predict slight frequency drift based on track history
                            freq_drift = 0
                            if len(formant_tracks[target_idx]) >= 2:
                                recent_freqs = [item['freq'] for item in formant_tracks[target_idx][-3:]]
                                if len(recent_freqs) > 1:
                                    freq_drift = (recent_freqs[-1] - recent_freqs[0]) / len(recent_freqs) * 0.3
                            
                            return_array[frame_idx, base_idx] = last_formant['freq'] + freq_drift
                            return_array[frame_idx, base_idx + 1] = last_formant['power'] * decay
                            return_array[frame_idx, base_idx + 2] = last_formant['width']
                            return_array[frame_idx, base_idx + 3] = last_formant['dissonance']
                            
                            # Add predicted point to track
                            predicted_formant = {
                                'freq': last_formant['freq'] + freq_drift,
                                'power': last_formant['power'] * decay,
                                'width': last_formant['width'],
                                'dissonance': last_formant['dissonance'],
                                'frame': frame_idx,
                                'predicted': True
                            }
                            formant_tracks[target_idx].append(predicted_formant)
                        else:
                            return_array[frame_idx, base_idx:base_idx + 4] = 0
                else:
                    # Too many missing frames - clear track and reset
                    return_array[frame_idx, base_idx:base_idx + 4] = 0
                    if missing_count[target_idx] > 2:
                        formant_tracks[target_idx] = []  # Clear track history
                        last_valid[target_idx] = None
    
    return return_array


def track_formants(segment_formants, formants_out=3, smoothing_strength=0.7, formant_ranges=None):
    '''
    Formant tracking with multiple smoothing techniques to reduce F0 jumps.
    
    Args:
        segment_formants: numpy array of shape (num_frames, max_formants, 4)
        formants_out: number of output formants to track (default 3 for F0, F1, F2)
        smoothing_strength: float 0-1, higher values = more smoothing
    
    Returns:
        numpy array of shape (num_frames, formants_out * 4) with smoothed formants
    '''
    num_frames = segment_formants.shape[0]
    max_formants = segment_formants.shape[1]
    return_array = np.zeros((num_frames, formants_out * 4), dtype=np.float64)

    if formant_ranges is None:
        formant_ranges = [
                (80, 400),    # F0 range
                (200, 1200),  # F1 range  
                (800, 3500),  # F2 range
            ]
    
    if num_frames == 0:
        return return_array
    
    # Tracking parameters
    formant_tracks = [[] for _ in range(formants_out)]
    missing_count = [0 for _ in range(formants_out)]
    last_valid = [None for _ in range(formants_out)]
    
    # Smoothing parameters
    power_scale = 10000
    track_history_length = max(7, int(10 * smoothing_strength))  # Longer history for more smoothing
    
    # F0-specific parameters (assuming F0 is formant 0)
    f0_momentum = 0.3 * smoothing_strength  # Inertia factor for F0
    f0_max_jump_ratio = 0.15 * (1 - smoothing_strength * 0.5)  # Smaller jumps with more smoothing
    
    # First pass: Basic assignment with continuity
    for frame_idx in range(num_frames):
        frame_formants = segment_formants[frame_idx]
        
        # Extract valid formants
        valid_formants = []
        for f_idx in range(max_formants):
            freq = frame_formants[f_idx, 0]
            if freq > 0:
                valid_formants.append({
                    'freq': freq,
                    'power': frame_formants[f_idx, 1],
                    'width': frame_formants[f_idx, 2], 
                    'dissonance': frame_formants[f_idx, 3],
                    'index': f_idx,
                    'frame': frame_idx
                })
        
        valid_formants.sort(key=lambda x: x['freq'])
        frame_assignments = [None for _ in range(formants_out)]
        
        # Assignment logic
        for target_idx in range(formants_out):
            f_min, f_max = formant_ranges[target_idx]
            best_candidate = None
            best_score = float('inf')
            
            candidates = [f for f in valid_formants if f_min <= f['freq'] <= f_max]
            
            if candidates:
                for candidate in candidates:
                    score = 0
                    
                    # F0 tracking (assuming target_idx == 0 is F0)
                    if target_idx == 0 and len(formant_tracks[target_idx]) > 0:
                        # F0-specific continuity scoring
                        recent_track = formant_tracks[target_idx][-track_history_length:]
                        last_freq = recent_track[-1]['freq']
                        freq_diff = abs(candidate['freq'] - last_freq)
                        
                        # Stricter F0 jump penalty
                        center_freq = (f_min + f_max) / 2
                        if freq_diff > center_freq * f0_max_jump_ratio:
                            score += 10.0  # Heavy penalty for F0 jumps
                        else:
                            score += freq_diff / 50.0  # Gentle penalty for small changes
                        
                        # F0 trend prediction with momentum
                        if len(recent_track) >= 3:
                            recent_freqs = [item['freq'] for item in recent_track[-5:]]
                            
                            # Weighted moving average for trend
                            weights = np.exp(np.linspace(-1, 0, len(recent_freqs)))
                            weights /= weights.sum()
                            weighted_trend = np.sum(np.diff(recent_freqs) * weights[1:])
                            
                            # Apply momentum to prediction
                            predicted_freq = last_freq + weighted_trend * f0_momentum
                            trend_deviation = abs(candidate['freq'] - predicted_freq)
                            score += trend_deviation / 30.0
                        
                        # F0 stability reward
                        if len(recent_track) >= 4:
                            recent_variance = np.var([item['freq'] for item in recent_track[-4:]])
                            if recent_variance < 50:  # Very stable F0
                                stability_bonus = 2.0 / (1.0 + recent_variance / 10.0)
                                score -= stability_bonus
                    
                    # General continuity scoring for all formants
                    elif len(formant_tracks[target_idx]) > 0:
                        recent_track = formant_tracks[target_idx][-track_history_length:]
                        last_freq = recent_track[-1]['freq']
                        freq_diff = abs(candidate['freq'] - last_freq)
                        score += freq_diff / 100.0
                        
                        # Power and trend analysis
                        if len(recent_track) >= 2:
                            last_power = recent_track[-1]['power']
                            power_diff = abs(candidate['power'] - last_power)
                            score += power_diff / power_scale
                            
                            # Trend analysis
                            recent_freqs = [item['freq'] for item in recent_track[-3:]]
                            if len(recent_freqs) > 1:
                                trend_slope = (recent_freqs[-1] - recent_freqs[0]) / len(recent_freqs)
                                predicted_freq = recent_freqs[-1] + trend_slope
                                trend_deviation = abs(candidate['freq'] - predicted_freq)
                                score += trend_deviation / 50.0
                    
                    # Power preference
                    score += (power_scale - candidate['power']) / power_scale
                    
                    if score < best_score:
                        best_score = score
                        best_candidate = candidate
                
                if best_candidate:
                    frame_assignments[target_idx] = best_candidate
                    valid_formants.remove(best_candidate)
        
        # Process assignments and handle missing formants
        for target_idx in range(formants_out):
            base_idx = target_idx * 4
            
            if frame_assignments[target_idx] is not None:
                formant = frame_assignments[target_idx]
                
                # Apply additional F0 smoothing
                if target_idx == 0 and len(formant_tracks[target_idx]) > 0:
                    last_freq = formant_tracks[target_idx][-1]['freq']
                    # Exponential moving average for F0
                    alpha = 0.4 * (1 - smoothing_strength * 0.5)
                    smoothed_freq = alpha * formant['freq'] + (1 - alpha) * last_freq
                    formant['freq'] = smoothed_freq
                # Apply additional F0 width smoothing
                formant = apply_f0_width_smoothing_during_tracking(formant, formant_tracks, target_idx, smoothing_strength)
                return_array[frame_idx, base_idx] = formant['freq']
                return_array[frame_idx, base_idx + 1] = formant['power']
                return_array[frame_idx, base_idx + 2] = formant['width']
                return_array[frame_idx, base_idx + 3] = formant['dissonance']
                
                formant_tracks[target_idx].append(formant.copy())
                last_valid[target_idx] = formant
                missing_count[target_idx] = 0
                
                if len(formant_tracks[target_idx]) > track_history_length * 2:
                    formant_tracks[target_idx] = formant_tracks[target_idx][-track_history_length:]
            
            else:
                # Interpolation for missing formants
                missing_count[target_idx] += 1
                
                if missing_count[target_idx] <= 3 and len(formant_tracks[target_idx]) > 0:  # Extended interpolation
                    # Look ahead for next valid formant
                    next_formant = None
                    look_ahead = min(4, num_frames - frame_idx - 1)
                    
                    for ahead_idx in range(1, look_ahead + 1):
                        if frame_idx + ahead_idx < num_frames:
                            future_frame = segment_formants[frame_idx + ahead_idx]
                            f_min, f_max = formant_ranges[target_idx]
                            
                            for f_idx in range(max_formants):
                                freq = future_frame[f_idx, 0]
                                if f_min <= freq <= f_max:
                                    if len(formant_tracks[target_idx]) > 0:
                                        last_freq = formant_tracks[target_idx][-1]['freq']
                                        # More lenient jump threshold for interpolation
                                        jump_threshold = (f_max - f_min) * 0.4
                                        if abs(freq - last_freq) < jump_threshold:
                                            next_formant = {
                                                'freq': freq,
                                                'power': future_frame[f_idx, 1],
                                                'width': future_frame[f_idx, 2],
                                                'dissonance': future_frame[f_idx, 3]
                                            }
                                            break
                            if next_formant:
                                break
                    
                    if next_formant:
                        # Smooth interpolation
                        last_formant = formant_tracks[target_idx][-1]
                        alpha = 1.0 / (missing_count[target_idx] + 1)
                        
                        # Cubic interpolation for F0, linear for others
                        if target_idx == 0 and len(formant_tracks[target_idx]) >= 2:
                            # Cubic spline-like interpolation for F0
                            recent_freqs = [item['freq'] for item in formant_tracks[target_idx][-3:]]
                            if len(recent_freqs) >= 2:
                                # Smooth interpolation with acceleration consideration
                                v1 = recent_freqs[-1] - recent_freqs[-2] if len(recent_freqs) >= 2 else 0
                                v2 = next_formant['freq'] - last_formant['freq']
                                
                                # Hermite interpolation
                                t = alpha
                                h1 = 2*t**3 - 3*t**2 + 1
                                h2 = -2*t**3 + 3*t**2
                                h3 = t**3 - 2*t**2 + t
                                h4 = t**3 - t**2
                                
                                interp_freq = (h1 * last_formant['freq'] + 
                                            h2 * next_formant['freq'] +
                                            h3 * v1 * 0.1 +  # Damped velocity
                                            h4 * v2 * 0.1)
                            else:
                                interp_freq = last_formant['freq'] * (1 - alpha) + next_formant['freq'] * alpha
                        else:
                            interp_freq = last_formant['freq'] * (1 - alpha) + next_formant['freq'] * alpha
                        
                        interp_power = last_formant['power'] * (1 - alpha) + next_formant['power'] * alpha
                        interp_width = last_formant['width'] * (1 - alpha) + next_formant['width'] * alpha
                        interp_dissonance = last_formant['dissonance'] * (1 - alpha) + next_formant['dissonance'] * alpha
                        
                        return_array[frame_idx, base_idx] = interp_freq
                        return_array[frame_idx, base_idx + 1] = interp_power
                        return_array[frame_idx, base_idx + 2] = interp_width
                        return_array[frame_idx, base_idx + 3] = interp_dissonance
                        
                        interp_formant = {
                            'freq': interp_freq,
                            'power': interp_power,
                            'width': interp_width,
                            'dissonance': interp_dissonance,
                            'frame': frame_idx,
                            'interpolated': True
                        }
                        formant_tracks[target_idx].append(interp_formant)
                    
                    else:
                        # Decay prediction
                        if len(formant_tracks[target_idx]) > 0:
                            last_formant = formant_tracks[target_idx][-1]
                            decay = 0.9 ** missing_count[target_idx]
                            
                            # Smooth frequency drift for F0
                            if target_idx == 0 and len(formant_tracks[target_idx]) >= 2:
                                recent_freqs = [item['freq'] for item in formant_tracks[target_idx][-3:]]
                                if len(recent_freqs) > 1:
                                    # Damped trend continuation
                                    trend = np.mean(np.diff(recent_freqs))
                                    freq_drift = trend * 0.2 * (0.8 ** missing_count[target_idx])
                                else:
                                    freq_drift = 0
                            else:
                                freq_drift = 0
                            
                            return_array[frame_idx, base_idx] = last_formant['freq'] + freq_drift
                            return_array[frame_idx, base_idx + 1] = last_formant['power'] * decay
                            return_array[frame_idx, base_idx + 2] = last_formant['width']
                            return_array[frame_idx, base_idx + 3] = last_formant['dissonance']
                            
                            predicted_formant = {
                                'freq': last_formant['freq'] + freq_drift,
                                'power': last_formant['power'] * decay,
                                'width': last_formant['width'],
                                'dissonance': last_formant['dissonance'],
                                'frame': frame_idx,
                                'predicted': True
                            }
                            formant_tracks[target_idx].append(predicted_formant)
                        else:
                            return_array[frame_idx, base_idx:base_idx + 4] = 0
                else:
                    # Reset after too many missing frames
                    return_array[frame_idx, base_idx:base_idx + 4] = 0
                    if missing_count[target_idx] > 3:
                        formant_tracks[target_idx] = []
                        last_valid[target_idx] = None
    
    # Post-processing: Additional smoothing passes
    return_array = apply_post_smoothing(return_array, formants_out, smoothing_strength)
    
    return return_array

def apply_post_smoothing(formant_array, formants_out, smoothing_strength):
    '''
    Apply additional smoothing filters to reduce remaining jumps.
    With F0 width smoothing.
    '''
    smoothed_array = formant_array.copy()
    
    # Parameters based on smoothing strength
    median_window = max(3, int(5 * smoothing_strength))
    if median_window % 2 == 0:
        median_window += 1  # Ensure odd window size
    
    savgol_window = max(5, int(7 * smoothing_strength))
    if savgol_window % 2 == 0:
        savgol_window += 1
    
    for formant_idx in range(formants_out):
        freq_idx = formant_idx * 4
        width_idx = formant_idx * 4 + 2  # Width is the 3rd feature (index 2)
        
        freq_data = formant_array[:, freq_idx]
        width_data = formant_array[:, width_idx]
        
        # Only smooth non-zero regions
        non_zero_mask = freq_data > 0
        if np.sum(non_zero_mask) > savgol_window:
            
            # Step 1: Median filter to remove outliers (especially for F0)
            if formant_idx == 0:  # F0 gets extra outlier removal
                freq_data_filtered = median_filter(freq_data, size=median_window)
                # Only apply where original data was non-zero
                smoothed_array[:, freq_idx] = np.where(non_zero_mask, freq_data_filtered, freq_data)
                freq_data = smoothed_array[:, freq_idx]
                
                # NEW: Also apply median filter to F0 width
                width_data_filtered = median_filter(width_data, size=median_window)
                smoothed_array[:, width_idx] = np.where(non_zero_mask, width_data_filtered, width_data)
                width_data = smoothed_array[:, width_idx]
            
            # Step 2: Savitzky-Golay filter for smooth curves
            try:
                if len(freq_data[non_zero_mask]) >= savgol_window:
                    # Apply Savitzky-Golay to non-zero segments
                    segments = find_continuous_segments(non_zero_mask)
                    
                    for start, end in segments:
                        if end - start >= savgol_window:
                            # Smooth frequency
                            segment_data = freq_data[start:end+1]
                            smoothed_segment = signal.savgol_filter(
                                segment_data, savgol_window, 3,
                                mode='interp'
                            )
                            smoothed_array[start:end+1, freq_idx] = smoothed_segment
                            
                            # NEW: Also smooth width for F0
                            if formant_idx == 0:  # F0 width smoothing
                                width_segment_data = width_data[start:end+1]
                                smoothed_width_segment = signal.savgol_filter(
                                    width_segment_data, savgol_window, 3,
                                    mode='interp'
                                )
                                smoothed_array[start:end+1, width_idx] = smoothed_width_segment
                        
            except Exception as e:
                # Fallback to simple moving average if Savitzky-Golay fails
                window_size = max(3, int(smoothing_strength * 5))
                smoothed_freq = moving_average(freq_data, window_size, non_zero_mask)
                smoothed_array[:, freq_idx] = smoothed_freq
                
                # NEW: Fallback moving average for F0 width
                if formant_idx == 0:
                    smoothed_width = moving_average(width_data, window_size, non_zero_mask)
                    smoothed_array[:, width_idx] = smoothed_width
        
        # Also smooth power for more stable tracking
        power_idx = formant_idx * 4 + 1
        power_data = formant_array[:, power_idx]
        if np.sum(power_data > 0) > 3:
            power_smoothed = moving_average(power_data, 3, power_data > 0)
            smoothed_array[:, power_idx] = power_smoothed
    
    return smoothed_array


# Additional function for F0 width smoothing during tracking
def apply_f0_width_smoothing_during_tracking(formant, formant_tracks, target_idx, smoothing_strength):
    '''
    Apply real-time F0 width smoothing during the tracking process.
    Call this when assigning F0 formants (target_idx == 0).
    '''
    if target_idx == 0 and len(formant_tracks[target_idx]) > 0:
        # F0 width smoothing with exponential moving average
        last_width = formant_tracks[target_idx][-1]['width']
        alpha_width = 0.5 * (1 - smoothing_strength * 0.3)  # Slightly less aggressive than frequency
        smoothed_width = alpha_width * formant['width'] + (1 - alpha_width) * last_width
        formant['width'] = smoothed_width
    
    return formant

def find_continuous_segments(mask):
    '''Find continuous True segments in a boolean mask.'''
    segments = []
    start = None
    
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            segments.append((start, i-1))
            start = None
    
    if start is not None:
        segments.append((start, len(mask)-1))
    
    return segments


def moving_average(data, window_size, mask):
    '''Apply moving average only to masked (valid) regions.'''
    result = data.copy()
    
    # Find continuous segments
    segments = find_continuous_segments(mask)
    
    for start, end in segments:
        if end - start + 1 >= window_size:
            segment = data[start:end+1]
            # Apply moving average
            averaged = np.convolve(segment, np.ones(window_size)/window_size, mode='same')
            result[start:end+1] = averaged
    
    return result


# Usage example with different smoothing levels:
def track_formants_pipeline(segment_formants, formants_out=3, smoothing_level='medium', formant_ranges=None):
    '''
    Convenience function with preset smoothing levels.
    
    Args:
        smoothing_level: 'light' (0.3), 'medium' (0.7), 'heavy' (0.9)
    '''
    smoothing_map = {
        'light': 0.3,
        'medium': 0.7,
        'heavy': 0.9
    }
    
    smoothing_strength = smoothing_map.get(smoothing_level, 0.7)
    return track_formants(segment_formants, formants_out, smoothing_strength, formant_ranges)