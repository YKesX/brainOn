#include <Arduino.h>
#include "filters.h"
#include "pins.h"
#include <cmath>

// Compile flags
#ifndef USE_60_HZ
#define USE_60_HZ 0
#endif

// Sampling rate assumption
#define FS 1000.0  // Hz

// Filter coefficients - IIR Notch filters
// 50Hz notch filter coefficients (Q=10, fs=1000Hz)
const float notch50_b[] = {0.9691, -1.2311, 0.9691};
const float notch50_a[] = {1.0000, -1.2311, 0.9382};

// 60Hz notch filter coefficients (Q=10, fs=1000Hz)  
const float notch60_b[] = {0.9691, -0.9135, 0.9691};
const float notch60_a[] = {1.0000, -0.9135, 0.9382};

// 4th-order Butterworth band-pass 20-450 Hz (implemented as cascaded biquads)
// Low-pass 450Hz coefficients
const float lp_b[] = {0.0067, 0.0134, 0.0067};
const float lp_a[] = {1.0000, -1.1430, 0.4128};

// High-pass 20Hz coefficients  
const float hp_b[] = {0.9150, -1.8299, 0.9150};
const float hp_a[] = {1.0000, -1.8227, 0.8372};

// Filter state variables
struct ChannelState {
  // Notch filter states
  float notch_x[3] = {0};  // x[n], x[n-1], x[n-2]
  float notch_y[3] = {0};  // y[n], y[n-1], y[n-2]
  
  // Band-pass filter states (cascaded)
  float lp_x[3] = {0};
  float lp_y[3] = {0};
  float hp_x[3] = {0};
  float hp_y[3] = {0};
  
  // RMS calculation (50ms window at 1kHz = 50 samples)
  uint16_t rms_buffer[50] = {0};
  uint8_t rms_idx = 0;
  uint32_t rms_sum = 0;
  uint16_t last_rms = 0;
  
  // Envelope follower state
  float envelope = 0;
  
  // State detection with hysteresis
  bool last_state = false;
};

ChannelState channels[CHANNEL_COUNT];

// Initialize filters
void filter_init() {
  for (uint8_t ch = 0; ch < CHANNEL_COUNT; ++ch) {
    // Clear all state variables
    memset(&channels[ch], 0, sizeof(ChannelState));
  }
}

// Apply IIR filter (Direct Form II)
float apply_iir(float input, const float* b, const float* a, float* x, float* y) {
  // Shift delay line
  x[2] = x[1];
  x[1] = x[0];
  x[0] = input;
  
  y[2] = y[1];
  y[1] = y[0];
  
  // Calculate output
  y[0] = b[0] * x[0] + b[1] * x[1] + b[2] * x[2] - a[1] * y[1] - a[2] * y[2];
  
  return y[0];
}

// Process one sample through all filters
uint16_t filter_sample(uint8_t ch, uint16_t raw) {
  if (ch >= CHANNEL_COUNT) return 0;
  
  ChannelState& state = channels[ch];
  float sample = (float)raw;
  
  // Apply notch filter (50Hz or 60Hz based on compile flag)
#if USE_60_HZ
  sample = apply_iir(sample, notch60_b, notch60_a, state.notch_x, state.notch_y);
#else
  sample = apply_iir(sample, notch50_b, notch50_a, state.notch_x, state.notch_y);
#endif
  
  // Apply band-pass filter (cascaded low-pass then high-pass)
  sample = apply_iir(sample, lp_b, lp_a, state.lp_x, state.lp_y);
  sample = apply_iir(sample, hp_b, hp_a, state.hp_x, state.hp_y);
  
  // Rectify for RMS calculation
  float rectified = fabs(sample);
  
  // Update RMS sliding window
  state.rms_sum -= state.rms_buffer[state.rms_idx];
  state.rms_buffer[state.rms_idx] = (uint16_t)rectified;
  state.rms_sum += state.rms_buffer[state.rms_idx];
  state.rms_idx = (state.rms_idx + 1) % 50;
  
  // Calculate RMS
  state.last_rms = sqrt(state.rms_sum / 50.0);
  
  // Envelope follower (10ms attack, 200ms release at 1kHz)
  const float attack_coeff = exp(-1.0 / (FS * 0.01));   // 10ms
  const float release_coeff = exp(-1.0 / (FS * 0.2));   // 200ms
  
  if (rectified > state.envelope) {
    state.envelope = attack_coeff * state.envelope + (1 - attack_coeff) * rectified;
  } else {
    state.envelope = release_coeff * state.envelope + (1 - release_coeff) * rectified;
  }
  
  return state.last_rms;
}

// Get current RMS value
uint16_t get_rms(uint8_t ch) {
  if (ch >= CHANNEL_COUNT) return 0;
  return channels[ch].last_rms;
}

// Get binary state with hysteresis
bool get_state(uint8_t ch, uint16_t threshold, uint16_t hysteresis) {
  if (ch >= CHANNEL_COUNT) return false;
  
  ChannelState& state = channels[ch];
  uint16_t current_rms = state.last_rms;
  
  // Apply hysteresis
  if (state.last_state) {
    // Currently ON - need to drop below (threshold - hysteresis) to turn OFF
    if (current_rms < (threshold - hysteresis)) {
      state.last_state = false;
    }
  } else {
    // Currently OFF - need to rise above threshold to turn ON
    if (current_rms > threshold) {
      state.last_state = true;
    }
  }
  
  return state.last_state;
} 