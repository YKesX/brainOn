#include <Arduino.h>
#include <Preferences.h>
#include <FS.h>
#include <SPIFFS.h>

//USE THIS AS MAIN.CPP BEFORE RUNNING ANYTHING
// Pin definitions for muscle detection
#define PIN_RIGHT_PEC  34  // bit 0
#define PIN_LEFT_PEC   35  // bit 1
#define PIN_RIGHT_QUAD 32  // bit 2
#define PIN_LEFT_LEG   33  // bit 3
#define CHANNEL_COUNT  4

// EEG pin definitions (12 pins for brainwave recording)
const int eeg_pins[12] = {18, 19, 21, 22, 23, 13, 14, 27, 26, 25, 2, 4};
#define EEG_CHANNEL_COUNT 12

const int pins[CHANNEL_COUNT] = {PIN_RIGHT_PEC, PIN_LEFT_PEC, PIN_RIGHT_QUAD, PIN_LEFT_LEG};
const char* channel_names[CHANNEL_COUNT] = {"RightPec", "LeftPec", "RightQuad", "LeftLeg"};
const char* eeg_channel_names[EEG_CHANNEL_COUNT] = {"EEG1", "EEG2", "EEG3", "EEG4", "EEG5", "EEG6", "EEG7", "EEG8", "EEG9", "EEG10", "EEG11", "EEG12"};

Preferences prefs;

// EEG data structures
struct EEGSample {
  unsigned long timestamp;
  int values[EEG_CHANNEL_COUNT];
};

#define MAX_SAMPLES 500
EEGSample baseline_samples[MAX_SAMPLES];
EEGSample unlock_samples[MAX_SAMPLES];
EEGSample transaction_samples[MAX_SAMPLES];
int baseline_count = 0;
int unlock_count = 0;
int transaction_count = 0;

void setup() {
  Serial.begin(115200);
  analogSetAttenuation(ADC_11db);
  
  // Initialize SPIFFS for file storage
  if (!SPIFFS.begin(true)) {
    Serial.println("SPIFFS Mount Failed");
    return;
  }
  
  // Initialize muscle pins
  for (int i = 0; i < CHANNEL_COUNT; i++) {
    pinMode(pins[i], INPUT);
  }
  
  // Initialize EEG pins
  for (int i = 0; i < EEG_CHANNEL_COUNT; i++) {
    pinMode(eeg_pins[i], INPUT);
  }
  
  prefs.begin("emg", false);
  
  Serial.println("=== Individual Muscle Calibration Tool ===");
  Serial.println("This will calibrate each muscle channel individually for optimal accuracy");
  Serial.println("\nFirst, let's measure your baseline (relaxed state)...");
  
  // Step 1: Global baseline measurement
  Serial.println("\nStep 1: Stay completely relaxed for 5 seconds...");
  Serial.println("Keep all muscles relaxed, don't touch any pins.");
  delay(3000);
  Serial.println("Starting baseline measurement in 3...");
  delay(1000);
  Serial.println("2...");
  delay(1000);
  Serial.println("1...");
  delay(1000);
  Serial.println("Measuring baseline - stay completely relaxed!");
  
  long baseline[CHANNEL_COUNT] = {0, 0, 0, 0};
  int samples = 0;
  
  unsigned long start_time = millis();
  while (millis() - start_time < 5000) {
    for (int ch = 0; ch < CHANNEL_COUNT; ch++) {
      baseline[ch] += analogRead(pins[ch]);
    }
    samples++;
    delay(10);
  }
  
  // Calculate baseline averages
  for (int ch = 0; ch < CHANNEL_COUNT; ch++) {
    baseline[ch] /= samples;
    Serial.printf("✓ %s baseline: %d\n", channel_names[ch], (int)baseline[ch]);
  }
  
  // Step 2: Individual muscle calibration
  int peak[CHANNEL_COUNT] = {0, 0, 0, 0};
  
  Serial.println("\nNow we'll calibrate each muscle individually...");
  delay(2000);
  
  for (int ch = 0; ch < CHANNEL_COUNT; ch++) {
    Serial.printf("\n=== Calibrating %s (Pin %d) ===\n", channel_names[ch], pins[ch]);
    
    if (ch == 0) {
      Serial.println("Instructions: Touch GPIO pin 34 firmly OR flex your right pectoral muscle");
    } else if (ch == 1) {
      Serial.println("Instructions: Touch GPIO pin 35 firmly OR flex your left pectoral muscle");
    } else if (ch == 2) {
      Serial.println("Instructions: Touch GPIO pin 32 firmly OR flex your right quadriceps");
    } else if (ch == 3) {
      Serial.println("Instructions: Touch GPIO pin 33 firmly OR flex your left leg muscle");
    }
    
    Serial.println("You have 8 seconds to activate this muscle/pin...");
    delay(3000);
    
    Serial.println("Starting in 3...");
    delay(1000);
    Serial.println("2...");
    delay(1000);
    Serial.println("1...");
    delay(1000);
    Serial.printf("ACTIVATE %s NOW!\n", channel_names[ch]);
    
    peak[ch] = baseline[ch];  // Start with baseline
    start_time = millis();
    
    while (millis() - start_time < 8000) {
      int reading = analogRead(pins[ch]);
      if (reading > peak[ch]) {
        peak[ch] = reading;
      }
      
      // Show live feedback every 200ms
      if ((millis() - start_time) % 200 < 20) {
        int progress = (millis() - start_time) / 100;  // Progress in 100ms units
        Serial.printf("Time: %ds | Current: %d | Peak: %d\n", 
                     progress / 10, reading, peak[ch]);
      }
      delay(10);
    }
    
    Serial.printf("✓ %s calibration complete! Peak: %d\n", channel_names[ch], peak[ch]);
    Serial.println("Relax for 3 seconds before next muscle...");
    delay(3000);
  }
  
  Serial.println("\n=== Calibration Results ===");
  
  // Calculate and store thresholds (35% of range above baseline)
  for (int ch = 0; ch < CHANNEL_COUNT; ch++) {
    int range = peak[ch] - baseline[ch];
    int threshold = baseline[ch] + (range * 35) / 100;
    
    // Store in preferences
    String thr_key = "thr" + String(ch);
    prefs.putInt(thr_key.c_str(), threshold);
    prefs.putInt("hyst", 80);  // Adaptive hysteresis
    
    Serial.printf("Channel %d (%s): baseline=%d, peak=%d, range=%d, threshold=%d\n", 
                 ch, channel_names[ch], (int)baseline[ch], peak[ch], range, threshold);
  }
  
  Serial.println("\n🎉 Individual muscle calibration complete!");
  Serial.println("✅ Thresholds saved to ESP32 memory");
  
  // Step 3: EEG Brainwave Recording
  Serial.println("\n=== Starting EEG Brainwave Recording ===");
  Serial.println("This will record your brainwave patterns for wallet authentication");
  delay(3000);
  
  // Phase 1: Baseline recording
  Serial.println("\n--- Phase 1: Baseline Recording ---");
  Serial.println("Please just breathe and don't move and don't think.");
  Serial.println("Clear your mind completely for the next 5 seconds.");
  delay(3000);
  Serial.println("Starting baseline recording in 3...");
  delay(1000);
  Serial.println("2...");
  delay(1000);
  Serial.println("1...");
  delay(1000);
  Serial.println("Recording baseline - stay completely relaxed and empty your mind!");
  
  baseline_count = 0;
  start_time = millis();
  while (millis() - start_time < 5000 && baseline_count < MAX_SAMPLES) {
    EEGSample sample;
    sample.timestamp = millis();
    for (int i = 0; i < EEG_CHANNEL_COUNT; i++) {
      sample.values[i] = analogRead(eeg_pins[i]);
    }
    baseline_samples[baseline_count++] = sample;
    delay(10);
  }
  Serial.printf("✓ Baseline recording complete: %d samples\n", baseline_count);
  
  // Phase 2: Wallet unlock thinking
  Serial.println("\n--- Phase 2: Wallet Unlock Thought Pattern ---");
  Serial.println("Now think about unlocking your wallet. Visualize entering your password,");
  Serial.println("seeing your balance, feeling secure access to your funds.");
  Serial.println("Focus intensely on this thought for 5 seconds.");
  delay(3000);
  Serial.println("Starting unlock recording in 3...");
  delay(1000);
  Serial.println("2...");
  delay(1000);
  Serial.println("1...");
  delay(1000);
  Serial.println("Think about UNLOCKING YOUR WALLET now!");
  
  unlock_count = 0;
  start_time = millis();
  while (millis() - start_time < 5000 && unlock_count < MAX_SAMPLES) {
    EEGSample sample;
    sample.timestamp = millis();
    for (int i = 0; i < EEG_CHANNEL_COUNT; i++) {
      sample.values[i] = analogRead(eeg_pins[i]);
    }
    unlock_samples[unlock_count++] = sample;
    delay(10);
  }
  Serial.printf("✓ Unlock thought recording complete: %d samples\n", unlock_count);
  
  // Phase 3: Transaction thinking
  Serial.println("\n--- Phase 3: Transaction Thought Pattern ---");
  Serial.println("Now think about making a transaction. Visualize sending money,");
  Serial.println("confirming the amount, approving the transfer, feeling the completion.");
  Serial.println("Focus intensely on this transaction process for 5 seconds.");
  delay(3000);
  Serial.println("Starting transaction recording in 3...");
  delay(1000);
  Serial.println("2...");
  delay(1000);
  Serial.println("1...");
  delay(1000);
  Serial.println("Think about MAKING A TRANSACTION now!");
  
    transaction_count = 0;
  start_time = millis();
  while (millis() - start_time < 5000 && transaction_count < MAX_SAMPLES) {
    EEGSample sample;
    sample.timestamp = millis();
    for (int i = 0; i < EEG_CHANNEL_COUNT; i++) {
      sample.values[i] = analogRead(eeg_pins[i]);
    }
    transaction_samples[transaction_count++] = sample;
    delay(10);
  }
  Serial.printf("✓ Transaction thought recording complete: %d samples\n", transaction_count);

  // Step 4: Save data to file
  Serial.println("\n=== Saving EEG Data to File ===");
  File file = SPIFFS.open("/eeg_patterns.csv", "w");
  if (!file) {
    Serial.println("Failed to open file for writing");
    return;
  }
  
  // Write header
  file.print("Phase,Timestamp");
  for (int i = 0; i < EEG_CHANNEL_COUNT; i++) {
    file.printf(",%s", eeg_channel_names[i]);
  }
  file.println();
  
  // Write baseline data
  for (int j = 0; j < baseline_count; j++) {
    file.printf("Baseline,%lu", baseline_samples[j].timestamp);
    for (int i = 0; i < EEG_CHANNEL_COUNT; i++) {
      file.printf(",%d", baseline_samples[j].values[i]);
    }
    file.println();
  }
  
  // Write unlock data
  for (int j = 0; j < unlock_count; j++) {
    file.printf("Unlock,%lu", unlock_samples[j].timestamp);
    for (int i = 0; i < EEG_CHANNEL_COUNT; i++) {
      file.printf(",%d", unlock_samples[j].values[i]);
    }
    file.println();
  }
  
  // Write transaction data
  for (int j = 0; j < transaction_count; j++) {
    file.printf("Transaction,%lu", transaction_samples[j].timestamp);
    for (int i = 0; i < EEG_CHANNEL_COUNT; i++) {
      file.printf(",%d", transaction_samples[j].values[i]);
    }
    file.println();
  }
  
  file.close();
  Serial.println("✅ EEG data saved to /eeg_patterns.csv");
  
  
  Serial.println("\n🎉🧠 Complete Calibration Finished! 🧠🎉");
  Serial.println("✅ Muscle thresholds calibrated and saved");
  Serial.println("✅ EEG brainwave patterns recorded for 3 states");
  Serial.println("✅ Data saved to /eeg_patterns.csv");
  Serial.println("✅ Ready for multi-modal authentication system");
}

void loop() {
  // Show live readings for verification
  Serial.print("Muscle readings: ");
  for (int ch = 0; ch < CHANNEL_COUNT; ch++) {
    Serial.print(analogRead(pins[ch]));
    if (ch < CHANNEL_COUNT - 1) Serial.print(", ");
  }
  Serial.print(" | EEG readings: ");
  for (int ch = 0; ch < EEG_CHANNEL_COUNT; ch++) {
    Serial.print(analogRead(eeg_pins[ch]));
    if (ch < EEG_CHANNEL_COUNT - 1) Serial.print(", ");
  }
  Serial.println();
  delay(500);
} 
