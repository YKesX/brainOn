#include <Arduino.h>
#include <Preferences.h>

// Pin definitions
#define PIN_RIGHT_PEC  34  // bit 0
#define PIN_LEFT_PEC   35  // bit 1
#define PIN_RIGHT_QUAD 32  // bit 2
#define PIN_LEFT_LEG   33  // bit 3
#define CHANNEL_COUNT  4

const int pins[CHANNEL_COUNT] = {PIN_RIGHT_PEC, PIN_LEFT_PEC, PIN_RIGHT_QUAD, PIN_LEFT_LEG};
const char* channel_names[CHANNEL_COUNT] = {"RightPec", "LeftPec", "RightQuad", "LeftLeg"};

// Threshold values
int thresholds[CHANNEL_COUNT] = {300, 300, 300, 300};
int hysteresis = 50;
bool muscle_state[CHANNEL_COUNT] = {false, false, false, false};

// Simple moving average filter
#define WINDOW_SIZE 10
int readings[CHANNEL_COUNT][WINDOW_SIZE];
int read_index = 0;
int totals[CHANNEL_COUNT] = {0, 0, 0, 0};

Preferences prefs;

void setup() {
  Serial.begin(115200);
  analogSetAttenuation(ADC_11db);
  
  // Load thresholds
  prefs.begin("emg", true);
  for (int ch = 0; ch < CHANNEL_COUNT; ch++) {
    String thr_key = "thr" + String(ch);
    thresholds[ch] = prefs.getInt(thr_key.c_str(), thresholds[ch]);
  }
  hysteresis = prefs.getInt("hyst", hysteresis);
  prefs.end();
  
  // Initialize pins and arrays
  for (int i = 0; i < CHANNEL_COUNT; i++) {
    pinMode(pins[i], INPUT);
    for (int j = 0; j < WINDOW_SIZE; j++) {
      readings[i][j] = 0;
    }
  }
  
  Serial.println("=== EMG Debug Monitor ===");
  Serial.println("Format: Raw0,Filtered0,State0,Raw1,Filtered1,State1,...");
  Serial.println("You can also type commands:");
  Serial.println("  'b' - Show binary output");
  Serial.println("  't' - Show thresholds");
  Serial.println("  'h' - Show this help");
  delay(2000);
}

void loop() {
  uint8_t output_byte = 0;
  
  // Process all channels
  for (int ch = 0; ch < CHANNEL_COUNT; ch++) {
    // Read and filter
    int raw = analogRead(pins[ch]);
    
    totals[ch] = totals[ch] - readings[ch][read_index];
    readings[ch][read_index] = raw;
    totals[ch] = totals[ch] + readings[ch][read_index];
    
    int filtered = totals[ch] / WINDOW_SIZE;
    
    // Apply hysteresis
    if (muscle_state[ch]) {
      if (filtered < (thresholds[ch] - hysteresis)) {
        muscle_state[ch] = false;
      }
    } else {
      if (filtered > thresholds[ch]) {
        muscle_state[ch] = true;
      }
    }
    
    if (muscle_state[ch]) {
      output_byte |= (1 << ch);
    }
  }
  
  read_index = (read_index + 1) % WINDOW_SIZE;
  
  // Check for serial commands
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'b') {
      Serial.print("Binary output: ");
      for (int i = 7; i >= 0; i--) {
        Serial.print((output_byte >> i) & 1);
      }
      Serial.printf(" (decimal: %d)\n", output_byte);
    } else if (cmd == 't') {
      Serial.println("Current thresholds:");
      for (int ch = 0; ch < CHANNEL_COUNT; ch++) {
        Serial.printf("  %s: %d (hyst: %d)\n", channel_names[ch], thresholds[ch], hysteresis);
      }
    } else if (cmd == 'h') {
      Serial.println("Commands: 'b'=binary, 't'=thresholds, 'h'=help");
    }
  }
  
  // Output CSV data for plotting
  for (int ch = 0; ch < CHANNEL_COUNT; ch++) {
    int raw = analogRead(pins[ch]);
    int filtered = totals[ch] / WINDOW_SIZE;
    Serial.printf("%d,%d,%d", raw, filtered, muscle_state[ch] ? 1 : 0);
    if (ch < CHANNEL_COUNT - 1) Serial.print(",");
  }
  Serial.println();
  
  delay(50);  // 20Hz update rate for readability
} 