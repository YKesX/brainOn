#include <Arduino.h>

// Pin definitions
#define PIN_RIGHT_PEC  34  // bit 0
#define PIN_LEFT_PEC   35  // bit 1
#define PIN_RIGHT_QUAD 32  // bit 2
#define PIN_LEFT_LEG   33  // bit 3
#define CHANNEL_COUNT  4

const int pins[CHANNEL_COUNT] = {PIN_RIGHT_PEC, PIN_LEFT_PEC, PIN_RIGHT_QUAD, PIN_LEFT_LEG};
const char* channel_names[CHANNEL_COUNT] = {"RightPec", "LeftPec", "RightQuad", "LeftLeg"};

void setup() {
  Serial.begin(115200);
  analogSetAttenuation(ADC_11db);
  
  for (int i = 0; i < CHANNEL_COUNT; i++) {
    pinMode(pins[i], INPUT);
  }
  
  Serial.println("=== Raw EMG Signal Test ===");
  Serial.println("This program outputs raw ADC readings from all 4 EMG channels");
  Serial.println("Connect your AD8232 sensors and flex muscles to test signal quality");
  Serial.println();
  Serial.println("Channel mapping:");
  for (int i = 0; i < CHANNEL_COUNT; i++) {
    Serial.printf("  Pin %d: %s\n", pins[i], channel_names[i]);
  }
  Serial.println();
  Serial.println("Data format: Ch0,Ch1,Ch2,Ch3 (raw ADC values 0-4095)");
  Serial.println("Good EMG signals should show:");
  Serial.println("  - Baseline around 1800-2000 when relaxed");
  Serial.println("  - Significant changes (±200-800) when flexing");
  Serial.println("  - Minimal noise when electrodes are properly attached");
  Serial.println();
  delay(3000);
}

void loop() {
  // Read all channels
  for (int ch = 0; ch < CHANNEL_COUNT; ch++) {
    int reading = analogRead(pins[ch]);
    Serial.print(reading);
    if (ch < CHANNEL_COUNT - 1) {
      Serial.print(",");
    }
  }
  Serial.println();
  
  // Fast sampling for good signal capture
  delay(5);  // 200Hz sampling rate
} 