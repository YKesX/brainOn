#include "pins.h"
void setup() {
  Serial.begin(115200);
  analogSetAttenuation(ADC_11db);
  for (uint8_t i = 0; i < CHANNEL_COUNT; ++i) pinMode(pins[i], INPUT);
}
void loop() {
  for (uint8_t ch = 0; ch < CHANNEL_COUNT; ++ch) {
    Serial.print(analogRead(pins[ch]));
    Serial.print(ch == CHANNEL_COUNT - 1 ? '\n' : ',');
  }
} 