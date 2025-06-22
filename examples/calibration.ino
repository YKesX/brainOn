#include "pins.h"
#include "filters.h"
#include <Preferences.h>
Preferences prefs;
const uint16_t HYST = 40;
void setup() {
  Serial.begin(115200);
  filter_init();
  prefs.begin("emg", false);
  Serial.println("Stay relaxed for 5 s…");
  uint32_t t0 = millis();
  uint32_t restSum[CHANNEL_COUNT] = {0};
  while (millis() - t0 < 5000) {
    for (uint8_t ch = 0; ch < CHANNEL_COUNT; ++ch)
      restSum[ch] += filter_sample(ch, analogRead(pins[ch]));
  }
  uint16_t restMean[CHANNEL_COUNT];
  for (uint8_t ch = 0; ch < CHANNEL_COUNT; ++ch) restMean[ch] = restSum[ch] / 5000;
  Serial.println("Now flex maximally for 5 s…");
  uint16_t peak[CHANNEL_COUNT] = {0};
  t0 = millis();
  while (millis() - t0 < 5000) {
    for (uint8_t ch = 0; ch < CHANNEL_COUNT; ++ch) {
      uint16_t rms = filter_sample(ch, analogRead(pins[ch]));
      if (rms > peak[ch]) peak[ch] = rms;
    }
  }
  for (uint8_t ch = 0; ch < CHANNEL_COUNT; ++ch) {
    uint16_t thr = restMean[ch] + (peak[ch] - restMean[ch]) * 35 / 100;
    prefs.putUShort((String("thr") + ch).c_str(), thr);
    prefs.putUShort((String("hyst") + ch).c_str(), HYST);
    Serial.printf("Ch%u thr=%u\n", ch, thr);
  }
  Serial.println("Calibration done.");
}
void loop() {} 