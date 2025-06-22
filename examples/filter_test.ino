#include "pins.h"
#include "filters.h"
const uint16_t HYST = 40;
uint16_t thr[CHANNEL_COUNT] = {200, 200, 200, 200};
bool notch50 = true, notch60 = false, bp = true, env = true;

void printHelp() {
  Serial.println(F("Commands:\n  f <n> on|off\n    name = notch50 | notch60 | bp | env\n  t <ch> <thr>  (0‑3)\n  h  help"));
}
void setup() {
  Serial.begin(115200);
  filter_init();
  printHelp();
}
void loop() {
  static uint32_t last = 0;
  if (millis() - last >= 5) {
    last = millis();
    for (uint8_t ch = 0; ch < CHANNEL_COUNT; ++ch) {
      uint16_t raw = analogRead(pins[ch]);
      uint16_t rms = filter_sample(ch, raw);
      bool active = get_state(ch, thr[ch], HYST);
      Serial.printf("%u,%u,%u%c", raw, rms, active, ch == CHANNEL_COUNT - 1 ? '\n' : ',');
    }
  }
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.startsWith("f ")) {
      int sp = cmd.indexOf(' ', 2);
      if (sp > 0) {
        String name = cmd.substring(2, sp);
        String val = cmd.substring(sp + 1);
        bool on = val == "on";
        if (name == "notch50") notch50 = on;
        else if (name == "notch60") notch60 = on;
        else if (name == "bp") bp = on;
        else if (name == "env") env = on;
        Serial.printf("%s %s\n", name.c_str(), on ? "ON" : "OFF");
      }
    } else if (cmd.startsWith("t ")) {
      int sp1 = cmd.indexOf(' ', 2);
      if (sp1 > 0) {
        uint8_t ch = cmd.substring(2, sp1).toInt();
        uint16_t v = cmd.substring(sp1 + 1).toInt();
        if (ch < CHANNEL_COUNT) thr[ch] = v;
        Serial.printf("thr[%u]=%u\n", ch, v);
      }
    } else if (cmd == "h") printHelp();
  }
} 