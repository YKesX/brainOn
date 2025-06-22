#pragma once
void filter_init();
int filter_sample(int ch, int raw);
int get_rms(int ch);
bool get_state(int ch, int threshold, int hysteresis); 