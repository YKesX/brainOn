#include <Arduino.h>
#include "BluetoothSerial.h"

// Check if Bluetooth is properly configured
#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif

// Bluetooth Serial object
BluetoothSerial SerialBT;

// Pin definitions for muscle detection (existing)
const int MUSCLE_PINS[4] = {34, 35, 32, 33};  // GPIO pins for muscle sensors
const char* MUSCLE_NAMES[4] = {"RightPec", "LeftPec", "RightQuad", "LeftLeg"};

// Pin definitions for EEG data (12 channels)
const int EEG_PINS[12] = {18, 19, 21, 22, 23, 13, 14, 27, 26, 25, 2, 4};

// Calibrated thresholds from user's calibration
const int THRESHOLDS[4] = {2625, 2627, 2617, 2355};
const int HYSTERESIS = 100;

// State variables for muscle detection
bool muscleStates[4] = {false, false, false, false};
unsigned long lastMuscleUpdate = 0;
unsigned long lastEEGUpdate = 0;
unsigned long lastStatusPrint = 0;

// Sampling rates (Hz)
const int MUSCLE_SAMPLE_RATE = 50;  // 50 Hz for muscle
const int EEG_SAMPLE_RATE = 100;    // 100 Hz for EEG
const int STATUS_PRINT_RATE = 2;    // 2 Hz for status updates

// Calculate intervals in milliseconds
const unsigned long MUSCLE_INTERVAL = 1000 / MUSCLE_SAMPLE_RATE;  // 20ms
const unsigned long EEG_INTERVAL = 1000 / EEG_SAMPLE_RATE;        // 10ms
const unsigned long STATUS_INTERVAL = 1000 / STATUS_PRINT_RATE;   // 500ms

// Command interface
bool eegRecording = true;  // EEG recording enabled by default
String commandBuffer = "";
String commandBufferBT = "";

// Function prototypes
void updateMuscleStates();
void readEEGData();
void handleSerialCommands();
void handleBluetoothCommands();
void printSystemStatus();
int readFilteredADC(int pin);
void dualPrint(const String& message);
void dualPrintln(const String& message);
void dualPrint(int value);
void dualPrint(unsigned long value);

// Dual output functions for both USB and Bluetooth
void dualPrint(const String& message) {
    Serial.print(message);
    SerialBT.print(message);
}

void dualPrintln(const String& message) {
    Serial.println(message);
    SerialBT.println(message);
}

void dualPrint(int value) {
    Serial.print(value);
    SerialBT.print(value);
}

void dualPrint(unsigned long value) {
    Serial.print(value);
    SerialBT.print(value);
}

void setup() {
    Serial.begin(115200);
    
    // Initialize Bluetooth Serial
    SerialBT.begin("BrainOn"); // Bluetooth device name
    
    // Configure ADC attenuation for all muscle pins
    for(int i = 0; i < 4; i++) {
        analogSetAttenuation(static_cast<adc_attenuation_t>(3)); // 11dB attenuation
    }
    
    // Configure ADC attenuation for all EEG pins
    for(int i = 0; i < 12; i++) {
        analogSetAttenuation(static_cast<adc_attenuation_t>(3)); // 11dB attenuation
    }
    
    delay(1000);  // Startup delay
    
    // Print startup message to both USB and Bluetooth
    dualPrintln("=== Biometric Authentication System ===");
    dualPrintln("🔗 USB Serial: Connected");
    dualPrintln("📶 Bluetooth: ESP32-BioAuth (discoverable)");
    dualPrintln("Commands: START_EEG, STOP_EEG, STATUS");
    dualPrintln("Muscle thresholds: [2625, 2627, 2617, 2355] DO YOUR CALIBRATION!!!");
    dualPrintln("Otherwise you will hate yourself");
    dualPrintln("EEG channels: 12 for now, Muscle channels: 4");
    dualPrintln("Data format:");
    dualPrintln("  MUSCLE:binary,decimal,timestamp");
    dualPrintln("  EEG:ch1,ch2,...,ch12,timestamp");
    dualPrintln("===========================================");
    
    // Also print USB-specific message
    Serial.println("Note: Data is being transmitted via both USB and Bluetooth");
    
    // Print Bluetooth connection info
    Serial.println("Bluetooth device name: ESP32-BioAuth");
    Serial.println("Pair with this device to receive wireless data");
}

void loop() {
    unsigned long currentTime = millis();
    
    // Handle commands from both USB and Bluetooth
    handleSerialCommands();
    handleBluetoothCommands();
    
    // Update muscle states at 50Hz
    if (currentTime - lastMuscleUpdate >= MUSCLE_INTERVAL) {
        updateMuscleStates();
        lastMuscleUpdate = currentTime;
    }
    
    // Update EEG data at 100Hz (only if recording enabled)
    if (eegRecording && (currentTime - lastEEGUpdate >= EEG_INTERVAL)) {
        readEEGData();
        lastEEGUpdate = currentTime;
    }
    
    // Print system status at 2Hz
    if (currentTime - lastStatusPrint >= STATUS_INTERVAL) {
        printSystemStatus();
        lastStatusPrint = currentTime;
    }
}

void updateMuscleStates() {
    static bool lastStates[4] = {false, false, false, false};
    bool stateChanged = false;
    
    // Read all muscle channels with hysteresis
    for(int i = 0; i < 4; i++) {
        int rawValue = readFilteredADC(MUSCLE_PINS[i]);
        
        // Apply hysteresis logic
        if (!muscleStates[i] && rawValue > THRESHOLDS[i]) {
            muscleStates[i] = true;
            stateChanged = true;
        } else if (muscleStates[i] && rawValue < (THRESHOLDS[i] - HYSTERESIS)) {
            muscleStates[i] = false;
            stateChanged = true;
        }
    }
    
    // Output muscle state data to both USB and Bluetooth
    // Format: MUSCLE:binary,decimal,timestamp
    int packedBits = 0;
    String binaryStr = "";
    
    for(int i = 0; i < 4; i++) {
        if(muscleStates[i]) {
            packedBits |= (1 << i);
            binaryStr += "1";
        } else {
            binaryStr += "0";
        }
    }
    
    // Send muscle data in protocol format to both outputs
    dualPrint("MUSCLE:");
    dualPrint(binaryStr);
    dualPrint(",");
    dualPrint(packedBits);
    dualPrint(",");
    dualPrint(millis());
    dualPrintln("");
}

void readEEGData() {
    // Format: EEG:ch1,ch2,ch3,...,ch12,timestamp
    dualPrint("EEG:");
    
    for(int i = 0; i < 12; i++) {
        int eegValue = readFilteredADC(EEG_PINS[i]);
        dualPrint(eegValue);
        if(i < 11) {
            dualPrint(",");
        }
    }
    
    dualPrint(",");
    dualPrint(millis());
    dualPrintln("");
}

void handleSerialCommands() {
    while(Serial.available()) {
        char c = Serial.read();
        if(c == '\n' || c == '\r') {
            // Process complete command
            commandBuffer.trim();
            commandBuffer.toUpperCase();
            
            if(commandBuffer == "START_EEG") {
                eegRecording = true;
                dualPrintln("CMD_ACK:EEG_RECORDING_STARTED");
            } else if(commandBuffer == "STOP_EEG") {
                eegRecording = false;
                dualPrintln("CMD_ACK:EEG_RECORDING_STOPPED");
            } else if(commandBuffer == "STATUS") {
                dualPrint("CMD_ACK:STATUS,");
                dualPrint("EEG:");
                dualPrint(eegRecording ? "ON" : "OFF");
                dualPrint(",MUSCLE:ON,UPTIME:");
                dualPrint(millis());
                dualPrint(",BLUETOOTH:");
                dualPrint(SerialBT.hasClient() ? "CONNECTED" : "DISCONNECTED");
                dualPrintln("");
            } else if(commandBuffer.length() > 0) {
                dualPrintln("CMD_ERR:UNKNOWN_COMMAND");
            }
            
            commandBuffer = "";
        } else {
            commandBuffer += c;
        }
    }
}

void handleBluetoothCommands() {
    while(SerialBT.available()) {
        char c = SerialBT.read();
        if(c == '\n' || c == '\r') {
            // Process complete command from Bluetooth
            commandBufferBT.trim();
            commandBufferBT.toUpperCase();
            
            if(commandBufferBT == "START_EEG") {
                eegRecording = true;
                dualPrintln("CMD_ACK:EEG_RECORDING_STARTED");
            } else if(commandBufferBT == "STOP_EEG") {
                eegRecording = false;
                dualPrintln("CMD_ACK:EEG_RECORDING_STOPPED");
            } else if(commandBufferBT == "STATUS") {
                dualPrint("CMD_ACK:STATUS,");
                dualPrint("EEG:");
                dualPrint(eegRecording ? "ON" : "OFF");
                dualPrint(",MUSCLE:ON,UPTIME:");
                dualPrint(millis());
                dualPrint(",BLUETOOTH:");
                dualPrint(SerialBT.hasClient() ? "CONNECTED" : "DISCONNECTED");
                dualPrintln("");
            } else if(commandBufferBT.length() > 0) {
                dualPrintln("CMD_ERR:UNKNOWN_COMMAND");
            }
            
            commandBufferBT = "";
        } else {
            commandBufferBT += c;
        }
    }
}

void printSystemStatus() {
    // Only print if there's muscle activity or every few seconds
    static unsigned long lastFullStatus = 0;
    bool hasActivity = false;
    
    for(int i = 0; i < 4; i++) {
        if(muscleStates[i]) {
            hasActivity = true;
            break;
        }
    }
    
    if(hasActivity || (millis() - lastFullStatus) > 5000) {
        // Create readable muscle state string
        String activeList = "";
        int activeCount = 0;
        
        for(int i = 0; i < 4; i++) {
            if(muscleStates[i]) {
                if(activeCount > 0) activeList += ", ";
                activeList += MUSCLE_NAMES[i];
                activeCount++;
            }
        }
        
        if(activeCount == 0) {
            activeList = "None";
        }
        
        // Create binary representation
        String binaryStr = "";
        int decimal = 0;
        for(int i = 0; i < 4; i++) {
            if(muscleStates[i]) {
                binaryStr += "1";
                decimal |= (1 << i);
            } else {
                binaryStr += "0";
            }
        }
        
        // Print human-readable status to both outputs
        String statusMessage = "STATUS: Muscle=" + binaryStr + " (" + String(decimal) + ") Active: " + activeList;
        statusMessage += " | EEG: " + String(eegRecording ? "Recording" : "Stopped");
        statusMessage += " | BT: " + String(SerialBT.hasClient() ? "Connected" : "Disconnected");
        statusMessage += " | Uptime: " + String(millis()/1000) + "s";
        
        dualPrintln(statusMessage);
        
        if(!hasActivity) {
            lastFullStatus = millis();
        }
    }
}

int readFilteredADC(int pin) {
    // Simple moving average filter (10 samples)
    const int numSamples = 10;
    long sum = 0;
    
    for(int i = 0; i < numSamples; i++) {
        sum += analogRead(pin);
        delayMicroseconds(100);  // Small delay between samples
    }
    
    return sum / numSamples;
} 
