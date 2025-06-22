# BrainOn 🧠🔐

**BrainOn** is an open‑source, multimodal biometric authentication framework that fuses **electroencephalography (EEG)** and **electromyography (EMG)** to unlock and authorize cryptocurrency wallets. By translating your unique brain‑wave and muscle‑activation signatures into cryptographically secure hashes, BrainOn delivers friction‑free, high‑assurance access control for digital assets.

This is a Stellar Hack Pera Hackathon project, project presentation:
https://www.canva.com/design/DAGrDmWBmlU/ejefM7qx-oT7NajoBhnfwA/edit
---

## ✨ Core Features

| Category          | Highlights                                                                               |
| ----------------- | ---------------------------------------------------------------------------------------- |
| **Functionality** | Can transfer money to a wallet, can swap XLM to USDC                                     |
| **Sensors**       | 12‑channel EEG @ 128 Hz · 4‑channel EMG @ 50 Hz on ESP32                                 |
| **ML Pipeline**   | Random‑Forest primary → CNN fallback · artifact rejection · temporal smoothing using PSD |
| **Latency**       | Target ≤ 50 ms end‑to‑end                                                                |
| **APIs**          | WebSocket streaming · REST hooks · CORS enabled                                          |
| **Front‑End**     | React (TypeScript) dashboard (in ``) with WalletConnect & live plots                     |
| **Run Modes**     | `demo` · `real` for rapid testing                                                        |

---

## 📂 Repository Layout (unchanged)

```
brainOn/
├── brainOnClient/       # React (wallet UI & live visualisation)
├── host/                # Host computer to work with ESP32 (optional)
├── src/                 # C++ firmware (ESP32 PlatformIO)
├── examples/            # Misc, has many uses(filters, calibration etc.)
├── *.py                 # Python back‑end scripts (classifier, websocket, etc.)
└── LICENSE              # GPL‑3.0
```

*The repo currently mixes Python back‑end, C++ firmware, and React front‑end in one place. Use the quick‑start below to run each part.*

---

## ⚡ Quick Start

### 1 · Flash the ESP32

```bash
# Build & upload firmware (requires PlatformIO ≥ 6.0)
platformio run -d src -t upload

##If done for the first time use simple_calibration.cpp
#to calibrate to your body
```

Update `EEG_PINS[]`, `BLUETOOTH_DEVICE_NAME`, etc. in `src/main.cpp` if needed.

### 2 · Install Python Dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # scikit‑learn, tensorflow, websockets …
```

### 3 · Launch Real‑Time Services

```bash
# WebSocket + REST gateway
python websocket_server.py --mode real --host 0.0.0.0 --port 5000 \
  --config examples/config.yaml

# Classifier (separate shell)
python eeg_classifier.py --config examples/config.yaml
```

### 4 · Start the Front‑End Dashboard

```bash
cd brainOnClient
npm i       # or pnpm i / yarn
npm run dev # Vite dev‑server → http://localhost:5173
```

The dashboard auto‑connects to `ws://localhost:5000`, visualises live EEG/EMG, shows current authentication state, and exposes **Stellar** wallet actions once confidence thresholds are met.

---

## 🔒 Security Model

- **Entropy** — brain & muscle signals are personal and stochastic.
- **Replay Protection** — session nonce + state hash → SHA‑256.
- **Confidence Thresholds** — adjustable per operation (`unlock`, `transaction`).
- **Auto‑Lock** — configurable timeouts & failed‑attempt limits.
- **Tamper Detection** — artifact flag ⇒ classification suppressed.

> BrainOn is a research prototype. Use at your own risk and keep recovery phrases offline.

---

## ⚙️ Security Model

![brainOn basic workflow](https://github.com/user-attachments/assets/b1df8819-8fab-4d6a-b7a2-d987eadf7938)


---

## 🗓️ Roadmap (matching open Issues)

-

---

## 🤝 Contributing

Pull requests and issue reports are welcome! Please open an issue before large‑scale changes.

---

## 📜 License

This project is released under the **GNU GPL v3**. See [LICENSE](LICENSE) for details.

