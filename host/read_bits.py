import serial, sys
ser = serial.Serial(sys.argv[1], 115200)
while True:
    b = ser.read(1)
    print(f"{b[0]:08b}") 
