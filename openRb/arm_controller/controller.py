import serial
import serial.tools.list_ports
import math
import time


def inverse_kinematics(x, y, z):
    b = math.atan2(z,x)*(180/math.pi)
    b = 360+b if b<0 else b

    u = math.atan2(y,x)*(180/math.pi)
    u = 360+u if u<0 else u
    return [b,u]

def ser_write(dat):
    ser.write(bytes(dat, 'utf-8')) 
    time.sleep(0.05) 
    fback = ser.readline() 
    return fback



ports = sorted(serial.tools.list_ports.comports())

for i,(port, desc, _) in enumerate(ports):
    print(f"{i} : {port} \t{desc}")

port_sel = int(input("Select Port Number: "))

ser = serial.Serial(port=ports[port_sel][0], baudrate=115200, timeout=0.1)

print(f"\nPort Selected: {ports[port_sel][0]} \n")



while True:
    coor_x = int(input("X: "))
    coor_y = int(input("Y: "))
    coor_z = int(input("Z: "))

    i_k = inverse_kinematics(coor_x, coor_y, coor_z)
    
    ser_write(f"{i_k[0]:.2f},{i_k[1]:.2f}")
    
