import serial
import serial.tools.list_ports
import math
import time

def get_port():
    ports = sorted(serial.tools.list_ports.comports())

    for i,(port, desc, hwid) in enumerate(ports):
        print(f"{i} : {port} \t{desc}\t {hwid}")

class ARM_ROBOT:
    def __init__(self, com_port="COM4", baudrate=115200, timeout=0.1) -> None:
        self.ser = None
        self.connected = False
        self.theta1_limit = [181, 287]

        try:
            self.ser = serial.Serial(port=com_port, baudrate=baudrate, timeout=timeout)
            self.connected = True
        except serial.SerialException:
            print("ERROR CONNECTING TO PORT")

    def write(self, data, get_feedback=True):
        self.ser.write(bytes(data, "utf-8"))
        time.sleep(0.15)

        if get_feedback:
            res = self.ser.readline()
            return res.decode()
        else:
            return None
        
    def inverse_kinematics(self, x=0, y=0, z=0):
        length = math.sqrt(math.pow(x,2) + math.pow(z,2))

        p = x-138.113
        q = y-92

        theta0 = math.atan2(z,x)
        theta1 = 1.5708 + math.atan2(q,p)
        theta2 = math.pi - theta1

        theta0 = theta0 * (180/math.pi)
        theta1 = theta1 * (180/math.pi)
        theta2 = theta2 * (180/math.pi)

        theta0 = 360+theta0 if theta0<0 else theta0
        theta1 = 360+theta1 if theta1<0 else theta1
        theta2 = 360+theta2 if theta2<0 else theta2

        return [theta0, theta1, theta2]
    
    def close(self):
        self.ser.close()
    