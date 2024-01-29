#include <DynamixelShield.h>
#include <Wire.h>
#include <avr/dtostrf.h>
#include <Math.h>
#include <StringSplitter.h>

#define delay_I2C                 10
#define NORMALIZATION_COUNT       16 //amount of data for normalization

//#define Addr                  0x0F  // MLX90393 I2C Address is 0x0F(12)
// #define OPEN_MAX_POSITION     253
// #define CLOSE_MAX_POSITION    177
// #define OFFSET_DEGREE_SENSOR   0
#define DEGREE_COMPENSATION   177
// #define MIN_STROKE            2
// #define MAX_STROKE            90

#define GUI_CONTROL           0
#define KALIBRASI             1
#define POTENSIO_CONTROL      2
#define CONTROL_DEMO          3
#define COLLECT_DATASET       4
#define ML_ACT1               5

#define READY                 0
#define REACH_TARGET          1
#define SAFE_GRASPING         2
#define GRIPPER_CLOSING       3
#define SQUEEZING             4
#define RETURN_FRUIT          5
#define FINISH                6

#define _1x1                 0
#define _4x4                 1

#define THRESHOLD_GRASP_LENGTH  5
#define THRESHOLD_GRASP_TORQUE  0.7   
//#define PERIOD_DELAY            5000    

#define POTENSIO_PIN          A0

#define DXL_ID                7
#define DXL_ID2               2
#define DXL_ID1               1
#define DXL_PROTOCOL_VERSION  2.0

#define RAD 57.29577951

const int GRIPPER_LIM[2] = {196,244};
const int UPPER_LIM[2] = {181,287};

#if defined(ARDUINO_AVR_UNO) || defined(ARDUINO_AVR_MEGA2560)
#include <SoftwareSerial.h>
SoftwareSerial soft_serial(7, 8); // DYNAMIXELShield UART RX/TX
#define DXL_SERIAL   Serial
#define DEBUG_SERIAL soft_serial
#elif defined(ARDUINO_SAM_DUE) || defined(ARDUINO_SAM_ZERO)
#define DEBUG_SERIAL SerialUSB
#elif defined(ARDUINO_OpenRB)  // When using OpenRB-150
//OpenRB does not require the DIR control pin.
#define DXL_SERIAL Serial1
//#define DEBUG_SERIAL Serial
#define DEBUG_SERIAL Serial
#else
#define DEBUG_SERIAL Serial
#endif

DynamixelShield dxl;
StringSplitter *stsp;

int x,y,z;
// using namespace ControlTableItem;

void setup() {
  DEBUG_SERIAL.begin(19200);
  dxl.begin(57600);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
  dxl.ping(DXL_ID);
  dxl.ping(DXL_ID2);
  dxl.ping(DXL_ID1);

  dxl.torqueOff(DXL_ID);  // Turn off torque when configuring items in EEPROM area
  dxl.torqueOff(DXL_ID2);  // Turn off torque when configuring items in EEPROM area
  dxl.torqueOff(DXL_ID1);  // Turn off torque when configuring items in EEPROM area

  dxl.setOperatingMode(DXL_ID, OP_POSITION);
  dxl.setOperatingMode(DXL_ID2, OP_POSITION);
  dxl.setOperatingMode(DXL_ID1, OP_POSITION);

  dxl.torqueOn(DXL_ID);
  dxl.torqueOn(DXL_ID2);
  dxl.torqueOn(DXL_ID1);

  gripper_init();
  DEBUG_SERIAL.print("Init finished");
  
  move_arm(0,0,0);
}

void loop() {
  if(DEBUG_SERIAL.available() > 0){
    String data = DEBUG_SERIAL.readString();
    data.trim();
    stsp = new StringSplitter(data, ',', 3);

    if(stsp->getItemCount() != 3){
      DEBUG_SERIAL.println(stsp->getItemCount());
      return;
    }

    x = stsp->getItemAtIndex(0).toInt();
    y = stsp->getItemAtIndex(1).toInt();
    z = stsp->getItemAtIndex(2).toInt();
    
    move_arm(x, y, z);
  }
}

void gripper_init(){
  dxl.writeControlTableItem(ControlTableItem::PROFILE_ACCELERATION, DXL_ID, 0);
  dxl.writeControlTableItem(ControlTableItem::PROFILE_VELOCITY, DXL_ID, 50);

  dxl.writeControlTableItem(ControlTableItem::PROFILE_ACCELERATION, DXL_ID1, 0);
  dxl.writeControlTableItem(ControlTableItem::PROFILE_VELOCITY, DXL_ID1, 50);

  dxl.writeControlTableItem(ControlTableItem::PROFILE_ACCELERATION, DXL_ID2, 0);
  dxl.writeControlTableItem(ControlTableItem::PROFILE_VELOCITY, DXL_ID2, 50);
}

void move_arm(double x, double y, double z){
  double b = atan2(z,x)*RAD; 
  b = b<0 ? 360+b : b;

  DEBUG_SERIAL.println(b);
  servo_degree(b, DXL_ID1);
}

void servo_degree(float target_val, uint8_t id){
  switch(id){
    case DXL_ID:
      target_val = constrain(target_val, GRIPPER_LIM[0], GRIPPER_LIM[1]);
      break;
    case DXL_ID2:
      target_val = constrain(target_val, UPPER_LIM[0], UPPER_LIM[1]);
      break;
    default:
      break;
  }

  dxl.torqueOn(id);
  
  DEBUG_SERIAL.println(String(id) + " " + String(target_val));

  dxl.setGoalPosition(id, target_val, UNIT_DEGREE);
}


