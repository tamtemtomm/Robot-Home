void setup() {
  Serial.begin(9600);
  pinMode(4, INPUT);
  pinMode(3, OUTPUT);
}

void loop() {
  Serial.println(digitalRead(4));
  if(digitalRead(4)) digitalWrite(3, HIGH);
  else digitalWrite(3,LOW);
}
