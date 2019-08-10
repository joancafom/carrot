/*
 * Program designed to test out the connections
 * between the Arduino board and the RC controller
 */

unsigned int testPin;

// Actions and pins to test them out
const unsigned int centroGas = 13;
const unsigned int izquierda = 11;
const unsigned int derecha = 9;
const unsigned int freno = 7;

// Delay after each pin write
const unsigned int highDelay = 1*1000;
const unsigned int lowDelay = 10*1000;

void setup() {
     testPin = derecha;
     pinMode(testPin, OUTPUT);
}

void loop() {
   digitalWrite(testPin, HIGH);
   delay(highDelay);
   digitalWrite(testPin, LOW);
   delay(lowDelay);
}
