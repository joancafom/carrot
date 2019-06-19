/*
 * Program to be executed on the Arduino Board.
 * 
 * Constantly listens to the Serial Port, where 
 * instructions coming from the computer are 
 * sent. When an action is received, 
 * the program first validates it and proceeds
 * to activate the corresponding pins attatched
 * to the RC Car remote controller.
 * 
 * The provided specification sheet provides 
 * more details about the hardware architecture.
 * 
 * RC Car remote controller has 4 physical buttons:
 * S1, S2, S3, S4
 * 
 * 
 * /------ SUPPORTED ACTIONS ------/
 * |                               |
 * |      'Izquierda'  ==> 0,      |
 * |      'Centro'     ==> 1,      |
 * |      'Derecha'    ==> 2,      |
 * |      'Centro-Gas' ==> 3,      |
 * |      'Freno'      ==> 4,      |
 * |                               |
 * /-------------------------------/
 * 
 * /----- ACTIONs to BUTTONs -----/
 * |                              |
 * |      'Izquierda'  ==> S2,    |
 * |      'Centro'     ==> null,  |
 * |      'Derecha'    ==> S3,    |
 * |      'Centro-Gas' ==> S1,    |
 * |      'Freno'      ==> S4,    |
 * |                              |
 * /------------------------------/
 * 
 * /------- BUTTONs to PINs ------/
 * |                              |
 * |          S1  ==> 13,         |
 * |          S2  ==> 11,         |
 * |          S3  ==> 9,          |
 * |          S4 ==> 7,           |
 * |                              |
 * /------------------------------/
 * 
 */

// ----- CONSTANTS -----

// BUTTONs to PINs
const unsigned int S1 = 13;
const unsigned int S2 = 11;
const unsigned int S3 = 9;
const unsigned int S4 = 7;

// ms to wait for a new action to come through
// SerialPort
const long waitingInverval = 500;

// ----- COMMON VARS -----
// Last action performed
static char lastAction;
// Last moment an action took place
static unsigned long previousMillis = 0;


void setup() 
{
   // Initialize Serial Communications on 9600
   Serial.begin(9600);

   // All PINs are outputs
   pinMode(S1, OUTPUT);
   pinMode(S2, OUTPUT);
   pinMode(S3, OUTPUT);
   pinMode(S4, OUTPUT);
   
}
 
void loop()
{   
   // Current time
   unsigned long currentMillis = millis();
   
   // If the PC has sent an action, we need to perform
   // it
   if(Serial.available()>0){
      
      // Read the data on the SerialPort
      // and update the last time
      // we received an action
      char action = Serial.read();
      previousMillis = currentMillis;

      // Assert instruction correctness.
      // We only truly perform an action if it
      // wasn't executed right before (as in that 
      // case, its corresponding output PINs are 
      // still enabled)
      if(action >= '0' && action < '5' && lastAction != action)
      {
         lastAction = action;

         // We need to disable all outputs to ensure
         // we only light the correct ones up.
         disableOutputs();
         
         if(action == '0'){

            // Izquierda
            digitalWrite(S2, HIGH);
            
            // Ensure the physical mechanisms have
            // time to execute the action 
            delay(500);
            
         }else if(action == '1'){

            // Centro
            // No-op
            ;
          
         }else if(action == '2'){

            // Derecha
            digitalWrite(S3, HIGH);
            delay(500);
            
         }else if(action == '3'){

            // Centro-Gas
            digitalWrite(S1, HIGH);
            delay(250); 
            
         } else if(action == '4'){
          
            // Freno
            digitalWrite(S4, HIGH);
            delay(250);
            digitalWrite(S4, LOW);

         }
      }

   // If no action is received, we keep the PINs' curent status
   // for a time no longer than the permitted. After that time
   // expires, we disable all PINs.
   }else if (currentMillis - previousMillis >= waitingInverval) {
    
      previousMillis = currentMillis;
      disableOutputs();
      
      // Null Character
      lastAction = '\0';
      
   }
}

// Set all output PINs to LOW
void disableOutputs(){
           
    // Izquierda
    digitalWrite(S2, LOW);

    // Derecha
    digitalWrite(S3, LOW);
  
    // Centro-Gas
    digitalWrite(S1, LOW);
  
    // Freno
    digitalWrite(S4, LOW);
  
  
}
