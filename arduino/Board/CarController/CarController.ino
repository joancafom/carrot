/*
 * Performs actions received over Serial port
 * on the actual RC Controller to move the car.
 * 
 * Supported actions: 
 * 
 *  ['Izquierda', 'Centro', 'Derecha', 'Centro-Gas', 'Freno']
 *  
 *  Actions->Buttons Mapping
 *  
 *  'Izquierda'-> S2
 *  'Centro'-> null
 *  'Derecha' -> S3
 *  'Centro-Gas' -> S1
 *  'Freno' -> S4
 *  
 *  Buttons->Pin Mapping
 *  
 *  S1 -> 13
 *  S2 -> 11
 *  S3 -> 9
 *  S4 -> 7
 */

const unsigned int S1 = 13;
const unsigned int S2 = 11;
const unsigned int S3 = 9;
const unsigned int S4 = 7;
 
void setup() 
{
   // Initialize Serial Communications on 9600
   Serial.begin(9600);

   // Set all pins to Output
   pinMode(S1, OUTPUT);
   pinMode(S2, OUTPUT);
   pinMode(S3, OUTPUT);
   pinMode(S4, OUTPUT);
   
}
 
void loop()
{
   // If the PC has sent instructions
   if(Serial.available()>0) 
   {
      char action = Serial.read();

      // Assert instruction correctness
      if(action >= '0' && action < '5')
      {

         if(action == '0'){

            // Izquierda
            digitalWrite(S2, HIGH);
            delay(100);
            digitalWrite(S2, LOW);
          
         }
         else if(action == '1'){

            // Centro
            ;
          
         }
         else if(action == '2'){

            // Derecha
            digitalWrite(S3, HIGH);
            delay(100);
            digitalWrite(S3, LOW);
          
         }
         else if(action == '3'){

            // Centro-Gas
            digitalWrite(S1, HIGH);
            delay(100);
            digitalWrite(S1, LOW);
          
         }
         else if(action == '4'){

            // Freno
            digitalWrite(S4, HIGH);
            delay(100);
            digitalWrite(S4, LOW);
          
         }
      }
   }
}
