const unsigned int pinLED = 13;
 
void setup() 
{
   // Initialize Serial Communications on 9600
   Serial.begin(9600);

   // Set Board's 13th PIN to OUTPUT
   pinMode(pinLED, OUTPUT);
}
 
void loop()
{
   // If the PC has sent instructions
   if (Serial.available()>0) 
   {
      char option = Serial.read();

      // Assert instruction correctness
      if (option >= '1' && option <= '9')
      {
         // Convert from ASCII Char to int
         int numOption = (int) (option - '0');
         Serial.println(numOption);

         // Make the LED blink several times
         for (int i = 0;i<numOption;i++) 
         {
            digitalWrite(pinLED, HIGH);
            delay(100);
            digitalWrite(pinLED, LOW);
            delay(200);
         }
      }
   }
}
