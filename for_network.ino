#include <Servo.h>
#include <Wire.h>                       
#include <LiquidCrystal_I2C.h> 

Servo servo;
LiquidCrystal_I2C LCD(0x27,16,2); 

double angle = 0; // начальное значение угла
int ledPin = 13; // Пин светодиода

void setup()
{
    pinMode(ledPin, OUTPUT);
    Serial.begin(9600);
    LCD.init();            
    LCD.backlight(); 
    servo.attach(3); // указываем номер пина, к которому подключен сервопривод
    servo.write(0); // устанавливаем начальное положение сервопривода
}

void loop()
{
  LCD.setCursor(0, 0);
  LCD.print("Face ");
   
    if (Serial.available())
    {
        char val = Serial.read();

        if (val == '0')
        {
           angle = 0; // если значение больше 0.5, устанавливаем угол 0 градусов
            digitalWrite(ledPin, LOW); // Выключить светодиод
            LCD.setCursor(0, 0);
            LCD.print("Face no");
            delay(3000);
            LCD.clear();
            delay(200);
        }
        else if (val == '1')
        {
         servo.write(90); // если значение меньше 0.5, устанавливаем угол 90 градусов
            digitalWrite(ledPin, HIGH); // Включить светодиод
            LCD.setCursor(0, 0);
            LCD.print("Face ok");
            delay(5000);
            servo.write(0);
            LCD.clear();
            delay(200);
        }
    }
}
