#include <Servo.h>
#include <Wire.h>                       
#include <LiquidCrystal_I2C.h> 
#include <SoftwareSerial.h>
#include <DFRobotDFPlayerMini.h>

Servo servo;
LiquidCrystal_I2C LCD(0x27,16,2); 
SoftwareSerial mySoftwareSerial(10, 11); // RX, TX
DFRobotDFPlayerMini myDFPlayer;

double angle = 0; // начальное значение угла
int ledPin = 13; // Пин светодиода

void setup()
{
    pinMode(ledPin, OUTPUT);
    Serial.begin(9600);
    LCD.init();            
    LCD.backlight(); 
    // Устанавливаем скорость соединения
  mySoftwareSerial.begin(9600);
     // Инициализируем DFPlayer mini
  if (!myDFPlayer.begin(mySoftwareSerial)) 
  {
    Serial.println(F("Не удалось инициализировать DFPlayer mini!"));
    

    while(1);
  }
    servo.attach(3); // указываем номер пина, к которому подключен сервопривод
    servo.write(0); // устанавливаем начальное положение сервопривода
}

void loop()
{
  LCD.setCursor(0, 0);
  LCD.print("Close ");
   
    if (Serial.available())
    {
        char val = Serial.read();

        if (val == '0')
        {
           angle = 0; // если значение больше 0.5, устанавливаем угол 0 градусов
            digitalWrite(ledPin, LOW); // Выключить светодиод
            LCD.setCursor(0, 0);
            LCD.print("Face isn't found");
            LCD.setCursor(0, 1);
            LCD.print("Close");
            myDFPlayer.play(2); // Проигрываем первую аудиозапись
            delay(5000);
            LCD.clear();
            delay(200);
        }
        else if (val == '1')
        {
         servo.write(90); // если значение меньше 0.5, устанавливаем угол 90 градусов
            digitalWrite(ledPin, HIGH); // Включить светодиод
            LCD.setCursor(0, 0);
            LCD.print("Face is found");
            LCD.setCursor(0, 1);
            LCD.print("Open");
            myDFPlayer.play(1); // Проигрываем первую аудиозапись
            delay(5000);
            servo.write(0);
            LCD.clear();
            delay(200);
        }
    }
}
