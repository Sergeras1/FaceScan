#include <Servo.h>

Servo servo;
double angle = 0; // начальное значение угла
int ledPin = 13; // Пин светодиода

void setup()
{
    pinMode(ledPin, OUTPUT);
    Serial.begin(9600);
    servo.attach(3); // указываем номер пина, к которому подключен сервопривод
    servo.write(0); // устанавливаем начальное положение сервопривода
}

void loop()
{
    if (Serial.available())
    {
        char val = Serial.read();

        if (val == '0')
        {
           angle = 0; // если значение больше 0.5, устанавливаем угол 0 градусов
            digitalWrite(ledPin, LOW); // Выключить светодиод
           
        }
        else if (val == '1')
        {
         servo.write(90); // если значение меньше 0.5, устанавливаем угол 90 градусов
            digitalWrite(ledPin, HIGH); // Включить светодиод
             delay(2000);
             servo.write(0);
        }
    }
    //servo.write(angle); // отправляем угол на сервопривод
  //delay(20); // задержка для стабильной работы
}
