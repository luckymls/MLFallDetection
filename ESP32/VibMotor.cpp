#include "VibMotor.h"


VibMotor::VibMotor(int motorPin){

    this->motorPin = motorPin;
    pinMode(motorPin, OUTPUT);

    this->maxTimeOn = 200;
    this->off();
}

void VibMotor::on(int delay_time){ // Delay in ms
    if(delay_time > this->maxTimeOn)
        delay_time = this->maxTimeOn;

    digitalWrite(this->motorPin, HIGH);
    delay(delay_time);
    this->off();
}

void VibMotor::off(){
    digitalWrite(this->motorPin, LOW);
}





