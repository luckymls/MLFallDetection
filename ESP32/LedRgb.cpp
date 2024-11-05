#include "LedRgb.h"

LedRgb::LedRgb(int redPin, int greenPin, int bluePin){

    this->redPin = redPin;
    this->greenPin = greenPin;
    this->bluePin = bluePin;

    pinMode(redPin, OUTPUT);
    pinMode(greenPin, OUTPUT);
    pinMode(bluePin, OUTPUT);

    off();
}

void LedRgb::setColor(int rVal, int gVal, int bVal){
    analogWrite(this->redPin, rVal);
    analogWrite(this->greenPin, gVal);
    analogWrite(this->bluePin, bVal);
}

void LedRgb::off(){
    analogWrite(this->redPin, 0);
    analogWrite(this->greenPin, 0);
    analogWrite(this->bluePin, 0);
}





