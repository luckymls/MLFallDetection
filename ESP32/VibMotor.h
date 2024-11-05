#ifndef VIBMOTOR_H
#define VIBMOTOR_H


#include <Arduino.h>

class VibMotor {
    private:
        int motorPin;
        int maxTimeOn;

    public:
        VibMotor(int motorPin);
        void on(int delay=100);
        void off();
};


#endif