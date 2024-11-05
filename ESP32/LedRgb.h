#ifndef LEDRGB_H
#define LEDRGB_H

#include <Arduino.h>

class LedRgb {
    private:
        int redPin;
        int greenPin;
        int bluePin;

    public:
        LedRgb(int rPin, int gPin, int bPin);
        void setColor(int rVal, int gVal, int bVal);
        void off();
};

#endif
