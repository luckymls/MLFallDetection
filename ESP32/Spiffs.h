#ifndef SPIFFS_H
#define SPIFFS_H

#include <Arduino.h>
#include <SPIFFS.h>

class Spiffs {
public:
    Spiffs();                       
    bool begin();                   // Initialize SPIFFS
    bool writeFile(const char* path, const char* message); // Write to a file
    String readFile(const char* path); // Read from a file
    bool appendToFile(const char* path, const char* message); // Append to a file
    bool deleteFile(const char* path); // Delete a file
};

#endif

