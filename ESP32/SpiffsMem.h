#ifndef SPIFFS_H
#define SPIFFS_H

#include <Arduino.h>
#include <SPIFFS.h>

class SpiffsMem {
public:
    Spiffs();                       
    bool begin();                   // Initialize SPIFFS
    bool writeFile(const char* path, const char* message); // Write to a file
    String readFile(const char* path); // Read from a file
    bool appendToFile(const char* path, const char* message); // Append to a file
    bool deleteFile(const char* path); // Delete a file
    size_t checkMemory(); // Get SPIFFS memory size in bytes 
    bool fileExists(const char* path); // Check if a file already exists in memory
    
};

#endif

