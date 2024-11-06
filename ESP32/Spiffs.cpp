#include "Spiffs.h"

Spiffs::Spiffs() {}

bool Spiffs::begin() {

    if (!SPIFFS.begin(true))
        return false;    
    return true;
}

bool Spiffs::writeFile(const char* path, const char* message) {

    File file = SPIFFS.open(path, FILE_WRITE);

    if (!file) {
        return false;
    }

    if (!file.print(message)) {
        file.close();
        return false;
    }
    
    file.close();
    return true;
}

String Spiffs::readFile(const char* path) {

    File file = SPIFFS.open(path, FILE_READ);
    String content;

    if (!file)
        return String(); // failed
    
    while (file.available()) {
        content += char(file.read());
    }

    file.close();
    return content;
}

bool Spiffs::appendToFile(const char* path, const char* message) {

    File file = SPIFFS.open(path, FILE_APPEND);

    if (!file) {
        Serial.printf("Failed to open file %s for appending\n", path);
        return false;
    }

    if (!file.print(message)) {
        file.close();
        return false;
    }

    file.close();
    return true;
}

bool Spiffs::deleteFile(const char* path) {

    if (SPIFFS.remove(path)) 
        return true;
    
    return false;
    
}
