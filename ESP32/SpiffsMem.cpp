#include "Spiffs.h"

SpiffsMem::Spiffs() {}

bool SpiffsMem::begin() {

    if (!SPIFFS.begin(true))
        return false;    
    return true;
}

bool SpiffsMem::writeFile(const char* path, const char* message) {

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

String SpiffsMem::readFile(const char* path) {

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

bool SpiffsMem::appendToFile(const char* path, const char* message) {

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

bool SpiffsMem::deleteFile(const char* path) {

    if (SPIFFS.remove(path)) 
        return true;
    
    return false;
    
}

size_t SpiffsMem::checkMemory() {
    size_t totalBytes = SPIFFS.totalBytes();
    size_t usedBytes = SPIFFS.usedBytes();
    size_t spaceLeftBytes = totalBytes - usedBytes;
    return spaceLeftBytes;
}