#include "LineReader.h"

void VectorizedTable::AddLine(int a, std::vector<float> e) {
    VectorizedTable::alldata.push_back(Line(a, e));
}

void VectorizedTable::AddLinesGlobal(std::ifstream& file, int totalLines) {
    for(int i = 0; i < totalLines; i++) {
        std::string tempLine;
        int actual;
        std::vector<float> tempVec;
        getline(file, tempLine);
        if(tempLine.size() < 5) continue;
        while(tempLine[tempLine.length()-2] != ']') {
            std::string tempTempLine;
            getline(file, tempTempLine);
            tempLine += " " + tempTempLine;
        }     
        actual = stoi(tempLine.substr(0));
        if(actual > 1) actual = 1;
        int startingIndex = 4;
        auto convert = [&startingIndex](std::string line, int index) -> float {
            int end = index;
            while(line[end] != ' ' && line[end] != ']') end++;
            startingIndex = end;
            return stof(line.substr(index, end-index));
        };
        while(startingIndex < tempLine.size()-2) {
            if(tempLine[startingIndex] == ']') break;
            if(tempLine[startingIndex] != ' ') {
                float tempF;
                tempF = convert(tempLine, startingIndex);
                tempVec.push_back(tempF); }
            else startingIndex++;
        }
        AddLine(actual, tempVec);
    }
}