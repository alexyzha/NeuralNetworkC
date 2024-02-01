#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

struct Line {
    public:
    int actual;
    std::vector<float> embedded;
    Line(int a, std::vector<float> e) : actual(a), embedded(e) {};
};

class VectorizedTable {
    public:
    std::vector<Line> alldata;
    VectorizedTable() : alldata() {}    
    //functions
    void AddLine(int a, std::vector<float> e);
    void AddLinesGlobal (std::ifstream& file, int totalLines);
};