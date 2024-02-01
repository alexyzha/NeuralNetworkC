#include "MachineLearningFinal.h"
#include "MLAccessories.h"

//to compile:
// g++ -std=c++11 -o MachineLearningFinal MLAccessories.cpp MachineLearningObjects.cpp LineReader.cpp MachineLearningFinal.cpp 

int main()
{
    //preinit
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL), std::cout.tie(NULL);
    //init network
    Network NET;
    NET.InitNetwork();

    //file loading
    VectorizedTable TrainData;
    std::string FilePath = "/PATH_TO/fixedRandom/100k.csv";
    std::ifstream file(FilePath);
    if(!file.is_open())
    {
        std::cout << "UNABLE TO OPEN FILE\n";
        return -1;
    }

    //adding lines to TrainData
    TrainData.AddLinesGlobal(file,100000);

    //train
    while(NET.GLOBAL_ITERATION <= 100000)
    {
        NET.ForwardPass(TrainData.alldata[NET.GLOBAL_ITERATION]);
        float predicted = NET.mOutputNode->output;
        float actual = TrainData.alldata[NET.GLOBAL_ITERATION].actual;
        NET.BackPropagate(actual, predicted);
        NET.GLOBAL_ITERATION++;
    }
    
    //training network
    float avgdif = 0.0f;
    float correct = 0;

    //test data
    VectorizedTable TestData;
    std::string FilePath2 = "/PATH_TO/randomBasic.csv";
    std::ifstream file2(FilePath2);
    if(!file.is_open())
    {
        std::cout << "FILE 2 BAD\n";
        return -2;
    }

    //test loop
    TestData.AddLinesGlobal(file2,20000);
    for(int i = 0; i < 20000; i++)
    {
        NET.ForwardPass(TestData.alldata[i]);
        float predicted = NET.mOutputNode->output;
        float temp = std::abs(TestData.alldata[i].actual-predicted);
        if(temp < 0.5f) correct++;
        avgdif += temp;
    }

    //results
    std::cout << "Correct: " << correct << std::endl;
    std::cout << "Percent: " << (correct/20000)*100.0f << std::endl;
    std::cout << "Avgdif: " << (avgdif/20000) << std::endl;

    return 0;
}
