#pragma GCC optimize("03")
#include "LineReader.h"

struct InputNode;
struct FirstHiddenLayer;
struct SecondHiddenLayer;
struct OutputNode;
class NodeNet;
const int NODE_COUNT = 25;
const float LEARNING_RATE = 0.0045f;

struct InputNode
{
public:
    std::vector<float> input;
    //functions
    void GetInputIn(std::vector<float>& in);
};

struct FirstHiddenLayer
{
public:
    std::vector<float> inputs;
    std::vector<float> weights;
    float bias;
    float output;
    float localGradient;
    //functions
    void Init(float upper, float lower); 
    void CalculateOutput();
    void SetInput(std::vector<float> in);
};

struct SecondHiddenLayer
{
public:
    std::vector<float> inputs;
    std::vector<float> weights;  
    float bias;
    float output;
    float localGradient;
    //functions
    void Init(float upper, float lower); 
    void CalculateOutput();
    void SetInput(std::vector<float>& in);
};

struct OutputNode
{
public:
    std::vector<float> inputs;
    std::vector<float> weights;
    float bias;
    float output;
    //functions
    void Init(float upper, float lower);
    void CalculateOutput();
    void SetInput(std::vector<float>& in);
};

class Network
{
public:
    InputNode* mInputNode;
    FirstHiddenLayer* mFirstHidden[NODE_COUNT];
    SecondHiddenLayer* mSecondHidden[NODE_COUNT];
    OutputNode* mOutputNode;
    void InitNetwork();
    void ForwardPass(Line& ln);
    //backpropagation
    //hyperparameters
    const float B1 = 0.9f;
    const float B2 = 0.999f;
    const float EPSILON = 1e-8f;
    int GLOBAL_ITERATION = 1;
    //first moment
    std::vector<std::vector<float>> AdamSecondLayerM1;
    std::vector<std::vector<float>> AdamFirstLayerM1;
    //second moment
    std::vector<std::vector<float>> AdamSecondLayerV1;
    std::vector<std::vector<float>> AdamFirstLayerV1;
    float CalculateLoss(float actual, float predicted);
    float CalculateGradient(float actual, float predicted);
    void BackPropagate(float actual, float predicted);
    void CalculateSecond();
    void CalculateFirst();
};