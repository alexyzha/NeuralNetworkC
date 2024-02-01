#include "MachineLearningFinal.h"
#include "MLAccessories.h"

/****** INPUT NODE *******/

void InputNode::GetInputIn(std::vector<float>& in)
{
    input = in;
}

/****** FIRST LAYER *******/

void FirstHiddenLayer::Init(float lower, float upper)
{
    for(int i = 0; i < 50; i++)
    {
        inputs.push_back(0.0f);
        weights.push_back(RandomFloat(lower, upper));
    }
    bias = RandomFloat(lower, upper);
    output = 0.0f;
}

void FirstHiddenLayer::CalculateOutput()
{
    int temp = 0.0f;
    for(int i = 0; i < 50; i++) (temp += inputs[i] * weights[i]);
    output = ReLU(temp + bias);
}

void FirstHiddenLayer::SetInput(std::vector<float> in)
{
    inputs = in;
}

/****** SECOND LAYER *******/

void SecondHiddenLayer::Init(float upper, float lower)
{
    for(int i = 0; i < NODE_COUNT; i ++)
    {
        inputs.push_back(0.0f);
        weights.push_back(RandomFloat(lower, upper));
    }
    bias = RandomFloat(lower, upper);
    output = 0.0f;
}

void SecondHiddenLayer::CalculateOutput()
{
    float temp = 0.0f;
    for(int i = 0; i < NODE_COUNT; i++) temp += (inputs[i] * weights[i]);
    output = ReLU(temp + bias);
}

void SecondHiddenLayer::SetInput(std::vector<float>& in)
{
    inputs = in;
}

/****** OUTPUT NODE *******/

void OutputNode::Init(float upper, float lower)
{
    for(int i = 0; i < NODE_COUNT; i ++)
    {
        inputs.push_back(0.0f);
        weights.push_back(RandomFloat(lower, upper));
    }
    bias = RandomFloat(lower, upper);
    output = 0.0f;
}

void OutputNode::CalculateOutput()
{
    float temp = 0.0f;
    for(int i = 0; i < NODE_COUNT; i++) temp += (inputs[i] * weights[i]);
    output = Sigmoid(temp + bias);
}

void OutputNode::SetInput(std::vector<float>& in)
{
    inputs = in;
}

/****** NETWORK *******/

void Network::InitNetwork()
{
    mInputNode = new InputNode;
    for(int i = 0; i < NODE_COUNT; i++) {
        mFirstHidden[i] = new FirstHiddenLayer;
        mSecondHidden[i] = new SecondHiddenLayer;
        mFirstHidden[i]->Init(50,25);
        mSecondHidden[i]->Init(25,25);
    }
    mOutputNode = new OutputNode;
    mOutputNode->Init(25,1);
    AdamSecondLayerM1 = std::vector<std::vector<float>>(25, std::vector<float>(25, 0.0f));
    AdamSecondLayerV1 = std::vector<std::vector<float>>(25, std::vector<float>(25, 0.0f));
    AdamFirstLayerM1 = std::vector<std::vector<float>>(25, std::vector<float>(50, 0.0f));
    AdamFirstLayerV1 = std::vector<std::vector<float>>(25, std::vector<float>(50, 0.0f));
}

/****** FORWARDS *******/

void Network::ForwardPass(Line& ln)
{
    mInputNode->InputNode::GetInputIn(ln.embedded);
    //calculating first layer
    std::vector<float> secondLayerInputs;
    for(int i = 0; i < NODE_COUNT; i++) {
        mFirstHidden[i]->SetInput(mInputNode->input);
        mFirstHidden[i]->CalculateOutput();
        secondLayerInputs.push_back(mFirstHidden[i]->output);
        mFirstHidden[i]->localGradient = 0.0f;
    }
    //calculating second layer
    std::vector<float> outputLayerInputs;
    for(int i = 0; i < NODE_COUNT; i++) {
        mSecondHidden[i]->SetInput(secondLayerInputs);
        mSecondHidden[i]->CalculateOutput();
        outputLayerInputs.push_back(mSecondHidden[i]->output);
        mSecondHidden[i]->localGradient = 0.0f;
    }
    //calculating output node
    mOutputNode->SetInput(outputLayerInputs);
    mOutputNode->CalculateOutput();
}

/****** BACKWARDS *******/

float Network::CalculateLoss(float actual, float predicted)
{
    //edge case clamping
    if(predicted <= 0.0001f) predicted = 0.0001f;
    if(predicted >= 0.9999f) predicted = 0.9999f;
    //binary cross entropy:
    return -(actual * log(predicted) + (1.0f - actual) * log(1.0f - predicted));
}

float Network::CalculateGradient(float actual, float predicted)
{
    //edge case clamping
    if(predicted <= 0.0001f) predicted = 0.0001f;
    if(predicted >= 0.9999f) predicted = 0.9999f;
    //calculate gradient
    return((predicted - actual)/(predicted * (1.0f - predicted)));
}

void Network::BackPropagate(float actual, float predicted)
{
    float loss = Network::CalculateLoss(actual, predicted);
    float gradient = Network::CalculateGradient(actual, predicted);
    for(int i = 0; i < NODE_COUNT; i++)
    {
        //change output node weights
        mOutputNode->weights[i] -= gradient * AnyMax(0.0f,mSecondHidden[i]->output) * LEARNING_RATE;
        //calculating partial gradient for second 2
        mSecondHidden[i]->localGradient += gradient * mOutputNode->weights[i] * (mSecondHidden[i]->output > 0.0f ? 1.0f : 0.0f);
    }
    CalculateSecond();
    CalculateFirst();
}

void Network::CalculateSecond()
{
    for(int i = 0; i < NODE_COUNT; i++)
    {
        for(int j = 0; j < NODE_COUNT; j++)
        {   
            //ADAM calculations
            float GRAD_WEIGHT = mFirstHidden[j]->output * mSecondHidden[i]->localGradient;
            AdamSecondLayerM1[i][j] = B1 * AdamSecondLayerM1[i][j] + (1 - B1) * GRAD_WEIGHT;
            AdamSecondLayerV1[i][j] = B2 * AdamSecondLayerV1[i][j] + (GRAD_WEIGHT * GRAD_WEIGHT);
            //calculating partial gradient for first layer
            mFirstHidden[j]->localGradient += mSecondHidden[i]->localGradient * mSecondHidden[i]->weights[j] * (mFirstHidden[j]->output > 0.0f ? 1.0f : 0.0f);
            //further ADAM calculations
            float MCorr = AdamSecondLayerM1[i][j] / (1 - pow(B1,GLOBAL_ITERATION));
            float VCorr = AdamSecondLayerV1[i][j] / (1 - pow(B2,GLOBAL_ITERATION));
            mSecondHidden[i]->weights[j] -= LEARNING_RATE * MCorr / (sqrt(VCorr) + EPSILON);
        }
    }
}

void Network::CalculateFirst()
{
    for(int i = 0; i < NODE_COUNT; i++)
    {
        for(int j = 0; j < 50; j++)
        {   
            //ADAM calculations
            float GRAD_WEIGHT = mInputNode->input[j] * mFirstHidden[i]->localGradient;
            AdamFirstLayerM1[i][j] = B1 * AdamFirstLayerM1[i][j] + (1 - B1) * GRAD_WEIGHT;
            AdamFirstLayerV1[i][j] = B2 * AdamFirstLayerV1[i][j] + (GRAD_WEIGHT * GRAD_WEIGHT);
            float MCorr = AdamFirstLayerM1[i][j] / (1 - pow(B1,GLOBAL_ITERATION));
            float VCorr = AdamFirstLayerV1[i][j] / (1 - pow(B2,GLOBAL_ITERATION));
            mFirstHidden[i]->weights[j] -= LEARNING_RATE * MCorr / (sqrt(VCorr) + EPSILON);
        }
    }
}