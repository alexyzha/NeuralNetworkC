#include "MLAccessories.h"

template<typename T>
T AnyMax(T a, T b)
{
    if(a > b) return a;
    return b;
}

float RandomFloat(float low, float high)
{
    //seed
    static std::random_device seed;
    static std::mt19937 gen(seed());
    std::normal_distribution<float> dist(0, sqrt(2.0f/(low + high)));
    return dist(gen);
}

float Sigmoid(float f)
{
    return 1.0f/(1.0f + exp(-f));
}

float SigDer(float f)
{
    return f*(1.0f - f);
}

float ReLU(float in)
{
    return AnyMax(in, 0.0f);
}