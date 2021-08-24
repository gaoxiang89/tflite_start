#include <iostream>
#include <chrono>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include "model_r6.h"

#define RUN_TIMES 1000
std::unique_ptr<tflite::FlatBufferModel> _model;
tflite::ops::builtin::BuiltinOpResolver _resolver;
std::unique_ptr<tflite::Interpreter> _interpreter;

int main()
{
    _model = tflite::FlatBufferModel::BuildFromBuffer((const char *)STARK_MODEL, sizeof(STARK_MODEL));
    if (!_model)
    {
        std::cout << "Cannot read tflite model" << std::endl;
        return -1;
    }
    else
    {
        std::cout << "load tflite model successfully. ";
    }

    if (tflite::InterpreterBuilder(*_model, _resolver)(&_interpreter) != kTfLiteOk)
    {
        std::cout << "Cannot create interpreter" << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Interpreterl successfully" << std::endl;
    }

    if (_interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cout << "Cannot allocate interpreter tensors" << std::endl;
        return -1;
    }
    else
    {
        std::cout << "allocate interpreter tensors successfully" << std::endl;
    }

    /* fake data. 100 samples, 8 channel */
    /* [1 2 3 4 5 6 7 8],[-1 -2 -3 -4 -5 -6 -7 -8],[1 2 3 4 5 6 7 8],[-1 -2 -3 -4 -5 -6 -7 -8], ...... */
    float input_data[100][8];
    float *input = _interpreter->typed_input_tensor<float>(0);
    for (int i = 0; i < 0 + 100; i++)
    {
        for (int ch = 0; ch < 8; ch++)
        {
            if (i % 2 == 0)
                input_data[i][ch] = ch + 1; /* 1 2 3 4 5 6 7 8*/
            else
                input_data[i][ch] = -ch - 1; /* -1 -2 -3 -4 -5 -6 -7 -8*/
        }
    }

    memcpy(input, input_data, 100 * 8 * sizeof(float));

    /*store output result */
    TfLiteTensor *output = _interpreter->output_tensor(0);
    size_t outputSize = output->bytes / sizeof(float);
    std::vector<float> algo_out(outputSize, 0.0);

    auto start = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < RUN_TIMES; n++)
    {
        if (_interpreter->Invoke() != kTfLiteOk)
        {
            std::cout << "Cannot invoke interpreter" << std::endl;
            return -1;
        }

        if (1) /* enable or disable log */
        {
            std::cout << "output size:" << outputSize << std::endl;
            std::cout << "output :";
            for (size_t i = 0; i < outputSize; i++)
            {
                algo_out[i] = output->data.f[i];
                std::cout << " " << algo_out[i] << ", ";
            }
            std::cout << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> tm = end - start; // 毫秒
    std::cout << "run " << RUN_TIMES << " times speed: " << tm.count() << "ms" << std::endl;
    return 0;
}