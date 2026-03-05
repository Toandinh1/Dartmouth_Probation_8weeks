/* Copyright 2024 johnos, TensorFlow Authors. All Rights Reserved.

This sketch is derived from the classic Hello World example of the general
TensorFlow Lite Micro library.
It was adapted and simplified by (based on original work by Chirale)
to conform to the typical style of Arduino sketches.
It has been tested on an ESP32 Dev Board.
The sketch implements a Deep Neural Network pre-trained on calculating
the function sin(x).
By sending a value between 0 and 2*Pi via the Serial Monitor,
both the value inferred by the DNN model and the actual value
calculated using the Arduino math library are displayed.

It shows how to use MicroTFLite Library to run a TensorFlow Lite model.

For more information read the library documentation
at: https://github.com/johnosbb/MicroTFLite

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// include main library header file
#include <MicroTFLite.h>

// include static array definition of pre-trained model
#include "model.h"

// The Tensor Arena memory area is used by TensorFlow Lite to store input, output and intermediate tensors
constexpr int kTensorArenaSize = 2000;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

void setup()
{
    // Initialize serial communications and wait for Serial Monitor to be opened
    Serial.begin(115200);
    while (!Serial)
        ;

    Serial.println("Sine(x) function inference example.");
    Serial.println("Initializing TensorFlow Lite Micro Interpreter...");

    // Initialize the model
    if (!ModelInit(model, tensor_arena, kTensorArenaSize))
    {
        Serial.println("Model initialization failed!");
        while (true)
            ;
    }
    Serial.println("Model initialization done.");

    // Optional: Print model metadata
    ModelPrintMetadata();

    // Optional: Print input and output tensor dimensions
    ModelPrintInputTensorDimensions();
    ModelPrintOutputTensorDimensions();

    Serial.println("");
    Serial.println("Please, input a float number between 0 and 6.28");
}

void loop()
{
    // Check if a value was sent from Serial Monitor
    if (Serial.available())
    {
        String inputValue = Serial.readString();
        float x = inputValue.toFloat(); // Evaluates to zero if the user input is not a valid number
        Serial.print("Your input value: ");
        Serial.println(x);

        // The model was trained in range 0 to 2*Pi, ensure value stays within this range
        if (x < 0)
            x = 0;
        if (x > 6.28)
            x = 6.28;
        Serial.print("Adapted input value: ");
        Serial.println(x);

        // Set the value in the model's input tensor
        if (!ModelSetInput(x, 0, true))
        {
            Serial.println("Failed to set input!");
            return;
        }

        // Run inference, and report if an error occurs
        if (!ModelRunInference())
        {
            Serial.println("RunInference Failed!");
            return;
        }

        // Obtain and print the output from the model's output tensor
        float y = ModelGetOutput(0);

        Serial.print("Inferred Sin of ");
        Serial.print(x);
        Serial.print(" = ");
        Serial.println(y, 2);
        Serial.print("Actual Sin of ");
        Serial.print(x);
        Serial.print(" = ");
        Serial.println(sin(x), 2);
    }
}
