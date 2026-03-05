/* Copyright 2024 John O'Sullivan, TensorFlow Authors. All Rights Reserved.

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

#include <MicroTFLite.h>
#include "model.h"

// Statistical Summary of Features Before Scaling and Balancing:
// RPM - Mean: 1603.866, Standard Deviation: 195.843
// Temperature (°C) - Mean: 24.354, Standard Deviation: 4.987
// Vibration (g) - Mean: 0.120, Standard Deviation: 0.020
// Current (A) - Mean: 3.494, Standard Deviation: 0.308
constexpr float tMean = 24.354f;
constexpr float rpmMean = 1603.866f;
constexpr float vMean = 0.120f;
constexpr float cMean = 3.494f;
constexpr float tStd = 4.987f;
constexpr float rpmStd = 195.843f;
constexpr float vStd = 0.020f;
constexpr float cStd = 0.308f;

// Define thresholds for failure conditions
const float highTempThreshold = 30.0f;            // High temperature threshold in °C
const float lowRpmThreshold = 1500.0f;            // Low RPM threshold
const float highVibrationThreshold = 0.60f;       // High vibration threshold in g
const float abnormalCurrentLowThreshold = 0.2f;   // Low current threshold in A
const float abnormalCurrentHighThreshold = 10.8f; // High current threshold in A

// Statistics counters
int totalPredictions = 0;
int truePositives = 0;
int falsePositives = 0;
int trueNegatives = 0;
int falseNegatives = 0;
float rollingAccuracy = 0.0f;
bool showStatistics = false;

constexpr int kTensorArenaSize = 4 * 1024;
alignas(16) uint8_t tensorArena[kTensorArenaSize];

void setup()
{
    // Initialize serial communications and wait for Serial Monitor to be opened
    Serial.begin(115200);
    while (!Serial)
        ;
    delay(5000);
    Serial.println("Preventative Maintenance Example.");
    Serial.println("Initializing TensorFlow Lite Micro Interpreter...");
    if (!ModelInit(model, tensorArena, kTensorArenaSize))
    {
        Serial.println("Model initialization failed!");
        while (true)
            ;
    }
    Serial.println("Model initialization done.");
    ModelPrintMetadata();
    ModelPrintTensorQuantizationParams();
    ModelPrintTensorInfo();
}

// Function to generate normally distributed random values using Box-Muller transform
float GenerateRandomValue(float mean, float stddev)
{
    float u1 = random(0, 10000) / 10000.0f;
    float u2 = random(0, 10000) / 10000.0f;
    // Box-Muller transform to generate normally distributed values
    float z0 = sqrt(-2.0f * log(u1)) * cos(2.0f * PI * u2);
    float value = mean + z0 * stddev;
    return value;
}

float ReadRpm()
{
    return GenerateRandomValue(rpmMean, rpmStd);
}

float ReadVibration()
{
    return GenerateRandomValue(vMean, vStd);
}

float ReadTemperature()
{
    return GenerateRandomValue(tMean, tStd);
}

float ReadCurrent()
{
    return GenerateRandomValue(cMean, cStd);
}

// Check failure conditions and return true if any are met
bool CheckFailureConditions(float temperature, float rpm, float vibration, float current)
{
    bool conditionMet = false;
    String failureReason = "";

    if (temperature > highTempThreshold)
    {
        conditionMet = true;
        failureReason += "High Temperature; ";
    }
    if (rpm < lowRpmThreshold)
    {
        conditionMet = true;
        failureReason += "Low RPM; ";
    }
    if (vibration > highVibrationThreshold)
    {
        conditionMet = true;
        failureReason += "High Vibration; ";
    }
    if (current < abnormalCurrentLowThreshold)
    {
        conditionMet = true;
        failureReason += "Low Current; ";
    }
    if (current > abnormalCurrentHighThreshold)
    {
        conditionMet = true;
        failureReason += "High Current; ";
    }

    if (conditionMet)
    {
        Serial.print("Note: Sensor readings indicate a potential failure condition. Reasons: ");
        Serial.println(failureReason);
    }

    return conditionMet; // Return true if failure conditions are met, false otherwise
}

void loop()
{
    float rpm = ReadRpm();
    float temperature = ReadTemperature();
    float current = ReadCurrent();
    float vibration = ReadVibration();
    bool inputSetFailed = false;

    // Check if actual failure conditions are met
    bool actualFailure = CheckFailureConditions(temperature, rpm, vibration, current);

    // Normalize the values
    float temperatureN = (temperature - tMean) / tStd;
    float rpmN = (rpm - rpmMean) / rpmStd;
    float currentN = (current - cMean) / cStd;
    float vibrationN = (vibration - vMean) / vStd;

    // Place the value in the model's input tensor
    if (!ModelSetInput(temperatureN, 0))
        inputSetFailed = true;
    if (!ModelSetInput(rpmN, 1))
        inputSetFailed = true;
    if (!ModelSetInput(currentN, 2))
        inputSetFailed = true;
    if (!ModelSetInput(vibrationN, 3))
        inputSetFailed = true;

    // Run inference, and report if an error occurs
    if (!ModelRunInference())
    {
        Serial.println("RunInference Failed!");
        return;
    }

    // Obtain the output from the model's output tensor
    float prediction = ModelGetOutput(0);
    bool predictedFailure = (prediction > 0.50f); // Model predicts failure if prediction > 0.5

    // Update prediction statistics
    if (predictedFailure && actualFailure)
    {
        truePositives++;
    }
    else if (predictedFailure && !actualFailure)
    {
        falsePositives++;
        showStatistics = true;
    }
    else if (!predictedFailure && actualFailure)
    {
        falseNegatives++;
        showStatistics = true;
    }
    else if (!predictedFailure && !actualFailure)
    {
        trueNegatives++;
    }

    totalPredictions++;

    // Calculate rolling accuracy (percentage of correct predictions)
    rollingAccuracy = (float)(truePositives + trueNegatives) / totalPredictions * 100.0f;

    // Show statistics for False Negatives and False Positives
    if (showStatistics)
    {
        Serial.println("-----------------------------");
        Serial.print("Prediction confidence: ");
        Serial.println(prediction);
        Serial.print("Predicted failure: ");
        Serial.println(predictedFailure);
        Serial.print("Actual failure: ");
        Serial.println(actualFailure);
        Serial.print("RPM: ");
        Serial.print(rpm);
        Serial.print(" | Temperature: ");
        Serial.print(temperature);
        Serial.print(" C");
        Serial.print(" | Current: ");
        Serial.print(current);
        Serial.print(" A");
        Serial.print(" | Vibration: ");
        Serial.print(vibration);
        Serial.print(" m/s^2\n");
        Serial.println("Total Predictions: " + String(totalPredictions) + ", True Positives: " + String(truePositives) + ", False Positives: " + String(falsePositives) + ", True Negatives: " + String(trueNegatives) + ", False Negatives: " + String(falseNegatives) + ", Rolling Accuracy (%): " + String(rollingAccuracy));

        showStatistics = false;
    }

    delay(10000); // Wait for 10 seconds before the next loop
}
