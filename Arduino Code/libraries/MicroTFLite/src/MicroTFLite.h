#ifndef ARDUINOTFLITE_H
#define ARDUINOTFLITE_H

#include <Arduino.h>
#include <MicroTensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// Initializes the TensorFlow Lite model and interpreter
bool ModelInit(const unsigned char *model, byte *tensorArena, int tensorArenaSize);

// Sets the input tensor with a given value, handling quantization if needed
bool ModelSetInput(float inputValue, int index, bool showQuantizedValue = false);

// Runs inference on the model
bool ModelRunInference();

// Retrieves the output value from the model
float ModelGetOutput(int index, bool showQuantizedValue = false);

// Prints the dimensions of the input tensor
void ModelPrintInputTensorDimensions();

// Prints the dimensions of the output tensor
void ModelPrintOutputTensorDimensions();

// Prints the quantization parameters for the input and output tensors
void ModelPrintTensorQuantizationParams();

// Prints metadata about the model, such as description and version
void ModelPrintMetadata();

// Prints information about the tensors, including type and dimensions
void ModelPrintTensorInfo();

#endif
