# MicroTFLite

![image](https://github.com/user-attachments/assets/9c2bd0e8-b827-4340-9ab9-8c6f60741699)

## A TensorFlow Lite Micro Library in Arduino Style

This library simplifies the use of **TensorFlow Lite Micro** on Arduino boards, offering APIs in the typical _Arduino style_. It avoids the use of _pointers_ or other C++ syntactic constructs that are discouraged within an Arduino sketch.
It was inspired in large part by [ArduTFLite](https://github.com/spaziochirale/ArduTFLite) but includes the latest tensor flow code; it can also work with quantized data or raw float values, detecting the appropriate processing depending on the models meta data. It is more geared towards those who require Arduino style APIs and who wish to learn about the process of deploying **TensorFlow** models on constrained edge devices. It provides a number of functions that provide insight into the process of deploying models and these functions are particularly useful in debugging issues with models.

**MicroTFLite** is designed to enable experimentation with **Tiny Machine Learning** on Arduino boards with constrained resources, such as **Arduino Nano 33 BLE**, **Arduino Nano ESP32**, **Arduino Nicla**, **Arduino Portenta**, **ESP32 based devices** and **Arduino Giga R1 WiFi**. Usage is simple and straightforward and You don't need an extensive TensorFlow expertise to code your sketches and the library provides an extensive API that allows you explore the internal meta data of the model.

![image](https://github.com/user-attachments/assets/a963ff73-9629-490a-ad5c-0a3d081f0162)


## Architecture

MicroTFLite consists of an Arduino style abstraction called **MicroTFLite** and a port of TensorFlow Lite for Arduino type boards.

## Installation

To install the in-development version of this library, you can use the latest version directly from the [GitHub repository](https://github.com/johnosbb/MicroTFLite). This requires you clone the repo into the folder that holds libraries for the Arduino IDE.

Once you're in that folder in the terminal, you can then grab the code using the git command line tool:

```
git clone https://github.com/johnosbb/MicroTFLite MicroTFLite
```

To update your clone of the repository to the latest code, use the following terminal commands:

```
cd MicroTFLite
git pull
```

## Checking your Installation

Once the library has been installed, you should then start the Arduino IDE. You will now see an `MicroTFLite` entry in the `File -> Examples` menu of the Arduino IDE. This submenu contains a list of sample projects you can try out. These examples show the abstraction layer in use.

## API Guide

A guide to the **MicroTFLite** API can be found [here](./docs/API.md).

## Usage examples

The examples included with the library show how to use the library. The examples come with their pre-trained models.

## General TinyML development process:

1. **Create an Arduino Sketch to collect data suitable for training**: First, create an Arduino sketch to collect data to be used as the training dataset.
2. **Define a DNN model**: Once the training dataset is acquired, create a neural network model using an external TensorFlow development environment, such as Google Colaboratory.
3. **Train the Model**: Import training data on the TensorFlow development environment and train the model on the training dataset.
4. **Convert and Save the Model**: Convert the trained model to TensorFlow Lite format and save it as a `model.h` file. This file should contain the definition of a static byte array corresponding to the binary format (flat buffer) of the TensorFlow Lite model.
5. **Prepare a new Arduino Sketch for Inference**
6. **Include Necessary Headers**: Include the `MicroTFLite.h` header file and the `model.h` file.
7. **Define Tensor Arena**: Define a globally sized byte array for the area called tensorArena.
8. **Initialize the Model**: Initialize the model using the `ModelInit()` function.
9. **Set Input Data**: Insert the input data into the model's input tensor using the `ModelSetInput()` function.
10. **Run Inference**: Invoke the inference operation using the `ModelRunInference()` function.
11. **Read Output Data**: Read the output data using the `ModelGetOutput()` function.

## Tensorflow versus TensorFlow-Lite

The difference between TensorFlow Lite and TensorFlow Lite for Microcontrollers is the platforms they target. TFLite is targeted at mobile use cases, such as Android & iOS devices. As the name suggests, TFLite-Micro is aimed at low-power microcontrollers and digital signal processors for embedded use cases. These can even be baremetal use cases, as TFLite-Micro has no dependencies on an operating system. MicroTFLite is built on top of TFLite-Micro.


### GitHub

The officially supported TensorFlow Lite Micro library for Arduino resides in the [tflite-micro-arduino-examples](https://github.com/tensorflow/tflite-micro-arduino-examples). This library is a fork of that project with the necessary refactoring required to allow the code build in an Arduino IDE environment. The latest version of this library can be found in the repository [MicroTFLite](https://github.com/johnosbb/MicroTFLite)

## Compatibility

This library is designed for the mbed based `Raspberry Pi Pico` and for a range of `ESP32 based boards` boards. The framework code for running machine learning models should be compatible with most Arm Cortex M-based boards.

## License

This code is made available under the Apache 2 license.

## Contributing

Forks of this library are welcome and encouraged. If you have bug reports or fixes to contribute, the source of this code is at [https://github.com/johnosbb/MicroTFLite](https://github.com/johnosbb/MicroTFLite) and all issues and pull requests should be directed there.
