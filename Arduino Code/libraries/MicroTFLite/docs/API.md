# MicroTFLite Library Documentation

This document provides a detailed description of the MicroTFLite abstraction layer.

![image](https://github.com/user-attachments/assets/919036d5-e5d0-4de9-a1d9-85b832dfdee9)


### **Function**

**bool ModelInit(const unsigned char* model, byte* tensorArena, int tensorArenaSize);**

_Initializes TensorFlow Lite Micro environment, instantiates an `AllOpsResolver`, and allocates input and output tensors._

**Parameters:**

- `model`: A pointer to the model data.
- `tensorArena`: A memory buffer to be used for tensor allocations.
- `tensorArenaSize`: The size of the tensor arena.

**Returns:**  
`true` = success, `false` = failure.

---

### **Function**

**bool ModelSetInput(float inputValue, int index, bool showQuantizedValue = false);**

_Writes `inputValue` in the position `index` of the input tensor, automatically handling quantization if the tensor is of type `int8`._

**Parameters:**

- `inputValue`: The input value to be written to the tensor.
- `index`: The position in the input tensor where the value is written.
- `showQuantizedValue`: (Optional) If `true`, prints the quantized value for debugging purposes.

**Returns:**  
`true` = success, `false` = failure.

---

### **Function**

**bool ModelRunInference();**

_Invokes the TensorFlow Lite Micro Interpreter and executes the inference algorithm._

**Returns:**  
`true` = success, `false` = failure.

---

### **Function**

**float ModelGetOutput(int index, bool showQuantizedValue = false);**

_Returns the output value from the position `index` of the output tensor, automatically handling dequantization if the tensor is of type `int8`._

**Parameters:**

- `index`: The position in the output tensor from which to retrieve the result.
- `showQuantizedValue`: (Optional) If `true`, prints the quantized value for debugging purposes.
- **Returns:**  
  The output value from the tensor, or `-1` if there was an error.

---

### **Function**

**void ModelPrintInputTensorDimensions();**

_Prints the dimensions of the input tensor._

**Description:**  
Prints the size and dimensionality of the input tensor to the Serial Monitor, allowing the user to check how the input data should be structured.

---

### **Function**

**void ModelPrintOutputTensorDimensions();**

_Prints the dimensions of the output tensor._

**Description:**  
Prints the size and dimensionality of the output tensor to the Serial Monitor, providing insight into the model’s output shape.

---

### **Function**

**void ModelPrintTensorQuantizationParams();**

_Prints the quantization parameters for both the input and output tensors._

**Description:**  
Prints the scale and zero-point values for the input and output tensors, which are crucial for understanding how floating-point values are converted to and from quantized integer values.

---

### **Function**

**void ModelPrintMetadata();**

_Prints metadata information about the model, including description and version._

**Description:**  
If the model contains a description, it will be printed along with the model version. If no description is available, it will indicate so.

---

### **Function**

**void ModelPrintTensorInfo();**

_Prints detailed information about the input and output tensors, including their types and dimensions._

**Description:**  
Prints the data type (e.g., `int8` or `float32`) and the size of each dimension for both the input and output tensors.

---

### **Helper Functions**

#### **Quantization Functions**

**int8_t QuantizeInput(float x, float scale, float zeroPoint);**

_Quantizes a floating-point value to an `int8_t` value based on the provided scale and zero point._

**Parameters:**

- `x`: The floating-point input value.
- `scale`: The quantization scale.
- `zeroPoint`: The quantization zero point.

**Returns:**  
The quantized `int8_t` value.

---

**float DequantizeOutput(int8_t quantizedValue, float scale, float zeroPoint);**

_Dequantizes an `int8_t` output value back to a floating-point value based on the provided scale and zero point._

**Parameters:**

- `quantizedValue`: The quantized `int8_t` output value.
- `scale`: The quantization scale.
- `zeroPoint`: The quantization zero point.

**Returns:**  
The dequantized floating-point value.

---
