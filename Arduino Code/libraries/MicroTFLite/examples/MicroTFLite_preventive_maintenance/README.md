# Preventative Maintenance Example

This sketch is an example of using sensor data to predict component failure. It is somewhat contrived and is provided to illustrate the ability of the **MicroTFLite** library to handle quantized and multiple input variables. Its purpose is to provide an example of the structure of a typical classification program that uses a pre-trained neural network model, whose _inference_ algorithm (i.e. input classification) is executed using the **TensorFlow Lite Micro library**.

On the board we emulate the sensor readings; this is for the convenience of those who do not have the appropriate sensors, but there is no reason why the emulation could not be replaced by actual sensor readings.

The **MicroTFLite** library handles models where values can be represented with _float_ numbers or in a quantized form, but in this example we are using quantized data.

The generation of the data is described in detail [here](https://github.com/johnosbb/Artificial-intelligence/tree/main/Embedded/PreventativeMaintenance)

The binary file in _.tflite_ format, containing the trained model, has been transformed into a constant char array so that it can be loaded into memory directly at compile time. In TensorFlow Lite applications developed on devices with an external file system, such as smartphones and single-board computers (e.g., Raspberry Pi), the model is stored on the external disk or SD Card as a binary file with a _.tflite_ extension and subsequently loaded into memory at runtime during the initialization phase. To enable Arduino boards, which lack an external file system, to use the model, the contents of the .tflite file have been included within an array that is compiled along with the code. This array, named _model_, is declared and initialized in **model.h** file.

## Model Information

### Neural Network Architecture and Training

#### Model Architecture

This example demonstrates a feedforward neural network implemented using TensorFlow's Keras API. The network is designed to handle a binary classification problem with a balanced or imbalanced dataset. The model consists of the following layers:

1. **Input Layer**:

   - **Dense Layer**: A fully connected layer with `NUMBER_OF_FEATURES` units. This layer uses the ReLU activation function (`relu`) and is initialized with the shape of `(len(f_names),)`, where `len(f_names)` represents the number of features in the input data.

2. **Dropout Layer**:

   - **Dropout Layer**: A dropout layer with a dropout rate of 0.3. This layer randomly sets a fraction of input units to 0 during training to help prevent overfitting.

3. **Hidden Layer**:

   - **Dense Layer**: A fully connected layer with 16 units and ReLU activation (`relu`). This layer introduces non-linearity and allows the network to learn complex patterns.

4. **Dropout Layer**:

   - **Dropout Layer**: Another dropout layer with a dropout rate of 0.3, used to further regularize the network and prevent overfitting.

5. **Output Layer**:
   - **Dense Layer**: A fully connected layer with a single unit and sigmoid activation (`sigmoid`). This layer outputs a probability score for the binary classification task.

The model summary is displayed using `model.summary()`, which provides details about the network's architecture, including the number of parameters in each layer.

#### Model Compilation

The model is compiled with the following settings:

- **Loss Function**: `binary_crossentropy` - This loss function is appropriate for binary classification tasks, as it measures the performance of the model based on how well it predicts the class labels.
- **Optimizer**: `adam` - An adaptive learning rate optimizer that helps the model converge faster and achieve better performance.
- **Metrics**: `accuracy` - The model's performance is evaluated based on accuracy, which measures the proportion of correctly predicted instances.

#### Handling Class Imbalance

To address class imbalance, class weights are computed using the `class_weight.compute_class_weight` function. The class weights are balanced based on the distribution of the training data (`y_train`). If Synthetic Minority Over-sampling Technique (SMOTE) is used (`USE_SMOTE` is `True`), class weights are not applied (`class_weights_dict` is set to `None`). Otherwise, class weights are incorporated into the training process to improve model performance on the minority class.

#### Training the Model

The model is trained using the `fit` method with the following parameters:

- **Training Data**: `X_train_balanced_scaled` and `y_train_balanced` - Scaled and balanced training features and labels.
- **Validation Data**: `X_validate_scaled` and `y_validate` - Scaled validation features and labels, used to evaluate the model's performance during training.
- **Epochs**: `NUM_EPOCHS` - The number of epochs to train the model.
- **Batch Size**: `BATCH_SIZE` - The number of samples per gradient update.
- **Class Weights**: Applied if `class_weights_dict` is not `None`, to address class imbalance.
- **Callbacks**: An `EarlyStopping` callback is used to monitor the validation loss (`val_loss`) and prevent overfitting by stopping training if the validation loss does not improve for `patience` epochs. The best weights are restored to avoid overfitting.

## Validating Results on the Board

### Validating Findings with `checkFailureConditions`

When deploying a quantized TensorFlow Lite model on an embedded system or microcontroller, it is crucial to validate that the model's performance meets the expected criteria. One of the key functions in this validation process is `checkFailureConditions`. This function is designed to ensure that the quantized model operates correctly and yields reliable results.

#### Purpose of `checkFailureConditions`

The `checkFailureConditions` function performs a series of checks to verify the integrity and performance of the quantized model. These checks help identify any discrepancies or issues that may arise from the quantization process, such as loss of accuracy or unexpected behavior on the hardware.

#### How `checkFailureConditions` Works

1. **Model Output Validation**:

   - The function compares the output of the quantized model against expected results or a baseline reference. This comparison helps ensure that the quantization process has not introduced significant deviations in model predictions.

2. **Input Data Checks**:

   - It verifies that the input data provided to the model is correctly processed and fed into the quantized model. This includes checking data preprocessing steps to confirm that they align with what the model expects.

3. **Quantization Error Analysis**:

   - `checkFailureConditions` evaluates the impact of quantization on the model’s performance. It assesses any potential errors or artifacts introduced during quantization, which can affect the model's accuracy and reliability.

4. **Hardware-Specific Validation**:

   - The function tests the model's performance on the actual hardware to ensure compatibility and correctness. This involves running the model on the board and checking for any hardware-specific issues, such as memory constraints or processing limitations.

5. **Logging and Reporting**:
   - During the validation process, `checkFailureConditions` logs any detected issues and generates a detailed report. This report provides insights into any failures or anomalies, helping developers troubleshoot and address potential problems.

#### Example Usage

In a typical workflow, after loading the quantized model onto the board and running inference, the `checkFailureConditions` function is invoked to validate the results. Here’s a simplified example:

```cpp
// Run inference with the quantized model
bool inferenceResult = ModelRunInference();

// Validate the results using checkFailureConditions
if (!checkFailureConditions()) {
    Serial.println("Model validation failed!");
    // Handle the failure (e.g., retry, report error, etc.)
} else {
    Serial.println("Model validation successful.");
}
```
