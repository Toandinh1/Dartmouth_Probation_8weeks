// ===== Includes =====
#include <Wire.h>
#include <math.h>

// MAX30105
#include "MAX30105.h"

// OLED
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// TFLite Micro
#include <Chirale_TensorFlowLite.h>
#include "cnn_1d_float_live.h"   // file name can be live; symbol inside is cnn_1d_float
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define DEBUG 1

// ================= OLED =================
#define OLED_ADDR 0x3C
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

static void oled_print_status(const char* title, const char* line2 = nullptr) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(2);
  display.setCursor(0, 0);
  display.println(title);

  if (line2) {
    display.setTextSize(1);
    display.setCursor(0, 30);
    display.println(line2);
  }
  display.display();
}

// ========  OLED display (3 lines) ========
// Line1: STRESS / NON-STRESS (or -- if never predicted yet)
// Line2: FINGER / NO FINGER
// Line3: Infer time
static void oled_show_3lines(int pred01, bool pred_valid, bool finger_on, uint32_t infer_us) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  // Line 1: Stress / Non-stress
  display.setCursor(0, 0);
  if (pred_valid) display.print(pred01 ? "STRESS" : "NON-STRESS");
  else            display.print("--");

  // Line 2: Finger / No finger
  display.setCursor(0, 16);
  display.print(finger_on ? "FINGER" : "NO FINGER");

  // Line 3: Inference time
  display.setCursor(0, 32);
  display.print("Infer: ");
  display.print(infer_us);
  display.print(" us");

  display.display();
}

// ================= MAX30105 =================
MAX30105 ppg;
static volatile uint32_t latestIR = 0;

// ================= TFLite =================
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

constexpr int kTensorArenaSize = 120 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// ================= Preprocessing =================
constexpr uint32_t IR_FINGER_THRESH = 5000;

// Debounce finger detection
static bool finger_on = false;
static uint8_t finger_cnt = 0;
constexpr uint8_t FINGER_ON_COUNT = 6;

static void update_finger_state(uint32_t ir) {
  bool now_on = (ir > IR_FINGER_THRESH);
  if (now_on) {
    if (finger_cnt < 255) finger_cnt++;
  } else {
    if (finger_cnt > 0) finger_cnt--;
  }
  if (!finger_on && finger_cnt >= FINGER_ON_COUNT) finger_on = true;
  if (finger_on && finger_cnt == 0) finger_on = false;
}

// HP filter settings
constexpr float FS_HZ = 64.0f;
constexpr float HP_HZ = 0.5f;

static float hp_a = 0.0f;
static float hp_y = 0.0f;
static float hp_x_prev = 0.0f;
static bool  hp_inited = false;

static void hp1_init() {
  float dt = 1.0f / FS_HZ;
  float RC = 1.0f / (2.0f * 3.1415926f * HP_HZ);
  hp_a = RC / (RC + dt);
  hp_y = 0.0f;
  hp_x_prev = 0.0f;
  hp_inited = false;
}

static float hp1_filter(float x) {
  if (!hp_inited) {
    hp_x_prev = x;
    hp_y = 0.0f;
    hp_inited = true;
    return 0.0f;
  }
  hp_y = hp_a * (hp_y + x - hp_x_prev);
  hp_x_prev = x;
  return hp_y;
}

// ================= Streaming params =================
constexpr int N = 512;     // 8s @ 64Hz
constexpr int HOP = 384;   // 6s hop
constexpr int RING = 1024; // >= 2*N

float ringbuf[RING];
uint32_t widx = 0;
uint32_t total_samples = 0;
uint32_t last_infer_sample = 0;

static inline void ring_push(float v) {
  ringbuf[widx] = v;
  widx = (widx + 1) % RING;
  total_samples++;
}

// ---- Normalization per Window -----

constexpr float NORM_EPS = 1e-6f;

static void compute_mean_std_lastN(float& mean, float& stdv) {
  float m = 0.0f, M2 = 0.0f;
  int k = 0;

  uint32_t end = widx; // exclusive
  uint32_t start = (end + RING - N) % RING;

  for (int i = 0; i < N; i++) {
    uint32_t idx = (start + (uint32_t)i) % RING;
    float x = ringbuf[idx];
    k++;
    float delta = x - m;
    m += delta / (float)k;
    float delta2 = x - m;
    M2 += delta * delta2;
  }

  mean = m;
  float var = (N > 1) ? (M2 / (float)(N - 1)) : 0.0f;
  stdv = sqrtf(var);
}

static void copy_last_window_to_input_zscore() {
  float mean = 0.0f, stdv = 1.0f;
  compute_mean_std_lastN(mean, stdv);
  float inv = 1.0f / (stdv + NORM_EPS);

  uint32_t end = widx; // exclusive
  uint32_t start = (end + RING - N) % RING;

  for (int i = 0; i < N; i++) {
    uint32_t idx = (start + (uint32_t)i) % RING;
    float x = ringbuf[idx];
    model_input->data.f[i] = (x - mean) * inv;
  }
}

static void get_pred(int& pred, float& conf) {
  int n_out = 1;
  for (int i = 0; i < model_output->dims->size; i++) {
    n_out *= model_output->dims->data[i];
  }

  if (n_out == 1) {
    float p = model_output->data.f[0];
    pred = (p >= 0.5f) ? 1 : 0;
    conf = pred ? p : (1.0f - p);
  } else {
    int best = 0;
    float bestv = model_output->data.f[0];
    for (int i = 1; i < n_out; i++) {
      float v = model_output->data.f[i];
      if (v > bestv) { bestv = v; best = i; }
    }
    pred = best;
    conf = bestv;
  }
}

static bool run_inference_once(uint32_t& infer_us_out, int& pred_out, float& conf_out) {
  infer_us_out = 0;
  pred_out = 0;
  conf_out = 0.0f;

  if (!model_input || !model_output) return false;
  if (model_input->type != kTfLiteFloat32 || model_output->type != kTfLiteFloat32) return false;
  if (total_samples < (uint32_t)N) return false;

  copy_last_window_to_input_zscore();

  uint32_t t0 = micros();
  TfLiteStatus st = interpreter->Invoke();
  uint32_t t1 = micros();
  if (st != kTfLiteOk) return false;

  infer_us_out = (t1 - t0);
  get_pred(pred_out, conf_out);

#if DEBUG
  Serial.print("PRED=");
  Serial.print(pred_out);
  Serial.print(" CONF=");
  Serial.print(conf_out, 4);
  Serial.print(" infer_us=");
  Serial.println(infer_us_out);
#endif
  return true;
}

// ================= Timing =================
static const uint32_t SAMPLE_PERIOD_US = (uint32_t)(1000000.0f / FS_HZ);
uint32_t next_sample_us = 0;

// ======= Keep last prediction/infer time =======
static bool g_pred_valid = false;
static int  g_last_pred01 = 0;
static uint32_t g_last_infer_us = 0;

void setup() {
  Serial.begin(115200);
#if DEBUG
  while (!Serial) {}
#endif

  Wire.begin();

  // OLED init
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("OLED init failed. Try OLED_ADDR=0x3D if needed.");
  } else {
    oled_print_status("BOOT", "Init MAX30105...");
  }

  // MAX30105 init
  if (!ppg.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30105 not found. Check wiring/contact + 3.3V.");
    if (display.width() > 0) oled_print_status("MAX ERR", "Check wiring/contact");
    while (1) delay(10);
  }

  // Configure MAX30105: Red + IR
  ppg.setup(
    0x2A,  // LED brightness
    4,     // sampleAverage
    2,     // ledMode: 2 = Red + IR
    100,   // sampleRate
    411,   // pulseWidth
    4096   // adcRange
  );

  // Turn ON RED LED (visible) + IR (signal)
  ppg.setPulseAmplitudeRed(0x2A);
  ppg.setPulseAmplitudeIR(0x2A);
  ppg.setPulseAmplitudeGreen(0x00);

  // TFLite init
  Serial.println("Initializing TFLite Micro...");
  tfl_model = tflite::GetModel(cnn_1d_float_live);

  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Schema mismatch!");
    while (1);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (1);
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  hp1_init();

  if (display.width() > 0) oled_print_status("READY", "Place finger");
  Serial.println("READY: MAX30105(IR) @64Hz -> CNN -> OLED");

  next_sample_us = micros();
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - next_sample_us) >= 0) {
    next_sample_us += SAMPLE_PERIOD_US;

    // Read sample
    uint32_t ir = (uint32_t)ppg.getIR();
    latestIR = ir;

    // Update finger state
    update_finger_state(ir);

    // If no finger: do not push samples; keep last prediction; show OLED
    if (!finger_on) {
      oled_show_3lines(g_last_pred01, g_pred_valid, false, g_last_infer_us);
      return;
    }

    // Preprocess + push
    float x  = (float)ir;
    float xf = hp1_filter(x);
    ring_push(xf);

    // Inference every HOP samples
    bool ran = false;
    uint32_t infer_us = 0;
    int pred = 0;
    float conf = 0.f;

    if (total_samples >= (uint32_t)N) {
      if (total_samples - last_infer_sample >= (uint32_t)HOP) {
        last_infer_sample += (uint32_t)HOP;
        ran = run_inference_once(infer_us, pred, conf);
      }
    }

    if (ran) {
      g_last_pred01 = (pred == 1) ? 1 : 0;
      g_last_infer_us = infer_us;
      g_pred_valid = true;
    }

    // OLED always shows your 3 lines
    oled_show_3lines(g_last_pred01, g_pred_valid, true, g_last_infer_us);
  }
}
