/* Copyright 2024 Chirale, TensorFlow Authors. All Rights Reserved.
==============================================================================*/

// ===== Includes =====
#include <Chirale_TensorFlowLite.h>
#include "cnn_1d_float.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <string.h>   // strcmp, strncmp, memcpy
#include <stdlib.h>   // strtof, atoi
#include <stdint.h>
#include <math.h>
#include <stdio.h>    // sscanf

// ===== OLED =====
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define OLED_ADDR 0x3C
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

#define DEBUG 1

// ===== mbed heap + cpu stats (Nano 33 BLE uses mbed OS) =====
#include "mbed_stats.h"

// ===== Binary streaming protocol =====
// Frame: [0xAA 0x55 mode len payload... checksum]
// checksum = XOR of bytes: mode, len, payload...
static const uint8_t SYNC1 = 0xAA;
static const uint8_t SYNC2 = 0x55;
static const uint8_t MODE_F32 = 0x01;
static const uint8_t MODE_I8  = 0x02;

enum StreamMode { STREAM_TEXT=0, STREAM_BIN_F32=1, STREAM_BIN_I8=2 };
static StreamMode g_stream_mode = STREAM_TEXT;

// If you stream int8 samples, define what they mean on the wire.
// Arduino will dequantize int8 -> float using these before filtering/windowing.
static float g_stream_i8_scale = 0.01f;
static int   g_stream_i8_zp    = 0;

// ===== Power/Energy profiling (software-only) =====
static uint64_t g_us_infer = 0;   // time in interpreter->Invoke()
static uint64_t g_us_fill  = 0;   // time in fill_input_last_window()
static uint64_t g_us_rx    = 0;   // time spent parsing serial bytes
static uint32_t g_infer_cnt = 0;
static uint32_t g_sample_cnt = 0;

// mbed CPU stats for active vs sleep (Nano 33 BLE = mbed OS)
static mbed_stats_cpu_t g_cpu_prev;
static bool g_cpu_prev_ok = false;

static uint32_t g_last_pwr_ms = 0;
static const uint32_t PWR_PERIOD_MS = 5000;

// Track peak heap (measured at runtime)
static uint32_t g_peak_heap = 0;
static uint32_t g_last_mem_print_ms = 0;

// ---- TFLite globals ----
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// ---- Tensor arena ----
constexpr int kTensorArenaSize = 96 * 1024;   // tune if needed
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// ==== Filter Settings =====
constexpr float FS_HZ = 64.0f;   // IMPORTANT: set this to match your stream Fs
constexpr float HP_HZ = 0.5f;    // cutoff

// ==============================
// Streaming / buffering params
// ==============================
constexpr int N = 512;            // window length
constexpr int HOP_DEFAULT = 384;  // default inference hop
constexpr int RING = 1024;        // ring buffer size

static volatile uint32_t g_hop = (uint32_t)HOP_DEFAULT;  // runtime hop set by command

// Ring buffer
float ringbuf[RING];
uint32_t widx = 0;
uint32_t total_samples = 0;
uint32_t last_infer_sample = 0;
uint32_t missed_infer = 0;

// ---- Normalization per Window -----
constexpr float NORM_EPS = 1e-6f;

// filter states
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

static void init_cpu_stats() {
  mbed_stats_cpu_get(&g_cpu_prev);
  g_cpu_prev_ok = true;
}

static void reset_power_counters() {
  g_us_infer = g_us_fill = g_us_rx = 0;
  g_infer_cnt = g_sample_cnt = 0;
  g_last_pwr_ms = millis();
  init_cpu_stats(); // refresh baseline for CPU active% calculation
}

static void report_power_energy() {
  uint32_t now = millis();
  if (g_last_pwr_ms == 0) { g_last_pwr_ms = now; return; }
  uint32_t dt_ms = now - g_last_pwr_ms;
  if (dt_ms < PWR_PERIOD_MS) return;

  // ---- CPU active vs sleep time from mbed ----
  mbed_stats_cpu_t cur;
  mbed_stats_cpu_get(&cur);

  if (!g_cpu_prev_ok) {
    g_cpu_prev = cur;
    g_cpu_prev_ok = true;
    g_last_pwr_ms = now;
    return;
  }

  // uptime, sleep_time, deep_sleep_time are in microseconds on mbed OS
  uint64_t du_uptime = cur.uptime - g_cpu_prev.uptime;
  uint64_t du_sleep  = cur.sleep_time - g_cpu_prev.sleep_time;
  uint64_t du_deep   = cur.deep_sleep_time - g_cpu_prev.deep_sleep_time;

  uint64_t du_idle = du_sleep + du_deep;
  uint64_t du_active = (du_uptime > du_idle) ? (du_uptime - du_idle) : 0;

  float active_pct = (du_uptime > 0) ? (100.0f * (float)du_active / (float)du_uptime) : 0.0f;

  // ---- Workload per second proxies ----
  float dt_s = dt_ms / 1000.0f;
  float samp_s = (dt_s > 0) ? (g_sample_cnt / dt_s) : 0.0f;
  float inf_s  = (dt_s > 0) ? (g_infer_cnt  / dt_s) : 0.0f;

  float infer_ms_s = (dt_s > 0) ? ((g_us_infer / 1000.0f) / dt_s) : 0.0f;
  float fill_ms_s  = (dt_s > 0) ? ((g_us_fill  / 1000.0f) / dt_s) : 0.0f;
  float rx_ms_s    = (dt_s > 0) ? ((g_us_rx    / 1000.0f) / dt_s) : 0.0f;

  float Eproxy_ms_s = infer_ms_s + fill_ms_s + rx_ms_s;

  Serial.print("[PWR] dt="); Serial.print(dt_s, 2); Serial.print("s  ");
  Serial.print("Fs="); Serial.print(FS_HZ, 1); Serial.print("Hz  ");
  Serial.print("HOP="); Serial.print((uint32_t)g_hop); Serial.print("  ");
  Serial.print("samp="); Serial.print(samp_s, 2); Serial.print("/s  ");
  Serial.print("inf="); Serial.print(inf_s, 4); Serial.print("/s  ");
  Serial.print("active="); Serial.print(active_pct, 1); Serial.print("%  ");
  Serial.print("infer_ms/s="); Serial.print(infer_ms_s, 3); Serial.print("  ");
  Serial.print("fill_ms/s=");  Serial.print(fill_ms_s, 3);  Serial.print("  ");
  Serial.print("rx_ms/s=");    Serial.print(rx_ms_s, 3);    Serial.print("  ");
  Serial.print("Eproxy_ms/s=");Serial.println(Eproxy_ms_s, 3);

  // reset window counters
  g_us_infer = g_us_fill = g_us_rx = 0;
  g_infer_cnt = g_sample_cnt = 0;

  g_cpu_prev = cur;
  g_last_pwr_ms = now;
}

// ===== Forward declarations =====
static bool run_inference_once();
static void handle_line(char* s);

// Serial line buffer (for TEXT mode + command lines)
static char linebuf[48];
static uint8_t line_i = 0;

// ===== Runtime memory helpers (mbed) =====
static void update_heap_stats(uint32_t &cur, uint32_t &maxv, uint32_t &reserved) {
  mbed_stats_heap_t heap;
  mbed_stats_heap_get(&heap);
  cur = heap.current_size;
  maxv = heap.max_size;
  reserved = heap.reserved_size;
  if (maxv > g_peak_heap) g_peak_heap = maxv;
}

static void reportMemorySerial(const char* tag) {
  uint32_t heap_cur=0, heap_max=0, heap_res=0;
  update_heap_stats(heap_cur, heap_max, heap_res);

  Serial.print("[MEM] "); Serial.print(tag);
  Serial.print(" heap_cur="); Serial.print(heap_cur);
  Serial.print(" heap_peak="); Serial.print(g_peak_heap);
  Serial.print(" heap_reserved="); Serial.print(heap_res);

  if (interpreter) {
    Serial.print(" arena_used=");
    Serial.print(interpreter->arena_used_bytes());
    Serial.print("/");
    Serial.print(kTensorArenaSize);
  }
  Serial.println();
}

// ===== OLED helper =====
static void oled_show_pred_and_timing(int pred01, uint32_t infer_us) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(2);
  display.setCursor(0, 0);
  display.print(pred01 ? "STRESS" : "NON");
  if (!pred01) {
    display.setCursor(0, 18);
    display.print("STRESS");
  }

  display.setTextSize(1);
  display.setCursor(0, 44);
  display.print("Infer: ");
  display.print(infer_us);
  display.print(" us");

  display.display();
}

// ===== Helpers =====
static inline void ring_push(float v) {
  ringbuf[widx] = v;
  widx = (widx + 1) % RING;
  total_samples++;
}

// ===== Binary RX state (global) =====
static uint8_t rx_state = 0;
static uint8_t rx_mode  = 0;
static uint8_t rx_len   = 0;
static uint8_t rx_buf[8];
static uint8_t rx_i     = 0;
static uint8_t rx_chk   = 0;

static inline void reset_rx() {
  rx_state = 0; rx_mode = 0; rx_len = 0; rx_i = 0; rx_chk = 0;
}

static inline float stream_dequant_i8(int8_t q) {
  return ((int)q - g_stream_i8_zp) * g_stream_i8_scale;
}

// ---- Stats over last N samples ----
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

static inline int8_t clamp_int8(int v) {
  if (v < -128) return (int8_t)-128;
  if (v >  127) return (int8_t) 127;
  return (int8_t)v;
}

// Fill model_input from last N samples.
// - If input is float32: write z-scored float to data.f
// - If input is int8:    quantize z-scored float using input scale/zero_point and write data.int8
static bool fill_input_last_window() {
  if (!model_input) return false;

  float mean = 0.0f, stdv = 1.0f;
  compute_mean_std_lastN(mean, stdv);
  float inv = 1.0f / (stdv + NORM_EPS);

  uint32_t end = widx; // exclusive
  uint32_t start = (end + RING - N) % RING;

  if (model_input->type == kTfLiteFloat32) {
    for (int i = 0; i < N; i++) {
      uint32_t idx = (start + (uint32_t)i) % RING;
      float x = ringbuf[idx];
      model_input->data.f[i] = (x - mean) * inv;
    }
    return true;
  }

  if (model_input->type == kTfLiteInt8) {
    const float scale = model_input->params.scale;
    const int   zp    = model_input->params.zero_point;
    if (scale <= 0) {
      Serial.println("ERR input scale <= 0");
      return false;
    }

    for (int i = 0; i < N; i++) {
      uint32_t idx = (start + (uint32_t)i) % RING;
      float x = ringbuf[idx];
      float z = (x - mean) * inv;
      int q = (int)lroundf(z / scale) + zp;
      model_input->data.int8[i] = clamp_int8(q);
    }
    return true;
  }

  Serial.print("ERR unsupported input type=");
  Serial.println(model_input->type);
  return false;
}

// One decoded sample (float) -> filter -> ring -> inference schedule
static void handle_sample_float(float v) {
  g_sample_cnt++;  // IMPORTANT for samp/s metric

  float vf = hp1_filter(v);
  ring_push(vf);

  if (total_samples >= (uint32_t)N) {
    while ((total_samples - last_infer_sample) >= (uint32_t)g_hop) {
      uint32_t hop = (uint32_t)g_hop;  // snapshot
      uint32_t due = total_samples - last_infer_sample;

      if (hop > 0 && due >= (2u * hop)) {
        missed_infer += (due / hop) - 1u;
      }

      last_infer_sample += hop;
      run_inference_once();
    }
  }
}

static void parse_binary_byte(uint8_t b) {
  switch (rx_state) {
    case 0: // wait SYNC1
      if (b == SYNC1) rx_state = 1;
      break;

    case 1: // wait SYNC2
      rx_state = (b == SYNC2) ? 2 : 0;
      break;

    case 2: // mode
      rx_mode = b;
      rx_chk = b;
      rx_state = 3;
      break;

    case 3: // len
      rx_len = b;
      rx_chk ^= b;
      rx_i = 0;
      if (rx_len > sizeof(rx_buf)) { reset_rx(); break; }
      rx_state = (rx_len == 0) ? 5 : 4;
      break;

    case 4: // payload
      rx_buf[rx_i++] = b;
      rx_chk ^= b;
      if (rx_i >= rx_len) rx_state = 5;
      break;

    case 5: // checksum
      if (b != rx_chk) { reset_rx(); break; }

      if (rx_mode == MODE_F32 && rx_len == 4) {
        float v;
        memcpy(&v, rx_buf, 4);  // little-endian
        handle_sample_float(v);
      } else if (rx_mode == MODE_I8 && rx_len == 1) {
        int8_t q = (int8_t)rx_buf[0];
        handle_sample_float(stream_dequant_i8(q));
      }
      reset_rx();
      break;

    default:
      reset_rx();
      break;
  }
}

// ===== Output printing =====
static void print_probs_and_pred(uint32_t infer_us) {
  int n_out = 1;
  for (int i = 0; i < model_output->dims->size; i++) {
    n_out *= model_output->dims->data[i];
  }

  int best = 0;
  float bestv = -1e9f;

  Serial.print("PROB ");

  if (model_output->type == kTfLiteFloat32) {
    for (int i = 0; i < n_out; i++) {
      float p = model_output->data.f[i];
      Serial.print(p, 6);
      if (i != n_out - 1) Serial.print(",");
      if (p > bestv) { bestv = p; best = i; }
    }
  } else if (model_output->type == kTfLiteInt8) {
    const float scale = model_output->params.scale;
    const int   zp    = model_output->params.zero_point;
    for (int i = 0; i < n_out; i++) {
      float p = (model_output->data.int8[i] - zp) * scale;
      Serial.print(p, 6);
      if (i != n_out - 1) Serial.print(",");
      if (p > bestv) { bestv = p; best = i; }
    }
  } else {
    Serial.print("ERR unsupported output type=");
    Serial.println(model_output->type);
    return;
  }

  Serial.println();

  Serial.print("PRED ");
  Serial.print(best);
  Serial.print(" CONF ");
  Serial.print(bestv, 6);
  Serial.print(" infer_us=");
  Serial.print(infer_us);
  Serial.print(" missed=");
  Serial.println(missed_infer);

  int pred01 = (best == 1) ? 1 : 0;
  oled_show_pred_and_timing(pred01, infer_us);

  reportMemorySerial("after_infer");
}

static bool run_inference_once() {
  if (!model_input || !model_output) return false;

  if (total_samples < (uint32_t)N) {
    Serial.println("ERR not enough samples");
    return false;
  }

  // ---- fill_input timing ----
  uint32_t t_fill0 = micros();
  if (!fill_input_last_window()) {
    Serial.println("ERR fill_input failed");
    return false;
  }
  uint32_t t_fill1 = micros();
  g_us_fill += (uint32_t)(t_fill1 - t_fill0);

  // ---- Invoke timing ----
  uint32_t t0 = micros();
  TfLiteStatus st = interpreter->Invoke();
  uint32_t t1 = micros();
  uint32_t infer_us = t1 - t0;

  g_us_infer += infer_us;
  g_infer_cnt++;

  if (st != kTfLiteOk) {
    Serial.println("ERR Invoke failed");
    return false;
  }

  print_probs_and_pred(infer_us);
  return true;
}

// ===== Command/text handler =====
static void do_reset_state() {
  widx = 0;
  total_samples = 0;
  last_infer_sample = 0;
  missed_infer = 0;
  hp1_init();
  g_peak_heap = 0;
  reset_rx();
  line_i = 0;

  reset_power_counters();

  Serial.println("OK RST");
  reportMemorySerial("after_rst");

  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("RST OK");
  display.display();
}

static void handle_line(char* s) {
  while (*s == ' ' || *s == '\t') s++;
  if (s[0] == '\0') return;

  if (strcmp(s, "RUN") == 0) { run_inference_once(); return; }
  if (strcmp(s, "RST") == 0) { do_reset_state(); return; }

  // Change hop (batch size axis)
  // Usage: "HOP 512"
  if (strncmp(s, "HOP ", 4) == 0) {
    int hop = atoi(s + 4);
    if (hop == 512 || hop == 384 || hop == 256 || hop == 128) {
      g_hop = (uint32_t)hop;
      last_infer_sample = total_samples; // avoid burst
      missed_infer = 0;
      reset_power_counters();            // clean profiling window
      Serial.print("OK HOP "); Serial.println((uint32_t)g_hop);
    } else {
      Serial.println("ERR HOP must be one of: 512 384 256 128");
    }
    return;
  }

  // streaming mode switch
  if (strcmp(s, "MODE TEXT") == 0)    { g_stream_mode = STREAM_TEXT;    Serial.println("OK MODE TEXT");    return; }
  if (strcmp(s, "MODE BIN_F32") == 0) { g_stream_mode = STREAM_BIN_F32;  Serial.println("OK MODE BIN_F32"); return; }
  if (strcmp(s, "MODE BIN_I8") == 0)  { g_stream_mode = STREAM_BIN_I8;   Serial.println("OK MODE BIN_I8");  return; }

  // set int8 stream quant params (for BIN_I8 dequant)
  // Example: "I8Q 0.05167 -15"
  if (strncmp(s, "I8Q ", 4) == 0) {
    float sc = 0.0f; int zp = 0;
    if (sscanf(s+4, "%f %d", &sc, &zp) == 2 && sc > 0) {
      g_stream_i8_scale = sc;
      g_stream_i8_zp = zp;
      Serial.print("OK I8Q scale="); Serial.print(g_stream_i8_scale, 8);
      Serial.print(" zp="); Serial.println(g_stream_i8_zp);
    } else {
      Serial.println("ERR I8Q usage: I8Q <scale> <zero_point>");
    }
    return;
  }

  // Default: treat as float sample line (TEXT mode or manual numeric line)
  char* endptr = nullptr;
  float v = strtof(s, &endptr);
  if (endptr == s) {
    Serial.print("ERR bad line: ");
    Serial.println(s);
    return;
  }
  handle_sample_float(v);
}

void setup() {
  Serial.begin(115200);
#if DEBUG
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0) < 3000) { }
#endif

  // OLED init
  Wire.begin();
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("OLED init failed");
    while (1) {}
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Booting...");
  display.display();

  // TFLM init
  Serial.println("Initializing TFLite Micro...");

  Serial.print("Model bytes: ");
  Serial.println(cnn_1d_float_len);
  Serial.print("Model KB: ");
  Serial.println(cnn_1d_float_len / 1024.0f, 2);

  tfl_model = tflite::GetModel(cnn_1d_float);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Schema mismatch!");
    while (1);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  Serial.print("Tensor arena bytes: ");
  Serial.println(kTensorArenaSize);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("TFLM alloc FAIL");
    display.display();
    while (1);
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  Serial.print("input type=");  Serial.println(model_input->type);
  Serial.print("output type="); Serial.println(model_output->type);

  Serial.print("input scale="); Serial.println(model_input->params.scale, 8);
  Serial.print("input zero=");  Serial.println(model_input->params.zero_point);

  Serial.print("Arena used bytes: ");
  Serial.println(interpreter->arena_used_bytes());

  hp1_init();

  // init CPU stats baseline + reset profiling window
  init_cpu_stats();
  reset_power_counters();

  Serial.println("READY.");
  Serial.println("Commands:");
  Serial.println("  RUN");
  Serial.println("  RST");
  Serial.println("  HOP 512|384|256|128");
  Serial.println("  MODE TEXT | MODE BIN_F32 | MODE BIN_I8");
  Serial.println("  I8Q <scale> <zp>   (defines BIN_I8 meaning)");

  reportMemorySerial("setup_done");

  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);
  display.println("READY");
  display.println("Streaming...");
  display.display();
}

void loop() {
  // periodic memory print (every 2s)
  uint32_t now = millis();
  if (now - g_last_mem_print_ms >= 2000) {
    g_last_mem_print_ms = now;
    reportMemorySerial("periodic");
  }

  // power/energy report (every 5s)
  report_power_energy();

  // If no incoming data, yield a bit.
  // This is important so active% can drop at lower Fs / lower hop load.
  if (!Serial.available()) {
    delay(1);
    return;
  }

  // Measure RX processing time
  uint32_t rx0 = micros();
  while (Serial.available()) {
    uint8_t b = (uint8_t)Serial.read();

    if (g_stream_mode == STREAM_TEXT) {
      // TEXT mode: parse lines
      char c = (char)b;
      if (c == '\n' || c == '\r') {
        linebuf[line_i] = '\0';
        line_i = 0;
        handle_line(linebuf);
      } else {
        if (line_i < sizeof(linebuf) - 1) linebuf[line_i++] = c;
        else line_i = 0;
      }
    } else {
      // BINARY mode: parse frames ONLY
      parse_binary_byte(b);
    }
  }
  uint32_t rx1 = micros();
  g_us_rx += (uint32_t)(rx1 - rx0);
}
