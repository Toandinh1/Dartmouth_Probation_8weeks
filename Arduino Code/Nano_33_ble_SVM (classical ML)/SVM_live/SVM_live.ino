/*
  SVM_live.ino (Collector-matched realtime inference)

  Pipeline:
    MAX30105 (IN_FS=100) -> FIFO drain -> RingBuffer (drop-oldest on overflow)
    -> 100 Hz tick stream (OUT_FS=100) with UNDERFLOW repeat and CATCHUP flag
    -> window N=512, hop=384 (128 overlap)
    -> features (5):
        0 ptp
        1 absmean
        2 log_bp_0p1_0p5
        3 log_bp_0p5_1p5
        4 spec_entropy_0_5
    -> linear SVM (MCU matched): uses linear_svm_params.h
*/

#include <Wire.h>
#include "MAX30105.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <math.h>

// ========================= USER SETTINGS =========================
static const uint32_t BAUD = 921600;

static const uint16_t IN_FS  = 100;
static const uint16_t OUT_FS = 100;
static const uint32_t OUT_PERIOD_US = 1000000UL / OUT_FS;

static const uint32_t IR_FINGER_TH = 30000;

// Turn on ASCII prints ONLY if you are not streaming binary
#define ASCII_DEBUG 1

// ================= OLED =================
#define OLED_ADDR 0x3C
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

static const uint32_t OLED_PERIOD_MS = 200;
static uint32_t lastOledMs = 0;

// ================= MAX30105 =================
MAX30105 sensor;

// ================= BUTTON ON/OFF =================
static const int BTN_PIN = 2;
static const uint32_t DEBOUNCE_MS = 60;
static bool run_enable = true;
static uint32_t lastBtnChangeMs = 0;
static int lastBtnRead = HIGH;

// ================= Scheduler =================
static uint32_t next_tick_us = 0;
static uint32_t out_idx = 0;

// ================= Ring Buffer (collector-matching) =================
typedef struct { uint32_t r, ir, g; } PPG_Sample;

static const uint16_t RB_SIZE = 512;
static volatile uint16_t rb_w = 0, rb_r = 0;
static PPG_Sample rb[RB_SIZE];
static uint32_t rb_overflow = 0;

static inline bool rbEmpty() { return rb_w == rb_r; }
static inline bool rbFull()  { return (uint16_t)((rb_w + 1) % RB_SIZE) == rb_r; }
static inline uint16_t rbUsed() {
  uint16_t w = rb_w, r = rb_r;
  return (w >= r) ? (w - r) : (RB_SIZE - (r - w));
}

// IMPORTANT: collector uses drop-oldest on overflow
static inline void rbPush(const PPG_Sample &s) {
  if (rbFull()) {
    rb_overflow++;
    rb_r = (rb_r + 1) % RB_SIZE; // drop oldest
  }
  rb[rb_w] = s;
  rb_w = (rb_w + 1) % RB_SIZE;
}

static inline bool rbPop(PPG_Sample &s) {
  if (rbEmpty()) return false;
  s = rb[rb_r];
  rb_r = (rb_r + 1) % RB_SIZE;
  return true;
}

static inline void drain_fifo_to_rb() {
  sensor.check();
  while (sensor.available()) {
    PPG_Sample s;
    s.r  = sensor.getRed();
    s.ir = sensor.getIR();
    s.g  = 0;
    sensor.nextSample();
    rbPush(s);
  }
}

// ================= Stats =================
static uint32_t lastRateMs = 0;
static uint32_t outThisSecond = 0;
static uint32_t hz_est = 0;

static PPG_Sample last_sample = {0,0,0};
static bool last_finger = false;

// ================= Windowing =================
static const int N = 512;
static const int HOP = 384; // 512-384 = 128 overlap

static float ring_buf[N];
static int ring_head = 0;
static uint32_t pushed = 0;

static inline void ring_reset() { ring_head = 0; pushed = 0; }

static inline void ring_push(float x) {
  ring_buf[ring_head] = x;
  ring_head = (ring_head + 1) % N;
  pushed++;
}

static inline void ring_get_window(float *dst) {
  int start = ring_head; // oldest
  for (int i = 0; i < N; i++) dst[i] = ring_buf[(start + i) % N];
}

// ================= SVM params include =================
// SVM_D/SVM_MEAN/SVM_SCALE/SVM_W/SVM_B/SVM_THR
// and inline svm_score(x_raw), svm_predict(x_raw)
#include "linear_svm_params.h"

// Use threshold from header directly
static const float DEPLOY_THR = (float)SVM_THR;

// ================= OLED UI (compact, keep last result) =================
static void oled_show(uint8_t pred01, bool pred_valid, bool finger,
                      uint32_t infer_us, float score) {
  // Keep last-known values (so screen doesn't jump to "--" between inferences)
  static bool     has_last = false;
  static uint8_t  last_pred01 = 0;
  static uint32_t last_infer_us = 0;
  static bool     last_finger = false;

  if (pred_valid) {
    last_pred01 = pred01;
    last_infer_us = infer_us;
    last_finger = finger;
    has_last = true;
  } else {
    // when no new inference, still update finger status live (optional)
    last_finger = finger;
  }

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // Line 1: result (big)
  display.setTextSize(2);
  display.setCursor(0, 0);
  if (!has_last) {
    display.print("--");
  } else {
    display.print(last_pred01 ? "FOCUS" : "NORM");
  }

  // Line 2-4: small text
  display.setTextSize(1);

  // Line 2: Hz
  display.setCursor(0, 22);
  display.print("Hz: ");
  display.print(hz_est);

  // Line 3: inference time
  display.setCursor(0, 34);
  display.print("inf: ");
  display.print(last_infer_us);
  display.print("us");

  // Line 4: finger
  display.setCursor(0, 46);
  display.print("Finger: ");
  display.print(last_finger ? "YES" : "NO");

  display.display();
}

// ================= Button toggle =================
static void power_on_peripherals() {
  sensor.wakeUp();
  sensor.clearFIFO();

  rb_w = rb_r = 0;
  rb_overflow = 0;
  out_idx = 0;

  next_tick_us = micros() + OUT_PERIOD_US;

  lastRateMs = millis();
  outThisSecond = 0;
  hz_est = 0;

  // prime last_sample
  drain_fifo_to_rb();
  PPG_Sample s;
  if (rbPop(s)) last_sample = s;

  ring_reset();
}

static void power_off_peripherals() {
  sensor.shutDown();

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(2);
  display.setCursor(28, 24);
  display.print("OFF");
  display.display();
}

static void handle_button_toggle() {
  int r = digitalRead(BTN_PIN);
  uint32_t now = millis();

  if (r != lastBtnRead) {
    lastBtnRead = r;
    lastBtnChangeMs = now;
  }

  if ((now - lastBtnChangeMs) > DEBOUNCE_MS) {
    if (r == LOW) {
      while (digitalRead(BTN_PIN) == LOW) { delay(1); }
      run_enable = !run_enable;
      if (run_enable) power_on_peripherals();
      else power_off_peripherals();
      delay(50);
    }
  }
}

// ================= Offline-matched feature extraction =================
// Matches Python training:
// - per-window z-score: (w - mean) / (std + EPS_Z)  :contentReference[oaicite:1]{index=1}
// - rfft power spectrum
// - band masks: 0.1<=f<0.5 and 0.5<=f<1.5  :contentReference[oaicite:2]{index=2}
// - log(bp + EPS_BP) and entropy = -sum(p log p) over 0..5Hz (UNNORMALIZED) :contentReference[oaicite:3]{index=3}
//
// IMPORTANT: call with a full window x_raw[N] sampled at OUT_FS==FS_TARGET (100Hz).
static inline void compute_features_offline_matched(
  const float *x_raw, int Nwin, float fs, float feat[5],
  float EPS_Z = 1e-6f, float EPS_BP = 1e-6f, float EPS_ENT = 1e-12f
){
  // ---- z-score window ----
  double mean = 0.0;
  for (int i = 0; i < Nwin; i++) mean += (double)x_raw[i];
  mean /= (double)Nwin;

  double var = 0.0;
  for (int i = 0; i < Nwin; i++) {
    double v = (double)x_raw[i] - mean;
    var += v * v;
  }
  var /= (double)Nwin;
  double stdv = sqrt(var);

  //reuse a static buffer to avoid large stack usage
  static float wz[512]; // assumes Nwin <= 512; change if needed
  float inv = 1.0f / (float)(stdv + (double)EPS_Z);
  for (int i = 0; i < Nwin; i++) {
    wz[i] = (float)((double)x_raw[i] - mean) * inv;
  }

  // ---- time-domain features on z-scored window ----
  float mn = wz[0], mx = wz[0];
  double absmean = 0.0;
  for (int i = 0; i < Nwin; i++) {
    float v = wz[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    absmean += fabs((double)v);
  }
  absmean /= (double)Nwin;

  feat[0] = mx - mn;          // ptp
  feat[1] = (float)absmean;   // absmean

  // ---- compute power spectrum bins up to 5 Hz ----
  const float df = fs / (float)Nwin;
  const int k_max = (int)floorf(5.0f / df);       // max k with f<=5
  const int k_cap = (k_max > (Nwin/2)) ? (Nwin/2) : k_max;

  // entropy uses Pm over 0..5Hz inclusive
  static double Pm[300]; // enough for k_cap ~ 25 at 100Hz/512; safe a bit larger
  double bp01 = 0.0, bp05 = 0.0;

  for (int k = 0; k <= k_cap; k++) {
    double re = 0.0, im = 0.0;

    const double ang_step = -2.0 * M_PI * (double)k / (double)Nwin;
    const double c_step = cos(ang_step);
    const double s_step = sin(ang_step);
    double c = 1.0, s = 0.0;

    for (int n = 0; n < Nwin; n++) {
      const double xn = (double)wz[n];
      re += xn * c;
      im += xn * s;

      const double c_new = c * c_step - s * s_step;
      const double s_new = s * c_step + c * s_step;
      c = c_new; s = s_new;
    }

    double Pk = re*re + im*im;
    double f = (double)k * (double)df;

    // for entropy (0..5 inclusive)
    Pm[k] = Pk + (double)EPS_ENT;

    // band masks EXACT (offline): 0.1<=f<0.5 and 0.5<=f<1.5 :contentReference[oaicite:4]{index=4}
    if (f >= 0.1 && f < 0.5) bp01 += Pk;
    if (f >= 0.5 && f < 1.5) bp05 += Pk;
  }

  feat[2] = logf((float)(bp01 + (double)EPS_BP));
  feat[3] = logf((float)(bp05 + (double)EPS_BP));

  // ---- entropy (UNNORMALIZED): -sum(p log p) over 0..5Hz :contentReference[oaicite:5]{index=5}
  double sP = 0.0;
  for (int k = 0; k <= k_cap; k++) sP += Pm[k];

  if (sP <= 0.0) {
    feat[4] = 0.0f;
  } else {
    double H = 0.0;
    for (int k = 0; k <= k_cap; k++) {
      double p = Pm[k] / sP;
      H += -p * log(p);
    }
    feat[4] = (float)H;
  }
}


// ================= Feature extraction =================
// Compute spectrum up to 5 Hz from a mean-removed window.
// Fs=100Hz, N=512 => bin resolution = Fs/N = 0.1953125 Hz
static inline void compute_features(const float *x_in, float feat[5]) {
  // 1) remove mean
  double mean = 0.0;
  for (int i = 0; i < N; i++) mean += x_in[i];
  mean /= (double)N;

  // 2) time-domain stats on mean-removed x
  float mn = (float)(x_in[0] - mean);
  float mx = mn;
  double absmean = 0.0;

  static float x[N];
  for (int i = 0; i < N; i++) {
    float v = (float)(x_in[i] - mean);
    x[i] = v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    absmean += fabs((double)v);
  }
  absmean /= (double)N;

  const float ptp = mx - mn;

  // 3) power spectrum bins 0..5Hz
  const int k_max = (int)floorf(5.0f * (float)N / (float)OUT_FS); // ~25
  static float P[64]; // enough for k_max ~ 25
  const float eps = 1e-12f;

  for (int k = 0; k <= k_max; k++) {
    double re = 0.0, im = 0.0;
    double ang_step = -2.0 * M_PI * (double)k / (double)N;
    double c = 1.0, s = 0.0;
    double c_step = cos(ang_step);
    double s_step = sin(ang_step);

    for (int n = 0; n < N; n++) {
      double xn = (double)x[n];
      re += xn * c;
      im += xn * s;

      double c_new = c * c_step - s * s_step;
      double s_new = s * c_step + c * s_step;
      c = c_new; s = s_new;
    }

    float pk = (float)(re*re + im*im);
    P[k] = pk + eps;
  }

  auto hz_to_k = [&](float hz) -> int {
    int k = (int)lroundf(hz * (float)N / (float)OUT_FS);
    if (k < 0) k = 0;
    if (k > k_max) k = k_max;
    return k;
  };

  int k01 = hz_to_k(0.1f);
  int k05 = hz_to_k(0.5f);
  int k15 = hz_to_k(1.5f);

  double bp_01_05 = 0.0;
  for (int k = k01; k <= k05; k++) bp_01_05 += (double)P[k];

  double bp_05_15 = 0.0;
  for (int k = k05; k <= k15; k++) bp_05_15 += (double)P[k];

  float log_bp_01_05 = logf((float)bp_01_05 + eps);
  float log_bp_05_15 = logf((float)bp_05_15 + eps);

  double sumP = 0.0;
  for (int k = 0; k <= k_max; k++) sumP += (double)P[k];

  double H = 0.0;
  int M = k_max + 1;
  for (int k = 0; k <= k_max; k++) {
    double p = (double)P[k] / (sumP + 1e-24);
    if (p > 1e-24) H += -p * log(p);
  }
  double Hn = H / log((double)M + 1e-24);

  feat[0] = ptp;
  feat[1] = (float)absmean;
  feat[2] = log_bp_01_05;
  feat[3] = log_bp_05_15;
  feat[4] = (float)Hn;
}

// ================= Setup =================
void setup() {
  Serial.begin(BAUD);
  while (!Serial) {}

  pinMode(BTN_PIN, INPUT_PULLUP);

  Wire.begin();
  Wire.setClock(400000);

  display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  display.clearDisplay();
  display.display();

  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
#if ASCII_DEBUG
    Serial.println("MAX30105 NOT FOUND (check wiring + 3.3V)");
#endif
    while (1) delay(10);
  }

  sensor.setup(
    0x2A,   // ledBrightness
    1,      // sampleAverage
    2,      // ledMode = 2 (Red + IR)
    IN_FS,  // sampleRate = 100
    411,    // pulseWidth
    4096    // adcRange
  );

  sensor.setPulseAmplitudeRed(0x2A);
  sensor.setPulseAmplitudeIR(0x2A);
  sensor.setPulseAmplitudeGreen(0x00);

  sensor.clearFIFO();
  sensor.shutDown();

#if ASCII_DEBUG
  Serial.println("READY: MAX30105 -> 100Hz tick -> feat(5) -> LinearSVM -> OLED");
  Serial.println("Label map: 0=NORM, 1=FOCUS");
  Serial.print("SVM_THR="); Serial.println(DEPLOY_THR, 6);
#endif

  run_enable = true;
  power_on_peripherals();

  lastOledMs = millis();
}

// ================= Loop =================
void loop() {
  handle_button_toggle();
  if (!run_enable) { delay(10); return; }

  drain_fifo_to_rb();

  uint32_t now_us = micros();
  if (next_tick_us == 0) next_tick_us = now_us + OUT_PERIOD_US;

  int catchup = 0;
  while ((int32_t)(now_us - next_tick_us) >= 0 && catchup < 10) {
    next_tick_us += OUT_PERIOD_US;
    catchup++;

    drain_fifo_to_rb();

    // pop sample; if none -> repeat last
    PPG_Sample s;
    bool ok = rbPop(s);
    if (!ok) s = last_sample;
    else last_sample = s;

    bool finger = (s.ir > IR_FINGER_TH);
    last_finger = finger;

    out_idx++;
    outThisSecond++;

    // ===== Finger gating to avoid mixing no-finger into windows =====
    static bool prevFinger = false;
    if (!finger) {
      if (prevFinger) ring_reset();
      prevFinger = false;
      continue;
    } else {
      if (!prevFinger) ring_reset();
      prevFinger = true;
    }

    // push IR sample (matches training if training used IR)
    ring_push((float)s.ir);

    // infer on hop schedule
    if (pushed >= (uint32_t)N) {
      uint32_t since_full = pushed - (uint32_t)N;
      if ((since_full % (uint32_t)HOP) == 0) {
        static float win[N];
        ring_get_window(win);

        float feat[5];
        compute_features_offline_matched(win, N, (float)OUT_FS, feat);


        uint32_t t0 = micros();
        float score = svm_score(feat);     // <-- from header (applies mean/scale + W/B)
        int pred = svm_predict(feat);      // <-- from header (uses SVM_THR)
        uint32_t t1 = micros();
        uint32_t infer_us = (uint32_t)(t1 - t0);

#if ASCII_DEBUG
        Serial.print("feat: ");
        for (int i = 0; i < 5; i++) {
          Serial.print(feat[i], 6);
          if (i < 4) Serial.print(", ");
        }
        Serial.print(" score=");
        Serial.print(score, 6);
        Serial.print(" pred=");
        Serial.println(pred);
#endif

        uint32_t nowMs2 = millis();
        if (nowMs2 - lastOledMs >= OLED_PERIOD_MS) {
          lastOledMs = nowMs2;
          oled_show((uint8_t)pred, true, finger, infer_us, score);
        }
      }
    }
  }

  // Hz estimate
  uint32_t nowMs = millis();
  if (nowMs - lastRateMs >= 1000) {
    hz_est = outThisSecond;
    outThisSecond = 0;
    lastRateMs += 1000;
  }

  // OLED refresh even when no inference
  if (nowMs - lastOledMs >= OLED_PERIOD_MS) {
    lastOledMs = nowMs;
    oled_show(0, false, last_finger, 0, 0.0f);
  }
}
