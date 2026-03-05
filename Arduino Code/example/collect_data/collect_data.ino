/*
  MAX30105 100 Hz DATA COLLECTOR - BINARY FRAMED OUTPUT (IMPROVED)
  Goals:
  - Keep 100 packets/sec output
  - Reduce UNDERFLOW/CATCHUP by improving scheduling and FIFO draining
  - Avoid OLED/I2C contention during recording
  - No ASCII in binary stream
*/

#include <Wire.h>
#include "MAX30105.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ================= CONFIG =================
static const uint16_t IN_FS  = 100;   // MAX30105 internal sampling rate
static const uint16_t OUT_FS = 100;   // output rate to PC
static const uint32_t OUT_PERIOD_US = 1000000UL / OUT_FS;

#define USE_GREEN 0

static const uint32_t IR_FINGER_TH = 30000;
static const uint32_t BAUD = 921600;

// OLED: make it lighter during recording (I2C contention)
#define USE_OLED 1
static const uint32_t OLED_PERIOD_MS_IDLE = 500;   // when not recording
static const uint32_t OLED_PERIOD_MS_REC  = 1500;  // when recording (less frequent)

// FIFO drain cap: avoid spending too long per call on I2C
static const uint16_t DRAIN_MAX_SAMPLES_PER_CALL = 32;

// ================= OLED =================
#define OLED_ADDR 0x3C
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

static uint32_t lastOledMs = 0;

// ================= MAX30105 =================
MAX30105 sensor;

// ================= BUTTON ON/OFF =================
static const int BTN_PIN = 2;
static const uint32_t DEBOUNCE_MS = 60;
static bool run_enable = false;
static uint32_t lastBtnChangeMs = 0;
static int lastBtnRead = HIGH;

// ================= Scheduler =================
static uint32_t next_tick_us = 0;

// ================= Ring Buffer =================
typedef struct { uint32_t r, ir, g; } PPG_Sample;

// bigger buffer helps absorb FIFO bursts
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

static inline void rbPushDropOldest(const PPG_Sample &s) {
  if (rbFull()) {
    rb_overflow++;
    rb_r = (rb_r + 1) % RB_SIZE; // drop oldest to keep newest
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

// ================= Stats =================
static uint32_t out_idx = 0;
static uint32_t lastRateMs = 0;
static uint32_t outThisSecond = 0;
static uint32_t hz_est = 0;

// diagnostic counters (per second)
static uint32_t ufThisSecond = 0;
static uint32_t ofThisSecond = 0;
static uint32_t cuThisSecond = 0;

static PPG_Sample last_sample = {0, 0, 0};
static bool last_finger = false;

// ================= Framing =================
// [0]=0xAA [1]=TYPE [2]=LEN [payload] [CHK] [0x55]
static const uint8_t FRAME_START = 0xAA;
static const uint8_t FRAME_END   = 0x55;

enum FrameType : uint8_t {
  FT_DATA  = 0x01,
  FT_START = 0x10,
  FT_STOP  = 0x11,
  FT_READY = 0x12,
  FT_STATS = 0x13  // optional stats once per second
};

// Flags for data packets
// bit0: FINGER
// bit1: UNDERFLOW (RB empty -> repeated last sample)
// bit2: RB_OVERFLOW happened since last packet
// bit3: CATCHUP (we were late and had to catch up)
static inline uint8_t checksum8(const uint8_t* p, uint16_t n) {
  uint8_t x = 0;
  for (uint16_t i = 0; i < n; i++) x ^= p[i];
  return x;
}

struct __attribute__((packed)) DataPayload {
  uint32_t idx;
  uint32_t t_us;
  uint32_t red;
  uint32_t ir;
  uint32_t green;
  uint16_t rb_used;
  uint8_t  flags;
};

struct __attribute__((packed)) StatsPayload {
  uint32_t hz;
  uint32_t underflow;
  uint32_t overflow;
  uint32_t catchup;
  uint16_t rb_used;
};

// pre-allocated checksum buffer (avoid stack alloc each send)
static uint8_t chkbuf[2 + 255];

// send a framed message
static void send_frame(uint8_t type, const uint8_t* payload, uint8_t len) {
  Serial.write(FRAME_START);
  Serial.write(type);
  Serial.write(len);

  if (len > 0 && payload) Serial.write(payload, len);

  // checksum covers: type, len, payload bytes
  chkbuf[0] = type;
  chkbuf[1] = len;
  for (uint8_t i = 0; i < len; i++) chkbuf[2 + i] = payload[i];
  uint8_t chk = checksum8(chkbuf, (uint16_t)(2 + len));

  Serial.write(chk);
  Serial.write(FRAME_END);
}

// ================= OLED helpers =================
#if USE_OLED
static void oled_status(bool recording) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  if (recording) {
    display.setCursor(0, 0);  display.print("REC");
    display.setCursor(40, 0); display.print("Hz: "); display.println(hz_est);
    display.setCursor(0, 12); display.print("idx: "); display.println(out_idx);
    display.setCursor(0, 24); display.print("RB: "); display.println(rbUsed());
    display.setCursor(0, 36); display.print("Finger: "); display.println(last_finger ? "YES" : "NO");
    display.setCursor(0, 48); display.print("UF/OF/CU: ");
    display.print(ufThisSecond); display.print("/");
    display.print(ofThisSecond); display.print("/");
    display.println(cuThisSecond);
  } else {
    display.setCursor(0, 20);
    display.setTextSize(2);
    display.print("READY");
    display.setTextSize(1);
    display.setCursor(0, 45);
    display.print("Press to START");
  }
  display.display();
}
#endif

// ================= Sensor FIFO -> RB =================
static void drain_fifo_to_rb_limited() {
  sensor.check();
  uint16_t n = 0;
  while (sensor.available() && n < DRAIN_MAX_SAMPLES_PER_CALL) {
    PPG_Sample s;
    s.r  = sensor.getRed();
    s.ir = sensor.getIR();
#if USE_GREEN
    s.g  = sensor.getGreen();
#else
    s.g  = 0;
#endif
    sensor.nextSample();
    rbPushDropOldest(s);
    n++;
  }
}

// ================= Power control =================
static void start_recording() {
  sensor.wakeUp();
  sensor.clearFIFO();

  rb_w = rb_r = 0;
  rb_overflow = 0;

  out_idx = 0;
  next_tick_us = micros() + OUT_PERIOD_US;

  lastRateMs = millis();
  outThisSecond = 0;
  hz_est = 0;

  ufThisSecond = ofThisSecond = cuThisSecond = 0;

  // prime last_sample from real data if possible
  drain_fifo_to_rb_limited();
  PPG_Sample s;
  if (rbPop(s)) last_sample = s;

  run_enable = true;

  send_frame(FT_START, nullptr, 0);
#if USE_OLED
  lastOledMs = millis();
  oled_status(true);
#endif
}

static void stop_recording() {
  run_enable = false;
  sensor.shutDown();

  send_frame(FT_STOP, nullptr, 0);

#if USE_OLED
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 20);
  display.print("STOPPED");
  display.setCursor(0, 35);
  display.print("Samples: ");
  display.println(out_idx);
  display.display();
  delay(500);
  oled_status(false);
#endif
}

// ================= Button =================
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

      if (run_enable) stop_recording();
      else start_recording();

      delay(50);
    }
  }
}

// ================= Setup =================
void setup() {
  Serial.begin(BAUD);
  while (!Serial) {}

  pinMode(BTN_PIN, INPUT_PULLUP);

  Wire.begin();
  Wire.setClock(400000);

#if USE_OLED
  display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
#endif

  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
#if USE_OLED
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 20);
    display.print("MAX30105 NOT FOUND");
    display.display();
#endif
    while (1) delay(10);
  }

  const int mode = 2; // Red + IR

  sensor.setup(
    0x2A,   // ledBrightness
    1,      // sampleAverage
    mode,   // ledMode
    IN_FS,  // sampleRate
    411,    // pulseWidth
    4096    // adcRange
  );

  sensor.setPulseAmplitudeRed(0x2A);
  sensor.setPulseAmplitudeIR(0x2A);
  sensor.setPulseAmplitudeGreen(0x00);

  sensor.clearFIFO();
  sensor.shutDown();

  send_frame(FT_READY, nullptr, 0);

#if USE_OLED
  lastOledMs = millis();
  oled_status(false);
#endif
}

// ================= Loop =================
void loop() {
  handle_button_toggle();

  if (!run_enable) {
    // even when idle, update OLED slowly
#if USE_OLED
    uint32_t nowMs = millis();
    if (nowMs - lastOledMs >= OLED_PERIOD_MS_IDLE) {
      lastOledMs = nowMs;
      oled_status(false);
    }
#endif
    delay(5);
    return;
  }

  // Keep RB filled (small chunks, frequent calls)
  drain_fifo_to_rb_limited();

  // scheduler: update now_us inside loop to avoid stale comparison
  int catchup = 0;
  while (catchup < 10) {
    uint32_t now_us = micros();
    if (next_tick_us == 0) next_tick_us = now_us + OUT_PERIOD_US;

    if ((int32_t)(now_us - next_tick_us) < 0) break;

    const uint32_t tick_us = next_tick_us;
    next_tick_us += OUT_PERIOD_US;
    catchup++;

    // drain a bit again before popping
    drain_fifo_to_rb_limited();

    uint8_t flags = 0;

    // overflow tracking (since last packet)
    static uint32_t last_rb_overflow = 0;
    if (rb_overflow != last_rb_overflow) {
      flags |= (1 << 2);
      last_rb_overflow = rb_overflow;
      ofThisSecond++;
    }

    // pop sample, else repeat last_sample
    PPG_Sample s;
    bool ok = rbPop(s);
    if (!ok) {
      s = last_sample;
      flags |= (1 << 1);
      ufThisSecond++;
    } else {
      last_sample = s;
    }

    bool finger = (s.ir > IR_FINGER_TH);
    if (finger) flags |= (1 << 0);
    last_finger = finger;

    if (catchup > 1) {
      flags |= (1 << 3);
      cuThisSecond++;
    }

    DataPayload pl;
    pl.idx = out_idx;
    pl.t_us = tick_us;
    pl.red = s.r;
    pl.ir = s.ir;
    pl.green = s.g;
    pl.rb_used = rbUsed();
    pl.flags = flags;

    send_frame(FT_DATA, (const uint8_t*)&pl, (uint8_t)sizeof(pl));

    out_idx++;
    outThisSecond++;
  }

  // rate stats per second
  uint32_t nowMs = millis();
  if (nowMs - lastRateMs >= 1000) {
    hz_est = outThisSecond;
    outThisSecond = 0;

    // send a compact stats frame once per second (optional but handy)
    StatsPayload st;
    st.hz = hz_est;
    st.underflow = ufThisSecond;
    st.overflow  = ofThisSecond;
    st.catchup   = cuThisSecond;
    st.rb_used   = rbUsed();
    send_frame(FT_STATS, (const uint8_t*)&st, (uint8_t)sizeof(st));

    ufThisSecond = ofThisSecond = cuThisSecond = 0;
    lastRateMs += 1000;
  }

#if USE_OLED
  const uint32_t oledPeriod = OLED_PERIOD_MS_REC;
  if (nowMs - lastOledMs >= oledPeriod) {
    lastOledMs = nowMs;
    oled_status(true);
  }
#endif
}
