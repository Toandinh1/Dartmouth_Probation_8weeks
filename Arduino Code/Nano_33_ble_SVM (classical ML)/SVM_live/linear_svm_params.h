#pragma once
// Auto-generated Linear SVM params (MCU-matched)
// BEST_C=0.05  CLASS_WEIGHT=NONE
// THR_MODE=balanced  THR_SHIFT_SIGMA=0.25
#define SVM_D 5

static const float SVM_MEAN[5] = {4.99656343e+00f, 7.99239635e-01f, 1.02661476e+01f, 9.73305225e+00f, 2.10602808e+00f};
static const float SVM_SCALE[5] = {9.96671200e-01f, 8.95191208e-02f, 2.39216518e+00f, 2.29901719e+00f, 4.88664836e-01f};
static const float SVM_W[5] = {1.67480692e-01f, 4.81572121e-01f, 4.86216486e-01f, -7.29776502e-01f, 7.00206995e-01f};
static const float SVM_B = -5.24247556e-02f;
static const float SVM_THR = 2.92626746e-01f;
//static const float SVM_THR = -6.92626746e-01f;


static inline float svm_score(const float *x_raw) {
  float score = SVM_B;
  for (int i = 0; i < SVM_D; i++) {
    float xn = (x_raw[i] - SVM_MEAN[i]) / (SVM_SCALE[i] + 1e-12f);
    score += SVM_W[i] * xn;
  }
  return score;
}

static inline int svm_predict(const float *x_raw) {
  float score = svm_score(x_raw);
  return (score >= SVM_THR) ? 1 : 0; // 1=scare, 0=normal
}
