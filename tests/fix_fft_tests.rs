//! Tests for Fixed-point in-place Fast Fourier Transform

use fixed_fft::{fft_radix2_q15, fft_radix2_real_q15, Direction};
use num_complex::{Complex, Complex64};
use std::f64;

type Sample = Complex<i16>;

/// Simple forward FFT test
///
/// [x, x, ..., x] must be transformed to [N*x, 0, ..., 0]
#[test]
fn fft2r_8() {
    let mut samples: [Sample; 8] = [Complex::new(1000, 0); 8];
    fft_radix2_q15(&mut samples, Direction::Forward).unwrap();
    assert_eq!(samples[0].re, 8000);
    assert_eq!(samples[0].im, 0);
    for s in samples.iter().skip(1) {
        assert_eq!(s.re, 0);
        assert_eq!(s.im, 0);
    }

    let mut samples: [Sample; 8] = [Complex::new(1000, 0); 8];
    fft_radix2_q15(&mut samples, Direction::ForwardScaled).unwrap();
    for s in samples.iter().skip(1) {
        assert_eq!(s.re, 0);
        assert_eq!(s.im, 0);
    }
    assert_eq!(samples[0].re, 1000);
    assert_eq!(samples[0].im, 0);
}

/// Simple inverse FFT test
///
/// [X, 0, ..., 0 ] must be inverse transformed to [X/N, X/N, ..., X/N]
#[test]
fn fft2r_inverse() {
    let mut samples: [Sample; 8] = [Complex::new(0, 0); 8];
    samples[0].re = 800;
    fft_radix2_q15(&mut samples, Direction::Inverse).unwrap();
    for s in samples.iter() {
        assert_eq!(s.re, 100);
        assert_eq!(s.im, 0);
    }
}

/// Forward FFT test with single pulse at t=0 and time shifted pulses
#[test]
fn fft2r_8_pulse() {
    let length = vec![2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    let ampl = vec![10, 100, 700, 1000, 10000, 20000, 32767, -32768];
    for a in ampl {
        for l in length.iter() {
            fft2r_shifted_pulse(*l, a, 0, true);
            if a >= 1000 && *l >= 8 {
                fft2r_shifted_pulse(*l, a, 1, true);
                fft2r_shifted_pulse_real(*l, a, 1, true);
                fft2r_shifted_pulse(*l, a, 2, true);
                fft2r_shifted_pulse_real(*l, a, 2, true);
                fft2r_shifted_pulse(*l, a, 3, true);
                fft2r_shifted_pulse_real(*l, a, 3, true);
                fft2r_shifted_pulse(*l, a, 4, true);

                fft2r_shifted_pulse(*l, a, 1, false);
                fft2r_shifted_pulse(*l, a, 2, false);

                fft2r_shifted_pulse(*l, a, 3, false);

                fft2r_shifted_pulse(*l, a, 4, false);
            }
        }
    }
}

fn fft2r_shifted_pulse(length: usize, amplitude: i16, shift: usize, scale: bool) {
    let mut samples = Vec::with_capacity(length);
    for i in 0..length {
        let re = if i == shift { amplitude } else { 0 };
        samples.push(Complex::new(re, 0));
    }
    println!(
        "shifted pulse test: length = {}, amplitude = {}, shift = {}, scale = {}",
        samples.len(),
        amplitude,
        shift,
        scale
    );
    let (dir, spec_ampl) = if scale {
        (Direction::ForwardScaled, amplitude as f64 / length as f64)
    } else {
        (Direction::Forward, amplitude as f64)
    };
    fft_radix2_q15(samples.as_mut_slice(), dir).unwrap();
    for (k, s) in samples.iter().enumerate() {
        // use the DFT shift theorem to calculate the FFT value using FP arithmetic
        let phi = -2.0 * f64::consts::PI * (shift as f64) * (k as f64) / length as f64;
        let fact = Complex64::new(0.0, phi).exp();
        let val = fact * Complex64::new(spec_ampl, 0.0);

        // we need to accept errors
        // Calculate relative error q for large values
        // Otherwise, calculate absolute error d
        if val.norm() > 100.0 {
            let q = (Complex64::new(s.re as f64, s.im as f64) - val).norm() / val.norm();
            println!("    k = {}, s = {}, val = {:.2}, q = {:.4}", k, s, val, q);
            assert!(q <= 0.01);
        } else {
            let d = (Complex64::new(s.re as f64, s.im as f64) - val).norm();
            println!("    k = {}, s = {}, val = {:.2}, d = {:.4}", k, s, val, d);
            assert!(d <= 2.0);
        }
    }
}

fn fft2r_shifted_pulse_real(length: usize, amplitude: i16, shift: usize, scale: bool) {
    let mut samples = Vec::with_capacity(length);
    for i in 0..length {
        let re = if i == shift { amplitude } else { 0 };
        samples.push(re);
    }
    let mut result = vec![Complex::new(0, 0); length / 2 + 1];
    let spec_ampl = if scale {
        amplitude as f64 / length as f64
    } else {
        amplitude as f64
    };

    println!(
        "shifted pulse test (real FFT): length = {}, amplitude = {}, shift = {}, scale = {}",
        samples.len(),
        amplitude,
        shift,
        scale
    );
    fft_radix2_real_q15(&mut samples, &mut result, scale).unwrap();
    for (k, s) in result.iter().enumerate() {
        // use the DFT shift theorem to calculate the FFT value using FP arithmetic
        let phi = -2.0 * f64::consts::PI * (shift as f64) * (k as f64) / length as f64;
        let fact = Complex64::new(0.0, phi).exp();
        let val = fact * Complex64::new(spec_ampl, 0.0);

        // we need to accept errors
        // Calculate relative error q for large values
        // Otherwise, calculate absolute error d
        if val.norm() > 100.0 {
            let q = (Complex64::new(s.re as f64, s.im as f64) - val).norm() / val.norm();
            println!("    k = {}, s = {}, val = {:.2}, q = {:.4}", k, s, val, q);
            assert!(q <= 0.1);
        } else {
            let d = (Complex64::new(s.re as f64, s.im as f64) - val).norm();
            println!("    k = {}, s = {}, val = {:.2}, d = {:.4}", k, s, val, d);
            assert!(d <= 2.0);
        }
    }
}

/// Simple test for real FFT
#[test]
fn fft_radix2_real_q15_simple() {
    let mut samples = [2000, 7000, 4000, 3000, 2000, 1500, 1200, 1000];
    let mut result = [Complex::new(0, 0); 5];

    let result_check = [
        Complex::new(21700, 0),
        Complex::new(2475, -8103),
        Complex::new(-1200, -4500),
        Complex::new(-2475, -2503),
        Complex::new(-3300, 0),
    ];

    fft_radix2_real_q15(&mut samples, &mut result, false).unwrap();

    println!("result = {:?}", &result);
    assert_eq!(result, result_check);
}
