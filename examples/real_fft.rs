//! Real FFT Example

use fixed_fft::fft_radix2_real_q15;
use num_complex::Complex;

fn main() {
    let mut samples = [1000; 8];
    let mut result = [Complex::new(0, 0); 5];

    println!("input data: {:?}", samples);
    fft_radix2_real_q15(&mut samples, &mut result, false).unwrap();
    println!("output data: {:?}", result);
}
