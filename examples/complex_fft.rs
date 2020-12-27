//! Complex FFT Example

use fixed_fft::{fft_radix2_q15, Direction};
use num_complex::Complex;

fn main() {
    let mut samples = [Complex::new(1000, 0); 8];

    println!("input data: {:?}", samples);
    fft_radix2_q15(&mut samples, Direction::Forward).unwrap();
    println!("output data: {:?}", samples);
}
