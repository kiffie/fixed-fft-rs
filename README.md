# Fixed-point Fast Fourier Transform

This "no-std" crate is intended for use with cores without an FPU and that can
perform a fixed point FFT more quickly. The FFT code uses
a signed 16 bit number format, which is interpreted as a Q15
format (i.e. one sign bit, 15 fractional bits).

The code was written under the assumption that a Count Leading Zeros (CLZ)
instruction is available on the target architecture and that
`leading_zeros()`uses this instruction.

This code was inspired by [fix_fft.c](https://gist.github.com/Tomwi/3842231),
which is a very simple fixed-point FFT function written in C. The idea is to provide
a simple, straightforward and target-independent FFT implementation.

## Example with complex input data

```rust
use fixed_fft::{fft_radix2_q15, Direction};
use num_complex::Complex;

fn main() {
    let mut samples = [Complex::new(1000, 0); 8];

    println!("input data: {:?}", samples);
    fft_radix2_q15(&mut samples, Direction::Forward).unwrap();
    println!("output data: {:?}", samples);
}
```

## Example with real input data

```rust
use fixed_fft::fft_radix2_real_q15;
use num_complex::Complex;

fn main() {
    let mut samples = [1000; 8];
    let mut result = [Complex::new(0, 0); 5];

    println!("input data: {:?}", samples);
    fft_radix2_real_q15(&mut samples, &mut result, false).unwrap();
    println!("output data: {:?}", result);
}
```
