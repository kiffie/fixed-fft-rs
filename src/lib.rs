//! Fixed-point Fast Fourier Transform
//!
//! This crate is intended for use with cores without an FPU and thus
//! can perform a fixed point FFT more quickly. The FFT code uses
//! a signed 16 bit number format, which is interpreted as a Q15
//! format (i.e. one sign bit, 15 fractional bits).
//!
//! The code was written under the assumption that a Count Leading Zeros (CLZ)
//! instruction is available on the target architecture.
//!
//! # Examples
//!
//! ## FFT with complex input data
//!
//! ```
//! use fixed_fft::{Direction, fft_radix2_q15};
//! use num_complex::Complex;
//!
//! let mut samples = [Complex::new(1000, 0); 4];
//! fft_radix2_q15(&mut samples, Direction::Forward).unwrap();
//! assert_eq!(samples[0], Complex::new(4000, 0));
//! assert_eq!(samples[1], Complex::new(0, 0));
//! assert_eq!(samples[2], Complex::new(0, 0));
//! assert_eq!(samples[3], Complex::new(0, 0));
//! ```
//!
//! ## FFT with real input data
//!
//! Note that the length of the output data is N / 2 + 1, where N is the FFT length.
//!
//! ```
//! use fixed_fft::fft_radix2_real_q15;
//! use num_complex::Complex;
//!
//! let mut samples = [1000; 4];
//! let mut result = [Complex::new(0, 0); 3];
//! fft_radix2_real_q15(&mut samples, &mut result, false).unwrap();
//! assert_eq!(result[0], Complex::new(4000, 0));
//! assert_eq!(result[1], Complex::new(0, 0));
//! assert_eq!(result[2], Complex::new(0, 0));
//! ```
//!

#![no_std]

use core::slice;
use num_complex::Complex;

type Q15 = i16;

pub enum Direction {
    /// Normal forward FFT
    Forward,

    /// Forward FFT with output values scaled by 1/N
    ///
    /// This variant can be used to avoid numerical overflows.
    ForwardScaled,

    /// Inverse FFT
    Inverse,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FFTError {
    InvalidFFTSize,
    InvalidDataSize,
}

const SINETAB_LD_N: usize = 10;
const SINETAB_N: usize = 1 << SINETAB_LD_N;

/// Sine table used to calculate twiddle factors
static SINETAB: [i16; SINETAB_N - SINETAB_N / 4] = [
    0, 201, 402, 603, 804, 1005, 1206, 1407, 1608, 1809, 2009, 2210, 2410, 2611, 2811, 3012, 3212,
    3412, 3612, 3811, 4011, 4210, 4410, 4609, 4808, 5007, 5205, 5404, 5602, 5800, 5998, 6195, 6393,
    6590, 6786, 6983, 7179, 7375, 7571, 7767, 7962, 8157, 8351, 8545, 8739, 8933, 9126, 9319, 9512,
    9704, 9896, 10087, 10278, 10469, 10659, 10849, 11039, 11228, 11417, 11605, 11793, 11980, 12167,
    12353, 12539, 12725, 12910, 13094, 13279, 13462, 13645, 13828, 14010, 14191, 14372, 14553,
    14732, 14912, 15090, 15269, 15446, 15623, 15800, 15976, 16151, 16325, 16499, 16673, 16846,
    17018, 17189, 17360, 17530, 17700, 17869, 18037, 18204, 18371, 18537, 18703, 18868, 19032,
    19195, 19357, 19519, 19680, 19841, 20000, 20159, 20317, 20475, 20631, 20787, 20942, 21096,
    21250, 21403, 21554, 21705, 21856, 22005, 22154, 22301, 22448, 22594, 22739, 22884, 23027,
    23170, 23311, 23452, 23592, 23731, 23870, 24007, 24143, 24279, 24413, 24547, 24680, 24811,
    24942, 25072, 25201, 25329, 25456, 25582, 25708, 25832, 25955, 26077, 26198, 26319, 26438,
    26556, 26674, 26790, 26905, 27019, 27133, 27245, 27356, 27466, 27575, 27683, 27790, 27896,
    28001, 28105, 28208, 28310, 28411, 28510, 28609, 28706, 28803, 28898, 28992, 29085, 29177,
    29268, 29358, 29447, 29534, 29621, 29706, 29791, 29874, 29956, 30037, 30117, 30195, 30273,
    30349, 30424, 30498, 30571, 30643, 30714, 30783, 30852, 30919, 30985, 31050, 31113, 31176,
    31237, 31297, 31356, 31414, 31470, 31526, 31580, 31633, 31685, 31736, 31785, 31833, 31880,
    31926, 31971, 32014, 32057, 32098, 32137, 32176, 32213, 32250, 32285, 32318, 32351, 32382,
    32412, 32441, 32469, 32495, 32521, 32545, 32567, 32589, 32609, 32628, 32646, 32663, 32678,
    32692, 32705, 32717, 32728, 32737, 32745, 32752, 32757, 32761, 32765, 32766, 32767, 32766,
    32765, 32761, 32757, 32752, 32745, 32737, 32728, 32717, 32705, 32692, 32678, 32663, 32646,
    32628, 32609, 32589, 32567, 32545, 32521, 32495, 32469, 32441, 32412, 32382, 32351, 32318,
    32285, 32250, 32213, 32176, 32137, 32098, 32057, 32014, 31971, 31926, 31880, 31833, 31785,
    31736, 31685, 31633, 31580, 31526, 31470, 31414, 31356, 31297, 31237, 31176, 31113, 31050,
    30985, 30919, 30852, 30783, 30714, 30643, 30571, 30498, 30424, 30349, 30273, 30195, 30117,
    30037, 29956, 29874, 29791, 29706, 29621, 29534, 29447, 29358, 29268, 29177, 29085, 28992,
    28898, 28803, 28706, 28609, 28510, 28411, 28310, 28208, 28105, 28001, 27896, 27790, 27683,
    27575, 27466, 27356, 27245, 27133, 27019, 26905, 26790, 26674, 26556, 26438, 26319, 26198,
    26077, 25955, 25832, 25708, 25582, 25456, 25329, 25201, 25072, 24942, 24811, 24680, 24547,
    24413, 24279, 24143, 24007, 23870, 23731, 23592, 23452, 23311, 23170, 23027, 22884, 22739,
    22594, 22448, 22301, 22154, 22005, 21856, 21705, 21554, 21403, 21250, 21096, 20942, 20787,
    20631, 20475, 20317, 20159, 20000, 19841, 19680, 19519, 19357, 19195, 19032, 18868, 18703,
    18537, 18371, 18204, 18037, 17869, 17700, 17530, 17360, 17189, 17018, 16846, 16673, 16499,
    16325, 16151, 15976, 15800, 15623, 15446, 15269, 15090, 14912, 14732, 14553, 14372, 14191,
    14010, 13828, 13645, 13462, 13279, 13094, 12910, 12725, 12539, 12353, 12167, 11980, 11793,
    11605, 11417, 11228, 11039, 10849, 10659, 10469, 10278, 10087, 9896, 9704, 9512, 9319, 9126,
    8933, 8739, 8545, 8351, 8157, 7962, 7767, 7571, 7375, 7179, 6983, 6786, 6590, 6393, 6195, 5998,
    5800, 5602, 5404, 5205, 5007, 4808, 4609, 4410, 4210, 4011, 3811, 3612, 3412, 3212, 3012, 2811,
    2611, 2410, 2210, 2009, 1809, 1608, 1407, 1206, 1005, 804, 603, 402, 201, 0, -201, -402, -603,
    -804, -1005, -1206, -1407, -1608, -1809, -2009, -2210, -2410, -2611, -2811, -3012, -3212,
    -3412, -3612, -3811, -4011, -4210, -4410, -4609, -4808, -5007, -5205, -5404, -5602, -5800,
    -5998, -6195, -6393, -6590, -6786, -6983, -7179, -7375, -7571, -7767, -7962, -8157, -8351,
    -8545, -8739, -8933, -9126, -9319, -9512, -9704, -9896, -10087, -10278, -10469, -10659, -10849,
    -11039, -11228, -11417, -11605, -11793, -11980, -12167, -12353, -12539, -12725, -12910, -13094,
    -13279, -13462, -13645, -13828, -14010, -14191, -14372, -14553, -14732, -14912, -15090, -15269,
    -15446, -15623, -15800, -15976, -16151, -16325, -16499, -16673, -16846, -17018, -17189, -17360,
    -17530, -17700, -17869, -18037, -18204, -18371, -18537, -18703, -18868, -19032, -19195, -19357,
    -19519, -19680, -19841, -20000, -20159, -20317, -20475, -20631, -20787, -20942, -21096, -21250,
    -21403, -21554, -21705, -21856, -22005, -22154, -22301, -22448, -22594, -22739, -22884, -23027,
    -23170, -23311, -23452, -23592, -23731, -23870, -24007, -24143, -24279, -24413, -24547, -24680,
    -24811, -24942, -25072, -25201, -25329, -25456, -25582, -25708, -25832, -25955, -26077, -26198,
    -26319, -26438, -26556, -26674, -26790, -26905, -27019, -27133, -27245, -27356, -27466, -27575,
    -27683, -27790, -27896, -28001, -28105, -28208, -28310, -28411, -28510, -28609, -28706, -28803,
    -28898, -28992, -29085, -29177, -29268, -29358, -29447, -29534, -29621, -29706, -29791, -29874,
    -29956, -30037, -30117, -30195, -30273, -30349, -30424, -30498, -30571, -30643, -30714, -30783,
    -30852, -30919, -30985, -31050, -31113, -31176, -31237, -31297, -31356, -31414, -31470, -31526,
    -31580, -31633, -31685, -31736, -31785, -31833, -31880, -31926, -31971, -32014, -32057, -32098,
    -32137, -32176, -32213, -32250, -32285, -32318, -32351, -32382, -32412, -32441, -32469, -32495,
    -32521, -32545, -32567, -32589, -32609, -32628, -32646, -32663, -32678, -32692, -32705, -32717,
    -32728, -32737, -32745, -32752, -32757, -32761, -32765, -32766,
];

/// complex multiplication in Q15 format
fn q15_cmul(a: Complex<Q15>, b: Complex<Q15>) -> Complex<Q15> {
    let rnd: i32 = 1 << 14; // constant for rounding
    Complex::new(
        ((a.re as i32 * b.re as i32 - a.im as i32 * b.im as i32 + rnd) >> 15) as i16,
        ((a.re as i32 * b.im as i32 + a.im as i32 * b.re as i32 + rnd) >> 15) as i16,
    )
}

fn check_size(size: usize) -> Result<(), FFTError> {
    if !(2..=SINETAB_N).contains(&size) {
        return Err(FFTError::InvalidFFTSize);
    }
    const MSB: usize = (usize::MAX >> 1) + 1;
    let allowed_size = MSB >> (size.leading_zeros());
    if allowed_size != size {
        return Err(FFTError::InvalidFFTSize);
    }
    Ok(())
}

/// Radix-2 in-place FFT
///
/// Perform forward/inverse fast Fourier transform using the radix-2 algorithm.
/// The the length of slice```data``` corresponds to the FFT size and must be a
/// power of 2. The maximum FFT size is 1024.
///
/// Returns `FFTError::InvalidFFTSize` if the requested FFT size is not supported.
pub fn fft_radix2_q15(data: &mut [Complex<Q15>], dir: Direction) -> Result<(), FFTError> {
    let fft_size = data.len();
    check_size(fft_size)?;
    let (inverse, shift) = match dir {
        Direction::Forward => (false, 0),
        Direction::ForwardScaled => (false, 1),
        Direction::Inverse => (true, 1),
    };

    // re-order data
    let mut mr = 0usize; // bit reverse counter
    for m in 1..fft_size {
        const MSB: usize = (usize::MAX >> 1) + 1;
        let l = MSB >> (fft_size - 1 - mr).leading_zeros();
        mr = (mr & (l - 1)) + l;
        if mr > m {
            data.swap(m, mr);
        }
    }

    let mut stage = 1;
    let mut sine_step = (SINETAB_LD_N - 1) as isize;
    // loop over stages
    while stage < fft_size {
        // loop over groups with the same twiddle factor
        let twiddle_step = stage << 1;
        for group in 0..stage {
            // calculate twiddle factor
            let sin_ndx = group << sine_step;
            let wr = SINETAB[sin_ndx + SINETAB_N / 4] >> shift;
            let wi = match inverse {
                false => -SINETAB[sin_ndx],
                true => SINETAB[sin_ndx],
            } >> shift;
            let w = Complex::new(wr, wi);
            // butterfly operations
            let mut i = group;
            while i < fft_size {
                let j = i + stage;
                let temp = q15_cmul(w, data[j]);

                let q = Complex::new(data[i].re >> shift, data[i].im >> shift);
                data[i] = q + temp;
                data[j] = q - temp;
                i += twiddle_step;
            }
        }
        sine_step -= 1;
        stage = twiddle_step;
    }
    Ok(())
}

/// Real value FFT using a complex valued FFT
///
/// The `input` data will be used as a buffer for an in-place CFFT and will thus be modified by
/// this function. The `output` slice will hold the result without redundant data.
/// The length of the slices must be as follows: `output.len() >= input.len() / 2 + 1`
/// The data will be scaled by 1/input.len() if `scale == true`.
pub fn fft_radix2_real_q15(
    input: &mut [i16],
    output: &mut [Complex<Q15>],
    scale: bool,
) -> Result<(), FFTError> {
    let fft_len = input.len();
    check_size(fft_len)?;
    let half_fft_len = fft_len / 2;
    if output.len() < half_fft_len + 1 {
        return Err(FFTError::InvalidDataSize);
    }

    // Assume that Complex<T> is a exactly a sequence of two values { re: T, im: T }
    // This is checked for T == Q15 by the test module below
    let ptr = input.as_mut_ptr() as *mut Complex<Q15>;
    let fft_out = unsafe { slice::from_raw_parts_mut(ptr, half_fft_len) };

    // Do half size CFFT
    fft_radix2_q15(
        &mut fft_out[0..half_fft_len],
        if scale {
            Direction::ForwardScaled
        } else {
            Direction::Forward
        },
    )?;

    let shift = scale as usize;
    // Post processing that yields fft_len / 2 + 1 complex values
    output[0] = Complex::new((fft_out[0].im + fft_out[0].re) >> shift, 0);
    for k in 1..half_fft_len {
        // calculate the twiddle factor w
        let sin_ndx = k * SINETAB_N / fft_len;
        let w = Complex::new(SINETAB[sin_ndx + SINETAB_N / 4], -SINETAB[sin_ndx]);

        // do one post processing calculation
        let zo = Complex::new(
            (fft_out[half_fft_len - k].im + fft_out[k].im) >> (shift + 1),
            (fft_out[half_fft_len - k].re - fft_out[k].re) >> (shift + 1),
        );
        let ze = Complex::new(
            (fft_out[k].re + fft_out[half_fft_len - k].re) >> (shift + 1),
            (fft_out[k].im - fft_out[half_fft_len - k].im) >> (shift + 1),
        );
        let r = ze + q15_cmul(zo, w);
        output[k] = r;
    }
    output[half_fft_len] = Complex::new((fft_out[0].re - fft_out[0].im) >> shift, 0);

    Ok(())
}

#[cfg(test)]
mod tests {

    extern crate alloc;
    use super::{check_size, FFTError, Q15, SINETAB, SINETAB_LD_N, SINETAB_N};
    use alloc::vec::Vec;
    use core::f64;
    use core::mem::size_of;
    use num_complex::Complex;

    /// check sine table
    #[test]
    fn verify_sinewave() {
        for (i, v) in SINETAB.iter().enumerate() {
            let fval = 32767.0 * f64::sin(2.0 * f64::consts::PI * (i as f64) / (SINETAB_N as f64));
            let rval = fval.round() as i16;
            assert_eq!(*v, rval);
        }
    }

    /// Check size of Complex<Q15> to make sure that the unsafe type cast in the real FFT function
    /// works correctly
    #[test]
    fn check_complex_lenth() {
        assert_eq!(size_of::<Complex<Q15>>(), 2 * size_of::<Q15>());
    }

    #[test]
    fn check_size_test() {
        let valid = (1..=SINETAB_LD_N).map(|k| 1 << k).collect::<Vec<usize>>();
        for s in 0..=SINETAB_N * 2 {
            let result = check_size(s);
            if valid.contains(&s) {
                assert_eq!(result, Ok(()));
            } else {
                assert_eq!(result, Err(FFTError::InvalidFFTSize))
            }
        }
    }
}
