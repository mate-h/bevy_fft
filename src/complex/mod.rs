use bevy_math::ops;
use encase::ShaderType;
use half::f16;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq)]
#[repr(align(4))]
pub struct c16 {
    pub real: f16,
    pub complex: f16,
}

impl c16 {
    pub fn new(real: f16, complex: f16) -> Self {
        Self { real, complex }
    }

    pub fn conjugate(self) -> Self {
        Self {
            real: self.real,
            complex: -self.complex,
        }
    }

    pub fn exp(theta: f32) -> Self {
        let (sin, cos) = ops::sin_cos(theta);
        Self {
            real: f16::from_f32(cos),
            complex: f16::from_f32(sin),
        }
    }
}

impl Add for c16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real + rhs.real,
            complex: self.complex + rhs.complex,
        }
    }
}

impl AddAssign for c16 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Neg for c16 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            real: -self.real,
            complex: -self.complex,
        }
    }
}

impl Sub for c16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real - rhs.real,
            complex: self.complex - rhs.complex,
        }
    }
}

impl SubAssign for c16 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for c16 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let real = self.real * rhs.real - self.complex * rhs.complex;
        let complex = self.real * rhs.complex + rhs.real * self.complex;
        Self { real, complex }
    }
}

impl Mul<f16> for c16 {
    type Output = Self;

    fn mul(self, rhs: f16) -> Self::Output {
        Self {
            real: self.real * rhs,
            complex: self.complex * rhs,
        }
    }
}

impl MulAssign for c16 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs
    }
}

impl Div for c16 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let c2_d2 = rhs.real * rhs.real + rhs.complex * rhs.complex;
        let real = (self.real * rhs.real + self.complex * rhs.complex) / c2_d2;
        let complex = (self.complex * rhs.real - self.real * rhs.complex) / c2_d2;
        Self { real, complex }
    }
}

impl Div<f16> for c16 {
    type Output = Self;

    fn div(self, rhs: f16) -> Self::Output {
        Self {
            real: self.real / rhs,
            complex: self.complex / rhs,
        }
    }
}

impl DivAssign for c16 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
