use std::{
    fmt::{self, Display, Formatter},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use bevy_math::ops;
use bevy_reflect::Reflect;
use bevy_render::render_resource::ShaderType;

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Reflect, PartialEq, Debug, ShaderType)]
#[repr(C)]
pub struct c32 {
    pub re: f32,
    pub im: f32,
}

impl c32 {
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub fn from_real(re: f32) -> Self {
        Self::new(re, 0.0)
    }

    pub fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn cis(theta: f32) -> Self {
        let (im, re) = ops::sin_cos(theta);
        Self { re, im }
    }
}

impl Display for c32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}i", self.re, self.im)
    }
}

impl From<f32> for c32 {
    fn from(value: f32) -> Self {
        Self::from_real(value)
    }
}

impl Add for c32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl AddAssign for c32 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Neg for c32 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Sub for c32 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl SubAssign for c32 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for c32 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + rhs.re * self.im;
        Self { re, im }
    }
}

impl Mul<f32> for c32 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl MulAssign for c32 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs
    }
}

impl Div for c32 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let c2_d2 = rhs.re * rhs.re + rhs.im * rhs.im;
        let re = (self.re * rhs.re + self.im * rhs.im) / c2_d2;
        let im = (self.im * rhs.re - self.re * rhs.im) / c2_d2;
        Self { re, im }
    }
}

impl Div<f32> for c32 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self {
            re: self.re / rhs,
            im: self.im / rhs,
        }
    }
}

impl DivAssign for c32 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
