use ndarray::{Axis, Array3, Array4};
use ort::Tensor;

use super::sam_transforms::ResizeLongestSide;

pub struct Predictor {
    pub transform: ResizeLongestSide,
    //model: Session, 
}

impl Predictor {
    pub fn new(target_length: usize) -> Self {
        Predictor {
            transform: ResizeLongestSide::new(target_length),
        }
    }
}