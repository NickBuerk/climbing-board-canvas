use std::collections::HashMap;

use ndarray::{Axis, Array3, concatenate};


pub struct MaskData {
    stats: HashMap<String, Array3<f32>>,
}

impl MaskData {
    pub fn new(tensor_map: Option<HashMap<String, Array3<f32>>>) -> Self {
        Self {
            stats: tensor_map.unwrap_or_default(),
        }
    }

    pub fn get(self, key: &String) -> Array3<f32> {
        return self.stats.get(key).unwrap().clone();
    }

    pub fn set(mut self, key: &String, value: Array3<f32>) {
        self.stats.insert(key.to_string(), value).unwrap();
    }

    pub fn cat(mut self, new_stats: HashMap<String, Array3<f32>>) {
        for (k, v) in new_stats.iter() {
            if !self.stats.contains_key(k) || self.stats.get(k).is_none() {
                self.stats.insert(k.clone(), v.clone());
            } else {
                self.stats.insert(k.clone(), concatenate![Axis(0), self.stats.get(k).unwrap().view(), v.view()]);
            }
        }
    }

    pub fn filter(mut self, keep: HashMap<String, Array3<f32>>) {
        for (k, v) in keep.iter() {
            self.stats.insert(k.clone(), v.clone());
        }
    }
}

// pub fn is_box_near_crop_edge(boxes: Tensor, crop_box: Vec<i32>, orig_box: Vec<i32>, atol: Option<f64>) -> Tensor {
//     let crop_box_torch = tch::Tensor::from_slice(&crop_box.to_owned()).to_dtype(Kind::Float, false, false).to_device(boxes.device());
//     let orig_box_torch = tch::Tensor::from_slice(&orig_box.to_owned()).to_dtype(Kind::Float, false, false).to_device(boxes.device());
//     let boxes = uncrop_boxes_xyxy(boxes, crop_box).to_dtype(Kind::Float, false, false);
//     let near_crop_edge = boxes.isclose(&crop_box_torch, 0_f64, atol.unwrap_or(20_f64), false);
//     let near_image_edge = boxes.isclose(&orig_box_torch, 0_f64, atol.unwrap_or(20_f64), false);
//     let near_cop_edge = near_crop_edge.logical_and(&near_image_edge.bitwise_not());
//     return near_cop_edge.any_dim(1, false);
// }

// pub fn box_xyxy_to_xywh(box_xyxy: Tensor) -> Tensor {
//     let x0 = box_xyxy.get(0).double_value(&[]);
//     let y0 = box_xyxy.get(1).double_value(&[]);
//     let x1 = box_xyxy.get(2).double_value(&[]);
//     let y1 = box_xyxy.get(3).double_value(&[]);
//     let width = x1 - x0;
//     let height = y1 - y0;

//     Tensor::from_slice(&[x0, y0, width, height])
//         .to_device(box_xyxy.device())
//         .to_kind(Kind::Float) 
// }

// pub fn uncrop_boxes_xyxy(boxes: Tensor, crop_box: Vec<i32>) -> Tensor {
//     let (x0, y0) = (crop_box.get(0).unwrap().clone(), crop_box.get(1).unwrap().clone());
//     let slice = vec![vec![x0, y0, x0, y0]];
//     let mut offset = Tensor::from_slice2::<i32, Vec<i32>>(slice.as_slice()).to_device(boxes.device());
//     if boxes.dim() == 3 {
//         offset = offset.unsqueeze(1);
//     }
//     return boxes + offset;
// }