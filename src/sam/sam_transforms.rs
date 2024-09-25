use image::{imageops, RgbImage, RgbaImage};
use ndarray::{s, Array, Array2, Array3, Array4, ArrayView4, AssignElem, Dimension, ShapeError};
use std::{cmp::max, fmt::Display, fs::File, io::{BufWriter, Write}, path::Path, u8};

pub struct ResizeLongestSide {
    target_length: usize,
    
}

impl ResizeLongestSide {
    pub fn new(target_length: usize) -> Self {
        ResizeLongestSide {
            target_length: target_length,
        }
    }

    pub fn apply_image(&self, img_arr: &Array3<u8>) -> Array3<f32> {
        let target_length = self.target_length;
        let (target_h, target_w) = self.get_preprocess_shape(img_arr.shape()[0], img_arr.shape()[1], target_length);
        let img = ndarray_to_image(img_arr);
        let resized = imageops::resize(&img, target_w as u32, target_h as u32, imageops::FilterType::Lanczos3);
        return image_to_ndarray(&resized).unwrap().map(|v| *v as f32);
    }

    pub fn apply_coords(self, mut coords: Array2<f32>, orig_w: usize, orig_h: usize) -> Array2<f32> {
        let target_length = self.target_length;
        let (new_h, new_w) = self.get_preprocess_shape(orig_h, orig_w, target_length);
        coords.slice_mut(s![.., 0]).iter_mut().for_each(|v| *v *= (new_w as f32 / orig_w as f32));
        coords.slice_mut(s![.., 1]).iter_mut().for_each(|v| *v *= (new_h as f32 / orig_h as f32));
        return coords;
    }

    // pub fn apply_boxes(self, boxes: Array2<u8>, orig_w: usize, orig_h: usize) -> Array2<f32> {
    //     let boxes = self.apply_coords(boxes.into_shape_with_order((-1, 2, 2)).unwrap(), orig_w, orig_h);
    //     return boxes.into_shape_with_order((-1, 4)).unwrap();
    // }

    pub fn get_preprocess_shape(&self, old_h: usize, old_w: usize, long_side_length: usize) -> (usize, usize) {
        let scale = long_side_length as f32 / max(old_h, old_w) as f32;
        let (new_h, new_w) = (old_h as f32 * scale, old_w as f32 * scale);
        let ret_w = (new_w + 0.5_f32) as usize;
        let ret_h = (new_h + 0.5_f32) as usize;
        return (ret_h, ret_w);
    }
}

// Converts image to ndarray with format HxWxC
pub fn image_to_ndarray(img: &RgbImage) -> Result<Array3<u8>, ShapeError> {
    let (width, height) = img.dimensions();
    let mut img_arr = Array3::<u8>::zeros((height as usize, width as usize, 3));
    for (x, y, pixel) in img.enumerate_pixels() {
        img_arr[[y as usize, x as usize, 0]] = pixel.0[0];
        img_arr[[y as usize, x as usize, 1]] = pixel.0[1];
        img_arr[[y as usize, x as usize, 2]] = pixel.0[2];
    }
    println!("{:?}", img_arr.len());

    Ok(img_arr)
}

// Converts 3d ndarray with format HxWxC to rgba image
pub fn ndarray_to_image(img_arr: &Array3<u8>) -> RgbImage {
    let (height, width, _) = img_arr.dim();
    let mut img = RgbImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let r = img_arr[[y, x, 0]];
            let g = img_arr[[y, x, 1]];
            let b = img_arr[[y, x, 2]];
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    return img;
}

pub fn ndarrayf32_to_image(img_arr: &Array3<f32>) -> RgbImage {
    let (height, width, _) = img_arr.dim();
    let mut img = RgbImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let r = img_arr[[y, x, 0]] as u8;
            let g = img_arr[[y, x, 1]] as u8;
            let b = img_arr[[y, x, 2]] as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    return img;
}

// Pads back of each dimension specified.
pub fn constant_pad_4d(array: ArrayView4<f32>, padding: (usize, usize, usize, usize)) -> Array4<f32> {
    let (batch, channels, height, width) = array.dim();

    let mut padded = Array4::<f32>::zeros((batch + padding.0, channels + padding.1, height + padding.2, width + padding.3));
    for (idx, v) in array.indexed_iter() {
        padded[idx] = *v;
    }

    padded
}

pub fn save_ndarray_to_text<A, D>(array: &Array<A, D>, path: &Path) 
where A: Clone + Display,
      D: Dimension, 
{
    let flat = array.clone().flatten().to_owned();
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);
    for val in flat.iter() {
        writeln!(writer, "{}", *val).unwrap();
    }
}