use ort::{inputs, Error, GraphOptimizationLevel, Session, SessionOutputs, Tensor, Value};
use image::{imageops::{grayscale, FilterType}, DynamicImage, GenericImage, GenericImageView, GrayImage, ImageBuffer, RgbImage, Rgba};
use std::{fmt::Display, io::{BufWriter, Write}, ops::Index, path::Path};
use ndarray::{array, s, Array, Array1, Array2, Array3, Array4, ArrayBase, ArrayView4, Axis, Dimension, Ix1, Ix4, IxDyn, OwnedRepr, Shape, ShapeArg};
use std::fs::File;
use std::cmp::max;

use crate::sam::sam_transforms::{self, image_to_ndarray, ndarray_to_image, ndarrayf32_to_image};

pub struct SegmentAnythingModel {
    pub encoder: Session,
    pub decoder: Session,
    model_width: usize,
    model_height: usize,
}

impl SegmentAnythingModel {
    pub fn new(encoder_path: &Path, decoder_path: &Path, model_width: usize, model_height: usize) -> Self {
        Self { 
            encoder: load_model(encoder_path).unwrap(), 
            decoder: load_model(decoder_path).unwrap(), 
            model_width, 
            model_height, 
        }
    }

    pub fn get_masks(mut self, image_path: &Path, x1: f32, y1: f32, x2: f32, y2: f32) -> Result<(), Error> {
        if let Ok(mut original_image) = image::open(image_path) {
            let (original_image_width, original_image_height) = (original_image.width(), original_image.height());
            
            //let img = original_image.resize_exact(self.model_width as u32, self.model_height as u32, FilterType::CatmullRom);
            let img_arr = image_to_ndarray(&original_image.to_rgb8()).unwrap();
            save_ndarray_to_text(&img_arr, Path::new("./images/o1rs.txt"));
            let embeddings: Array4<f32> = self.get_image_embeddings(&img_arr)?;
            save_ndarray_to_text(&embeddings, Path::new("./images/embeddingsrs.txt"));
            
            let point_coords = Value::from_array(array![[
                [x1*(self.model_width as f32 / original_image_width as f32), y1*(self.model_height as f32 / original_image_height as f32)], 
                [x2*(self.model_width as f32 / original_image_width as f32), y2*(self.model_height as f32 / original_image_height as f32)]]])?;
            let masks = self.decode(embeddings, point_coords, original_image_width as f32, original_image_height as f32)?;
            let raw_out = GrayImage::from_raw(original_image_width, original_image_height, masks.clone().into_raw_vec_and_offset().0).unwrap();
            raw_out.save("images/raw_out.jpg").unwrap();
            println!("orig: ({:?},{:?}). mask dims: {:?}", original_image_width, original_image_height, masks.dim());
            for ((_, _, y, x), value) in masks.indexed_iter() {
                if *value == 1_u8 {
                    original_image.put_pixel(x as u32, y as u32, Rgba([0, 0, 255, 0]));
                    // println!("Putting mask pixel at ({:?}, {:?})", y, x)
                }
            }

            original_image.save("images/out.jpg").unwrap();
            return Ok(());
        }

        Err(Error::new("Error while opening image")) // TODO: Make me more detailed
    }

    fn get_image_embeddings(&mut self, img: &Array3<u8>) -> Result<Array4<f32>, Error> {

        let (orig_h, orig_w, _) = img.dim();
        let rls = sam_transforms::ResizeLongestSide::new(1024);
        let img_applied = rls.apply_image(&img);
        save_ndarray_to_text(&img_applied, Path::new("./images/o2rs.txt"));
        // Permute
        let permuted = img_applied.permuted_axes((2, 0, 1));
        let added_dim0 = permuted.insert_axis(Axis(0));
        save_ndarray_to_text(&added_dim0, Path::new("./images/o3rs.txt"));
        // Preprocess
        let mean = array![123.675, 116.28, 103.53];
        let std = array![58.395, 57.120, 57.375];
        let (_, _, height, width) = added_dim0.dim();
        let mean_broadcasted: Array4<f32> = Array::from_shape_fn((1, 3, height, width), |(_, c, _, _)| mean[c]);
        let std_broadcasted: Array4<f32> = Array::from_shape_fn((1, 3, height, width), |(_, c, _, _)| std[c]);
        let normalized = (added_dim0 - &mean_broadcasted) / &std_broadcasted;
        save_ndarray_to_text(&normalized, Path::new("./images/o4rs.txt"));
        let (h, w) = (normalized.shape()[2], normalized.shape()[3]);
        let max_length = 1024; // TODO: FIX ME - should be input based on the size of the model. Maybe get rid of model_height/width and make it just a length
        let padh = max_length - h;
        let padw = max_length - w;
        let padded = constant_pad_4d(normalized.view(), (0, 0, padh, padw));
        save_ndarray_to_text(&padded, Path::new("./images/o5rs.txt"));
        
        // let mut input = Array::zeros((1, 3, self.model_width, self.model_height));
        // for pixel in img.pixels() {
        //     let x = pixel.0 as _;
        //     let y = pixel.1 as _;
        //     let [r, g, b, _] = pixel.2.0;
        //     input[[0, 0, y, x]] = (r as f32 - mean[0]) / std[0];
        //     input[[0, 1, y, x]] = (g as f32 - mean[1]) / std[1];
        //     input[[0, 2, y, x]] = (b as f32 - mean[2]) / std[2];
        // }

        let input_tensor = Value::from_array(padded)?;
        let encoder_inputs = inputs!["images" => input_tensor]?;
        let outputs = self.encoder.run(encoder_inputs)?;
        let embeddings = outputs.get("embeddings").unwrap().try_extract_tensor::<f32>()?.into_owned().into_dimensionality::<Ix4>().unwrap();

        save_ndarray_to_text(&embeddings, Path::new("./images/o6rs.txt"));
        
        return Ok(embeddings);
    }

    fn decode(self, image_embeddings: Array<f32, Ix4>, point_coords: Tensor<f32>, orig_img_w: f32, orig_img_h: f32) -> Result<(Array4<f32>, Array4<f32>), Error> {
        let point_labels: Tensor<f32> = Value::from_array(array![[2.0, 3.0]])?;
        let orig_im_size: Tensor<f32> = Value::from_array(array![orig_img_h, orig_img_w])?;
        let mask_input: Tensor<f32> = Value::from_array(Array::zeros((1, 1, 256, 256)))?;
        let has_mask_input: Tensor<f32> = Value::from_array(array![0 as f32])?;
    
        let decoder_inputs = inputs![
            "image_embeddings" => Value::from_array(image_embeddings)?,
            "point_coords" => point_coords,
            "point_labels" => point_labels,
            "orig_im_size" => orig_im_size,
            "mask_input" => mask_input,
            "has_mask_input" => has_mask_input,
        ]?;
        let outputs = self.decoder.run(decoder_inputs)?;
        println!("{:?}", outputs.keys());
        let masks = outputs.get("masks").unwrap().try_extract_tensor::<f32>()?.into_owned().into_dimensionality::<Ix4>().unwrap();
        let iou_preds = outputs.get("iou_predictions").unwrap().try_extract_tensor::<f32>()?.into_owned().into_dimensionality::<Ix4>().unwrap();
        let softmax_mask = masks.map(|item|  if *item > 0.0 { 1_u8 } else {0_u8});
    
        Ok(masks, iou_preds)
    }
}

fn load_model(model_path: &Path) -> Result<Session, Error> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    Ok(model)
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

fn save_ndarray_to_text<A, D>(array: &Array<A, D>, path: &Path) 
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

