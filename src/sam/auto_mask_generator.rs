use std::{cmp::min, collections::HashMap, path::Path, thread::yield_now};

use image::{GenericImageView, ImageError, ImageResult};
use ndarray::{array, s, stack, Array, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, Axis, Dim, Ix3, Ix4, Shape, ShapeError};
use ort::{inputs, Error, Session, Value};
use std::iter::zip;
extern crate itertools;
use itertools::iproduct;

use crate::sam::mask_data::MaskData;
use crate::sam::predictor::Predictor;

use super::{mask_data, sam_transforms::{self, ResizeLongestSide}};

pub struct SamAutomaticMaskGenerator {
    encoder: Session,
    decoder: Session,
    predictor: Predictor,
    points_per_side: Option<i32>,
    points_per_batch: usize,
    pred_iou_thresh: f32,
    stability_score_thresh: f32,
    stability_score_offset: f32,
    box_nms_thresh: f32,
    crop_n_layers: i32,
    crop_nms_thresh: f32,
    crop_overlap_ratio: f32,
    crop_n_points_downscale_factor: i32,
    point_grids: Option<Vec<Array2<f32>>>,
    min_mask_region_area: i32,
    output_mode: String,
}

impl SamAutomaticMaskGenerator {
    pub fn new(encoder: Session, decoder: Session) -> Self {
        let mut sam = SamAutomaticMaskGenerator {
            encoder: encoder,
            decoder: decoder,
            predictor: Predictor::new(1024),
            points_per_side: Some(32),
            points_per_batch: 64,
            pred_iou_thresh: 0.88,
            stability_score_thresh: 0.95,
            stability_score_offset: 1.0,
            box_nms_thresh: 0.7,
            crop_n_layers: 0,
            crop_nms_thresh:  0.7,
            crop_overlap_ratio: 0.34133333333,
            crop_n_points_downscale_factor: 1,
            point_grids: None,
            min_mask_region_area: 0,
            output_mode: "binary_mask".to_string(),
        };

        assert!(sam.points_per_side.is_some() || sam.point_grids.is_some(), "Exactly one of points_per_side or point_grid must be provided.");
        if sam.points_per_side.is_some() {
            sam.point_grids = Some(build_all_layer_point_grids(
                sam.points_per_side.unwrap(),
                sam.crop_n_layers,
                sam.crop_n_points_downscale_factor,
            ));
        } else if sam.point_grids.is_none() {
            panic!("Can't have both points_per_side and point_grid be None.");
        }
            
        return sam;
    }

    // image format: HWC uint8
    fn generate(self, image: Array3<u8>) {
        let mask_data = self.generate_masks(image);
    }

    pub fn generate_masks(&self, image: Array3<u8>) {
        let (orig_h, orig_w) = (image.dim().1, image.dim().2);
        let crop_n_layers = self.crop_n_layers;
        let crop_overlap_ratio = self.crop_overlap_ratio;
        let (crop_boxes, layer_idxs) = self.generate_crop_boxes(orig_w, orig_h, crop_n_layers as usize, crop_overlap_ratio);
        println!("crop_boxes: {:?}", crop_boxes);
        println!("layer_idxs: {:?}", layer_idxs);

        // TODO: make sure crop boxes and layer idxs are generated correctly.
        let data = MaskData::new(None);
        for (crop_box, layer_idx) in zip(crop_boxes, layer_idxs) {
            let crop_data = self.process_crop(&image, crop_box, layer_idx, orig_w, orig_h);
            data.cat(crop_data);
        }

    }

    fn generate_crop_boxes(&self, im_w: usize, im_h: usize, n_layers: usize, overlap_ratio: f32) -> (Vec<[i32; 4]>, Vec<i32>) {
        let mut crop_boxes = vec![];
        let mut layer_idxs = vec![];
        let short_side = min(im_h, im_w);

        crop_boxes.push([0, 0, im_w as i32, im_h as i32]);
        layer_idxs.push(0);

        fn crop_len(orig_len: usize, n_crops: usize, overlap: i32) -> i32 {
            return (f32::ceil((overlap as f32 * (n_crops - 1) as f32 + orig_len as f32) / n_crops as f32)) as i32;
        }

        for i_layer in 0..n_layers as u32 {
            let n_crops_per_side = 2_usize.pow(i_layer + 1);
            let overlap = (overlap_ratio * short_side as f32 * (2.0f32 / n_crops_per_side as f32)) as i32;

            let crop_w = crop_len(im_w, n_crops_per_side, overlap);
            let crop_h = crop_len(im_h, n_crops_per_side, overlap);

            let mut crop_box_x0 = vec![];
            let mut crop_box_y0 = vec![];
            for i in 0..n_crops_per_side {
                crop_box_x0.push(((crop_w - overlap) as f32 * i as f32) as i32);
                crop_box_y0.push(((crop_h - overlap) as f32 * i as f32) as i32);
            }

            for (x0, y0) in iproduct!(crop_box_x0, crop_box_y0) {
                let cbox = [x0, y0, min(x0 + crop_w, im_w as i32), min(y0 + crop_h, im_h as i32)];
                crop_boxes.push(cbox);
                layer_idxs.push(i_layer as i32 + 1);
            }
        }

        return (crop_boxes, layer_idxs);
    }

    fn process_crop(&self, image: &Array3<u8>, crop_box: [i32; 4], crop_layer_idx: usize, orig_w: usize, orig_h: usize) -> MaskData {
        let (x0, y0, x1, y1) = (crop_box[0], crop_box[1], crop_box[2], crop_box[3]);
        let cropped_im = image.slice(s![y0..y1, x0..x1, ..]).to_owned();
        let (cropped_h, cropped_w) = (cropped_im.dim().1, cropped_im.dim().2);
        let embeddings = self.get_image_embeddings(&cropped_im).unwrap();

        let points_scale = array![cropped_h as f32, cropped_w as f32].insert_axis(Axis(0));
        let points_for_image = self.point_grids.unwrap()[crop_layer_idx] * points_scale;

        let data = MaskData::new(None);
        for points in batch_iterator(self.points_per_batch, points_for_image.view()) {
            let batch_data = self.process_batch(points_for_image, cropped_h, cropped_w, crop_box, orig_h, orig_w, embeddings).unwrap();
            data.cat(batch_data);
        }
        
        return data;
    }

    fn process_batch(&self, points: Array2<f32>, cropped_h: usize, cropped_w: usize, crop_box: [i32; 4], orig_h: usize, orig_w: usize, embeddings: Array4<f32>) -> Result<MaskData, Error> {
        let transformed_points = self.predictor.transform.apply_coords(points, cropped_h, cropped_w);
        let in_labels = Array1::<i32>::ones(points.dim().0); // TODO: Make this a member?
        let mask_input = Value::from_array(Array4::<f32>::zeros((1, 1, 256, 256)))?;
        let has_mask_input = Value::from_array(array![0 as f32])?;
        let decoder_inputs = inputs![
            "image_embeddings" => Value::from_array(embeddings)?,
            "point_coords" => Value::from_array(transformed_points)?,
            "point_labels" => Value::from_array(in_labels)?,
            "orig_im_size" => Value::from_array(array![orig_h as i32, orig_w as i32])?,
            "mask_input" => mask_input,
            "has_mask_input" => has_mask_input,
        ]?;

        let outputs = self.decoder.run(decoder_inputs)?;
        println!("{:?}", outputs.keys());
        let masks = outputs.get("masks").unwrap().try_extract_tensor::<f32>()?.into_owned().into_dimensionality::<Ix4>().unwrap();
        let iou_preds = outputs.get("iou_predictions").unwrap().try_extract_tensor::<f32>()?.into_owned().into_dimensionality::<Ix4>().unwrap();

        let masks_dim = masks.dim();
        let masks_flattened = masks.into_shape_with_order((masks_dim.0 * masks_dim.1, masks_dim.2, masks_dim.3)).unwrap();

        let iou_preds_dim = iou_preds.dim();
        let iou_preds_flattened = iou_preds.into_shape_with_order((iou_preds_dim.0 * iou_preds_dim.1, iou_preds_dim.2, iou_preds_dim.3)).unwrap();

        let repeat_count = masks_dim.1 / points.dim().0;
        let mut points_repeated = Array2::zeros((masks_dim.1, points.dim().1));
        for row in 0..points.dim().0 {
            let src_row = points.slice(s![row, ..]);
            for i in 0..repeat_count {
                points_repeated.slice_mut(s![row * repeat_count + i, ..]).assign(&src_row);
            }
        }

        let keep_mask: Option<Array3<bool>> = None;
        if self.pred_iou_thresh > 0.0_f32 {
            keep_mask = Some(iou_preds_flattened.mapv(|x| x > self.pred_iou_thresh));
        }
        
        let mask_threshold = 0.0f32;
        let stability_score = calculate_stability_score(masks_flattened, mask_threshold, self.stability_score_offset);
        let keep_stability = None;
        if self.stability_score_thresh > 0.0_f32 {
            keep_stability = Some(stability_score.mapv(|x| x >= self.stability_score_thresh));
        }

        let keep_masks = masks_flattened.mapv(|x| x > mask_threshold);
        let boxes = batched_mask_to_box(keep_masks);


        
    }

    fn batched_mask_to_box(masks: &Array3<f32>) -> Array3<f32> {
        let shape = masks.dim();
        let (h, w) = (shape.1, shape.2);
        
        let in_height = masks
                                                        .axis_iter(Axis(2))
                                                        .map(|row| row.fold(std::f32::MIN, |a, &b| a.max(b)))
                                                        .collect::<Array1<f32>>();

        let in_height_coords: Array1<usize> = in_height * Array1::from_iter(0..h);
        let bottom_edges = in_height_coords
            .axis_iter(Axis(2))
            .map(|row| row.fold(std::f32::MIN, |a, &b| a.max(b)))
            .collect::<Array1<f32>>();
        in_height_coords = in_height_coords + h * in_height;


    }

    fn get_image_embeddings(&mut self, img: &Array3<u8>) -> Result<Array4<f32>, Error> {

        let (orig_h, orig_w, _) = img.dim();
        let rls = ResizeLongestSide::new(1024);
        let img_applied = rls.apply_image(&img);
        // Permute
        let permuted = img_applied.permuted_axes((2, 0, 1));
        let added_dim0 = permuted.insert_axis(Axis(0));
        // Preprocess
        let mean = array![123.675, 116.28, 103.53];
        let std = array![58.395, 57.120, 57.375];
        let (_, _, height, width) = added_dim0.dim();
        let mean_broadcasted: Array4<f32> = Array::from_shape_fn((1, 3, height, width), |(_, c, _, _)| mean[c]);
        let std_broadcasted: Array4<f32> = Array::from_shape_fn((1, 3, height, width), |(_, c, _, _)| std[c]);
        let normalized = (added_dim0 - &mean_broadcasted) / &std_broadcasted;
        let (h, w) = (normalized.shape()[2], normalized.shape()[3]);
        let max_length = 1024; // TODO: FIX ME - should be input based on the size of the model. Maybe get rid of model_height/width and make it just a length
        let padh = max_length - h;
        let padw = max_length - w;
        let padded = sam_transforms::constant_pad_4d(normalized.view(), (0, 0, padh, padw));

        let input_tensor = Value::from_array(padded)?;
        let encoder_inputs = inputs!["images" => input_tensor]?;
        let outputs = self.encoder.run(encoder_inputs)?;
        let embeddings = outputs.get("embeddings").unwrap().try_extract_tensor::<f32>()?.into_owned().into_dimensionality::<Ix4>().unwrap();

        return Ok(embeddings);
    }
}

fn calculate_stability_score(masks: Array3<f32>, mask_threshold: f32, threshold_offset: f32) -> Array1<f32> {
    let high_threshold = mask_threshold + threshold_offset;
    let low_threshold = mask_threshold - threshold_offset;

    // Create binary masks for intersections and unions
    let intersections = masks.mapv(|x| if x > high_threshold { 1.0 } else { 0.0 });
    let unions = masks.mapv(|x| if x > low_threshold { 1.0 } else { 0.0 });

    // Sum over the last two axes to get intersections and unions
    let intersection_sums = intersections.sum_axis(Axis(1)).sum_axis(Axis(1));
    let union_sums = unions.sum_axis(Axis(1)).sum_axis(Axis(1));

    // Calculate stability score
    intersection_sums / union_sums
}

fn batch_iterator<'a, T>(batch_size: usize, data: ArrayView2<'a, T>) -> impl Iterator<Item = Array2<T>> + 'a
where  
    T: Clone,
{
    (0..(data.dim().0 + batch_size - 1) / batch_size).map(move |i| {
        let start = i * batch_size;
        let end = std::cmp::min(start + batch_size, data.dim().0);
        data.slice(s![start..end, ..]).to_owned()
    })
}

fn build_all_layer_point_grids(n_per_side: i32, n_layers: i32, scale_per_layer: i32) -> Vec<Array2<f32>> {
    let mut points_by_layer: Vec<Array2<f32>> = vec![];
    for i in 0 as u32..(n_layers + 1) as u32 {
        let n_points = n_per_side / (scale_per_layer.pow(i));
        points_by_layer.push(build_point_grid(n_points as usize).unwrap());
    }

    return points_by_layer;
}

fn build_point_grid(n_per_side: usize) -> Result<Array2<f32>, ShapeError> {
    let offset = 1.0_f32 / (2 * n_per_side) as f32;
    let points_one_side = Array::linspace(offset, 1.0_f32 - offset, n_per_side as usize);
    let mut points = Vec::with_capacity(n_per_side * n_per_side * 2);
    for i in 0..n_per_side {
        for j in 0..n_per_side {
            points.push(points_one_side[j]);
            points.push(points_one_side[i]);
        }
    }

    let points_arr = Array2::from_shape_vec((n_per_side * n_per_side, 2), points).unwrap();
    return Ok(points_arr);
}

pub fn load_image(image_path: &Path) -> Result<Array3<u8>, ImageError> {
    let original_image = image::open(image_path)?;
    let mut image_arr: Array3<u8> = Array::zeros((original_image.height() as usize, original_image.width() as usize, 3_usize));
    for pixel in original_image.pixels() {
        let mean = vec![123.675, 116.28, 103.53];
        let std = vec![58.395, 57.120, 57.375];
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2.0;
        image_arr[[y, x, 0]] = ((r as f32 - mean[0]) / std[0]) as u8;
        image_arr[[y, x, 1]] = ((g as f32 - mean[1]) / std[1]) as u8;
        image_arr[[y, x, 2]] = ((b as f32 - mean[2]) / std[2]) as u8;
    }

    return Ok(image_arr)
}