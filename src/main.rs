/* TODO:
	1. Do some sort of opencv polygon finding / edge detection. 
	2. Outline all holds. Don't outline shadows.
	3. Implement interactable canvas in some web framework.
	4. Transfer opencv polygon/outline data to interactable canvas.
	5. Implement selecting holds to save as routes.
	6. Save/retrieve data for canvas and saved interactables in postgres.
 */

use std::env::args;
use std::result::Result;
use std::path::Path;

use image::imageops;
use sam::sam::SegmentAnythingModel;
use sam::sam_transforms::{image_to_ndarray, ndarray_to_image};

// use opencv::core::{find_file, Point, Scalar, Vector};
// use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
// use opencv::imgproc::{approx_poly_dp, arc_length, canny, cvt_color_def, draw_contours, find_contours_with_hierarchy, median_blur, CHAIN_APPROX_NONE, COLOR_BGR2GRAY, LINE_8, RETR_TREE};
// use opencv::prelude::*;
// use opencv::Result;

mod sam;

fn main() {
	// Get image path from hard coded or arg
	let images_dir_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("images");
	let models_dir_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
	let image = image::open(images_dir_path.join("resized.png")).unwrap();
	let sam = SegmentAnythingModel::new(&models_dir_path.join("vit_b_encoder.onnx"), &models_dir_path.join("vit_b_decoder.onnx"), 1024, 1024);
	sam.get_masks(&images_dir_path.join("moonboard.jpg"), 0.0, 0.0, 600.0, 600.0).unwrap();
}