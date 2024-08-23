/* TODO:
	1. Do some sort of opencv polygon finding / edge detection. 
	2. Outline all holds. Don't outline shadows.
	3. Implement interactable canvas in some web framework.
	4. Transfer opencv polygon/outline data to interactable canvas.
	5. Implement selecting holds to save as routes.
	6. Save/retrieve data for canvas and saved interactables in postgres.
 */

use std::env::args;

use opencv::core::{find_file, Point, Scalar, Vector};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::{approx_poly_dp, arc_length, canny, cvt_color_def, draw_contours, find_contours_with_hierarchy, median_blur, CHAIN_APPROX_NONE, COLOR_BGR2GRAY, LINE_8, RETR_TREE};
use opencv::prelude::*;
use opencv::Result;

fn main() -> Result<()> {
	// Get image path from hard coded or arg
	let mut image_name = "/workspaces/board_image_proc_dev/images/moonboard.jpg".to_string();
	if let Some(image_name_arg) = args().nth(1) {
		image_name = image_name_arg;
	}
	
	// Read the image
	let src = imread(&find_file(&image_name, false, false)?, IMREAD_COLOR)?;
	if src.empty() {
		return Ok(());
	}

	// Process Image
	if let Ok(processed_img) = process_image(src) {
		// Write img to file with '_out' suffix
		let output_filename: String = image_name.replace(".jpg", "_out.jpg");
		let params: Vector<i32> = vec![].into();
		let result = imwrite(&output_filename, &processed_img, &params);
		match result {
			Ok(..) => { println!("Success processing and writing file: {}", output_filename) }
			Err(err) => { println!("{}", err); }
		}
	}

    Ok(())
}

fn process_image(mat: Mat) -> Result<Mat> {
	let mut output = Mat::default();
	mat.copy_to(&mut output)?;

	let mut colored = Mat::default();
	let color: i32 = COLOR_BGR2GRAY;
	cvt_color_def(&mat, &mut colored, color)?;
	imwrite("images/colored.jpg", &colored, &Vector::default())?;

	let mut blurred = Mat::default();
	median_blur(&colored, &mut blurred, 5)?;
	imwrite("images/blurred.jpg", &blurred, &Vector::default())?;

	let mut edges = Mat::default();
	canny(&blurred, &mut edges, 85.0, 255.0, 3, false)?;
	imwrite("images/edges.jpg", &edges, &Vector::default())?;

	let mut img_contours: Vector<Vector<Point>> = Vector::default();
	let mut heirarchy = Mat::default();
	find_contours_with_hierarchy(&edges, &mut img_contours, &mut heirarchy, RETR_TREE, CHAIN_APPROX_NONE, Point::new(0, 0))?;

	for i in 0..img_contours.len() {
		let contour = img_contours.get(i)?;
		let mut approx: Vector<Point> = Vector::default();
		let arc_len = arc_length(&contour, true)?;
		approx_poly_dp(&contour, &mut approx, 0.02 * arc_len, true)?;
		if approx.len() > 2 {
			let draw: Vector<Vector<Point>> = Vector::from_iter(vec![approx].into_iter());
            draw_contours(
                &mut output,
                &draw,
                0,
                Scalar::from_array([255.0, 0.0, 0.0, 0.0]),
                2,
                LINE_8,
                &heirarchy,
                0,
                Point::new(0, 0),
            )?;
		}
	}

	Ok(output)
}