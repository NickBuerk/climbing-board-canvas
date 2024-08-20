
use std::env::args;

use opencv::core::{find_file, Vector};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::{cvt_color_def, COLOR_BGR2GRAY,};
use opencv::prelude::*;
use opencv::Result;

fn main() -> Result<()> {
	// Read the image
	let mut image_name = "/workspaces/board_image_proc_dev/images/test.jpg".to_string();
	if let Some(image_name_arg) = args().nth(1) {
		image_name = image_name_arg;
	}

	let src = imread(&find_file(&image_name, false, false)?, IMREAD_COLOR)?;
	if src.empty() {
		return Ok(());
	}

	// Pass the image to gray
	let mut src_gray = Mat::default();
	cvt_color_def(&src, &mut src_gray, COLOR_BGR2GRAY)?;
    let params: Vector<i32> = vec![].into();
    let result = imwrite(&image_name, &src_gray, &params);
    match result {
        Ok(success) => { println!("{}", success); }
        Err(err) => { println!("{}", err); }
    }
    

    Ok(())
}