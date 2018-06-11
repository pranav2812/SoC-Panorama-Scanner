# Work done till now:

## 1. OpenCv tutorials

## 2. Feature detection using ORB (Oriented FAST and Rotated BRIEF)

### Results:

https://github.com/pranav2812/SoC-Panorama-Scanner/blob/Test_images/orbs.jpg

## 3.Mapping:

Used BFMatcher for mapping.
Tried Descriptor and FLANN.

#### very good for similar lighting conditions

https://github.com/pranav2812/SoC-Panorama-Scanner/blob/Test_images/matched1.jpg

#### poor for changed lighting conditions and scenery

https://github.com/pranav2812/SoC-Panorama-Scanner/blob/Test_images/matched%20(1).jpg

Poor results maybe because of direct application of orb creation and mapping on grayscale image.Applying transformations may help.Got to test it thoroughly.

## Observations till now

### Masking only makes it worse

Orb detection on grayscale is much accurate on grayscale images.(without masking)

https://github.com/pranav2812/SoC-Panorama-Scanner/blob/Test_images/matches_without_mask.jpg

With masking:

https://github.com/pranav2812/SoC-Panorama-Scanner/blob/Test_images/matched_with_mask.jpg

### Goes haywire on text images with mask

https://github.com/pranav2812/SoC-Panorama-Scanner/blob/Test_images/On_text.jpg

#### Without mask:

https://github.com/pranav2812/SoC-Panorama-Scanner/blob/Test_images/On_text%20without_mask.jpg

### BFMatcher is better than DescriptorMatcher and FLANN.

## Done with stitching two images with feature mapping and homography
### HSV matching is remaining 

https://github.com/pranav2812/SoC-Panorama-Scanner/blob/Test_images/final.jpg





