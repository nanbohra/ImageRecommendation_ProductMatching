"""
Extracts images from given PDF catalog and saves to local directory.
@File    : image_extraction.py
@Date    : 2025-03-04
@Author  : Nandini Bohra
@Contact : nbohra@ucsd.edu

"""

import os
from pdf2image import convert_from_path
from PIL import Image

root = "/Users/nandinibohra/Desktop/VSCodeFiles/ImageRecommendation_ProductMatching/Product_Catalog"
hs_pdf_path = root + "/IRPM_Catalog_HalfSamples.pdf"
fs_pdf_path = root + "/IRPM_Catalog_FullSamples.pdf"
prod_pdf_path = root + "/IRPM_Catalog_Products.pdf"

parent_dir = root + "/all_product_images"
subdirectories = [parent_dir+path for path in ["/product_images", "/sample_images"]]

os.makedirs(parent_dir, exist_ok=True)
for subdir in subdirectories:
    os.makedirs(subdir, exist_ok=True)

hs_images = convert_from_path(hs_pdf_path)
fs_images = convert_from_path(fs_pdf_path)
prod_images = convert_from_path(prod_pdf_path)

sample_images_dir = subdirectories[1] 

width, height = hs_images[0].size

# left, top, right, bottom
# 920 length x 1700 width
top_crop = (50, 230, width-50, (height//2)-50)  # (50,230,1750,1150)
bottom_crop = (50, (height//2)+130, width-50, height-150) # (50,1330,1750,2350)
i = 0 # Image index


# Crop and save half-page samples to 'half_sample_images' directory
for img in hs_images:
    top_img = img.crop(top_crop)
    bottom_img = img.crop(bottom_crop)
 
    top_img_path = os.path.join(sample_images_dir, f"sample_{i}.jpg")
    i+=1 
    bottom_img_path = os.path.join(sample_images_dir, f"sample_{i}.jpg")
    i+=1 

    top_img.save(top_img_path, "JPEG")
    bottom_img.save(bottom_img_path, "JPEG")


# Crop and save full-page samples to 'half_sample_images' directory
# Crop images to match half-page sample size
for img in (fs_images):
    crop_img = img.crop(top_crop)
    crop_img_path = os.path.join(sample_images_dir, f"sample_{i}.jpg")
    i+=1
    crop_img.save(crop_img_path, "JPEG")