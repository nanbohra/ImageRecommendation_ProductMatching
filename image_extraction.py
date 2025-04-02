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
import pytesseract
import re


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

top_text_crop = (600, height//2, width, (height//2)+80)
bottom_text_crop = (600, height - 100, width, height)

# Initialize a dictionary to store sample data
sample_data = {}

def extract_material_color(text):
    material, color = None, None
    
    # style_match = re.search(r"STYLE\s*:\s*([\w\s-]+)", text)
    design_match = re.search(r"DESIGN\s*:\s*([\w\s-]+)", text)

    # style = style_match.group(1).strip() if style_match else None

    if design_match:
        design_parts = design_match.group(1).strip().split("-")
        if len(design_parts) == 2:
            material, color = design_parts
    if design_match:
        design_parts = re.split(r"[-\s]", design_match.group(1).strip())
        if len(design_parts) >= 2:
            material, color = design_parts[0], "".join(design_parts[1:])
        else:
            material, color = design_parts[0], None
    
    # return [style, material, color]
    return [material, color]

# Crop and save half-page samples to 'half_sample_images' directory
for img in hs_images:
    # Extract top and bottom image from the page
    top_img = img.crop(top_crop)
    bottom_img = img.crop(bottom_crop)

    # Extract text from the top and bottom images
    text_region = img.crop(top_text_crop)
    top_text = pytesseract.image_to_string(text_region)

    text_region = img.crop(bottom_text_crop)
    bottom_text = pytesseract.image_to_string(text_region)

    top_info = extract_material_color(top_text)
    bottom_info = extract_material_color(bottom_text)

    top_img_path = os.path.join(sample_images_dir, f"sample_{i:02d}.jpg")
    sample_data[top_img_path] = top_info
    top_img.save(top_img_path, "JPEG")
    i+=1 

    bottom_img_path = os.path.join(sample_images_dir, f"sample_{i:02d}.jpg")
    sample_data[bottom_img_path] = bottom_info
    bottom_img.save(bottom_img_path, "JPEG")
    i+=1 


# Crop and save full-page samples to 'half_sample_images' directory
# Crop images to match half-page sample size
for img in (fs_images):
    crop_img = img.crop(top_crop)

    text_region = img.crop(bottom_text_crop)
    extracted_text = pytesseract.image_to_string(text_region)

    crop_img_path = os.path.join(sample_images_dir, f"sample_{i:02d}.jpg")
    info = extract_material_color(extracted_text)
    sample_data[crop_img_path] = info

    crop_img.save(crop_img_path, "JPEG")
    i+=1



print(sample_data)