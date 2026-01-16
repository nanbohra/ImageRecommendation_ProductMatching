from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

print("Loading CLIP model.")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print("CLIP loaded.")


# Create prompt templates for text embebeddings in CLIP
# FOllows attributes describes in textile_attributes.py
def create_prompt(category, value):
    templates = {
        'color': f"a photo of {value} colored fabric",
        'texture': f"a photo of fabric with {value} surface texture",
        'pattern': f"a photo of fabric with {value} patterns",
        'material': f"a close-up photo of {value} fabric texture"
    }
    return templates.get(category, f"a photo of {value} fabric")


# Classify which attribute label fits the image best for the given cateogry
def classify_attribute(image, category, options):
    # make a textual description 
    prompts = [create_prompt(category, option) for option in options]
    
    # process the prompt and the image together
    # getting embeddings for the text and the image
    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    # predict which attribute option fits best
    with torch.no_grad():
        outputs = model(**inputs) # run image and all text emb through CLIP to see semantic similarities
        logits = outputs.logits_per_image[0] # extract sim scores btw image and text descriptions
        probs = logits.softmax(dim=0) # convert to probs
    
    # return best match
    best_idx = torch.argmax(probs).item() 
    confidence = probs[best_idx].item()
    
    return options[best_idx], confidence


# Get all the attributes of an image 
def extract_all_attributes(image_path, attribute_definitions):
    image = Image.open(image_path).convert('RGB')
    
    results = {}
    scores = {}
    
    for attr_name, options in attribute_definitions.items():
        value, confidence = classify_attribute(image, attr_name, options)
        results[attr_name] = value
        scores[attr_name] = confidence
    
    return results, scores



# Test on one image
if __name__ == "__main__":
    from textile_attributes import TEXTILE_ATTRIBUTES
    
    test_image = "Product_Catalog/all_product_images/sample_images/sample_10.jpg" 
    print(f"Extracting features for sample #{test_image[-6:-4]}.")
    attributes, scores = extract_all_attributes(test_image, TEXTILE_ATTRIBUTES)
    
    print("="*60)
    print("EXTRACTED ATTRIBUTES")
    print("="*60)
    for attr_name, value in attributes.items():
        confidence = scores[attr_name]
        print(f"{attr_name.capitalize():<12}: {value:<20} ({confidence:.1%})")