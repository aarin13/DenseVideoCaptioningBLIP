import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",torch_dtype=torch.float16).to("cuda")


def generate_caption(pil_image):
    global previous_caption
    try:
        inputs = processor(pil_image, return_tensors="pt").to("cuda", torch.float16)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        previous_caption = caption
        return caption
    
    except:
        return "Unable to process image."
    


    #model takes the input and converts into encoding
    #processor takes the encoding and converts into outputs
