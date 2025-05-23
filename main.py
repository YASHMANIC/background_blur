import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
import cv2
from PIL import Image
import argparse

class BackgroundBlurrer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = deeplabv3_resnet50(pretrained=True, progress=True)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Define image transformations
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def segment_image(self, img):
        """Perform semantic segmentation on the input image"""
        # Convert to RGB if needed
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Apply transformations - this converts to tensor and normalizes
        input_tensor = self.transform(img)
        
        # Ensure input is a tensor and add batch dimension (unsqueeze) and move to device
        input_tensor = torch.as_tensor(input_tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # Get the segmentation mask (person class is usually 15 in COCO)
        segmentation_mask = output.argmax(0)
        person_mask = (segmentation_mask == 15).cpu().numpy()
        
        return person_mask
    
    def blur_background(self, img_path, output_path, blur_strength=30):
        """Blur the background of an image while keeping the foreground sharp"""
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image from {img_path}")
        
        # Ensure blur strength is positive and odd
        blur_strength = max(1, blur_strength)  # At least 1
        blur_strength = blur_strength + 1 if blur_strength % 2 == 0 else blur_strength  # Make odd
        
        # Get segmentation mask
        mask = self.segment_image(img)
        
        # Apply blur to entire image
        blurred_img = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)
        
        # Combine original and blurred images based on mask
        mask_3d = np.dstack([mask]*3)
        result = np.where(mask_3d, img, blurred_img)
        
        # Save the result
        cv2.imwrite(output_path, result)
        print(f"Processed image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blur image background while keeping foreground sharp")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_image", help="Path to save the output image")
    parser.add_argument("--blur_strength", type=int, default=30, 
                    help="Strength of the blur effect (higher values = more blur)")
    
    args = parser.parse_args()
    
    blurrer = BackgroundBlurrer()
    blurrer.blur_background(args.input_image, args.output_image, args.blur_strength)