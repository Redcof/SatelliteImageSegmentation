import cv2
import numpy as np
import matplotlib.pyplot as plt

image_file = r'C:\Users\dndlssardar\OneDrive - Smiths Group\Documents\Projects\Dataset\Sixray_easy\train\JPEGImages\P00513.jpg'


# !pip install git+https://github.com/facebookresearch/segment-anything.git
# !pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

def show_boundary(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.ones(3), [0.35]])
        
        img[m] = color_mask
    ax.imshow(img)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    print(img.shape)
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [.3]])
        # print(color_mask)
        img[m] = color_mask
    ax.imshow(img)


image = cv2.imread(image_file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# download SAM models form https://github.com/facebookresearch/segment-anything#model-checkpoints
sam = sam_model_registry["default"](checkpoint="./models/sam_vit_h_4b8939.pth")
device = "cuda:0"
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

print("Generating...")
masks = mask_generator.generate(image)
print("Done")

sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
img[:, :, :] = 0
for ann in sorted_anns:
    m = ann['segmentation']
    color_mask = np.concatenate([np.random.random(3)])
    # print(color_mask)
    img[m] = color_mask

plt.figure(figsize=(20, 20))
plt.subplot(131)
plt.imshow(image)

img2 = (np.mean(img, axis=2) * 255).astype('uint8')
plt.subplot(132)
plt.imshow(img2)
edges = cv2.Canny(img2, 0, 20)
plt.subplot(133)
plt.imshow(edges, cmap='gray')
cv2.imshow("edges", edges)
cv2.waitKey(-1)
cv2.destroyAllWindows()
