from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import utils
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from models.iresnet import get_model
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import os
import glob

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transforms.ToTensor()(image)
    image = TF.resize(image, (256, 256), interpolation=TF.InterpolationMode.BICUBIC).unsqueeze(0)
    return image

def extract_face(imgs, batch_boxes, mtcnn):
    image_size = imgs.shape[-1]
    faces = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        if batch_boxes[i] is not None:
            box = batch_boxes[i][0]
            margin = [
                mtcnn.margin * (box[2] - box[0]) / (112 - mtcnn.margin),
                mtcnn.margin * (box[3] - box[1]) / (112 - mtcnn.margin),
            ]

            box = [
                int(max(box[0] - margin[0] / 2, 0)),
                int(max(box[1] - margin[1] / 2, 0)),
                int(min(box[2] + margin[0] / 2, image_size)),
                int(min(box[3] + margin[1] / 2, image_size)),
            ]
            crop_face = img[None, :, box[1]:box[3], box[0]:box[2]]
        else:
            # crop_face = img[None, :, :, :]
            return None

        faces.append(F.interpolate(crop_face, size=112, mode='bicubic'))
    new_faces = torch.cat(faces)

    return (new_faces - 127.5) / 128.0

def get_faces(x, mtcnn):
    img = (x + 1.0) * 0.5 * 255.0
    img = img.permute(0, 2, 3, 1)
    with torch.no_grad():
        batch_boxes, batch_probs, batch_points = mtcnn.detect(img, landmarks=True)
        # Select faces
        batch_boxes, batch_probs, batch_points = mtcnn.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=mtcnn.selection_method
        )

    img = img.permute(0, 3, 1, 2)
    faces = extract_face(img, batch_boxes, mtcnn)
    return faces



def get_embedding(image, mtcnn, resnet, device, id):
    #x = TF.resize(image, (256, 256), interpolation=TF.InterpolationMode.BICUBIC)

    # Detect face and get face crop
    face = get_faces(image, mtcnn=mtcnn)
    
    if face is None:
        print(f"No face detected in the image")
        return None
    
    # Move face tensor to the same device as the model
    face = face.to(device)
    
    # Get embedding
    embedding = resnet(face)
    return embedding.detach()

def calculate_l1_distance(emb1, emb2):
    # Calculate L1 distance
    dist = F.l1_loss(emb1, emb2, reduction='none').mean(dim=1)
    return dist.item()

def calculate_cosine_similarity(emb1, emb2):
    return F.cosine_similarity(emb1, emb2, dim=1).item()

def calculate_arcface_loss(emb1, emb2, s=64.0, m=0.5):
    # Normalize embeddings
    emb1_norm = F.normalize(emb1, p=2, dim=1)
    emb2_norm = F.normalize(emb2, p=2, dim=1)
    
    # Calculate cosine similarity
    cos_similarity = F.cosine_similarity(emb1_norm, emb2_norm, dim=1)
    
    # Calculate theta
    theta = torch.acos(torch.clamp(cos_similarity, -1.0 + 1e-7, 1.0 - 1e-7))
    
    # Calculate ArcFace logits
    arcface_logits = s * torch.cos(theta + m)
    
    # Calculate ArcFace loss (assuming same identity, target=1)
    arcface_loss = -torch.log(torch.exp(arcface_logits) / (torch.exp(arcface_logits) + 1))
    
    return arcface_loss.item()

def compare_embeddings(embedding1, embedding2, model_name):
    l1_distance = calculate_l1_distance(embedding1, embedding2)
    cosine_similarity = calculate_cosine_similarity(embedding1, embedding2)
    arcface_loss = calculate_arcface_loss(embedding1, embedding2)

    print(f"{model_name} Results:")
    #print(f"L1 distance: {l1_distance:.4f}")
    print(f"Cosine similarity: {cosine_similarity:.4f}")
    #print(f"ArcFace loss: {arcface_loss:.4f}")
    print()

def plot_and_save_images(ground_truth_path, image1_path, image2_path, image3_path, vggface_sim1, vggface_sim2, vggface_sim3, casia_sim1, casia_sim2, casia_sim3, iresnet_sim1, iresnet_sim2, iresnet_sim3, i, path):
    # Increase figure height to accommodate text
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 10))
    
    ground_truth = plt.imread(ground_truth_path)
    img1 = plt.imread(image1_path)
    img2 = plt.imread(image2_path)
    img3 = plt.imread(image3_path)
    
    ax1.imshow(ground_truth)
    ax1.set_title("Ground Truth")
    ax1.axis('off')
    
    ax2.imshow(img1)
    ax2.set_title(f"VGGFace")
    ax2.axis('off')
    
    ax3.imshow(img2)
    ax3.set_title(f"CASIA")
    ax3.axis('off')
    
    ax4.imshow(img3)
    ax4.set_title(f"Arcface_ms2mv3")
    ax4.axis('off')
    
    plt.suptitle(f"Image {i}", y=0.95, fontsize=14)
    
    similarity_text = (f"VGGFace similarities: {vggface_sim1:.4f}, {vggface_sim2:.4f}, {vggface_sim3:.4f}\n"
                      f"CASIA similarities: {casia_sim1:.4f}, {casia_sim2:.4f}, {casia_sim3:.4f}\n"
                      f"Arcface_ms2mv3 similarities: {iresnet_sim1:.4f}, {iresnet_sim2:.4f}, {iresnet_sim3:.4f}")
    
    plt.figtext(0.5, 0.05, similarity_text, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.25)
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/comparison_{i}.png")
    plt.close()

def create_gif(image_folder, output_path, duration=200):
    images = []
    for file_name in sorted(glob.glob(f"{image_folder}/*.png")):
        img = Image.open(file_name)
        images.append(img)
    
    if images:
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
        print(f"GIF saved at {output_path}")
    else:
        print(f"No images found in {image_folder}")

def main(image1_path, image2_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    mtcnn = MTCNN(image_size=160, margin=0, device=device)
    
    resnet_vggface = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet_casia = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
    resnet_iresnet = get_model('r50', fp16=False).to(device)
    resnet_iresnet.load_state_dict(torch.load('models/weights/ms2mv3_r50.pth'))
    resnet_iresnet.eval()

    
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)
    
    embedding1_vggface = get_embedding(image1, mtcnn, resnet_vggface, device, "1")
    embedding2_vggface = get_embedding(image2, mtcnn, resnet_vggface, device, "2")
    
    embedding1_casia = get_embedding(image1, mtcnn, resnet_casia, device, "1")
    embedding2_casia = get_embedding(image2, mtcnn, resnet_casia, device, "2")
    
    embedding1_iresnet = get_embedding(image1, mtcnn, resnet_iresnet, device, "1")
    embedding2_iresnet = get_embedding(image2, mtcnn, resnet_iresnet, device, "2")
    
    if embedding2_vggface is None or embedding2_casia is None or embedding2_iresnet is None:
        return None, None, None
    
    vggface_sim = calculate_cosine_similarity(embedding1_vggface, embedding2_vggface)
    casia_sim = calculate_cosine_similarity(embedding1_casia, embedding2_casia)
    iresnet_sim = calculate_cosine_similarity(embedding1_iresnet, embedding2_iresnet)
    return vggface_sim, casia_sim, iresnet_sim

if __name__ == "__main__":

    output_folder = f"/scratch1/fxiao/noise/figures/1"
    os.makedirs(f"{output_folder}/vggface", exist_ok=True)
    os.makedirs(f"{output_folder}/casia", exist_ok=True)
    os.makedirs(f"{output_folder}/arcface_ms2mv3", exist_ok=True)

    for i in range(10):
        os.makedirs(output_folder, exist_ok=True)

        # root1: VGGFace
        # root2: CASIA
        # root3: Arcface ms2mv3
        root1 = "/scratch1/fxiao/noise/2_3_test_1"
        #root2 = "/scratch1/fxiao/noise/1_2"
        #root3 = "/scratch1/fxiao/noise/1_3"
        ground_truth = f"data/face_data/img_1.jpg"
        image1 = f"{root1}/new_img_{i}.png"
        #image2 = f"{root2}/new_img_{i}.png"
        #image3 = f"{root3}/new_img_{i}.png"
        print(f"IMAGE NUMBER {i}")
        vggface_sim1, casia_sim1, iresnet_sim_1 = main(ground_truth, image1)
        if vggface_sim1 is not None and casia_sim1 is not None and iresnet_sim_1 is not None:
            print(f"VGGFace similarity: {vggface_sim1:.4f}")
            print(f"CASIA similarity: {casia_sim1:.4f}")    
            print(f"iResNet similarity: {iresnet_sim_1:.4f}")
        '''
        if vggface_sim1 is not None and vggface_sim2 is not None and vggface_sim3 is not None: 
            plot_and_save_images(ground_truth, image1, image2, image3, vggface_sim1, vggface_sim2, vggface_sim3, casia_sim1, casia_sim2, casia_sim3, iresnet_sim_1, iresnet_sim_2, iresnet_sim_3, i, output_folder)
            print(f"VGG Max Run")
            print(f"VGGFace similarity: {vggface_sim1:.4f}")
            print(f"CASIA similarity: {casia_sim1:.4f}")    
            print(f"iResNet similarity: {iresnet_sim_1:.4f}")
            print(f"Casia Face 2")
            print(f"VGGFace similarity: {vggface_sim2:.4f}")
            print(f"CASIA similarity: {casia_sim2:.4f}")    
            print(f"iResNet similarity: {iresnet_sim_2:.4f}")
            print(f"iResNet Face 3")
            print(f"VGGFace similarity: {vggface_sim3:.4f}")
            print(f"CASIA similarity: {casia_sim3:.4f}")    
            print(f"iResNet similarity: {iresnet_sim_3:.4f}")
    

        # Create GIFs for VGGFace
        vggface_intermediate_folder = f"{root1}/intermediates/image{i}"
        vggface_gif_path = f"{output_folder}/vggface/image_{i}.gif"
        create_gif(vggface_intermediate_folder, vggface_gif_path)

        # Create GIFs for CASIA
        casia_intermediate_folder = f"{root2}/intermediates/image{i}"
        casia_gif_path = f"{output_folder}/casia/image_{i}.gif"
        create_gif(casia_intermediate_folder, casia_gif_path)

        # Create GIFs for iResNet
        iresnet_intermediate_folder = f"{root3}/intermediates/image{i}"
        iresnet_gif_path = f"{output_folder}/arcface_ms2mv3/image_{i}.gif"
        create_gif(iresnet_intermediate_folder, iresnet_gif_path)
        '''