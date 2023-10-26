from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os



def detect_Face(frame):
    mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # keep_all=False
    mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
    resnet = InceptionResnetV1(pretrained='vggface2').eval() 
    dataset = datasets.ImageFolder("C://Users//HP//Desktop//photos") # photos folder path 

    load_data = torch.load('C:/Users//HP//Desktop//CVR project//multimodal//live_captioning//data.pt') 
    embedding_list = load_data[0] 
    name_list = load_data[1]   
    name = "unrecognized"
    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
    
    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)
        detected = False
                
        for i, prob in enumerate(prob_list):
            if prob > 0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                
                dist_list = []  # list of matched distances, minimum distance is used to identify the person
                
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list)  # get minimum dist value
                min_dist_idx = dist_list.index(min_dist)  # get minimum dist index
                name = name_list[min_dist_idx]  # get name corresponding to minimum dist
                detected = True

        if not detected:
            name = "unrecognized"
            
    return name

if __name__ == "__main__":
   while True:
    vid = cv2.VideoCapture(0)
    ret, frame = vid.read()
    if not ret:
        print("fail to grab frame, try again")
        break
    cv2.imshow("IMG", frame)
    print(detect_Face(frame))
