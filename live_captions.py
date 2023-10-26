import cv2
import time
import threading
import numpy as np
from caption_generation import generate_caption
import PIL
from face_recon import detect_Face

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

last_process_time = time.time()
last_generation_time = time.time()
previous_caption = ""  
previous_captions = []
name = ""
lock = threading.Lock() #multithreaded

def f2pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(img)
    return img

def process_live(frame):
    global last_generation_time, previous_captions, name
    pil_im = f2pil(frame)
    caption = generate_caption(pil_im)
    name = detect_Face(frame)

    cur_time = time.time()
    if cur_time - last_generation_time > 1:
        if caption and caption not in previous_captions:
            previous_captions.append(caption) 
            if len(previous_captions) > 50:  
                previous_captions.pop(0)
            last_generation_time = cur_time


def text_box(frame, caption, name):
    _, frame_width = frame.shape[:2]
    text = f"I can see {caption}"
    text1 = f""
    #This person is {name}, i can see that you are : sad
    text_box_height = 50
    text_box = np.zeros((text_box_height, frame_width, 3), dtype=np.uint8)
    text_box1 = np.zeros((text_box_height, frame_width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    font_color = (255, 255, 255)
    text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
    text_x = (frame_width - text_size[0]) // 2
    text_y = (text_box_height + text_size[1]) // 2
    cv2.putText(text_box, text, (text_x, text_y), font, font_scale, font_color, 2, cv2.LINE_AA)
    cv2.putText(text_box1, text1, (text_x, text_y), font, font_scale, font_color, 2, cv2.LINE_AA)
    combined_frame = np.vstack((frame, text_box))
    combined_frame = np.vstack((combined_frame, text_box1))

    return combined_frame

def display_rt(frame):
    global previous_captions
    global name
    with lock:
        for caption in previous_captions[-1:][::1]:
            frame = text_box(frame, caption, name)
        cv2.namedWindow("frame", cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("frame", frame)
        
def main_loop():
    global last_process_time
    while True:
        ret, frame = vid.read()
        if not ret:
            print("Error capturing frame, exiting.")
            break
        
        current_time = time.time()
        if current_time - last_process_time >= 1:
            t = threading.Thread(target=process_live, args=(frame,))
            t.start()
            last_process_time = current_time
            

        display_rt(frame)
    
        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_loop()
    print(previous_captions)