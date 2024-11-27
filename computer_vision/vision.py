import cv2
import ollama
import os
def capture_image(image_path='./image1.jpg'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return None
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(image_path, frame)
        print(f"Image saved at {image_path}")
    else:
        print("Error: Could not capture an image.")
    
    cap.release()
    cv2.destroyAllWindows()
    return image_path

def ollama_chat_with_image(image_path):
    res = ollama.chat(
        model='llava:13b',
        messages=[
            {'role': 'user',
            'content': 'You are J.A.R.V.I.S, an AI created by Swarnavo Mukherjee whom you refer to as boss. Do not break character and describe what you see based on how Jarvis would reply, mentioning by starting with "Boss, Here I see..."',
            'images': [image_path]}
        ]
    )
    return res['message']['content']

def delete_image(image_path):
    if os.path.exists(image_path):
        os.remove(image_path)
        
    else:
        print(f"Error: {image_path} does not exist.")

def main():
    image_path = './image1.jpg'

    captured_image = capture_image(image_path)
    

    if captured_image:
        response = ollama_chat_with_image(captured_image)
        print(response)
        
        delete_image(captured_image)

main()