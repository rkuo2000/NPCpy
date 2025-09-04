


def process_video(file_path, table_name):
    
    import cv2
    import base64

    embeddings = []
    texts = []
    try:
        video = cv2.VideoCapture(file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(frame_count):
            ret, frame = video.read()
            if not ret:
                break

            
            n = 10  
        
        return embeddings, texts

    except Exception as e:
        print(f"Error processing video: {e}")
        return [], []  
