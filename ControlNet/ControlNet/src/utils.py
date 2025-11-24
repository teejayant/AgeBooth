def load_image(image_path):
    from PIL import Image
    return Image.open(image_path)

def resize_image(image, size=(1024, 1024)):
    return image.resize(size)

def create_canny_edge_map(image_np, low_threshold=100, high_threshold=200):
    import cv2
    canny_image_np = cv2.Canny(image_np, low_threshold, high_threshold)
    canny_image_np = canny_image_np[:, :, None]
    return np.concatenate([canny_image_np, canny_image_np, canny_image_np], axis=2)