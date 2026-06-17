from PIL import Image

def verify_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"Verified Size: {width}x{height}")
            return width, height
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Usage
verify_image_size("sample_000014.png")   