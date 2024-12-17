import serial
import numpy as np
import cv2  # OpenCV library
import os

# Global variable for the image storage path
IMAGE_SAVE_PATH = "images"

# Ensure the directory exists
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

def parse_image_rgb565(img, x, y):
    pixels = np.zeros((x * y, 3), dtype=np.uint8)
    for i in range(0, x * y * 2, 2):
        # Decode RGB565
        b = (img[i + 1] & 0x1f) << 3
        g = (((img[i] & 0x07) << 3) | ((img[i + 1] & 0xe0) >> 5)) << 2
        r = (img[i] & 0xf8)
        pixels[i // 2] = [b, g, r]
    # Reshape to 2D array with 3 color channels
    pixels = np.reshape(pixels, (y, x, 3), order='C')
    return pixels

# Configure the serial port
ser = serial.Serial(
    '/dev/serial/by-id/usb-ARM_DAPLink_CMSIS-DAP_04441701c0e38ade00000000000000000000000097969906-if01',
    115200,
    timeout=30
)
print(f"Successfully opened: {ser.name}")

# Flush the buffer
ser.read_all()

img_token = b"New image\n"

try:
    while True:
        buffer = bytearray()
        # Wait for the start message
        while img_token not in buffer:
            buffer = ser.readline()

        # Read image dimensions
        img_resol = ser.read(5)
        x_size = int.from_bytes(img_resol[0:2], byteorder='big')
        y_size = int.from_bytes(img_resol[2:4], byteorder='big')
        pixel_format_len = int.from_bytes(img_resol[4:5], byteorder='big')
        print(f"X dimension: {x_size}, Y dimension: {y_size}")

        # Read pixel format
        img_format = ser.read(pixel_format_len).decode("utf-8")
        print("Image format: ", img_format)

        # Read image length
        img_dim = ser.read(4)
        img_dim = int.from_bytes(img_dim, byteorder='big')
        print("Total image dimension: ", img_dim)

        # Read image data
        img_data = ser.read(img_dim)
        img = parse_image_rgb565(img_data, x_size, y_size)

        # Save the image to disk with a unique name
        filename = os.path.join(IMAGE_SAVE_PATH, f"captured_image_{int.from_bytes(os.urandom(4), 'big')}.png")
        cv2.imwrite(filename, img)
        print(f"Image saved as {filename}")

except KeyboardInterrupt:
    print("Program interrupted by user.")
finally:
    ser.close()  # Ensure the serial port is closed
