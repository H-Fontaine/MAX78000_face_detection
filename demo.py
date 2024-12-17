import serial
import numpy as np
import cv2  # OpenCV library
import re  # Regular expression library to extract class and percentage

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
result_token = b"Classification results:\n"

try:
    # Initialize the OpenCV window once
    cv2.namedWindow("Image Viewer", cv2.WINDOW_NORMAL)

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

        # Resize the image (scaling factor can be adjusted)
        scale_factor = 10  # Adjust scale factor to change size
        resized_img = cv2.resize(img, (x_size * scale_factor, y_size * scale_factor))

        # Resize the window to the size of the resized image
        cv2.resizeWindow("Image Viewer", resized_img.shape[1], resized_img.shape[0])

        # Now, parse the serial data for the class and percentage
        detected_class = None
        percentages = [0, 0]
        line = ser.readline()
        for i in range(2):
            line = ser.readline().decode("utf-8")
            print(line.strip())
            regex_percentage = re.search(r"\[\s*(-?\d+)\] -> Class (\d+): (\d+)\.(\d+)%", line)
            if regex_percentage:
                percentages[int(regex_percentage.group(2))] = float(f"{regex_percentage.group(3)}.{regex_percentage.group(4)}")
        line = ser.readline().decode("utf-8")
        regex_detected = re.search(r"Detected : (\d+)", line)
        if regex_detected:
            detected_class = int(regex_detected.group(1))

        # Update the title
        title = f"Class {detected_class} at {percentages[detected_class ]}%"
        cv2.setWindowTitle("Image Viewer", title)

        # Display the image in the existing window
        cv2.imshow("Image Viewer", resized_img)

        # Check for keypress to close (esc key or window close)
        key = cv2.waitKey(1)
        if key == 27 or cv2.getWindowProperty("Image Viewer", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed. Exiting.")
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")
finally:
    ser.close()  # Ensure the serial port is closed
    cv2.destroyAllWindows()  # Close OpenCV windows
