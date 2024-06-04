import os
import time
import cv2
import threading
import pykinect_azure as pykinect

def create_directory(person_name):
    directory = os.path.join(os.getcwd(), person_name)
    os.makedirs(directory, exist_ok=True)
    return directory

def initialize_kinect():
    pykinect.initialize_libraries()
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    device = pykinect.start_device(config=device_config)
    return device

def capture_images(device, directory, duration, label, fps=5):
    frame_count = 0
    start_time = time.time()
    while time.time() - start_time < duration:
        capture = device.update()
        ir_image = capture.get_ir_image()[1]  # Get the infrared image
        rgb_image = capture.get_color_image()[1]  # Get the RGB image
        
        file_name_rgb = os.path.join(directory, f'{label}_rgb_{frame_count:03d}.jpg')
        file_name_ir = os.path.join(directory, f'{label}_ir_{frame_count:03d}.jpg')
        
        # Simulating IR image as a grayscale image for demo purposes
        ir_image_gray = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
        
        cv2.imwrite(file_name_rgb, rgb_image)
        cv2.imwrite(file_name_ir, ir_image_gray)
        
        frame_count += 1
        time.sleep(1 / fps)

def print_animation(message, duration):
    print(message, end='')
    for _ in range(int(duration * 10)):  # Print 10 characters per second
        print("=", end='', flush=True)
        time.sleep(0.1)
    print()

def capture_and_animate(device, directory, duration, label, message, fps=5):
    capture_thread = threading.Thread(target=capture_images, args=(device, directory, duration, label, fps))
    animation_thread = threading.Thread(target=print_animation, args=(message, duration))

    capture_thread.start()
    animation_thread.start()

    capture_thread.join()
    animation_thread.join()

def main():
    person_name = input("Enter your name: ").strip()
    directory = create_directory(person_name)
    
    device = initialize_kinect()

    if not device.is_opened():
        print("Error: Could not open PyKinect Azure device.")
        return

    # Look straight
    input("Please look straight and press Enter to start capturing images for 10 seconds...")
    print("Capturing images while looking straight...")
    capture_and_animate(device, directory, 10, "straight", "Please look straight")
    
    # Turn right
    input("Please slowly turn your head to the right and press Enter to start capturing images for 5 seconds...")
    print("Capturing images while turning right...")
    capture_and_animate(device, directory, 5, "right", "Please turn right")
    
    # Turn left
    input("Please slowly turn your head to the left and press Enter to start capturing images for 5 seconds...")
    print("Capturing images while turning left...")
    capture_and_animate(device, directory, 5, "left", "Please turn left")
    
    print("Image capture completed.")
    device.stop()

if __name__ == "__main__":
    main()
