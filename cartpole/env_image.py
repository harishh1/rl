import cv2 

def getImage(env):
    img = env.render(mode='rgb_array')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb_resized = cv2.resize(img_rgb, (240, 160), interpolation=cv2.INTER_CUBIC)
    img_rgb_resized[img_rgb_resized < 255] = 0
    img_rgb_resized = img_rgb_resized / 255
    return img_rgb_resized