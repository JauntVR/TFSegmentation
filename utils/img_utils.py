from PIL import Image
import numpy as np
import matplotlib.cm as cm

# colour map
label_colours_global = [(255,255,255),  # 'nal'
                        (255, 106, 0),  # 'body'
                        (255, 0, 0),  # 'neck'
                        #right arm
                        (255, 178, 127),
                        (255, 127, 127),
                        (182, 255, 0),
                        (218, 255, 127),
                        (255, 216, 0),
                        (107, 63, 127),
                        #left arm
                        (255, 233, 127),
                        (0, 148, 255),
                        (255, 0, 110),
                        (48, 48, 48),
                        (76, 255, 0),
                        (63, 73, 127),
                        #head, hips
                        (0, 255, 33),
                        (0, 255, 255),
                        #right leg
                        (0, 255, 144),
                        (127, 116, 63),
                        (127, 201, 255),
                        (165, 255, 127),
                        (214, 127, 255),
                        #left leg
                        (178, 0, 255),
                        (127, 63, 63),
                        (127, 255, 255),
                        (127, 255, 197),
                        (161, 127, 255),
                        #not a user
                        (72, 0, 255),]

label_colours_reduced = [(255,255,255),  # 'nal'
                        (255, 106, 0),  # 'body'
                        (255, 0, 0),  # 'neck'
                        #right arm upper
                        (255, 178, 127),
                        #right arm lower
                        (218, 255, 127),
                        #right hand
                        (107, 63, 127),
                        #left arm upper
                        (255, 233, 127),
                        #left arm lower
                        (48, 48, 48),
                        #left hand
                        (63, 73, 127),
                        #head
                        (0, 255, 33),
                        #hips
                        (0, 255, 255),
                        #right leg upper
                        (0, 255, 144),
                        #right leg lower
                        (127, 201, 255),
                        (165, 255, 127),
                        (214, 127, 255),
                        #left leg upper
                        (178, 0, 255),
                        #left leg lower
                        (127, 255, 255),

                        #not a user
                        (72, 0, 255),]

def decode_labels(mask, num_classes):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking **argmax**.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
#    num_classes= num_classes+1
    # init colours array
    if num_classes == 28:
        colours = label_colours_global
    elif num_classes == 16:
        colours = label_colours_reduced
    else:
        colours = label_colours_global[:num_classes]
    # Check the length of the colours with num_classes

    # Get the shape of the mask
    n, h, w = mask.shape
    n = min(n,3)
    # Create the output numpy array
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    # Loop on images
    for i in range(n):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = colours[k]
        outputs[i] = np.array(img)
    return outputs

def decode_input(imm):
    n, h, w, _ = imm.shape
    n = min(n,3)
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        for c in range(3):
            outputs[i,:,:,c] = np.array(imm[i, :, :, 0])
    return outputs

def decode_conf(imm):
    n, h, w = imm.shape
    n = min(n,3)
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    cmap = cm.get_cmap('jet')
    for i in range(n):
        colored = np.round(256 * cmap(np.array(imm[i, :, :])))
        outputs[i,:,:,:] =  colored[:,:,:3]
    return outputs
