from IPython.display import display, HTML
import re
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# image
def preprocess_image(img):
    image = img.copy()
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    image = image[np.newaxis, ...]
    return torch.tensor(image, requires_grad=True)
def to0_1(mask):
    # 将mask中的最小值减去，然后除以最大值减去最小值的差，将mask中的值映射到0-1之间
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    return mask
def read_imgs(file_name, device='cuda'):
    img = cv2.imread(file_name)
    img = img[:, :, ::-1]
    img = np.float32(cv2.resize(img, (224,224))) / 255
    input = preprocess_image(img)
    input = input.to(device)
    img_show = img
    del img
    return input, img_show
def show_cam_on_img_GBR(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    camImg = np.uint8(255 * cam)
    camImg = camImg[:, :, ::-1]
    return camImg
def show_cam_on_img_Seismic(img, mask):
    heatmap = cm.seismic(mask)
    heatmap = np.uint8(heatmap * 255)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGBA2BGR)
    cam = 1.0 * np.float32(heatmap / 255) + 1 * np.float32(img)
    cam = cam / np.max(cam)
    camImg = np.uint8(255 * cam)
    camImg = camImg[:, :, ::-1]
    return camImg
def visualizing_CAM(img, mask, work_type='GBR'):
    if work_type == 'GBR':
        return show_cam_on_img_GBR(img, mask)
    elif work_type == 'Seismic':
        return show_cam_on_img_Seismic(img, mask)
    else:
        print('Error')
def show_vit_explanation(img_show, emap, color='Seismic'):
    plt.figure(figsize=(30, 14))
    mask = emap.sum(0)
    mask = mask.data.cpu().numpy().astype("float")
    mask = cv2.resize(mask, (img_show.shape[1],img_show.shape[0]))
    mask = to0_1(mask)
    summask = visualizing_CAM(img_show, mask, color)
    return summask
# text
def preprocess_caption(caption, max_words=50):
    caption = re.sub(r"([.!\"()*#:;~])", ' ', caption, )
    caption = re.sub(r"\s{2,}", ' ', caption,)
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption
def preprocess_tokens(sentence, device='cuda'):
    global text_processed, text_tokens_decoded
    import clip
    tokenize = clip.tokenize
    from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
    _tokenizer = _Tokenizer()
    if isinstance(sentence, str):
        text_processed = tokenize([sentence]).to(device)
        text_tokens = _tokenizer.encode(sentence)
        text_tokens_decoded = [_tokenizer.decode([a]) for a in text_tokens]
    elif isinstance(sentence, list):
        text_processed = tokenize(sentence).to(device)
        texts = sentence
        text_tokens_decoded = []
        for text in texts:
            text_tokens = _tokenizer.encode(text)
            text_tokens_decoded.append([_tokenizer.decode([a]) for a in text_tokens])
    return text_processed, text_tokens_decoded
def show_text_attr(expln, str_list, style=None, enhance_rate=2):
    def rgb_no_relu(x):
        if x >= 0: rgb = '0,255,0'
        else: rgb = '255,0,0'
        return rgb
    def alpha_no_relu(x):
        x = abs(x) * enhance_rate
        return x
    def judge_alpha(x):
        if x == None: x = 0
        elif x >= 1: x = 1
        return x
    attrs = list(expln)
    tokens = str_list # tokens
    token_marks = [
            f'<mark style="background-color:rgba({rgb_no_relu(attr)},{judge_alpha(alpha_no_relu(attr))})">{token}</mark>'
            for token, attr in zip(tokens, attrs)
    ]
    if style == None:
        style = """
        font-size: 26px; 
        font-family: 'Calibri', sans-serif;  
        line-height: 1.5; 
        font-weight: 580;  
        text-align: center;  
        """
    mark_style = """mark {margin: 0.3px;}"""
    html_data = HTML(f'<style>"{mark_style}"</style><p style="{style}">{" ".join(token_marks)}</p>')
    display(html_data)
    return html_data