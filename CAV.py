import torch
import torch.nn.functional as F
import cav_utils
import math

# image version
class Forward_Function:
    def __init__(self, model):
        self.model = model
    def run(self, img_feature=None, text_feature=None):
        img_embedding_norm = F.normalize(img_feature, dim=-1)
        text_embedding_norm = F.normalize(text_feature, dim=-1)
        confidence = (100. * img_embedding_norm @ text_embedding_norm.T)[0]
        return confidence

def attention_layer(q, k, v, num_heads=1, patch_size=16):
    tgt_len, bsz, embed_dim = q.shape
    head_dim = embed_dim // num_heads
    scale_factor = float(head_dim) ** -0.5
    q = q.transpose(0, 1).contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.transpose(0, 1).contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.transpose(0, 1).contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    attn_output_weights_preSoftmax = torch.bmm(q, k.transpose(1, 2))
    base_len = (224/patch_size) ** 2 + 1
    Temperature_Rate = tgt_len / base_len
    scale_factor = scale_factor * (Temperature_Rate ** 0.5)
    attn_output_weights_preSoftmax = attn_output_weights_preSoftmax * scale_factor
    attn_output_weights = F.softmax(attn_output_weights_preSoftmax, dim=-1)
    MHA = torch.bmm(attn_output_weights, v)
    assert list(MHA.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = MHA.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    return attn_output, MHA, attn_output_weights, (q, k, v)
def visual_forward(clipmodel, x, n, embedding_width=64):
    global attn_score, MHA, attn_output, q, k, v
    clip_inres = clipmodel.visual.input_resolution
    clip_ksize = clipmodel.visual.conv1.kernel_size
    patch_size = clip_ksize[0]
    vision_width = clipmodel.visual.transformer.width
    x = x.half()
    x = clipmodel.visual.conv1(x)
    feah, feaw = x.shape[-2:]
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    class_embedding = clipmodel.visual.class_embedding.to(x.dtype)
    x = torch.cat([class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1]).to(x), x], dim=1)
    pos_embedding = clipmodel.visual.positional_embedding.to(x.dtype)
    tok_pos, img_pos = pos_embedding[:1, :], pos_embedding[1:, :]
    pos_h = clip_inres // clip_ksize[0]
    pos_w = clip_inres // clip_ksize[1]
    vision_heads = vision_width // embedding_width
    assert img_pos.size(0) == (pos_h * pos_w), \
        f"the size of pos_embedding ({img_pos.size(0)}) does not match resolution shape pos_h ({pos_h}) * pos_w ({pos_w})"
    img_pos = img_pos.reshape(1, pos_h, pos_w, img_pos.shape[1]).permute(0, 3, 1, 2)
    img_pos = torch.nn.functional.interpolate(img_pos, size=(feah, feaw), mode='bicubic', align_corners=False)
    img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)
    pos_embedding = torch.cat((tok_pos[None, ...], img_pos), dim=1)
    x = x + pos_embedding
    x = clipmodel.visual.ln_pre(x)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = torch.nn.Sequential(*clipmodel.visual.transformer.resblocks[:n])(x)
    target_resblock = clipmodel.visual.transformer.resblocks[n]
    if target_resblock == clipmodel.visual.transformer.resblocks[n]:
        x_in = x
        x = target_resblock.ln_1(x_in)
        linear = torch._C._nn.linear
        q_in, k_in, v_in = linear(x, target_resblock.attn.in_proj_weight, target_resblock.attn.in_proj_bias).chunk(3, dim=-1)
        attn_output, MHA, attn_score, (q, k, v) = attention_layer(q_in, k_in, v_in, vision_heads, patch_size)
        x_after_attn = linear(attn_output, target_resblock.attn.out_proj.weight, target_resblock.attn.out_proj.bias)
        x = x_after_attn + x_in
        x = x + target_resblock.mlp(target_resblock.ln_2(x))
    if (n + 1) != len(clipmodel.visual.transformer.resblocks):
        x = torch.nn.Sequential(*clipmodel.visual.transformer.resblocks[n+1:])(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clipmodel.visual.ln_post(x)  # [:, 0, :])
    x = x[:, 0, :] @ clipmodel.visual.proj
    return x, (attn_output, MHA, attn_score), (q, k, v), (feah, feaw)
def CAV_simplifyAW(q, k):
    q_cls = q[:, :1, :]
    k_patch = k[:, 1:, :]
    cosine_qk = (q_cls * k_patch).sum(-1)
    cosine_qk_max = cosine_qk.max(dim=-1, keepdim=True)[0]
    cosine_qk_min = cosine_qk.min(dim=-1, keepdim=True)[0]
    cosine_qk = (cosine_qk-cosine_qk_min) / (cosine_qk_max-cosine_qk_min)
    return cosine_qk
def CAV_img(clipmodel, inputs, text_embedding=None, forward_function=None, in_resolutions=None, text_index=0, layers=11,  embedding_width=64,):
    global q_i, k_i
    if in_resolutions == None:
        resolutions = [224,324,424,524]
    else:
        resolutions = in_resolutions
    attention_output, attention_scores, MHAs, qs, ks, vs, Grads = {}, {}, {}, {}, {}, {}, {}
    for index in range(len(resolutions)):
        resolution = resolutions[index]
        input_i = F.interpolate(inputs, (resolution, resolution), mode='bilinear', align_corners=False)
        input_i = input_i.to(inputs.device)
        outputs, (attention_scores[resolution], MHAs[resolution], attention_output[resolution]), (qs[resolution], ks[resolution], vs[resolution]), map_size = \
            visual_forward(clipmodel=clipmodel, x=input_i, n=layers, embedding_width=embedding_width, )
        cosine = forward_function.run(img_feature=outputs, text_feature=text_embedding)
        Grads[resolution] = (torch.autograd.grad(outputs = cosine[text_index], inputs=vs[resolution], retain_graph=False))[0]
    count, fused_cls_map = 1, None
    for resolution in resolutions:
        v_i = vs[resolution]
        k_i = ks[resolution]
        q_i = qs[resolution]
        grad = Grads[resolution]
        f_g_cls = grad[:, 0, :]
        v_i_patch = v_i[:, 1:, :]
        cosin_qk = CAV_simplifyAW(q_i, k_i)
        cls_map = f_g_cls.unsqueeze(1) * v_i_patch * cosin_qk.unsqueeze(-1)
        cls_map = cls_map.sum(-1)
        map = cls_map
        re_dim = int(math.sqrt(map.shape[1]))
        cls_map = cls_map.reshape(map.shape[0], re_dim, re_dim).unsqueeze(0)
        cls_map = F.interpolate(cls_map, size=(224, 224), mode='bilinear', align_corners=False)[0]
        if count == 1:
            fused_cls_map = cls_map
        else:
            fused_cls_map += cls_map
        count += 1
    map = fused_cls_map
    map = F.relu_(map)
    return map
# text version
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
def attention_layer_text(q, k, v, num_heads=8, text_width=None, attn_mask=None):
    tgt_len, bsz, embed_dim = q.shape
    if text_width is not None:
        head_dim = text_width
        num_heads = embed_dim // head_dim
    else:
        head_dim = embed_dim // num_heads
    scaling = float(head_dim) ** -0.5
    q = q * scaling
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    if attn_mask is not None:
        attn_output_weights += attn_mask
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    MHA = torch.bmm(attn_output_weights, v)
    attn_output = MHA
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().reshape(tgt_len, bsz, embed_dim)
    return attn_output, MHA, attn_output_weights, (q, k, v)
def clip_encode_text_single(clipmodel, text, n, num_heads=8, text_width=None):
    x = clipmodel.token_embedding(text).type(clipmodel.dtype)
    attn_mask = clipmodel.build_attention_mask().to(dtype=x.dtype, device=x.device)
    x = x + clipmodel.positional_embedding.type(clipmodel.dtype)
    x = x.permute(1, 0, 2)
    x = torch.nn.Sequential(*clipmodel.transformer.resblocks[:n])(x)
    TR = clipmodel.transformer.resblocks[n]
    x_in = x
    x = TR.ln_1(x_in)
    linear = torch._C._nn.linear
    q_in, k_in, v_in = linear(x, TR.attn.in_proj_weight, TR.attn.in_proj_bias).chunk(3, dim=-1)
    attn_output, MHSA, attn, (q, k, v) = attention_layer_text(q_in, k_in, v_in, attn_mask=attn_mask, text_width=text_width, num_heads=num_heads)
    x_after_attn = linear(attn_output, TR.attn.out_proj.weight, TR.attn.out_proj.bias)
    x = x_after_attn + x_in
    x = x + TR.mlp(TR.ln_2(x))
    if n+1 < len(clipmodel.transformer.resblocks):
        x = torch.nn.Sequential(*clipmodel.transformer.resblocks[n+1:])(x)
    x = x.permute(1, 0, 2)
    x = clipmodel.ln_final(x).type(clipmodel.dtype)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
    x = x @ clipmodel.text_projection
    return x, x_in, (attn, MHSA, attn_output), (q, k, v)
def text_per_layer(clipmodel, text_tensor, ori_img_embedding, start_layer_n, num_heads, text_width=None, device='cuda'):
    global grad_in
    current_text = text_tensor.unsqueeze(0).to(device)
    eos_position = current_text.argmax(dim=-1)
    attns, atten_outs, MHSAs, vs, qs, ks, vs_in, qs_in, ks_in, grads = [], [], [], [], [], [], [], [], [], []
    for i in range(start_layer_n, len(clipmodel.transformer.resblocks)):
        text_embedding, _, (attn_i, MHSA_i, atten_out_i), (q_i, k_i, v_i) \
            = clip_encode_text_single(clipmodel, current_text, n=i, num_heads=num_heads, text_width=text_width)
        attns.append(attn_i), MHSAs.append(MHSA_i), atten_outs.append(atten_out_i),
        qs.append(q_i), ks.append(k_i), vs.append(v_i),
        text_embedding  = F.normalize(text_embedding, dim=-1)
        if ori_img_embedding == None:
            assert 'no image embedding'
        img_embedding = F.normalize(ori_img_embedding)
        cosine = (img_embedding @ text_embedding.T)
        c = cosine[0]
        grad_in = v_i
        grad_i = torch.autograd.grad(c, grad_in, retain_graph=False)[0]
        grads.append(grad_i.detach())
    return grads, qs, ks, vs, attns, atten_outs, MHSAs, eos_position
def text_explainer(clipmodel, explan_method, texts, wanted_text_indexs, forward_function=text_per_layer,
                   start_layer=6, ori_img_embedding=None, num_heads=8, text_width=None,
                   show_multiHeads = False, return_visualization=False, enhance_rate=1.2, device='cuda',):
    text_processed, tokens_decoded = preprocess_tokens(texts, device=device) #
    cam_results, html_results = [], []
    for i in wanted_text_indexs:
        grads, qs, ks, vs, attns, atten_outs, MHSAs, eos_position = forward_function(clipmodel, text_processed[i], ori_img_embedding, start_layer, num_heads, text_width,  device=device)
        emap = explan_method(grads, qs, ks, vs, attns, atten_outs, MHSAs, eos_position,)
        del grads
        emap = (emap - emap.min())/(emap.max()-emap.min())
        if return_visualization:
            html_data = cav_utils.show_text_attr(emap, tokens_decoded[i], enhance_rate=enhance_rate,).data
            html_results.append(html_data)
        cam_results.append(emap)
    return cam_results, html_results
def CAV_text(grads, qs, ks, vs, attns, attn_outputs, MHSAs, eos_position,):
    tmp_maps = []
    for i in range(len(qs)):
        grad, q, k, v, attn, attn_output, MHSA = grads[i], qs[i], ks[i], vs[i], attns[i], attn_outputs[i], MHSAs[i]
        explain_layer = grad * v
        tmp_maps.append(explain_layer)
    emap = torch.stack(tmp_maps, dim=0).sum(0)
    emap = F.relu_(emap)
    emap = emap.sum(-1) # sum embedding dim
    emap = emap[:, 1: eos_position - 1]
    emap = emap.sum(0) # sum head
    emap = (emap - emap.min()) / (emap.max() - emap.min())
    return emap