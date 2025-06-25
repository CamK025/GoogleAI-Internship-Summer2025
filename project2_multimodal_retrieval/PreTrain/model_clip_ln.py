import torch, torch.nn.functional as F
from clip.clip import load as clip_load, tokenize as clip_tokenize
import numpy as np

class CLIPEncoder:
    """Encapsulate both text & image encoding, output 512-d, normalized"""
    def __init__(self, device="cuda", arch="ViT-B/32"):
        self.device = torch.device(device)
        self.model , self.prep = clip_load(arch, device=self.device)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)

    @torch.inference_mode()
    def encode_text(self, texts):
        toks = clip_tokenize(texts).to(self.device)
        feat = self.model.encode_text(toks)
        return F.normalize(feat, dim=-1).cpu().numpy()

    @torch.inference_mode()
    def encode_pil(self, pil_imgs):
        feats=[]
        for img in pil_imgs:
            feats.append(self.prep(img).to(self.device))
        feat = self.model.encode_image(torch.stack(feats))
        return F.normalize(feat, dim=-1).cpu().numpy()
    
    @torch.inference_mode()
    def encode_pil(self, pil_imgs):
        feats = []
        valid = []
        for img in pil_imgs:
            try:
                feats.append(self.prep(img).to(self.device))
                valid.append(True)
            except ZeroDivisionError:         
                valid.append(False)

        if not any(valid):          
            return np.zeros((len(pil_imgs), 512), dtype="float32")

        feats = self.model.encode_image(torch.stack(feats))
        feats = F.normalize(feats, dim=-1).cpu().numpy()

        full = np.zeros((len(pil_imgs), 512), dtype="float32")
        j = 0
        for i, ok in enumerate(valid):
            if ok:
                full[i] = feats[j]; j += 1
        return full

