import pickle
from annoy import AnnoyIndex
from configs.default import INDEX_FILE, META_FILE, DEVICE_DEFAULT, TOPK_DEFAULT
from model_clip_ln import CLIPEncoder

class Retriever:
    def __init__(self, device=DEVICE_DEFAULT):
        self.enc  = CLIPEncoder(device=device)
        self.meta = pickle.load(open(META_FILE,"rb"))
        self.idx  = AnnoyIndex(512,"angular")
        self.idx.load(str(INDEX_FILE))

    def query(self, caption:str, k:int=TOPK_DEFAULT):
        vec=self.enc.encode_text([caption])[0]
        ids,dists=self.idx.get_nns_by_vector(vec,k,include_distances=True)
        outs=[]
        for i,d in zip(ids,dists):
            rec=self.meta[i].copy()
            rec["score"]=1-d/2
            outs.append(rec)
        return outs
