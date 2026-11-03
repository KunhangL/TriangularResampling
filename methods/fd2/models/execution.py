"""Bound memory for the frozen text encoder without changing training batches."""


class MicrobatchedTextEncoder:
    def __init__(self,encoder,batch_size):
        if batch_size<=0: raise ValueError('text_microbatch_size must be positive')
        self.encoder,self.batch_size=encoder,int(batch_size)

    def __getattr__(self,name):
        return getattr(self.encoder,name)

    def __call__(self,texts,device):
        features=[]
        for start in range(0,len(texts),self.batch_size):
            features.extend(self.encoder(texts[start:start+self.batch_size],device))
        return features
