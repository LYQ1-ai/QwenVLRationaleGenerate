import argparse
import os

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor, \
    CLIPModel

import Util
from data_loader import load_data


parser = argparse.ArgumentParser(description="计算相似度")
parser.add_argument("--root_path", type=str, help="数据集根目录", default="/home/lyq/DataSet/FakeNews/gossipcop")
parser.add_argument("--model_dir", type=str, help="模型目录", default='/home/lyq/Model/LongCLIP-GmP-ViT-L-14')
parser.add_argument("--dataset", type=str, help="数据集名称", default='gossipcop')
parser.add_argument("--batch_size", type=int, help="批量大小",default=64)
parser.add_argument("--device", type=str, help="设备",default='cuda')
parser.add_argument("--max_length", type=int, help="输入文本最大长度",default=77)
args = parser.parse_args()





class CLIP(nn.Module):

    def __init__(self, model_dir, **kwargs):
        super().__init__()
        clip_model = CLIPModel.from_pretrained(model_dir)
        self.text_model = clip_model.text_model
        self.image_model = clip_model.vision_model

    def forward(self, texts_input, images_input):
        text_features = self.text_model(**texts_input)
        image_features = self.image_model(**images_input)
        return nn.functional.cosine_similarity(text_features.pooler_output, image_features.pooler_output,dim=1)

def read_image_url_list(file_paths):
    return [Util.read_image_from_url(url) for url in file_paths]


def calculate_similarity(texts,images,model,image_processor,text_tokenizer,device):
    text_inputs = text_tokenizer(texts, padding='max_length',truncation=True,max_length=args.max_length, return_tensors='pt').to(device=device)
    image_inputs = image_processor(images=images, return_tensors='pt').to(device=device)
    similarity = model(text_inputs,image_inputs)
    return similarity


if __name__ == '__main__':
    data_iter,_ = load_data(args.dataset,args.root_path,batch_size=args.batch_size,collect_fn=None)
    model = CLIP(args.model_dir, max_length=args.max_length)
    device = torch.device(args.device)
    image_processor = AutoProcessor.from_pretrained(args.model_dir)
    text_tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model.to(device=device)
    result = {
        'id':[],
        'similarity':[]
    }
    for batch in tqdm(data_iter):
        texts = batch['text']
        images = read_image_url_list(batch['image_url'])
        batch_id = batch['id']
        batch_similarity = calculate_similarity(texts,images,model,image_processor,text_tokenizer,device).tolist()
        result['id'].extend(batch_id)
        result['similarity'].extend(batch_similarity)

    df = pd.DataFrame(result)
    df.to_csv(os.path.join(args.root_path,f'{args.dataset}_similarity.csv'),index=False)


