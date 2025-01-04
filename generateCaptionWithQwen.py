import os


import pandas as pd
from tqdm import tqdm
import pickle
import data_loader
from qwen import Qwen2VL


caption_prompt = "Please create a short and descriptive caption in {max_length} words or less."


def wrapper_caption_msg(batch):
    batch_size = len(batch['id'])
    max_length = batch['max_length'][0]
    messages = []
    for i in range(batch_size):
        url = batch['image_url'][i]
        msg = {
            "role": "user",
            "content": [
                {'type': 'image', 'image': url},
                {"type": "text", "text": caption_prompt.format(max_length=max_length)},
            ]
        }

        messages.append(msg)

    return messages,max_length


def generate_image_caption(data_iter,similarity_df,model):
    result = {
        'source_id':[],
        'image_caption':[],
    }
    source_id_set = set()
    cache_file = '/home/lyq/PycharmProjects/llamaRationaleGenerate/cache/gossipcop/qwen_caption.pkl'

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            result = pickle.load(f)
            source_id_set.update(result['source_id'])

    save_flag = 0
    for batch in tqdm(data_iter):
        if batch['id'][0] in source_id_set:
            continue
        batch['max_length'] = similarity_df[similarity_df['id']==batch['id'][0]]['caption_max_length'].tolist()
        msgs ,max_length= wrapper_caption_msg(batch)
        outputs = model.chat(msgs,max_length)
        print(outputs[0])
        result['source_id'].append(batch['id'][0])
        source_id_set.add(batch['id'][0])
        result['image_caption'].append(outputs[0])
        if (save_flag+1)% 100 == 0:
            with open(cache_file, 'wb') as f:
                pickle.dump(result,f)
        save_flag += 1

    return pd.DataFrame(result)







if __name__ == "__main__":
    llm = Qwen2VL()
    similarity_df = pd.read_csv('/home/lyq/PycharmProjects/llamaRationaleGenerate/gossipcop_similarity.csv')

    data_iter = data_loader.load_en_image_text_pair_goss('/home/lyq/DataSet/FakeNews/gossipcop')
    df = generate_image_caption(data_iter,similarity_df, llm)
    df.to_csv('gossipcop_qwen_image_caption.csv',index=False)

