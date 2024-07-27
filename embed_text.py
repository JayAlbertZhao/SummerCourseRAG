from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import faiss
import numpy as np
import pandas as pd
import json
import os
from PIL import Image
import matplotlib.pyplot as plt

index_dim = 3584

device_name = "cuda:2"  # Specify a GPU to use
device = torch.device(device_name)

model_path = "/data1n1/"
model_name = "Mistral-7B-Instruct-v0.1"
# model_name = "Qwen2-7B" # 千问模型


# Specify the paths
emoji_json_path = '/data1n1/emo-visual-data/data.json'
emoji_images_path = '/data1n1/emo-visual-data/emo'
emoji_matrix_path = '/home/team_e'
paper_matrix_path = 'data/paper/NLP_matrix'
paper_json_path = 'data/paper/NLP_text_json'


# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path + model_name)
model = AutoModelForCausalLM.from_pretrained(model_path + model_name,
                                             output_hidden_states=True,
                                             output_attentions=True,
                                             torch_dtype=torch.float16,
                                             device_map=device_name)
model.eval()


## Push the GPT model to GPU:
model.to(device_name, dtype=torch.float16)
model.eval()


# Load the JSON data
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Load image data into a DataFrame
def prepare_emoji_data(json_path, images_folder):
    data = load_data(json_path)
    records = []
    for item in data:
        filename = item['filename']
        content = item['content']
        filepath = os.path.join(images_folder, filename)
        records.append({
            'filename': filename,
            'content': content,
            'filepath': filepath
        })
    return pd.DataFrame(records)


def embed_text(text):
    """
    Convert text into vector through Embedding.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    last_hidden_state = outputs.hidden_states[-1]
    embeddings = last_hidden_state.mean(dim=1).cpu().detach().numpy()
    return embeddings


def build_emoji_index(df):
    index = faiss.IndexFlatL2(index_dim)  # Make sure to use the correct dimension
    embeddings = np.vstack(df['content'].apply(embed_text).values)
    # print(f"Embedding shape: {embeddings.shape}")
    index.add(embeddings)
    return index


def build_paper_index(data_dir):
    abstract_keywords_index = faiss.IndexFlatL2(index_dim)
    abstract_keywords_embeddings = []

    for json_file in os.listdir(data_dir):
        with open(os.path.join(data_dir, json_file), 'r') as f:
            paper = json.load(f)
            abstract = paper.get("Abstract", "")
            keywords = paper.get("Keywords", "")
            combined_text = abstract + " " + keywords
            abstract_keywords_embedding = embed_text(combined_text)
            abstract_keywords_embeddings.append(abstract_keywords_embedding)

            # 构建每篇论文的段落索引
            paragraphs = paper.get("Body", [])
            paragraphs_embeddings = [embed_text(p) for p in paragraphs]
            if paragraphs_embeddings:
                paragraphs_index = faiss.IndexFlatL2(index_dim)
                paragraphs_embeddings = np.vstack(paragraphs_embeddings)
                paragraphs_index.add(paragraphs_embeddings)

                # 存储每篇论文的段落索引
                paper_id = os.path.splitext(json_file)[0]
                faiss.write_index(paragraphs_index, os.path.join(paper_matrix_path, f'{paper_id}_paragraphs.index'))

    # 存储所有论文的Abstract和Keywords的索引
    if abstract_keywords_embeddings:
        abstract_keywords_embeddings = np.vstack(abstract_keywords_embeddings)
        abstract_keywords_index.add(abstract_keywords_embeddings)
        faiss.write_index(abstract_keywords_index, os.path.join(paper_matrix_path, 'abstract_keywords.index'))


def get_hidden_states(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    # len(hidden_states) = number of layers of the GPT model
    return hidden_states


def retrieve_image(query, k=5):
    """
    Retrieve the top-k most relevant images from the data.
    """

    # 读文件
    index = faiss.read_index('/home/team_e/emo_faiss_index.index')
    df = pd.read_csv('/home/team_e/emo_df.csv')

    query_embedding = embed_text(query)
    distances, indices = index.search(query_embedding, k)
    retrieved_images = df.iloc[indices[0]]['filepath'].tolist()
    return retrieved_images


def retrieve_paper(query, k=2, m=2):
    """
    Retrieve the top-k most relevant papers and top-m most relevant chunks from each paper.
    Returns the chunks with a window size of 1 (chunk + previous and next chunk).
    """
    # 载入索引
    abstract_keywords_index_path = os.path.join(paper_matrix_path, 'abstract_keywords.index')
    abstract_keywords_index = faiss.read_index(abstract_keywords_index_path)

    # 查询前k个相关的论文
    query_embedding = embed_text(query)
    distances, paper_indices = abstract_keywords_index.search(query_embedding, k)

    retrieved_chunks = []
    data_dir = paper_matrix_path

    for paper_idx in paper_indices[0]:
        paper_id = os.listdir(data_dir)[paper_idx].split('.')[0]  # 假设文件名就是论文ID

        # 载入对应论文的段落索引
        paper_index_path = os.path.join(paper_matrix_path, f'{paper_id}_paragraphs.index')
        paper_index = faiss.read_index(paper_index_path)

        # 获取该论文中的段落
        with open(os.path.join(data_dir, f'{paper_id}.json'), 'r') as f:
            paper = json.load(f)
            paragraphs = paper.get("Body", [])

        # 查询前m个相关的段落
        _, paragraph_indices = paper_index.search(query_embedding, m)

        # 获取段落及其前后窗口
        for idx in paragraph_indices[0]:
            start = max(0, idx - 1)
            end = min(len(paragraphs), idx + 2)
            retrieved_chunks.append(paragraphs[start:end])

    return retrieved_chunks


def generate_response(query):
    """
    Generate a response based on the query.
    """
    inputs = tokenizer(query, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def model_response(input_text):
    """
    Generate a response using a pre-trained language model.
    """
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def show_images(image_paths):
    """
    Display the images from the given file paths in a single row.
    """
    num_images = len(image_paths)
    num_cols = min(num_images, 5)  # Display up to 5 images per row
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5))
    axes = axes.flatten()  # Flatten the 2D array of axes to make indexing easier

    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        axes[i].imshow(image)
        axes[i].axis('off')  # Hide axes

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()  # Adjust spacing between images
    plt.show()


def query_paper(input_query, context):
    """
    Script to demonstrate the use of the retrieve_paper function with a fixed prompt,
    sending the generated text to a model and returning the response along with chunk list.
    """

    # 固定的 prompt
    prompt = '基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。'

    # 如果 context 是 retrieved_chunks，格式化为字符串
    if isinstance(context, list):
        context = "\n\n".join(["\n".join(chunk) for chunk in context])

    # 组合生成模型输入文本
    combined_input = f"{prompt}\n\n已知内容:\n{context}\n\n问题:\n{input_query}"

    # 调用模型生成回复
    response = generate_response(combined_input)

    # 打印并返回模型回复和 chunk 列表
    print(f"模型回复: {response}")
    print(f"chunk 列表: {context}")

    return response, context


def run_paper(query):
    if not os.listdir(paper_matrix_path):
        build_paper_index(paper_json_path)

    query_paper(query, retrieve_paper(query))


def run_emoji(query):
    if not os.listdir(paper_matrix_path):
        # Prepare the data
        df = prepare_emoji_data(emoji_json_path, emoji_images_path)

        index = build_emoji_index(df)

        # 存文件
        faiss.write_index(index, os.path.join(emoji_matrix_path, 'emo_Qwen_faiss_index.index'))
        df.to_csv(os.path.join(emoji_matrix_path, 'emo_Qwen_df.csv'), index=False)

    response = generate_response(query)
    print(f"Model Response: {response}")
    retrieved_images = retrieve_image(response)
    print(f"Retrieved Image(s): {retrieved_images}")

    show_images(retrieved_images)

