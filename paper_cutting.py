import os
import json
import re
import PyPDF2
import pandas as pd


def extract_text_from_pdf(file_path: str):
    # Extract text content from a PDF file.
    contents = []
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            page_text = page.extract_text().strip()
            raw_text = [text.strip() for text in page_text.splitlines() if text.strip()]
            new_text = ''
            for text in raw_text:
                new_text += text
                if text[-1] in ['.', '!', '?', '。', '！', '？', '…', ';', '；', ':', '：', '”', '’', '）', '】', '》', '」',
                                '』', '〕', '〉', '》', '〗', '〞', '〟', '»', '"', "'", ')', ']', '}']:
                    contents.append(new_text)
                    new_text = ''
            if new_text:
                contents.append(new_text)
    return contents


def extract_abstract_from_text(text: list[str]):
    res = ""
    body_start = ""
    para_index = 0
    while para_index < len(text):
        para = text[para_index]
        para_index += 1
        start = para.find("Abstract") * para.find("ABSTRACT") * (-1)
        end = para.find("Introduction") * para.find("INTRODUCTION") * (-1)
        if start != -1:
            if end != -1:
                res += para[start + 8: end - 2]
            else:
                res += para[start + 8:]
            break

    while para_index < len(text):
        para = text[para_index]
        para_index += 1
        end = para.find("Introduction") * para.find("INTRODUCTION") * (-1)
        if end == -1:
            res += para
        else:
            res += para[:end - 2]
            body_start += para[end:]
            break

    return res, body_start, para_index


def extract_keywords_from_abstract(para: str):
    keywords = None
    start = para.find("Keywords")
    if start != -1:
        keywords = para[start + 8:]
    else:
        start = para.find("INDEX TERMS")
        if start != -1:
            keywords = para[start + 11:]
    return keywords


def cut_references_from_text(text: list[str], body_start_index: int):
    para_index = len(text) - 1
    while para_index >= 0:
        para = text[para_index]
        para_index -= 1
        if "References" in para or "REFERENCES" in para:
            stop = para_index
            break
    cutted = text[body_start_index: stop]
    return cutted


def replace_ligatures(text):
    ligature_map = {
        '\ufb00': 'ff',
        '\ufb01': 'fi',
        '\ufb02': 'fl',
        '\ufb03': 'ffi',
        '\ufb04': 'ffl',
        '\ufb05': 'st',
        '\ufb06': 'st',
        '\u00e6': 'ae',  # æ
        '\u0153': 'oe',  # œ
        '\u2019': '\'',
    }

    for ligature, replacement in ligature_map.items():
        text = re.sub(ligature, replacement, text)

    return text


def under_non_alpha_ratio(text: str, threshold: float = 0.5):
    if len(text) == 0:
        return False
    alpha_count = len([char for char in text if char.strip() and char.isalpha()])
    total_count = len([char for char in text if char.strip()])
    try:
        ratio = alpha_count / total_count
        return ratio < threshold
    except ZeroDivisionError:
        return False


def split_text(text, max_length):
    paragraphs = []
    while len(text) > max_length:
        split_pos = max_length
        while split_pos > 0 and text[split_pos] not in ' \n.,;':
            split_pos -= 1
        if split_pos == 0:
            split_pos = max_length
        paragraphs.append(text[:split_pos].strip())
        text = text[split_pos:].strip()
    if text:
        paragraphs.append(text)
    return paragraphs


directory = "data/paper/NLP/"
output_path = "data/paper/NLP_text_json/"

for file_name in os.listdir(directory):
    file_path = os.path.join(directory, file_name)
    print(file_name)
    text = extract_text_from_pdf(file_path)
    for i, para in enumerate(text):
        text[i] = replace_ligatures(para)
    abstract, body_start, body_start_index = extract_abstract_from_text(text)
    keywords = extract_keywords_from_abstract(abstract)
    body = [body_start] + cut_references_from_text(text, body_start_index)

    # 过滤掉非字母字符比例不符合要求的段落
    filtered_body = [para for para in body if not under_non_alpha_ratio(para, threshold=0.5)]

    max_length = 2000
    split_body = []
    for para in filtered_body:
        split_body.extend(split_text(para, max_length))

    result = {"Abstract": abstract, "Keywords": keywords, "Body": split_body}

    output_file = os.path.join(output_path, file_name[:-3] + "json")
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=4, ensure_ascii=False)
