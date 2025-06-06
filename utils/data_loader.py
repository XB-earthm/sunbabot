import json
import random
import os
import jieba
from typing import List, Tuple, Optional

class SunbaDataLoader:
    def __init__(self, data_paths: List[str]):
        self.data = []
        self.keyword_index = {}  # 關鍵詞索引
        
        for path in data_paths:
            if not os.path.exists(path):
                print(f"警告：文件 {path} 不存在")
                continue
                
            with open(path, 'r', encoding='utf-8') as f:
                self.data.extend(json.load(f))
        
        self._build_index()
        self.stop_words = self._load_stop_words()

    def _load_stop_words(self):
        return {'的', '了', '是', '我', '你', '他'}

    def _build_index(self):
        for idx, item in enumerate(self.data):
            text = f"{item.get('instruction', '')} {item.get('output', '')}"
            words = set(jieba.cut(text))
            for word in words:
                if len(word) > 1:
                    if word not in self.keyword_index:
                        self.keyword_index[word] = []
                    self.keyword_index[word].append(idx)

    def get_training_pairs(self) -> List[Tuple[str, str]]:
        return [
            (item['instruction'], item['output'])
            for item in self.data
            if 'instruction' in item and 'output' in item
        ]

    def get_random_response(self, input_text=None):
        if input_text:
            input_text = input_text.strip()
            words = set(jieba.cut(input_text))
            matched = []
            for word in words:
                if word in self.keyword_index:
                    matched.extend(self.keyword_index[word])
            if matched:
                return random.choice(self.data)[matched[0]]['output']
        return random.choice(self.data)['output']