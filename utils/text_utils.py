import jieba

def jieba_tokenizer(text):
    """统一的中文分词函数"""
    return jieba.lcut(text)