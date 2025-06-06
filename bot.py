import os
import json
import random
import joblib
from pathlib import Path
from utils.data_loader import SunbaDataLoader
from utils.text_utils import jieba_tokenizer
import jieba

class SunbaBot:
    def __init__(self, use_model=True):
        """
        初始化孙吧机器人
        :param use_model: 是否使用机器学习模型
        """
        # 初始化jieba分词
        jieba.initialize()
        
        # 加载数据
        data_paths = [
            'data/dataset1.json',
            'data/dataset2.json'
        ]
        self.loader = SunbaDataLoader(data_paths)
        
        # 加载模型
        self.use_model = use_model
        if use_model:
            model_path = Path('models/sunba_model.pkl')
            answers_path = Path('models/answers.json')
            
            if model_path.exists() and answers_path.exists():
                try:
                    # 加载模型参数
                    model_data = joblib.load(model_path)
                    self.vectorizer = model_data['vectorizer']
                    self.model = model_data['model']
                    
                    # 从单独文件加载答案集
                    with open(answers_path, 'r', encoding='utf-8') as f:
                        self.model_answers = json.load(f)
                        
                except Exception as e:
                    self.use_model = False
                    print(f"⚠️ 模型加载失败: {str(e)}，已切换至规则模式")
            else:
                self.use_model = False
                print("⚠️ 模型文件缺失，已切换至规则模式")

    def model_respond(self, user_input):
        """
        使用模型生成响应
        :param user_input: 用户输入文本
        :return: 模型生成的响应或None
        """
        try:
            vec = self.vectorizer.transform([user_input])
            distances, indices = self.model.kneighbors(vec)
            
            # 只返回相似度高于阈值的回答
            if distances[0][0] < 0.7:  # 相似度阈值70%
                return self.model_answers[indices[0][0]]
        except Exception as e:
            print(f"⚠️ 模型响应出错: {str(e)}")
        return None
    
    def respond(self, user_input):
        """
        生成回复（混合模型和规则）
        :param user_input: 用户输入
        :return: 生成的回复
        """
        # 预处理输入
        user_input = user_input.strip()
        if not user_input:
            return "你倒是说话啊（流汗黄豆）"
        
        # 20%概率完全随机回复
        if random.random() < 0.2:
            return self.loader.get_random_response() + random.choice(["！", "~"])
        
        # 模型匹配
        if self.use_model:
            model_response = self.model_respond(user_input)
            if model_response:
                return model_response
        
        # 规则匹配
        response = self.loader.get_random_response(user_input)
        
        # 添加孙吧特色后缀
        suffixes = ["嗷", "啊", "！", "~", "（流汗黄豆）", "（恼）"]
        return response + random.choice(suffixes)

if __name__ == "__main__":
    print("="*40)
    print("孙吧Bot增强版 v1.0")
    print("输入'退出'结束对话")
    print("="*40)
    
    bot = SunbaBot(use_model=True)
    
    while True:
        try:
            user_input = input("你: ")
            if user_input.lower() in ['退出', 'exit', 'quit']:
                break
                
            response = bot.respond(user_input)
            print("Bot:", response)
            
        except KeyboardInterrupt:
            print("\n🛑 用户中断对话")
            break
        except Exception as e:
            print(f"❌ 出错: {str(e)}")
            continue