import os
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from utils.data_loader import SunbaDataLoader
import jieba
from pathlib import Path

# 定义独立的分词函数（解决pickle问题）
def jieba_tokenizer(text):
    """使用jieba进行中文分词"""
    return jieba.lcut(text)

class SunbaTrainer:
    def __init__(self, model_dir="models"):
        """
        初始化训练器
        :param model_dir: 模型保存目录
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # 配置数据集路径
        self.data_paths = [
            "data/dataset1.json",
            "data/dataset2.json"
        ]
        
        # 初始化模型组件
        self.vectorizer = TfidfVectorizer(
            tokenizer=jieba_tokenizer,  # 使用独立函数
            max_features=5000,
            ngram_range=(1, 2),
            analyzer='word'
        )
        self.model = NearestNeighbors(
            n_neighbors=5,
            metric='cosine',
            algorithm='brute'
        )
        
        # 加载数据
        self.loader = SunbaDataLoader(self.data_paths)
        self.training_data = None
        self.test_data = None

    def prepare_data(self, test_size=0.2):
        """准备训练和测试数据"""
        try:
            pairs = self.loader.get_training_pairs()
            if not pairs:
                raise ValueError("没有获取到有效的训练数据")
                
            questions, answers = zip(*pairs)
            
            # 划分训练测试集
            (X_train, X_test, 
             y_train, y_test) = train_test_split(
                questions, answers,
                test_size=test_size,
                random_state=42
            )
            
            self.training_data = (X_train, y_train)
            self.test_data = (X_test, y_test)
            
            print("\n🔍 数据统计:")
            print(f"- 总样本数: {len(questions)}")
            print(f"- 训练集: {len(X_train)}")
            print(f"- 测试集: {len(X_test)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"❌ 准备数据时出错: {str(e)}")
            raise

    def train(self):
        """执行完整训练流程"""
        print("\n🚀 开始训练孙吧对话模型...")
        
        try:
            # 准备数据
            X_train, _, y_train, _ = self.prepare_data()
            
            # 特征提取
            print("🔧 正在提取文本特征...")
            X_train_vec = self.vectorizer.fit_transform(X_train)
            
            # 训练模型
            print("🤖 正在训练KNN模型...")
            self.model.fit(X_train_vec)
            
            # 保存模型
            self._save_model(y_train)
            
            # 评估模型
            self.evaluate()
            
            print("\n🎉 训练完成！")
            
        except Exception as e:
            print(f"❌ 训练过程中出错: {str(e)}")
            raise

    def _save_model(self, answers):
        """保存模型到文件"""
        model_path = self.model_dir / "sunba_model.pkl"
        answer_path = self.model_dir / "answers.json"
        
        # 保存模型（使用joblib高效序列化）
        joblib.dump({
            'vectorizer': self.vectorizer,
            'model': self.model
        }, model_path, compress=3)
        
        # 保存答案集（确保是列表格式）
        if isinstance(answers, tuple):
            answers = list(answers)
        with open(answer_path, 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 模型已保存到: {model_path}")
        print(f"💾 答案集已保存到: {answer_path}")

    def evaluate(self):
        """评估模型性能"""
        if not self.test_data:
            print("⚠️ 警告: 没有测试数据可用")
            return
            
        try:
            X_test, y_test = self.test_data
            X_test_vec = self.vectorizer.transform(X_test)
            
            print("\n📊 正在评估模型...")
            distances, indices = self.model.kneighbors(X_test_vec)
            
            # 计算平均相似度
            avg_sim = np.mean(1 - distances[:, 0])
            print(f"- 平均相似度: {avg_sim:.2%}")
            
            # 计算top-1准确率
            correct = sum(
                1 for i, idxs in enumerate(indices)
                if y_test[i] in [self.training_data[1][idx] for idx in idxs]
            )
            accuracy = correct / len(X_test)
            print(f"- Top-1 准确率: {accuracy:.2%}")
            
            # 示例测试
            self._print_example_test()
            
        except Exception as e:
            print(f"❌ 评估过程中出错: {str(e)}")

    def _print_example_test(self):
        """打印示例测试结果"""
        test_cases = [
            "孙笑川",
            "打胶",
            "抽象",
            "小米粥",
            "李赣",
            "带节奏",
            "流汗黄豆"
        ]
        
        print("\n🔎 示例测试:")
        for query in test_cases:
            try:
                vec = self.vectorizer.transform([query])
                distances, indices = self.model.kneighbors(vec)
                
                print(f"\n📝 输入: '{query}'")
                print("🔍 最匹配的回复:")
                for i, idx in enumerate(indices[0]):
                    similarity = 1 - distances[0][i]
                    if similarity > 0.5:  # 显示相似度高于50%的结果
                        print(f"{i+1}. {self.training_data[1][idx]} (相似度: {similarity:.1%})")
            except Exception as e:
                print(f"测试用例 '{query}' 出错: {str(e)}")

def main():
    try:
        # 初始化jieba分词
        jieba.initialize()
        
        # 初始化训练器
        trainer = SunbaTrainer()
        
        # 开始训练
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断训练")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {str(e)}")

if __name__ == "__main__":
    main()