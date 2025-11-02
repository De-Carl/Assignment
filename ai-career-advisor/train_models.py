# -*- coding: utf-8 -*-
"""训练所有模型的脚本"""
import sys
import os
from pathlib import Path

# 添加src目录到路径
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from data_processing.data_loader import DataLoader
from data_processing.feature_engineering import FeatureEngineer
from models.salary_predictor import SalaryPredictor
from models.skill_recommender import SkillRecommender
from models.job_matcher import JobMatcher

def main():
    print("="*60)
    print("AI职业顾问系统 - 模型训练")
    print("="*60)
    
    # 1. 加载数据
    print("\n[步骤 1/5] 加载数据...")
    data_path = "data/raw/ai_job_market_cleaned.csv"
    loader = DataLoader(data_path)
    df = loader.load_data()
    df = loader.parse_list_columns()
    
    # 2. 特征工程
    print("\n[步骤 2/5] 特征工程...")
    engineer = FeatureEngineer()
    df_featured = engineer.fit_transform(df)
    
    # 保存处理后的数据
    os.makedirs('data/processed', exist_ok=True)
    df_featured.to_csv('data/processed/feature_engineered.csv', index=False)
    engineer.save('data/models/feature_engineer.pkl')
    
    # 3. 训练薪资预测模型
    print("\n[步骤 3/5] 训练薪资预测模型...")
    salary_predictor = SalaryPredictor(model_type='random_forest')
    results = salary_predictor.train(df_featured)
    
    os.makedirs('data/models', exist_ok=True)
    salary_predictor.save('data/models/salary_predictor.pkl')
    
    # 4. 训练技能推荐系统
    print("\n[步骤 4/5] 训练技能推荐系统...")
    skill_recommender = SkillRecommender()
    skill_recommender.train(df)
    skill_recommender.save('data/models/skill_recommender.pkl')
    
    # 5. 训练职位匹配系统
    print("\n[步骤 5/5] 训练职位匹配系统...")
    job_matcher = JobMatcher()
    job_matcher.train(df)
    job_matcher.save('data/models/job_matcher.pkl')
    
    print("\n" + "="*60)
    print(" 所有模型训练完成!")
    print("="*60)
    print("\n模型文件保存在: data/models/")
    print("- salary_predictor.pkl")
    print("- skill_recommender.pkl")
    print("- job_matcher.pkl")
    print("- feature_engineer.pkl")

if __name__ == "__main__":
    main()
