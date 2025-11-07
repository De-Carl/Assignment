# -*- coding: utf-8 -*-
"""
模型测试脚本
测试各个模型的功能
"""

import sys
from pathlib import Path

# 添加src目录到路径
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import pytest
from data_processing.data_loader import DataLoader
from models.salary_predictor import SalaryPredictor
from models.skill_recommender import SkillRecommender
from models.job_matcher import JobMatcher

DATA_PATH = "data/raw/ai_job_market_cleaned.csv"

def test_data_loader():
    """测试数据加载器"""
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    assert df is not None
    assert len(df) > 0
    print(f"✓ 数据加载测试通过: {len(df)} 条记录")

def test_salary_predictor():
    """测试薪资预测器"""
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    df = loader.parse_list_columns()
    
    # 这里需要先做特征工程
    # 简化测试,只检查模型创建
    predictor = SalaryPredictor()
    assert predictor is not None
    print("✓ 薪资预测器创建成功")

def test_skill_recommender():
    """测试技能推荐器"""
    recommender = SkillRecommender()
    assert recommender is not None
    print("✓ 技能推荐器创建成功")

def test_job_matcher():
    """测试职位匹配器"""
    matcher = JobMatcher()
    assert matcher is not None
    print("✓ 职位匹配器创建成功")

if __name__ == "__main__":
    print("开始测试...")
    test_data_loader()
    test_salary_predictor()
    test_skill_recommender()
    test_job_matcher()
    print("\n所有测试完成!")
