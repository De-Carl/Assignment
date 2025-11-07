"""
特征工程模块
负责从原始数据中提取和构造用于机器学习的特征
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List
import pickle


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self):
        """初始化特征工程器"""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.all_skills = []
        self.all_tools = []
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        拟合并转换数据
        
        Args:
            df: 输入数据框
            
        Returns:
            DataFrame: 特征工程后的数据
        """
        df_featured = df.copy()
        
        # 1. 提取所有唯一技能和工具
        self._extract_all_skills_tools(df_featured)
        
        # 2. 创建技能和工具的二进制特征
        df_featured = self._create_skill_features(df_featured)
        df_featured = self._create_tool_features(df_featured)
        
        # 3. 编码分类变量
        df_featured = self._encode_categorical_features(df_featured)
        
        # 4. 创建派生特征
        df_featured = self._create_derived_features(df_featured)
        
        return df_featured
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        仅转换数据(使用已拟合的编码器)
        
        Args:
            df: 输入数据框
            
        Returns:
            DataFrame: 特征工程后的数据
        """
        df_featured = df.copy()
        
        # 创建技能和工具特征
        df_featured = self._create_skill_features(df_featured)
        df_featured = self._create_tool_features(df_featured)
        
        # 编码分类变量
        for col, encoder in self.label_encoders.items():
            if col in df_featured.columns:
                df_featured[f'{col}_encoded'] = df_featured[col].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
        
        # 创建派生特征
        df_featured = self._create_derived_features(df_featured)
        
        return df_featured
    
    def _extract_all_skills_tools(self, df: pd.DataFrame):
        """提取所有唯一的技能和工具"""
        # 提取技能
        all_skills_set = set()
        if 'skills_list' in df.columns:
            for skills in df['skills_list']:
                if isinstance(skills, list):
                    all_skills_set.update(skills)
        self.all_skills = sorted(list(all_skills_set))
        
        # 提取工具
        all_tools_set = set()
        if 'tools_list' in df.columns:
            for tools in df['tools_list']:
                if isinstance(tools, list):
                    all_tools_set.update(tools)
        self.all_tools = sorted(list(all_tools_set))
        
        print(f"提取到 {len(self.all_skills)} 个唯一技能")
        print(f"提取到 {len(self.all_tools)} 个唯一工具")
    
    def _create_skill_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建技能的二进制特征 (One-Hot编码)"""
        # 只选择最常见的技能以避免维度爆炸
        top_n = 30
        skill_counts = {}
        
        for skills in df['skills_list']:
            if isinstance(skills, list):
                for skill in skills:
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_skill_names = [s[0] for s in top_skills]
        
        # 创建二进制特征
        for skill in top_skill_names:
            df[f'skill_{skill}'] = df['skills_list'].apply(
                lambda x: 1 if isinstance(x, list) and skill in x else 0
            )
        
        return df
    
    def _create_tool_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建工具的二进制特征"""
        top_n = 20
        tool_counts = {}
        
        for tools in df['tools_list']:
            if isinstance(tools, list):
                for tool in tools:
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        top_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_tool_names = [t[0] for t in top_tools]
        
        # 创建二进制特征
        for tool in top_tool_names:
            df[f'tool_{tool}'] = df['tools_list'].apply(
                lambda x: 1 if isinstance(x, list) and tool in x else 0
            )
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码分类特征"""
        categorical_cols = [
            'job_category', 'experience_level', 'employment_type',
            'industry', 'company_size', 'region', 'salary_category'
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([str(x)])[0] 
                        if str(x) in self.label_encoders[col].classes_ else -1
                    )
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建派生特征"""
        # 薪资相关特征
        if 'salary_min' in df.columns and 'salary_max' in df.columns:
            df['salary_range'] = df['salary_max'] - df['salary_min']
            df['salary_range_pct'] = df['salary_range'] / df['salary_avg']
        
        # 时间特征
        if 'posted_year' in df.columns:
            df['posting_age'] = 2025 - df['posted_year']
        
        # 技能和工具数量
        if 'num_skills_required' not in df.columns and 'skills_list' in df.columns:
            df['num_skills_required'] = df['skills_list'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        if 'num_tools_preferred' not in df.columns and 'tools_list' in df.columns:
            df['num_tools_preferred'] = df['tools_list'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        # 总技能数
        df['total_tech_count'] = df['num_skills_required'] + df['num_tools_preferred']
        
        return df
    
    def save(self, filepath: str):
        """保存特征工程器"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'all_skills': self.all_skills,
                'all_tools': self.all_tools
            }, f)
        print(f"特征工程器已保存到: {filepath}")
    
    def load(self, filepath: str):
        """加载特征工程器"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.label_encoders = data['label_encoders']
            self.scaler = data['scaler']
            self.all_skills = data['all_skills']
            self.all_tools = data['all_tools']
        print(f"特征工程器已从 {filepath} 加载")


if __name__ == "__main__":
    # 测试代码
    from data_loader import DataLoader
    
    loader = DataLoader("../../data/raw/ai_job_market_cleaned.csv")
    df = loader.load_data()
    df = loader.parse_list_columns()
    
    engineer = FeatureEngineer()
    df_featured = engineer.fit_transform(df)
    
    print("\n=== 特征工程后的列 ===")
    print(df_featured.columns.tolist())
    print(f"\n总特征数: {len(df_featured.columns)}")