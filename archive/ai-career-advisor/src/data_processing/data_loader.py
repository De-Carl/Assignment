"""
数据加载模块
负责从CSV文件加载和初步清洗数据
"""

import pandas as pd
import numpy as np
from typing import Optional
import ast


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, data_path: str):
        """
        初始化数据加载器
        
        Args:
            data_path: CSV文件路径
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        加载CSV数据
        
        Returns:
            DataFrame: 加载的数据
        """
        print(f"正在加载数据: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"数据加载完成! 形状: {self.df.shape}")
        return self.df
    
    def parse_list_columns(self) -> pd.DataFrame:
        """
        解析字符串格式的列表字段
        如: "['Python', 'SQL']" -> ['Python', 'SQL']
        
        Returns:
            DataFrame: 处理后的数据
        """
        if self.df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        list_columns = ['skills_list', 'tools_list']
        
        for col in list_columns:
            if col in self.df.columns:
                print(f"正在解析列: {col}")
                self.df[col] = self.df[col].apply(self._safe_parse_list)
        
        return self.df
    
    @staticmethod
    def _safe_parse_list(value):
        """
        安全地解析列表字符串
        
        Args:
            value: 字符串或列表
            
        Returns:
            list: 解析后的列表
        """
        if pd.isna(value):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except:
                # 如果解析失败,尝试简单分割
                return [s.strip() for s in value.strip('[]').replace("'", "").split(',')]
        return []
    
    def get_basic_stats(self) -> dict:
        """
        获取数据集基本统计信息
        
        Returns:
            dict: 统计信息字典
        """
        if self.df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        stats = {
            "总记录数": len(self.df),
            "总列数": len(self.df.columns),
            "缺失值统计": self.df.isnull().sum().to_dict(),
            "数值列统计": self.df.describe().to_dict(),
            "分类列统计": {
                "行业分布": self.df['industry'].value_counts().to_dict() if 'industry' in self.df.columns else {},
                "职位分布": self.df['job_category'].value_counts().to_dict() if 'job_category' in self.df.columns else {},
                "经验等级": self.df['experience_level'].value_counts().to_dict() if 'experience_level' in self.df.columns else {},
            }
        }
        
        return stats
    
    def filter_by_conditions(self, 
                            job_category: Optional[str] = None,
                            experience_level: Optional[str] = None,
                            industry: Optional[str] = None,
                            min_salary: Optional[float] = None,
                            max_salary: Optional[float] = None) -> pd.DataFrame:
        """
        根据条件筛选数据
        
        Args:
            job_category: 职位类别
            experience_level: 经验等级
            industry: 行业
            min_salary: 最低薪资
            max_salary: 最高薪资
            
        Returns:
            DataFrame: 筛选后的数据
        """
        if self.df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        filtered_df = self.df.copy()
        
        if job_category:
            filtered_df = filtered_df[filtered_df['job_category'] == job_category]
        
        if experience_level:
            filtered_df = filtered_df[filtered_df['experience_level'] == experience_level]
        
        if industry:
            filtered_df = filtered_df[filtered_df['industry'] == industry]
        
        if min_salary is not None:
            filtered_df = filtered_df[filtered_df['salary_avg'] >= min_salary]
        
        if max_salary is not None:
            filtered_df = filtered_df[filtered_df['salary_avg'] <= max_salary]
        
        print(f"筛选后数据量: {len(filtered_df)}")
        return filtered_df
    
    def get_unique_values(self, column: str) -> list:
        """
        获取某列的唯一值
        
        Args:
            column: 列名
            
        Returns:
            list: 唯一值列表
        """
        if self.df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        return sorted(self.df[column].unique().tolist())
    
    def get_all_skills(self) -> list:
        """
        获取所有技能的列表
        
        Returns:
            list: 去重后的技能列表
        """
        if self.df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        all_skills = set()
        for skills in self.df['skills_list']:
            if isinstance(skills, list):
                all_skills.update(skills)
        
        return sorted(list(all_skills))
    
    def get_all_tools(self) -> list:
        """
        获取所有工具的列表
        
        Returns:
            list: 去重后的工具列表
        """
        if self.df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        all_tools = set()
        for tools in self.df['tools_list']:
            if isinstance(tools, list):
                all_tools.update(tools)
        
        return sorted(list(all_tools))


if __name__ == "__main__":
    # 测试代码
    loader = DataLoader("../../data/raw/ai_job_market_cleaned.csv")
    df = loader.load_data()
    df = loader.parse_list_columns()
    
    print("\n=== 基本统计 ===")
    stats = loader.get_basic_stats()
    print(f"总记录数: {stats['总记录数']}")
    print(f"总列数: {stats['总列数']}")
    
    print("\n=== 所有技能 (前20个) ===")
    skills = loader.get_all_skills()
    print(skills[:20])
    
    print("\n=== 所有工具 (前10个) ===")
    tools = loader.get_all_tools()
    print(tools[:10])