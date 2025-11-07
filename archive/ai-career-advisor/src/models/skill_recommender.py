# -*- coding: utf-8 -*-
"""
技能推荐系统
基于关联规则挖掘和薪资分析推荐技能
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from collections import defaultdict, Counter
import pickle


class SkillRecommender:
    """技能推荐器类"""
    
    def __init__(self, min_support=0.05, min_confidence=0.3):
        """
        初始化技能推荐器
        
        Args:
            min_support: 最小支持度
            min_confidence: 最小置信度
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.skill_associations = {}
        self.skill_salary_impact = {}
        self.job_category_skills = defaultdict(Counter)
        self.is_trained = False
    
    def train(self, df: pd.DataFrame):
        """
        训练推荐系统
        
        Args:
            df: 包含技能和薪资信息的数据框
        """
        print('\n开始训练技能推荐系统...')
        
        # 1. 分析技能关联关系
        self._analyze_skill_associations(df)
        
        # 2. 分析技能对薪资的影响
        self._analyze_skill_salary_impact(df)
        
        # 3. 分析职位类别和技能的关系
        self._analyze_job_category_skills(df)
        
        self.is_trained = True
        print('技能推荐系统训练完成!')
    
    def _analyze_skill_associations(self, df: pd.DataFrame):
        """分析技能之间的关联关系"""
        print('  分析技能关联...')
        
        # 构建技能共现矩阵
        all_skillsets = []
        for skills in df['skills_list']:
            if isinstance(skills, list) and len(skills) > 0:
                all_skillsets.append(set(skills))
        
        # 计算支持度
        skill_counts = Counter()
        for skillset in all_skillsets:
            for skill in skillset:
                skill_counts[skill] += 1
        
        total_count = len(all_skillsets)
        
        # 计算技能关联规则
        for skill in skill_counts:
            if skill_counts[skill] / total_count < self.min_support:
                continue
            
            self.skill_associations[skill] = {}
            
            # 找出与该技能经常一起出现的其他技能
            co_occurrence = Counter()
            for skillset in all_skillsets:
                if skill in skillset:
                    for other_skill in skillset:
                        if other_skill != skill:
                            co_occurrence[other_skill] += 1
            
            # 计算置信度
            for other_skill, count in co_occurrence.items():
                confidence = count / skill_counts[skill]
                if confidence >= self.min_confidence:
                    self.skill_associations[skill][other_skill] = confidence
        
        print(f'    发现 {len(self.skill_associations)} 个技能的关联规则')
    
    def _analyze_skill_salary_impact(self, df: pd.DataFrame):
        """分析技能对薪资的影响"""
        print('  分析技能薪资影响...')
        
        # 为每个技能计算平均薪资
        skill_salaries = defaultdict(list)
        
        for idx, row in df.iterrows():
            if isinstance(row['skills_list'], list) and pd.notna(row['salary_avg']):
                for skill in row['skills_list']:
                    skill_salaries[skill].append(row['salary_avg'])
        
        # 计算每个技能的平均薪资和薪资提升
        overall_avg_salary = df['salary_avg'].mean()
        
        for skill, salaries in skill_salaries.items():
            if len(salaries) >= 10:  # 至少10个样本
                skill_avg = np.mean(salaries)
                salary_lift = ((skill_avg - overall_avg_salary) / overall_avg_salary) * 100
                
                self.skill_salary_impact[skill] = {
                    'avg_salary': skill_avg,
                    'salary_lift_pct': salary_lift,
                    'count': len(salaries),
                    'median_salary': np.median(salaries)
                }
        
        print(f'    分析了 {len(self.skill_salary_impact)} 个技能的薪资影响')
    
    def _analyze_job_category_skills(self, df: pd.DataFrame):
        """分析每个职位类别的常见技能"""
        print('  分析职位类别技能...')
        
        for idx, row in df.iterrows():
            if isinstance(row['skills_list'], list) and pd.notna(row['job_category']):
                job_cat = row['job_category']
                for skill in row['skills_list']:
                    self.job_category_skills[job_cat][skill] += 1
        
        print(f'    分析了 {len(self.job_category_skills)} 个职位类别')
    
    def recommend_skills_by_association(self, current_skills: List[str], top_n=5) -> List[Dict]:
        """
        基于技能关联推荐新技能
        
        Args:
            current_skills: 当前已掌握的技能列表
            top_n: 推荐数量
            
        Returns:
            推荐技能列表
        """
        if not self.is_trained:
            raise ValueError('模型尚未训练')
        
        recommendations = defaultdict(float)
        
        # 对于每个已掌握的技能,找出关联技能
        for skill in current_skills:
            if skill in self.skill_associations:
                for related_skill, confidence in self.skill_associations[skill].items():
                    if related_skill not in current_skills:
                        recommendations[related_skill] = max(
                            recommendations[related_skill], confidence
                        )
        
        # 排序并返回前N个
        sorted_recs = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        result = []
        for skill, confidence in sorted_recs:
            rec_dict = {
                'skill': skill,
                'confidence': round(confidence, 3),
                'reason': f'与您已掌握的技能关联度高 ({confidence:.1%})'
            }
            
            # 添加薪资信息
            if skill in self.skill_salary_impact:
                rec_dict['avg_salary'] = round(
                    self.skill_salary_impact[skill]['avg_salary'], 2
                )
                rec_dict['salary_lift_pct'] = round(
                    self.skill_salary_impact[skill]['salary_lift_pct'], 2
                )
            
            result.append(rec_dict)
        
        return result
    
    def recommend_skills_for_salary(self, current_skills: List[str], 
                                   target_salary: float, top_n=5) -> List[Dict]:
        """
        推荐有助于达到目标薪资的技能
        
        Args:
            current_skills: 当前已掌握的技能列表
            target_salary: 目标薪资
            top_n: 推荐数量
            
        Returns:
            推荐技能列表
        """
        if not self.is_trained:
            raise ValueError('模型尚未训练')
        
        # 找出薪资高于目标的技能
        high_salary_skills = []
        
        for skill, impact in self.skill_salary_impact.items():
            if skill not in current_skills and impact['avg_salary'] >= target_salary:
                high_salary_skills.append({
                    'skill': skill,
                    'avg_salary': impact['avg_salary'],
                    'salary_lift_pct': impact['salary_lift_pct'],
                    'count': impact['count'],
                    'reason': f"平均薪资 ${impact['avg_salary']:,.0f} (高出市场 {impact['salary_lift_pct']:.1f}%)"
                })
        
        # 按平均薪资排序
        result = sorted(
            high_salary_skills,
            key=lambda x: x['avg_salary'],
            reverse=True
        )[:top_n]
        
        return result
    
    def recommend_skills_for_job(self, job_category: str, 
                                 current_skills: List[str], top_n=5) -> List[Dict]:
        """
        推荐适合某职位类别的技能
        
        Args:
            job_category: 目标职位类别
            current_skills: 当前已掌握的技能列表
            top_n: 推荐数量
            
        Returns:
            推荐技能列表
        """
        if not self.is_trained:
            raise ValueError('模型尚未训练')
        
        if job_category not in self.job_category_skills:
            return []
        
        # 获取该职位类别的热门技能
        job_skills = self.job_category_skills[job_category]
        total_jobs = sum(job_skills.values())
        
        recommendations = []
        for skill, count in job_skills.most_common():
            if skill not in current_skills:
                frequency = count / total_jobs
                
                rec_dict = {
                    'skill': skill,
                    'frequency_pct': round(frequency * 100, 1),
                    'job_count': count,
                    'reason': f'在 {job_category} 岗位中出现频率 {frequency:.1%}'
                }
                
                # 添加薪资信息
                if skill in self.skill_salary_impact:
                    rec_dict['avg_salary'] = round(
                        self.skill_salary_impact[skill]['avg_salary'], 2
                    )
                    rec_dict['salary_lift_pct'] = round(
                        self.skill_salary_impact[skill]['salary_lift_pct'], 2
                    )
                
                recommendations.append(rec_dict)
                
                if len(recommendations) >= top_n:
                    break
        
        return recommendations
    
    def save(self, filepath: str):
        """保存推荐系统"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'skill_associations': self.skill_associations,
                'skill_salary_impact': self.skill_salary_impact,
                'job_category_skills': dict(self.job_category_skills),
                'is_trained': self.is_trained,
                'min_support': self.min_support,
                'min_confidence': self.min_confidence
            }, f)
        print(f'\n技能推荐系统已保存到: {filepath}')
    
    def load(self, filepath: str):
        """加载推荐系统"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.skill_associations = data['skill_associations']
            self.skill_salary_impact = data['skill_salary_impact']
            self.job_category_skills = defaultdict(Counter, data['job_category_skills'])
            self.is_trained = data['is_trained']
            self.min_support = data.get('min_support', 0.05)
            self.min_confidence = data.get('min_confidence', 0.3)
        print(f'技能推荐系统已从 {filepath} 加载')
