# -*- coding: utf-8 -*-
"""
可视化工具
提供数据可视化功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_salary_distribution(df, save_path=None):
    """绘制薪资分布图"""
    plt.figure(figsize=(10, 6))
    
    sns.histplot(data=df, x='salary_avg', bins=30, kde=True)
    plt.title('AI职位薪资分布', fontsize=16, fontweight='bold')
    plt.xlabel('平均薪资 ($)', fontsize=12)
    plt.ylabel('职位数量', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_skill_importance(skill_salary_impact, top_n=15, save_path=None):
    """绘制技能薪资影响图"""
    # 排序并选择top N
    sorted_skills = sorted(
        skill_salary_impact.items(),
        key=lambda x: x[1]['salary_lift_pct'],
        reverse=True
    )[:top_n]
    
    skills = [s[0] for s in sorted_skills]
    lifts = [s[1]['salary_lift_pct'] for s in sorted_skills]
    
    plt.figure(figsize=(12, 8))
    colors = ['green' if l > 0 else 'red' for l in lifts]
    
    plt.barh(skills, lifts, color=colors, alpha=0.7)
    plt.title(f'Top {top_n} 技能对薪资的影响', fontsize=16, fontweight='bold')
    plt.xlabel('薪资提升百分比 (%)', fontsize=12)
    plt.ylabel('技能', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(True, alpha=0.3, axis='x')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_job_categories(df, save_path=None):
    """绘制职位类别分布图"""
    plt.figure(figsize=(12, 6))
    
    category_counts = df['job_category'].value_counts().head(15)
    
    plt.bar(range(len(category_counts)), category_counts.values, alpha=0.7)
    plt.xticks(range(len(category_counts)), category_counts.index, rotation=45, ha='right')
    plt.title('Top 15 职位类别', fontsize=16, fontweight='bold')
    plt.xlabel('职位类别', fontsize=12)
    plt.ylabel('职位数量', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
