# -*- coding: utf-8 -*-
"""
模型模块
包含薪资预测、技能推荐和职位匹配功能
"""

from .salary_predictor import SalaryPredictor
from .skill_recommender import SkillRecommender
from .job_matcher import JobMatcher

__all__ = ['SalaryPredictor', 'SkillRecommender', 'JobMatcher']
