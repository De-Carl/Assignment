# -*- coding: utf-8 -*-
"""职位匹配系统"""
import pandas as pd
import numpy as np
from typing import List, Dict
import pickle

class JobMatcher:
    def __init__(self):
        self.job_profiles = []
        self.is_trained = False
    
    def train(self, df):
        print('\n开始训练职位匹配系统...')
        for idx, row in df.iterrows():
            self.job_profiles.append({
                'job_title': row['job_title'],
                'company_name': row['company_name'],
                'industry': row['industry'],
                'experience_level': row['experience_level'],
                'location': row['location'],
                'salary_avg': row['salary_avg'],
                'salary_min': row['salary_min'],
                'salary_max': row['salary_max'],
                'skills': row['skills_list'] if isinstance(row['skills_list'], list) else []
            })
        self.is_trained = True
        print(f'训练完成! 共 {len(self.job_profiles)} 个职位')
    
    def find_matching_jobs(self, user_skills, min_salary=0, top_n=10):
        if not self.is_trained:
            return []
        matches = []
        for job in self.job_profiles:
            if job['salary_avg'] < min_salary:
                continue
            matched = set(user_skills) & set(job['skills'])
            score = len(matched) / len(job['skills']) if job['skills'] else 0
            if score > 0:
                missing = list(set(job['skills']) - set(user_skills))[:5]
                matches.append({
                    'job_title': job['job_title'],
                    'company': job['company_name'],
                    'salary': job['salary_avg'],
                    'match_score': round(score * 100, 1),
                    'matched_skills': list(matched),
                    'missing_skills': missing
                })
        return sorted(matches, key=lambda x: x['match_score'], reverse=True)[:top_n]
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.job_profiles, f)
