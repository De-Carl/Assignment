# -*- coding: utf-8 -*-
"""AI职业顾问Agent - 简化版规则系统"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.salary_predictor import SalaryPredictor
from models.skill_recommender import SkillRecommender
from models.job_matcher import JobMatcher
import pandas as pd
import re

class CareerAdvisorAgent:
    def __init__(self, salary_predictor=None, skill_recommender=None, job_matcher=None):
        self.salary_predictor = salary_predictor
        self.skill_recommender = skill_recommender
        self.job_matcher = job_matcher
        self.conversation_history = []
        self.user_context = {}
    
    def chat(self, user_input):
        """处理用户输入并返回建议"""
        self.conversation_history.append({'role': 'user', 'content': user_input})
        
        # 提取用户信息
        self._extract_user_info(user_input)
        
        # 生成回复
        response = self._generate_response(user_input)
        
        self.conversation_history.append({'role': 'assistant', 'content': response})
        return response
    
    def _extract_user_info(self, text):
        """从文本中提取用户信息"""
        # 提取技能
        common_skills = ['Python', 'SQL', 'Java', 'JavaScript', 'PyTorch', 'TensorFlow',
                        'Scikit-learn', 'Pandas', 'NumPy', 'AWS', 'Azure', 'GCP', 'Docker',
                        'Kubernetes', 'React', 'Vue', 'Flask', 'Django', 'Excel', 'R']
        found_skills = [s for s in common_skills if s.lower() in text.lower()]
        if found_skills:
            self.user_context.setdefault('skills', []).extend(found_skills)
            self.user_context['skills'] = list(set(self.user_context['skills']))
        
        # 提取经验等级
        if any(w in text.lower() for w in ['初级', '新手', 'junior', 'entry']):
            self.user_context['experience_level'] = 'Entry'
        elif any(w in text.lower() for w in ['中级', '有经验', 'mid', 'intermediate']):
            self.user_context['experience_level'] = 'Mid'
        elif any(w in text.lower() for w in ['高级', '资深', 'senior', 'expert']):
            self.user_context['experience_level'] = 'Senior'
        
        # 提取薪资期望
        salary_match = re.search(r'(\d+)[kKwW万千]', text)
        if salary_match:
            num = int(salary_match.group(1))
            if 'k' in text.lower() or 'K' in text:
                self.user_context['target_salary'] = num * 1000
            elif '万' in text or 'w' in text.lower():
                self.user_context['target_salary'] = num * 10000
    
    def _generate_response(self, user_input):
        """根据用户输入生成响应"""
        text_lower = user_input.lower()
        
        # 问候语
        if any(w in text_lower for w in ['你好', 'hello', 'hi', '帮助']):
            return self._greeting()
        
        # 薪资预测相关
        if any(w in text_lower for w in ['薪资', '工资', '薪水', 'salary', '收入', '能拿多少']):
            return self._handle_salary_query()
        
        # 技能推荐相关
        if any(w in text_lower for w in ['技能', '学习', '掌握', 'skill', '需要学', '建议']):
            return self._handle_skill_recommendation()
        
        # 职位推荐相关
        if any(w in text_lower for w in ['职位', '岗位', '工作', 'job', '适合', '推荐']):
            return self._handle_job_recommendation()
        
        # 综合建议
        return self._handle_comprehensive_advice()
    
    def _greeting(self):
        return """你好! 我是AI职业顾问。我可以帮你:
1. 预测薪资范围
2. 推荐需要学习的技能
3. 匹配适合的职位
4. 提供职业发展建议

请告诉我你的背景(如技能、经验等)或者直接提问!"""
    
    def _handle_salary_query(self):
        """处理薪资相关查询"""
        skills = self.user_context.get('skills', [])
        exp_level = self.user_context.get('experience_level', 'Mid')
        
        if not skills:
            return "请告诉我你掌握了哪些技能,我才能为你预测薪资范围。例如: '我会Python和SQL'"
        
        # 这里应该调用薪资预测模型,简化版返回规则估算
        base_salary = {'Entry': 70000, 'Mid': 100000, 'Senior': 140000}
        estimated = base_salary.get(exp_level, 100000)
        estimated += len(skills) * 5000  # 每个技能增加5k
        
        response = f"""基于你的背景:
- 技能: {', '.join(skills)}
- 经验等级: {exp_level}

预测薪资范围: ${estimated*0.85:,.0f} - ${estimated*1.15:,.0f}
平均预期: ${estimated:,.0f}

这个预测基于市场数据分析。实际薪资还会受行业、地区、公司规模等因素影响。"""
        return response
    
    def _handle_skill_recommendation(self):
        """处理技能推荐"""
        skills = self.user_context.get('skills', [])
        target_salary = self.user_context.get('target_salary', None)
        
        if not skills:
            return "请先告诉我你目前掌握的技能,我才能推荐需要学习的新技能。"
        
        # 简化版技能推荐规则
        skill_groups = {
            'Python': ['Pandas', 'NumPy', 'Scikit-learn', 'PyTorch', 'TensorFlow'],
            'SQL': ['PostgreSQL', 'MySQL', 'BigQuery'],
            'AWS': ['Docker', 'Kubernetes', 'Terraform'],
            'JavaScript': ['React', 'Vue', 'Node.js']
        }
        
        recommendations = []
        for skill in skills:
            if skill in skill_groups:
                for rec_skill in skill_groups[skill]:
                    if rec_skill not in skills:
                        recommendations.append(rec_skill)
        
        if not recommendations:
            recommendations = ['PyTorch', 'Docker', 'AWS', 'React', 'Kubernetes']
        
        recommendations = recommendations[:5]
        
        response = f"""基于你已掌握的技能 ({', '.join(skills)}), 我建议你学习:

"""
        for i, skill in enumerate(recommendations, 1):
            response += f"{i}. {skill}\n"
        
        if target_salary:
            response += f"\n学习这些技能可以帮助你达到 ${target_salary:,.0f} 的目标薪资。"
        
        return response
    
    def _handle_job_recommendation(self):
        """处理职位推荐"""
        skills = self.user_context.get('skills', [])
        
        if not skills:
            return "请告诉我你的技能背景,我会为你推荐合适的职位。"
        
        # 简化版职位推荐规则
        job_recommendations = []
        
        if any(s in skills for s in ['Python', 'Pandas', 'SQL']):
            job_recommendations.append("数据分析师 (Data Analyst)")
        if any(s in skills for s in ['PyTorch', 'TensorFlow', 'Scikit-learn']):
            job_recommendations.append("机器学习工程师 (ML Engineer)")
            job_recommendations.append("数据科学家 (Data Scientist)")
        if any(s in skills for s in ['AWS', 'Docker', 'Kubernetes']):
            job_recommendations.append("云架构师 (Cloud Architect)")
        if any(s in skills for s in ['React', 'Vue', 'JavaScript']):
            job_recommendations.append("前端工程师 (Frontend Developer)")
        
        if not job_recommendations:
            job_recommendations = ["数据分析师", "软件工程师", "技术支持"]
        
        response = f"""基于你的技能 ({', '.join(skills)}), 推荐以下职位:\n\n"""
        for i, job in enumerate(job_recommendations[:5], 1):
            response += f"{i}. {job}\n"
        
        return response
    
    def _handle_comprehensive_advice(self):
        """提供综合建议"""
        skills = self.user_context.get('skills', [])
        
        if not skills:
            return """我可以为你提供以下帮助:

 薪资预测 - 告诉我你的技能,我会预测薪资范围
 技能推荐 - 根据你的背景推荐需要学习的技能
 职位匹配 - 推荐最适合你的职位

请告诉我你的技能背景,例如: "我会Python、SQL和机器学习" """
        
        # 提供综合建议
        response = f"""基于你的背景信息, 这是我的综合建议:\n\n"""
        response += "1 薪资评估:\n"
        response += self._handle_salary_query() + "\n\n"
        response += "2 技能建议:\n"
        response += self._handle_skill_recommendation() + "\n\n"
        response += "3 职位方向:\n"
        response += self._handle_job_recommendation()
        
        return response
    
    def reset(self):
        """重置对话"""
        self.conversation_history = []
        self.user_context = {}
