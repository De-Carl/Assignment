# -*- coding: utf-8 -*-
"""
提示词模板
为智能体提供标准化的提示词
"""

WELCOME_MESSAGE = """你好!我是AI职业顾问。

我可以帮助你:
1. 预测薪资水平
2. 推荐学习技能
3. 匹配合适职位

请告诉我你的情况,比如:
- 你会哪些技能?
- 有多少年经验?
- 想要的薪资目标?
"""

SALARY_PREDICTION_TEMPLATE = """根据你的背景信息:
- 技能: {skills}
- 经验: {experience}年
- 行业: {industry}

预测薪资: ${predicted_salary:,.0f}
置信区间: ${lower:,.0f} - ${upper:,.0f}
"""

SKILL_RECOMMENDATION_TEMPLATE = """为你推荐以下技能:

{skills_list}

这些技能可以帮助你:
- 提升竞争力
- 获得更高薪资
- 拓展职业发展空间
"""

JOB_MATCHING_TEMPLATE = """找到 {count} 个匹配的职位:

{jobs_list}

建议优先关注匹配度高的职位!
"""
