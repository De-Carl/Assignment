#  AI职业顾问系统 - 快速开始

##  系统功能

1. ** 薪资预测** - 根据技能、经验预测薪资范围
2. ** 技能推荐** - 推荐需要学习的新技能
3. ** 职位匹配** - 匹配最适合的职位
4. ** 智能对话** - 自然语言交互式建议

##  环境准备

### 1. 激活虚拟环境

```powershell
# 如果你已创建虚拟环境,激活它
conda activate your_env_name
# 或者
.\venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

主要依赖:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

##  使用步骤

### 步骤1: 训练模型

首次使用需要训练模型(约需5-10分钟):

```powershell
python train_models.py
```

这将训练4个模型:
- 薪资预测模型 (Random Forest)
- 技能推荐系统 (关联规则)
- 职位匹配系统 (相似度计算)
- 特征工程器

模型文件将保存在 `data/models/` 目录。

### 步骤2: 测试Agent

```powershell
python test_agent.py
```

##  使用示例

启动后,你可以这样提问:

### 示例1: 薪资查询
```
你: 我会Python和SQL,能拿多少薪水?

Agent: 基于你的背景:
- 技能: Python, SQL
- 经验等级: Mid

预测薪资范围: $89,250 - $120,750
平均预期: $105,000
```

### 示例2: 技能推荐
```
你: 我会Python,想学习新技能提高薪资

Agent: 基于你已掌握的技能 (Python), 我建议你学习:
1. Pandas
2. NumPy
3. Scikit-learn
4. PyTorch
5. TensorFlow
```

### 示例3: 职位推荐
```
你: 我会Python、SQL和机器学习,推荐适合的职位

Agent: 基于你的技能, 推荐以下职位:
1. 数据分析师 (Data Analyst)
2. 机器学习工程师 (ML Engineer)
3. 数据科学家 (Data Scientist)
```

### 示例4: 综合建议
```
你: 我是中级工程师,会Python、SQL、Docker,想拿15万年薪

Agent: (提供薪资评估、技能建议、职位方向的综合建议)
```

##  项目结构

```
ai-career-advisor/
 data/
    raw/                    # 原始数据
    processed/              # 处理后的数据
    models/                 # 训练好的模型
 src/
    data_processing/        # 数据处理
       data_loader.py
       feature_engineering.py
    models/                 # ML模型
       salary_predictor.py
       skill_recommender.py
       job_matcher.py
    agent/                  # AI Agent
        career_advisor_agent.py
 train_models.py             # 训练脚本
 test_agent.py               # 测试脚本
 requirements.txt            # 依赖列表
```

##  技术细节

### 薪资预测模型
- 算法: Random Forest Regressor
- 特征: 技能、经验、行业、地区等
- 评估指标: RMSE, MAE, R

### 技能推荐系统
- 方法: 关联规则挖掘
- 分析: 技能共现、薪资影响
- 推荐策略: 基于关联度和薪资提升

### 职位匹配系统
- 方法: 技能匹配度计算
- 评分: 已掌握技能 / 所需技能
- 输出: 匹配职位 + 缺失技能

### AI Agent
- 类型: 简化版规则系统
- 功能: 自动提取用户信息、意图识别、生成建议
- 可扩展: 可升级为OpenAI/LangChain版本

##  注意事项

1. **首次运行** - 必须先运行 `train_models.py` 训练模型
2. **数据位置** - 确保数据文件在 `data/raw/ai_job_market_cleaned.csv`
3. **虚拟环境** - 建议使用虚拟环境避免依赖冲突
4. **Python版本** - 需要 Python 3.8+

##  常见问题

### Q: ModuleNotFoundError
A: 检查是否安装了所有依赖: `pip install -r requirements.txt`

### Q: 数据文件找不到
A: 确保 `data/raw/ai_job_market_cleaned.csv` 存在

### Q: 模型未训练
A: 运行 `python train_models.py` 训练模型

### Q: 想要更智能的对话
A: 可以将Agent升级为OpenAI版本,需要API Key

##  进阶使用

### 在Notebook中使用

```python
from src.agent.career_advisor_agent import CareerAdvisorAgent

agent = CareerAdvisorAgent()
response = agent.chat("我会Python和SQL,想知道能拿多少薪水?")
print(response)
```

### 自定义配置

编辑 `config.yaml` 修改模型参数、训练配置等。

##  课程演示建议

1. **演示Agent对话** - 展示不同场景的交互
2. **展示模型性能** - 显示训练结果和评估指标
3. **代码结构讲解** - 说明模块设计和技术选型
4. **实际应用价值** - 讨论职业咨询的实用性

##  问题反馈

如有问题,请检查:
1. Python版本是否 >= 3.8
2. 所有依赖是否安装
3. 数据文件是否存在
4. 模型是否已训练

祝使用愉快! 
