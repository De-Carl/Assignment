"""
薪资预测模型
使用机器学习算法预测职位薪资
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from typing import Dict, List, Tuple


class SalaryPredictor:
    """薪资预测器类"""
    
    def __init__(self, model_type='random_forest'):
        """
        初始化薪资预测器
        
        Args:
            model_type: 模型类型 ('random_forest' or 'gradient_boosting')
        """
        self.model_type = model_type
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        self.feature_columns = []
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备训练特征
        
        Args:
            df: 特征工程后的数据框
            
        Returns:
            DataFrame: 仅包含模型特征的数据框
        """
        # 选择特征列
        feature_candidates = [
            # 编码后的分类特征
            'job_category_encoded', 'experience_level_encoded', 
            'employment_type_encoded', 'industry_encoded', 
            'company_size_encoded', 'region_encoded',
            # 数值特征
            'num_skills_required', 'num_tools_preferred', 'total_tech_count',
            'posted_year', 'posted_quarter'
        ]
        
        # 添加技能特征 (skill_xxx)
        skill_cols = [col for col in df.columns if col.startswith('skill_')]
        feature_candidates.extend(skill_cols)
        
        # 添加工具特征 (tool_xxx)
        tool_cols = [col for col in df.columns if col.startswith('tool_')]
        feature_candidates.extend(tool_cols)
        
        # 只保留存在的列
        self.feature_columns = [col for col in feature_candidates if col in df.columns]
        
        return df[self.feature_columns]
    
    def train(self, df: pd.DataFrame, target_col='salary_avg', test_size=0.2):
        """
        训练模型
        
        Args:
            df: 特征工程后的数据框
            target_col: 目标列名
            test_size: 测试集比例
            
        Returns:
            dict: 训练结果和评估指标
        """
        print(f"\n开始训练 {self.model_type} 模型...")
        
        # 准备特征和目标
        X = self.prepare_features(df)
        y = df[target_col]
        
        # 移除缺失值
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        print(f"训练数据形状: X={X.shape}, y={y.shape}")
        print(f"特征数量: {len(self.feature_columns)}")
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # 预测
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # 评估指标
        results = {
            'train': {
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'mae': mean_absolute_error(y_train, y_train_pred),
                'r2': r2_score(y_train, y_train_pred)
            },
            'test': {
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'mae': mean_absolute_error(y_test, y_test_pred),
                'r2': r2_score(y_test, y_test_pred)
            }
        }
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X_train, y_train, 
                                    cv=5, scoring='r2', n_jobs=-1)
        results['cv_r2_mean'] = cv_scores.mean()
        results['cv_r2_std'] = cv_scores.std()
        
        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = sorted(
                zip(self.feature_columns, importances),
                key=lambda x: x[1], reverse=True
            )
            results['feature_importance'] = feature_importance[:20]
        
        self._print_results(results)
        
        return results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        预测薪资
        
        Args:
            df: 特征工程后的数据框
            
        Returns:
            array: 预测的薪资值
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练,请先调用 train() 方法")
        
        X = df[self.feature_columns]
        return self.model.predict(X)
    
    def predict_single(self, features: Dict) -> Dict:
        """
        预测单个样本的薪资
        
        Args:
            features: 特征字典
            
        Returns:
            dict: 预测结果,包含薪资预测值和置信区间
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 创建数据框
        df = pd.DataFrame([features])
        
        # 确保所有特征列都存在
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        X = df[self.feature_columns]
        prediction = self.model.predict(X)[0]
        
        # 计算置信区间 (使用随机森林的树预测标准差)
        if self.model_type == 'random_forest':
            # 获取所有树的预测
            tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            std = tree_predictions.std()
            
            result = {
                'predicted_salary': round(prediction, 2),
                'confidence_interval_lower': round(prediction - 1.96 * std, 2),
                'confidence_interval_upper': round(prediction + 1.96 * std, 2),
                'std': round(std, 2)
            }
        else:
            # 对于梯度提升,使用简单的百分比区间
            result = {
                'predicted_salary': round(prediction, 2),
                'confidence_interval_lower': round(prediction * 0.85, 2),
                'confidence_interval_upper': round(prediction * 1.15, 2)
            }
        
        return result
    
    def _print_results(self, results: Dict):
        """打印训练结果"""
        print("\n" + "="*50)
        print("模型训练完成!")
        print("="*50)
        
        print("\n训练集性能:")
        print(f"  RMSE: ${results['train']['rmse']:,.2f}")
        print(f"  MAE:  ${results['train']['mae']:,.2f}")
        print(f"  R²:   {results['train']['r2']:.4f}")
        
        print("\n测试集性能:")
        print(f"  RMSE: ${results['test']['rmse']:,.2f}")
        print(f"  MAE:  ${results['test']['mae']:,.2f}")
        print(f"  R²:   {results['test']['r2']:.4f}")
        
        print(f"\n交叉验证 R²: {results['cv_r2_mean']:.4f} (+/- {results['cv_r2_std']:.4f})")
        
        if 'feature_importance' in results:
            print("\n前10个重要特征:")
            for i, (feat, imp) in enumerate(results['feature_importance'][:10], 1):
                print(f"  {i}. {feat}: {imp:.4f}")
    
    def save(self, filepath: str):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }, f)
        print(f"\n模型已保存到: {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.model_type = data['model_type']
            self.feature_columns = data['feature_columns']
            self.is_trained = data['is_trained']
        print(f"模型已从 {filepath} 加载")


if __name__ == "__main__":
    # 测试代码
    from data_processing.data_loader import DataLoader
    from data_processing.feature_engineering import FeatureEngineer
    
    # 加载数据
    loader = DataLoader("../../data/raw/ai_job_market_cleaned.csv")
    df = loader.load_data()
    df = loader.parse_list_columns()
    
    # 特征工程
    engineer = FeatureEngineer()
    df_featured = engineer.fit_transform(df)
    
    # 训练模型
    predictor = SalaryPredictor(model_type='random_forest')
    results = predictor.train(df_featured)
    
    # 测试预测
    sample_features = df_featured.iloc[0].to_dict()
    prediction = predictor.predict_single(sample_features)
    print(f"\n预测薪资: ${prediction['predicted_salary']:,.2f}")
    print(f"置信区间: ${prediction['confidence_interval_lower']:,.2f} - ${prediction['confidence_interval_upper']:,.2f}")