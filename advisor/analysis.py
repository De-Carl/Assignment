import pandas as pd

# 读取数据集
data_path = r'd:\大学\大二上\数据科学编程\项目\Assignment\dataset\ai_job_market_unified.csv'
df = pd.read_csv(data_path)

# 统计每个工作岗位的数据记录数
job_title_counts = df['job_title'].value_counts()

# 按照数量降序排列
job_title_counts_sorted = job_title_counts.sort_values(ascending=False)

print("=" * 80)
print("每个工作岗位的数据记录数量统计")
print("=" * 80)
print(f"\n总共有 {len(job_title_counts)} 个不同的工作岗位")
print(f"总数据记录数: {len(df)} 条\n")

print("-" * 80)
print(f"{'工作岗位':<40} {'数据记录数':>15} {'占比':>10}")
print("-" * 80)

for job_title, count in job_title_counts_sorted.items():
    percentage = (count / len(df)) * 100
    print(f"{job_title:<40} {count:>15} {percentage:>9.2f}%")

print("-" * 80)
print(f"{'总计':<40} {len(df):>15} {100.0:>9.2f}%")
print("=" * 80)

# 保存结果到CSV文件
output_path = r'd:\大学\大二上\数据科学编程\项目\Assignment\dataset\job_title_statistics.csv'
job_title_stats_df = pd.DataFrame({
    '工作岗位': job_title_counts_sorted.index,
    '数据记录数': job_title_counts_sorted.values,
    '占比(%)': (job_title_counts_sorted.values / len(df) * 100).round(2)
})
job_title_stats_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n统计结果已保存到: {output_path}")
