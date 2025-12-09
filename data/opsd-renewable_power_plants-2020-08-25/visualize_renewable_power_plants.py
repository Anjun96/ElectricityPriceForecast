import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
DATA_DIR = Path("F:/ElectricityPriceForecast/data/opsd-renewable_power_plants-2020-08-25")

# 可用的国家数据文件
COUNTRY_FILES = {
    'DE': "renewable_power_plants_DE.csv",
    'DK': "renewable_power_plants_DK.csv", 
    'FR': "renewable_power_plants_FR.csv",
    'PL': "renewable_power_plants_PL.csv",
    'UK': "renewable_power_plants_UK.csv",
    'CH': "renewable_power_plants_CH.csv",
    'SE': "renewable_power_plants_SE.csv",
    'CZ': "renewable_power_plants_CZ.csv"
}

def load_data():
    """加载可再生能源发电厂数据"""
    print("正在加载数据...")
    
    all_data = []
    
    for country_code, filename in COUNTRY_FILES.items():
        file_path = DATA_DIR / filename
        if not file_path.exists():
            print(f"警告: 文件 {filename} 不存在，跳过")
            continue
            
        try:
            print(f"正在加载 {country_code} 数据...")
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            df['country'] = country_code
            all_data.append(df)
            print(f"{country_code} 数据加载完成: {len(df)} 条记录")
        except Exception as e:
            print(f"加载 {country_code} 数据时出错: {e}")
            continue
    
    if not all_data:
        raise ValueError("没有成功加载任何数据")
    
    # 合并所有数据
    df_all = pd.concat(all_data, ignore_index=True)
    
    # 清理数据 - 确保electrical_capacity列存在且为数值类型
    if 'electrical_capacity' not in df_all.columns:
        raise ValueError("数据中缺少 'electrical_capacity' 列")
    
    df_all['electrical_capacity'] = pd.to_numeric(df_all['electrical_capacity'], errors='coerce')
    df_all = df_all.dropna(subset=['electrical_capacity'])
    
    # 过滤掉容量为0或负值的记录
    df_all = df_all[df_all['electrical_capacity'] > 0]
    
    print(f"数据加载完成！")
    print(f"- 总记录数: {len(df_all):,}")
    print(f"- 国家数量: {df_all['country'].nunique()}")
    print(f"- 总装机容量: {df_all['electrical_capacity'].sum():,.2f} MW")
    print(f"- 平均装机容量: {df_all['electrical_capacity'].mean():.2f} MW")
    
    return df_all

def plot_energy_source_distribution(df):
    """按能源类型分布的装机容量"""
    print("\n生成能源类型分布图...")
    
    # 使用energy_source_level_2作为主要能源类型分类
    energy_col = 'energy_source_level_2'
    if energy_col not in df.columns:
        print(f"警告: 列 '{energy_col}' 不存在")
        return
    
    # 按能源类型汇总
    energy_stats = df.groupby(energy_col)['electrical_capacity'].agg(['sum', 'count']).reset_index()
    energy_stats = energy_stats.sort_values('sum', ascending=False)
    
    # Matplotlib版本
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 装机容量饼图
    ax1.pie(energy_stats['sum'], labels=energy_stats[energy_col], 
            autopct='%1.1f%%', startangle=90)
    ax1.set_title('可再生能源类型装机容量分布 (MW)', fontsize=14, fontweight='bold')
    
    # 发电厂数量柱状图
    bars = ax2.bar(range(len(energy_stats)), energy_stats['count'])
    ax2.set_xticks(range(len(energy_stats)))
    ax2.set_xticklabels(energy_stats[energy_col], rotation=45, ha='right')
    ax2.set_title('可再生能源类型发电厂数量', fontsize=14, fontweight='bold')
    ax2.set_ylabel('发电厂数量')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('renewable_energy_source_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly交互式版本
    fig_plotly = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=('装机容量分布 (MW)', '发电厂数量')
    )
    
    fig_plotly.add_trace(
        go.Pie(labels=energy_stats[energy_col], 
               values=energy_stats['sum'],
               name="装机容量"),
        row=1, col=1
    )
    
    fig_plotly.add_trace(
        go.Bar(x=energy_stats[energy_col], 
               y=energy_stats['count'],
               name="发电厂数量"),
        row=1, col=2
    )
    
    fig_plotly.update_layout(
        title_text="可再生能源类型分布分析",
        title_x=0.5,
        showlegend=False,
        height=600
    )
    
    fig_plotly.write_html("renewable_energy_source_distribution_interactive.html")
    print("能源类型分布图已保存到 output/ 目录")

def plot_country_distribution(df):
    """按国家分布的装机容量"""
    print("\n生成国家分布图...")
    
    # 按国家汇总
    country_stats = df.groupby('country')['electrical_capacity'].agg(['sum', 'count']).reset_index()
    country_stats = country_stats.sort_values('sum', ascending=False)
    
    # 添加国家全名
    country_names = {
        'DE': '德国', 'DK': '丹麦', 'FR': '法国', 'PL': '波兰',
        'UK': '英国', 'CH': '瑞士', 'SE': '瑞典', 'CZ': '捷克'
    }
    country_stats['country_name'] = country_stats['country'].map(country_names)
    
    # Matplotlib版本
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars = ax.bar(country_stats['country_name'], country_stats['sum'], 
                  color=plt.cm.Set3(np.linspace(0, 1, len(country_stats))))
    ax.set_title('各国可再生能源装机容量分布 (MW)', fontsize=16, fontweight='bold')
    ax.set_xlabel('国家', fontsize=12)
    ax.set_ylabel('装机容量 (MW)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('renewable_country_capacity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly交互式版本
    fig_plotly = px.bar(
        country_stats, 
        x='country_name', 
        y='sum',
        color='sum',
        title='各国可再生能源装机容量分布 (MW)',
        labels={'sum': '装机容量 (MW)', 'country_name': '国家'},
        color_continuous_scale='Viridis'
    )
    
    fig_plotly.update_layout(
        title_x=0.5,
        xaxis_tickangle=-45,
        height=600
    )
    
    fig_plotly.write_html("renewable_country_capacity_distribution_interactive.html")
    print("国家分布图已保存到 output/ 目录")

def plot_geographic_distribution(df):
    """发电厂地理分布"""
    print("\n生成地理分布图...")
    
    # 检查是否有坐标数据
    if 'lat' not in df.columns or 'lon' not in df.columns:
        print("警告: 数据中缺少经纬度坐标")
        return
    
    # 过滤有坐标的数据
    df_geo = df.dropna(subset=['lat', 'lon'])
    
    if len(df_geo) == 0:
        print("警告: 没有可用的地理坐标数据")
        return
    
    # 确保lat和lon是数值类型
    df_geo['lat'] = pd.to_numeric(df_geo['lat'], errors='coerce')
    df_geo['lon'] = pd.to_numeric(df_geo['lon'], errors='coerce')
    df_geo = df_geo.dropna(subset=['lat', 'lon'])
    
    if len(df_geo) == 0:
        print("警告: 地理坐标数据转换失败")
        return
    
    # 确定能源类型列
    energy_col = 'energy_source_level_2'
    if energy_col not in df_geo.columns:
        energy_col = 'energy_source_level_1'
    
    # 确定hover_name列
    hover_name_col = 'municipality'
    if 'site_name' in df_geo.columns:
        hover_name_col = 'site_name'
    
    # 按国家分别绘制
    fig = px.scatter_map(
        df_geo,
        lat="lat",
        lon="lon",
        color=energy_col,
        size="electrical_capacity",
        hover_name=hover_name_col,
        hover_data={
            "electrical_capacity": True, 
            "country": True, 
            "technology": True,
            "commissioning_date": True
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
        zoom=3,
        height=800,
        title="可再生能源发电厂地理分布地图"
    )
    
    fig.update_layout(
        title_x=0.5,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    fig.write_html("renewable_geographic_distribution_map.html")
    print("地理分布图已保存到 output/ 目录")

def plot_temporal_trends(df):
    """按投产年份的趋势分析"""
    print("\n生成时间趋势图...")
    
    # 检查是否有投产日期数据
    date_col = 'commissioning_date'
    if date_col not in df.columns:
        print("警告: 数据中缺少投产日期信息")
        return
    
    # 清理投产年份数据
    df_temporal = df.dropna(subset=[date_col]).copy()
    df_temporal.loc[:, 'year'] = pd.to_datetime(df_temporal[date_col], errors='coerce').dt.year
    df_temporal = df_temporal.dropna(subset=['year'])
    
    # 只保留合理的年份（1990-2025）
    df_temporal = df_temporal[
        (df_temporal['year'] >= 1990) & 
        (df_temporal['year'] <= 2025)
    ]
    
    if len(df_temporal) == 0:
        print("警告: 没有可用的投产年份数据")
        return
    
    # 确定能源类型列
    energy_col = 'energy_source_level_2'
    if energy_col not in df_temporal.columns:
        energy_col = 'energy_source_level_1'
    
    # 按年份和能源类型汇总
    yearly_stats = df_temporal.groupby(['year', energy_col])['electrical_capacity'].sum().reset_index()
    
    # 获取前5大能源类型
    top_energy_sources = df_temporal.groupby(energy_col)['electrical_capacity'].sum().nlargest(5).index
    
    # Matplotlib版本
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for energy in top_energy_sources:
        data = yearly_stats[yearly_stats[energy_col] == energy]
        if not data.empty:
            ax.plot(data['year'], data['electrical_capacity'], 
                   marker='o', label=energy, linewidth=2, markersize=4)
    
    ax.set_title('不同可再生能源类型装机容量投产年份趋势', fontsize=16, fontweight='bold')
    ax.set_xlabel('投产年份', fontsize=12)
    ax.set_ylabel('装机容量 (MW)', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('renewable_temporal_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly交互式版本
    fig_plotly = px.line(
        yearly_stats[yearly_stats[energy_col].isin(top_energy_sources)],
        x='year',
        y='electrical_capacity',
        color=energy_col,
        title='不同可再生能源类型装机容量投产年份趋势',
        labels={
            'year': '投产年份',
            'electrical_capacity': '装机容量 (MW)',
            energy_col: '能源类型'
        }
    )
    
    fig_plotly.update_layout(
        title_x=0.5,
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    fig_plotly.write_html("renewable_temporal_trends_interactive.html")
    print("时间趋势图已保存到 output/ 目录")

def plot_technology_distribution(df):
    """技术类型分布"""
    print("\n生成技术类型分布图...")
    
    # 检查是否有技术类型数据
    if 'technology' not in df.columns:
        print("警告: 数据中缺少技术类型信息")
        return
    
    # 按技术类型汇总
    tech_stats = df.groupby('technology')['electrical_capacity'].agg(['sum', 'count']).reset_index()
    tech_stats = tech_stats.sort_values('sum', ascending=False)
    
    # 只显示前10个主要技术类型
    top_10_tech = tech_stats.head(10)
    
    # Matplotlib版本
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # 装机容量柱状图
    bars1 = ax1.barh(range(len(top_10_tech)), top_10_tech['sum'], 
                     color=plt.cm.viridis(np.linspace(0, 1, len(top_10_tech))))
    ax1.set_yticks(range(len(top_10_tech)))
    ax1.set_yticklabels(top_10_tech['technology'])
    ax1.set_title('Top 10 可再生能源技术类型装机容量 (MW)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('装机容量 (MW)')
    
    # 添加数值标签
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.0f}', ha='left', va='center', fontsize=9)
    
    # 发电厂数量柱状图
    bars2 = ax2.barh(range(len(top_10_tech)), top_10_tech['count'], 
                     color=plt.cm.plasma(np.linspace(0, 1, len(top_10_tech))))
    ax2.set_yticks(range(len(top_10_tech)))
    ax2.set_yticklabels(top_10_tech['technology'])
    ax2.set_title('Top 10 可再生能源技术类型发电厂数量', fontsize=14, fontweight='bold')
    ax2.set_xlabel('发电厂数量')
    
    # 添加数值标签
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{int(width)}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('renewable_technology_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("技术类型分布图已保存到 output/ 目录")

def plot_capacity_distribution(df):
    """装机容量分布分析"""
    print("\n生成装机容量分布图...")
    
    # 创建容量区间
    df = df.copy()
    df['capacity_category'] = pd.cut(
        df['electrical_capacity'],
        bins=[0, 0.1, 0.5, 1, 5, 10, 50, 100, float('inf')],
        labels=['<0.1MW', '0.1-0.5MW', '0.5-1MW', '1-5MW', '5-10MW', '10-50MW', '50-100MW', '>100MW']
    )
    
    # 按容量区间和国家汇总
    capacity_by_country = df.groupby(['country', 'capacity_category'])['electrical_capacity'].sum().unstack(fill_value=0)
    
    # Matplotlib版本 - 堆叠柱状图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    capacity_by_country.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_title('各国可再生能源装机容量分布', fontsize=16, fontweight='bold')
    ax.set_xlabel('国家', fontsize=12)
    ax.set_ylabel('装机容量 (MW)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='容量区间', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('renewable_capacity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("装机容量分布图已保存到 output/ 目录")

def generate_summary_report(df):
    """生成数据摘要报告"""
    print("\n生成数据摘要报告...")
    
    # 确定能源类型列
    energy_col = 'energy_source_level_2'
    if energy_col not in df.columns:
        energy_col = 'energy_source_level_1'
    
    # 国家名称映射
    country_names = {
        'DE': '德国', 'DK': '丹麦', 'FR': '法国', 'PL': '波兰',
        'UK': '英国', 'CH': '瑞士', 'SE': '瑞典', 'CZ': '捷克'
    }
    
    report = f"""
# 可再生能源发电厂数据分析报告

## 数据概览
- **总记录数**: {len(df):,}
- **国家数量**: {df['country'].nunique()}
- **总装机容量**: {df['electrical_capacity'].sum():,.2f} MW
- **平均装机容量**: {df['electrical_capacity'].mean():.2f} MW
- **最大装机容量**: {df['electrical_capacity'].max():.2f} MW
- **最小装机容量**: {df['electrical_capacity'].min():.2f} MW

## 按国家统计
{df.groupby('country')['electrical_capacity'].agg(['count', 'sum', 'mean']).round(2).to_string()}

## 按能源类型统计
{df.groupby(energy_col)['electrical_capacity'].agg(['count', 'sum']).round(2).to_string()}

## 数据完整性
- 有坐标数据: {df[['lat', 'lon']].notna().all(axis=1).sum():,} ({df[['lat', 'lon']].notna().all(axis=1).mean()*100:.1f}%)
- 有投产日期: {df['commissioning_date'].notna().sum():,} ({df['commissioning_date'].notna().mean()*100:.1f}%)
- 有技术类型: {df['technology'].notna().sum():,} ({df['technology'].notna().mean()*100:.1f}%)
"""
    
    with open('renewable_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("数据摘要报告已保存到 output/renewable_analysis_report.md")

def main():
    """主函数"""
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    # 加载数据
    df_all = load_data()
    
    # 生成各种可视化
    plot_energy_source_distribution(df_all)
    plot_country_distribution(df_all)
    plot_geographic_distribution(df_all)
    plot_temporal_trends(df_all)
    plot_technology_distribution(df_all)
    plot_capacity_distribution(df_all)
    
    # 生成摘要报告
    generate_summary_report(df_all)
    
    print("\n" + "="*50)
    print("所有可再生能源可视化图表已生成完成！")
    print("请查看 output/ 目录下的文件：")
    print("- PNG格式的静态图表")
    print("- HTML格式的交互式图表")
    print("- Markdown格式的分析报告")
    print("="*50)

if __name__ == "__main__":
    main()