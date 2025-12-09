import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
DATA_DIR = Path("F:/ElectricityPriceForecast/data/opsd-conventional_power_plants-2020-10-01")
DE_FILE = DATA_DIR / "conventional_power_plants_DE.csv"
EU_FILE = DATA_DIR / "conventional_power_plants_EU.csv"

def load_data():
    """加载发电厂数据"""
    print("正在加载数据...")
    
    # 加载德国数据 - 使用正则表达式分隔符，跳过有问题的行，指定编码
    try:
        df_de = pd.read_csv(DE_FILE, sep=r',\s+', engine='python', 
                            on_bad_lines='skip', encoding='utf-8')
    except:
        # 备用方法：尝试不同的编码
        df_de = pd.read_csv(DE_FILE, sep=r',\s+', engine='python', 
                            on_bad_lines='skip', encoding='latin1')
    
    # 清理列名（去除首尾空格）
    df_de.columns = df_de.columns.str.strip()
    
    df_de['country'] = 'DE'
    # 德国数据使用capacity_net_bnetza作为容量，重命名为capacity
    if 'capacity_net_bnetza' in df_de.columns:
        df_de['capacity'] = pd.to_numeric(df_de['capacity_net_bnetza'], errors='coerce')
    
    # 加载欧盟数据
    try:
        df_eu = pd.read_csv(EU_FILE, sep=r',\s+', engine='python', 
                            on_bad_lines='skip', encoding='utf-8')
    except:
        df_eu = pd.read_csv(EU_FILE, sep=r',\s+', engine='python', 
                            on_bad_lines='skip', encoding='latin1')
    
    # 清理列名（去除首尾空格）
    df_eu.columns = df_eu.columns.str.strip()
    
    # 合并数据
    df_all = pd.concat([df_de, df_eu], ignore_index=True)
    
    # 清理数据 - 确保capacity列存在
    if 'capacity' not in df_all.columns:
        print("警告: 'capacity'列不存在，尝试使用其他容量列")
        # 尝试其他可能的容量列名
        possible_cols = ['capacity_net_bnetza', 'capacity_gross_uba', 'capacity']
        for col in possible_cols:
            if col in df_all.columns:
                df_all['capacity'] = pd.to_numeric(df_all[col], errors='coerce')
                print(f"使用列 '{col}' 作为容量数据")
                break
    else:
        df_all['capacity'] = pd.to_numeric(df_all['capacity'], errors='coerce')
    
    df_all = df_all.dropna(subset=['capacity'])
    
    print(f"加载完成！共 {len(df_all)} 个发电厂记录")
    print(f"国家数量: {df_all['country'].nunique()}")
    print(f"总装机容量: {df_all['capacity'].sum():.2f} MW")
    
    return df_all, df_de, df_eu

def plot_energy_source_distribution(df):
    """按能源类型分布的装机容量"""
    print("\n生成能源类型分布图...")
    
    # 按能源类型汇总
    energy_stats = df.groupby('energy_source')['capacity'].agg(['sum', 'count']).reset_index()
    energy_stats = energy_stats.sort_values('sum', ascending=False)
    
    # Matplotlib版本
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 装机容量饼图
    top_10_energy = energy_stats.head(10)
    ax1.pie(top_10_energy['sum'], labels=top_10_energy['energy_source'], 
            autopct='%1.1f%%', startangle=90)
    ax1.set_title('Top 10 能源类型装机容量分布 (MW)', fontsize=14, fontweight='bold')
    
    # 发电厂数量柱状图
    bars = ax2.bar(range(len(top_10_energy)), top_10_energy['count'])
    ax2.set_xticks(range(len(top_10_energy)))
    ax2.set_xticklabels(top_10_energy['energy_source'], rotation=45, ha='right')
    ax2.set_title('Top 10 能源类型发电厂数量', fontsize=14, fontweight='bold')
    ax2.set_ylabel('发电厂数量')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('output/energy_source_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly交互式版本
    fig_plotly = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=('装机容量分布 (MW)', '发电厂数量')
    )
    
    fig_plotly.add_trace(
        go.Pie(labels=top_10_energy['energy_source'], 
               values=top_10_energy['sum'],
               name="装机容量"),
        row=1, col=1
    )
    
    fig_plotly.add_trace(
        go.Bar(x=top_10_energy['energy_source'], 
               y=top_10_energy['count'],
               name="发电厂数量"),
        row=1, col=2
    )
    
    fig_plotly.update_layout(
        title_text="能源类型分布分析",
        title_x=0.5,
        showlegend=False,
        height=600
    )
    
    fig_plotly.write_html("output/energy_source_distribution_interactive.html")
    print("能源类型分布图已保存到 output/ 目录")

def plot_country_distribution(df):
    """按国家分布的装机容量"""
    print("\n生成国家分布图...")
    
    # 按国家汇总
    country_stats = df.groupby('country')['capacity'].agg(['sum', 'count']).reset_index()
    country_stats = country_stats.sort_values('sum', ascending=False)
    
    # Matplotlib版本
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars = ax.bar(country_stats['country'], country_stats['sum'], 
                  color=plt.cm.Set3(np.linspace(0, 1, len(country_stats))))
    ax.set_title('各国装机容量分布 (MW)', fontsize=16, fontweight='bold')
    ax.set_xlabel('国家', fontsize=12)
    ax.set_ylabel('装机容量 (MW)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('output/country_capacity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly交互式版本
    fig_plotly = px.bar(
        country_stats, 
        x='country', 
        y='sum',
        color='sum',
        title='各国装机容量分布 (MW)',
        labels={'sum': '装机容量 (MW)', 'country': '国家'},
        color_continuous_scale='Viridis'
    )
    
    fig_plotly.update_layout(
        title_x=0.5,
        xaxis_tickangle=-45,
        height=600
    )
    
    fig_plotly.write_html("output/country_capacity_distribution_interactive.html")
    print("国家分布图已保存到 output/ 目录")

def plot_geographic_distribution(df):
    """发电厂地理分布"""
    print("\n生成地理分布图...")
    
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
    
    # 确定hover_name列
    hover_name_col = 'company'
    if 'name' in df_geo.columns:
        hover_name_col = 'name'
    elif 'name_bnetza' in df_geo.columns:
        hover_name_col = 'name_bnetza'
    
    # 按国家分别绘制 - 使用scatter_map替代已弃用的scatter_mapbox
    fig = px.scatter_map(
        df_geo,
        lat="lat",
        lon="lon",
        color="energy_source",
        size="capacity",
        hover_name=hover_name_col,
        hover_data={"capacity": True, "country": True, "technology": True},
        color_discrete_sequence=px.colors.qualitative.Set2,
        zoom=3,
        height=800,
        title="发电厂地理分布地图"
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
    
    fig.write_html("output/geographic_distribution_map.html")
    print("地理分布图已保存到 output/ 目录")

def plot_temporal_trends(df):
    """按投产年份的趋势分析"""
    print("\n生成时间趋势图...")
    
    # 清理投产年份数据
    df_temporal = df.dropna(subset=['commissioned']).copy()
    df_temporal['commissioned'] = pd.to_numeric(df_temporal['commissioned'], 
                                                errors='coerce')
    df_temporal = df_temporal.dropna(subset=['commissioned'])
    
    # 只保留合理的年份（1900-2025）
    df_temporal = df_temporal[
        (df_temporal['commissioned'] >= 1900) & 
        (df_temporal['commissioned'] <= 2025)
    ]
    
    if len(df_temporal) == 0:
        print("警告: 没有可用的投产年份数据")
        return
    
    # 按年份和能源类型汇总
    yearly_stats = df_temporal.groupby(['commissioned', 'energy_source'])['capacity'].sum().reset_index()
    
    # 获取前5大能源类型
    top_energy_sources = df_temporal.groupby('energy_source')['capacity'].sum().nlargest(5).index
    
    # Matplotlib版本
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for energy in top_energy_sources:
        data = yearly_stats[yearly_stats['energy_source'] == energy]
        if not data.empty:
            ax.plot(data['commissioned'], data['capacity'], 
                   marker='o', label=energy, linewidth=2, markersize=4)
    
    ax.set_title('不同能源类型装机容量投产年份趋势', fontsize=16, fontweight='bold')
    ax.set_xlabel('投产年份', fontsize=12)
    ax.set_ylabel('装机容量 (MW)', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/temporal_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly交互式版本
    fig_plotly = px.line(
        yearly_stats[yearly_stats['energy_source'].isin(top_energy_sources)],
        x='commissioned',
        y='capacity',
        color='energy_source',
        title='不同能源类型装机容量投产年份趋势',
        labels={
            'commissioned': '投产年份',
            'capacity': '装机容量 (MW)',
            'energy_source': '能源类型'
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
    
    fig_plotly.write_html("output/temporal_trends_interactive.html")
    print("时间趋势图已保存到 output/ 目录")

def plot_technology_distribution(df):
    """技术类型分布"""
    print("\n生成技术类型分布图...")
    
    # 按技术类型汇总
    tech_stats = df.groupby('technology')['capacity'].agg(['sum', 'count']).reset_index()
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
    ax1.set_title('Top 10 技术类型装机容量 (MW)', fontsize=14, fontweight='bold')
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
    ax2.set_title('Top 10 技术类型发电厂数量', fontsize=14, fontweight='bold')
    ax2.set_xlabel('发电厂数量')
    
    # 添加数值标签
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{int(width)}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('output/technology_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("技术类型分布图已保存到 output/ 目录")

def generate_summary_report(df):
    """生成数据摘要报告"""
    print("\n生成数据摘要报告...")
    
    report = f"""
# 传统发电厂数据分析报告

## 数据概览
- **总记录数**: {len(df):,}
- **国家数量**: {df['country'].nunique()}
- **总装机容量**: {df['capacity'].sum():,.2f} MW
- **平均装机容量**: {df['capacity'].mean():.2f} MW
- **最大装机容量**: {df['capacity'].max():.2f} MW
- **最小装机容量**: {df['capacity'].min():.2f} MW

## 按国家统计
{df.groupby('country')['capacity'].agg(['count', 'sum', 'mean']).round(2).to_string()}

## 按能源类型统计
{df.groupby('energy_source')['capacity'].agg(['count', 'sum']).round(2).to_string()}

## 数据完整性
- 有坐标数据: {df[['lat', 'lon']].notna().all(axis=1).sum():,} ({df[['lat', 'lon']].notna().all(axis=1).mean()*100:.1f}%)
- 有投产年份: {df['commissioned'].notna().sum():,} ({df['commissioned'].notna().mean()*100:.1f}%)
- 有技术类型: {df['technology'].notna().sum():,} ({df['technology'].notna().mean()*100:.1f}%)
"""
    
    with open('output/analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("数据摘要报告已保存到 output/analysis_report.md")

def main():
    """主函数"""
    # 创建输出目录
    import os
    os.makedirs('output', exist_ok=True)
    
    # 加载数据
    df_all, df_de, df_eu = load_data()
    
    # 生成各种可视化
    plot_energy_source_distribution(df_all)
    plot_country_distribution(df_all)
    plot_geographic_distribution(df_all)
    plot_temporal_trends(df_all)
    plot_technology_distribution(df_all)
    
    # 生成摘要报告
    generate_summary_report(df_all)
    
    print("\n" + "="*50)
    print("所有可视化图表已生成完成！")
    print("请查看 output/ 目录下的文件：")
    print("- PNG格式的静态图表")
    print("- HTML格式的交互式图表")
    print("- Markdown格式的分析报告")
    print("="*50)

if __name__ == "__main__":
    main()
