import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from datetime import datetime

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
DATA_DIR = Path(__file__).parent
TIME_SERIES_FILE = DATA_DIR / "time_series_60min_singleindex.csv"
OUTPUT_DIR = DATA_DIR / "output"

def load_data():
    """加载时间序列数据"""
    print("正在加载时间序列数据...")
    
    # 由于数据量大，使用chunk方式加载
    try:
        # 先读取前几行确定列名
        df_sample = pd.read_csv(TIME_SERIES_FILE, nrows=5)
        columns = df_sample.columns
        
        # 读取完整数据
        df = pd.read_csv(TIME_SERIES_FILE, parse_dates=['utc_timestamp', 'cet_cest_timestamp'])
        
        print(f"数据加载完成！共 {len(df)} 条记录")
        print(f"时间范围: {df['utc_timestamp'].min()} 至 {df['utc_timestamp'].max()}")
        print(f"列数: {len(df.columns)}")
        
        return df
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def plot_load_comparison(df, countries=['DE', 'FR', 'GB', 'IT', 'ES']):
    """比较不同国家的电力负荷"""
    print("\n生成电力负荷比较图...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for country in countries:
        load_col = f"{country}_load_actual_entsoe_transparency"
        if load_col in df.columns:
            # 按日聚合数据以减少绘图点数
            daily_data = df[['utc_timestamp', load_col]].copy()
            daily_data['date'] = daily_data['utc_timestamp'].dt.date
            daily_avg = daily_data.groupby('date')[load_col].mean()
            
            ax.plot(daily_avg.index, daily_avg.values, label=country, linewidth=2)
    
    ax.set_title('主要国家日均电力负荷比较 (MW)', fontsize=16, fontweight='bold')
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('电力负荷 (MW)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 设置x轴日期格式
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'load_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly交互式版本
    fig_plotly = go.Figure()
    
    for country in countries:
        load_col = f"{country}_load_actual_entsoe_transparency"
        if load_col in df.columns:
            # 使用月平均数据减少数据点
            monthly_data = df[['utc_timestamp', load_col]].copy()
            monthly_data['month'] = monthly_data['utc_timestamp'].dt.to_period('M')
            monthly_avg = monthly_data.groupby('month')[load_col].mean()
            
            fig_plotly.add_trace(go.Scatter(
                x=monthly_avg.index.astype(str),
                y=monthly_avg.values,
                mode='lines+markers',
                name=country,
                line=dict(width=2)
            ))
    
    fig_plotly.update_layout(
        title='主要国家月均电力负荷比较 (MW)',
        xaxis_title='月份',
        yaxis_title='电力负荷 (MW)',
        hovermode='x unified',
        height=600
    )
    
    fig_plotly.write_html(OUTPUT_DIR / "load_comparison_interactive.html")
    print(f"电力负荷比较图已保存到 {OUTPUT_DIR} 目录")

def plot_renewable_generation(df, country='DE'):
    """可视化可再生能源发电量"""
    print(f"\n生成{country}可再生能源发电量图...")
    
    # 检查相关列是否存在
    solar_col = f"{country}_solar_generation_actual"
    wind_col = f"{country}_wind_generation_actual"
    wind_offshore_col = f"{country}_wind_offshore_generation_actual"
    wind_onshore_col = f"{country}_wind_onshore_generation_actual"
    
    available_cols = []
    if solar_col in df.columns:
        available_cols.append(('太阳能', solar_col))
    if wind_col in df.columns:
        available_cols.append(('风能', wind_col))
    if wind_offshore_col in df.columns:
        available_cols.append(('海上风能', wind_offshore_col))
    if wind_onshore_col in df.columns:
        available_cols.append(('陆上风能', wind_onshore_col))
    
    if not available_cols:
        print(f"警告: 没有找到{country}的可再生能源数据列")
        return
    
    # Matplotlib版本 - 使用周平均数据
    fig, ax = plt.subplots(figsize=(16, 8))
    
    df_weekly = df[['utc_timestamp']].copy()
    df_weekly['week'] = df_weekly['utc_timestamp'].dt.to_period('W')
    
    for name, col in available_cols:
        weekly_data = df[['utc_timestamp', col]].copy()
        weekly_data['week'] = weekly_data['utc_timestamp'].dt.to_period('W')
        weekly_avg = weekly_data.groupby('week')[col].mean()
        
        ax.plot(weekly_avg.index.astype(str), weekly_avg.values, 
                label=name, linewidth=2, marker='o', markersize=3)
    
    ax.set_title(f'{country}可再生能源周平均发电量 (MW)', fontsize=16, fontweight='bold')
    ax.set_xlabel('周', fontsize=12)
    ax.set_ylabel('发电量 (MW)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 设置x轴标签显示间隔
    step = max(1, len(available_cols[0][1]) // 20)  # 显示约20个标签
    ax.set_xticks(ax.get_xticks()[::step])
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{country}_renewable_generation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly交互式版本 - 使用月平均数据
    fig_plotly = go.Figure()
    
    for name, col in available_cols:
        monthly_data = df[['utc_timestamp', col]].copy()
        monthly_data['month'] = monthly_data['utc_timestamp'].dt.to_period('M')
        monthly_avg = monthly_data.groupby('month')[col].mean()
        
        fig_plotly.add_trace(go.Scatter(
            x=monthly_avg.index.astype(str),
            y=monthly_avg.values,
            mode='lines+markers',
            name=name,
            line=dict(width=2)
        ))
    
    fig_plotly.update_layout(
        title=f'{country}可再生能源月平均发电量 (MW)',
        xaxis_title='月份',
        yaxis_title='发电量 (MW)',
        hovermode='x unified',
        height=600
    )
    
    fig_plotly.write_html(OUTPUT_DIR / f"{country}_renewable_generation_interactive.html")
    print(f"{country}可再生能源发电量图已保存到 {OUTPUT_DIR} 目录")

def plot_price_analysis(df, countries=['DE_LU', 'FR', 'GB_GBN']):
    """分析电价趋势"""
    print("\n生成电价分析图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, country in enumerate(countries[:4]):
        price_col = f"{country}_price_day_ahead"
        if price_col in df.columns:
            # 计算月平均电价
            monthly_data = df[['utc_timestamp', price_col]].copy()
            monthly_data['month'] = monthly_data['utc_timestamp'].dt.to_period('M')
            monthly_avg = monthly_data.groupby('month')[price_col].mean()
            
            axes[i].plot(monthly_avg.index.astype(str), monthly_avg.values, 
                        linewidth=2, marker='o', markersize=3)
            axes[i].set_title(f'{country} 月均电价 (EUR/MWh)', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('月份')
            axes[i].set_ylabel('电价 (EUR/MWh)')
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
        else:
            axes[i].text(0.5, 0.5, f'无{country}电价数据', 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    # 隐藏多余的子图
    for j in range(len(countries), 4):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'price_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly交互式版本
    fig_plotly = go.Figure()
    
    for country in countries:
        price_col = f"{country}_price_day_ahead"
        if price_col in df.columns:
            monthly_data = df[['utc_timestamp', price_col]].copy()
            monthly_data['month'] = monthly_data['utc_timestamp'].dt.to_period('M')
            monthly_avg = monthly_data.groupby('month')[price_col].mean()
            
            fig_plotly.add_trace(go.Scatter(
                x=monthly_avg.index.astype(str),
                y=monthly_avg.values,
                mode='lines+markers',
                name=country,
                line=dict(width=2)
            ))
    
    fig_plotly.update_layout(
        title='主要国家月均电价比较 (EUR/MWh)',
        xaxis_title='月份',
        yaxis_title='电价 (EUR/MWh)',
        hovermode='x unified',
        height=600
    )
    
    fig_plotly.write_html(OUTPUT_DIR / "price_analysis_interactive.html")
    print(f"电价分析图已保存到 {OUTPUT_DIR} 目录")

def plot_seasonal_patterns(df, country='DE'):
    """分析季节性模式"""
    print(f"\n生成{country}季节性模式分析图...")
    
    # 检查数据列
    load_col = f"{country}_load_actual_entsoe_transparency"
    solar_col = f"{country}_solar_generation_actual"
    wind_col = f"{country}_wind_generation_actual"
    
    if load_col not in df.columns:
        print(f"警告: 没有找到{country}的负荷数据")
        return
    
    # 准备数据
    df_seasonal = df[['utc_timestamp', load_col]].copy()
    df_seasonal['hour'] = df_seasonal['utc_timestamp'].dt.hour
    df_seasonal['month'] = df_seasonal['utc_timestamp'].dt.month
    df_seasonal['season'] = df_seasonal['month'].map({
        12: '冬季', 1: '冬季', 2: '冬季',
        3: '春季', 4: '春季', 5: '春季',
        6: '夏季', 7: '夏季', 8: '夏季',
        9: '秋季', 10: '秋季', 11: '秋季'
    })
    
    # 计算季节性平均负荷
    seasonal_hourly = df_seasonal.groupby(['season', 'hour'])[load_col].mean().unstack(level=0)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for season in ['春季', '夏季', '秋季', '冬季']:
        if season in seasonal_hourly.columns:
            ax.plot(seasonal_hourly.index, seasonal_hourly[season], 
                   label=season, linewidth=2, marker='o', markersize=4)
    
    ax.set_title(f'{country}不同季节日内负荷模式 (MW)', fontsize=16, fontweight='bold')
    ax.set_xlabel('小时', fontsize=12)
    ax.set_ylabel('平均负荷 (MW)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{country}_seasonal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly交互式版本
    fig_plotly = go.Figure()
    
    for season in ['春季', '夏季', '秋季', '冬季']:
        if season in seasonal_hourly.columns:
            fig_plotly.add_trace(go.Scatter(
                x=seasonal_hourly.index,
                y=seasonal_hourly[season],
                mode='lines+markers',
                name=season,
                line=dict(width=2)
            ))
    
    fig_plotly.update_layout(
        title=f'{country}不同季节日内负荷模式 (MW)',
        xaxis_title='小时',
        yaxis_title='平均负荷 (MW)',
        hovermode='x unified',
        height=600
    )
    
    fig_plotly.write_html(OUTPUT_DIR / f"{country}_seasonal_patterns_interactive.html")
    print(f"{country}季节性模式分析图已保存到 {OUTPUT_DIR} 目录")

def plot_correlation_matrix(df, country='DE'):
    """分析变量间的相关性"""
    print(f"\n生成{country}变量相关性矩阵...")
    
    # 收集相关变量
    variables = {}
    
    # 负荷数据
    load_col = f"{country}_load_actual_entsoe_transparency"
    load_forecast_col = f"{country}_load_forecast_entsoe_transparency"
    if load_col in df.columns:
        variables['实际负荷'] = load_col
    if load_forecast_col in df.columns:
        variables['预测负荷'] = load_forecast_col
    
    # 可再生能源数据
    solar_col = f"{country}_solar_generation_actual"
    wind_col = f"{country}_wind_generation_actual"
    if solar_col in df.columns:
        variables['太阳能发电'] = solar_col
    if wind_col in df.columns:
        variables['风能发电'] = wind_col
    
    # 电价数据
    price_col = f"{country}_price_day_ahead"
    if price_col in df.columns:
        variables['电价'] = price_col
    
    if len(variables) < 2:
        print(f"警告: {country}可用变量不足，无法计算相关性")
        return
    
    # 准备数据
    corr_data = df[list(variables.values())].copy()
    corr_data.columns = list(variables.keys())
    
    # 计算相关性矩阵
    corr_matrix = corr_data.corr()
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    ax.set_title(f'{country}电力系统变量相关性矩阵', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{country}_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly交互式版本
    fig_plotly = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig_plotly.update_layout(
        title=f'{country}电力系统变量相关性矩阵',
        width=600,
        height=600
    )
    
    fig_plotly.write_html(OUTPUT_DIR / f"{country}_correlation_matrix_interactive.html")
    print(f"{country}变量相关性矩阵已保存到 {OUTPUT_DIR} 目录")

def generate_summary_report(df):
    """生成数据摘要报告"""
    print("\n生成数据摘要报告...")
    
    # 基本统计信息
    report = f"""
# 时间序列数据分析报告

## 数据概览
- **总记录数**: {len(df):,}
- **时间范围**: {df['utc_timestamp'].min()} 至 {df['utc_timestamp'].max()}
- **时间分辨率**: 60分钟
- **列数**: {len(df.columns)}

## 可用国家/地区
"""
    
    # 提取国家代码
    countries = set()
    for col in df.columns:
        if '_' in col and not col.startswith(('utc_', 'cet_')):
            country = col.split('_')[0]
            countries.add(country)
    
    report += f"- **国家/地区数量**: {len(countries)}\n"
    report += f"- **国家/地区列表**: {', '.join(sorted(countries))}\n\n"
    
    # 数据类型统计
    load_cols = [col for col in df.columns if 'load_actual' in col]
    solar_cols = [col for col in df.columns if 'solar_generation' in col]
    wind_cols = [col for col in df.columns if 'wind_generation' in col]
    price_cols = [col for col in df.columns if 'price_day_ahead' in col]
    
    report += f"""
## 数据类型统计
- **负荷数据**: {len(load_cols)} 个国家/地区
- **太阳能发电数据**: {len(solar_cols)} 个国家/地区
- **风能发电数据**: {len(wind_cols)} 个国家/地区
- **电价数据**: {len(price_cols)} 个国家/地区

## 数据完整性
"""
    
    # 计算数据完整性
    for data_type, cols in [('负荷', load_cols), ('太阳能', solar_cols), 
                            ('风能', wind_cols), ('电价', price_cols)]:
        if cols:
            completeness = (df[cols].notna().sum().sum() / (len(df) * len(cols))) * 100
            report += f"- **{data_type}数据完整性**: {completeness:.1f}%\n"
    
    with open(OUTPUT_DIR / 'time_series_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"数据摘要报告已保存到 {OUTPUT_DIR / 'time_series_analysis_report.md'}")

def main():
    """主函数"""
    # 创建输出目录
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    df = load_data()
    if df is None:
        print("数据加载失败，退出程序")
        return
    
    # 生成各种可视化
    plot_load_comparison(df)
    plot_renewable_generation(df, 'DE')
    plot_renewable_generation(df, 'GB_GBN')
    plot_price_analysis(df)
    plot_seasonal_patterns(df, 'DE')
    plot_correlation_matrix(df, 'DE')
    
    # 生成摘要报告
    generate_summary_report(df)
    
    print("\n" + "="*50)
    print("所有时间序列可视化图表已生成完成！")
    print(f"请查看 {OUTPUT_DIR} 目录下的文件：")
    print("- PNG格式的静态图表")
    print("- HTML格式的交互式图表")
    print("- Markdown格式的分析报告")
    print("="*50)

if __name__ == "__main__":
    main()