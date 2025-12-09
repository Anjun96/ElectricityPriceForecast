import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from datetime import datetime
import calendar

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
DATA_DIR = Path(__file__).parent
WEATHER_FILE = DATA_DIR / "weather_data.csv"
OUTPUT_DIR = DATA_DIR / "output"

def load_data():
    """加载天气数据"""
    print("正在加载天气数据...")
    
    try:
        # 由于数据量大，使用chunk方式加载
        # 先读取前几行确定列名
        df_sample = pd.read_csv(WEATHER_FILE, nrows=5)
        columns = df_sample.columns
        
        # 读取完整数据
        df = pd.read_csv(WEATHER_FILE, parse_dates=['utc_timestamp'])
        
        print(f"数据加载完成！共 {len(df)} 条记录")
        print(f"时间范围: {df['utc_timestamp'].min()} 至 {df['utc_timestamp'].max()}")
        print(f"列数: {len(df.columns)}")
        
        return df
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def plot_temperature_comparison(df, countries=['DE', 'FR', 'GB', 'IT', 'ES'], years=[2019]):
    """比较不同国家的温度"""
    print("\n生成温度比较图...")
    
    fig, axes = plt.subplots(len(years), 1, figsize=(16, 6*len(years)))
    if len(years) == 1:
        axes = [axes]
    
    for i, year in enumerate(years):
        # 筛选指定年份的数据
        year_data = df[df['utc_timestamp'].dt.year == year].copy()
        
        for country in countries:
            temp_col = f"{country}_temperature"
            if temp_col in year_data.columns:
                # 计算月平均温度
                monthly_data = year_data[['utc_timestamp', temp_col]].copy()
                monthly_data['month'] = monthly_data['utc_timestamp'].dt.month
                monthly_avg = monthly_data.groupby('month')[temp_col].mean()
                
                axes[i].plot(monthly_avg.index, monthly_avg.values, 
                           label=country, linewidth=2, marker='o', markersize=4)
        
        axes[i].set_title(f'{year}年主要国家月平均温度比较 (°C)', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('月份', fontsize=12)
        axes[i].set_ylabel('温度 (°C)', fontsize=12)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xticks(range(1, 13))
        axes[i].set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temperature_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly交互式版本 - 只显示最近一年
    fig_plotly = go.Figure()
    
    latest_year = df['utc_timestamp'].dt.year.max()
    year_data = df[df['utc_timestamp'].dt.year == latest_year].copy()
    
    for country in countries:
        temp_col = f"{country}_temperature"
        if temp_col in year_data.columns:
            monthly_data = year_data[['utc_timestamp', temp_col]].copy()
            monthly_data['month'] = monthly_data['utc_timestamp'].dt.month
            monthly_avg = monthly_data.groupby('month')[temp_col].mean()
            
            fig_plotly.add_trace(go.Scatter(
                x=monthly_avg.index,
                y=monthly_avg.values,
                mode='lines+markers',
                name=country,
                line=dict(width=2)
            ))
    
    fig_plotly.update_layout(
        title=f'{latest_year}年主要国家月平均温度比较 (°C)',
        xaxis_title='月份',
        yaxis_title='温度 (°C)',
        hovermode='x unified',
        height=600
    )
    
    fig_plotly.write_html(OUTPUT_DIR / "temperature_comparison_interactive.html")
    print(f"温度比较图已保存到 {OUTPUT_DIR} 目录")

def plot_radiation_analysis(df, country='DE', years=[2019]):
    """分析太阳辐射数据"""
    print(f"\n生成{country}太阳辐射分析图...")
    
    # 检查相关列是否存在
    direct_col = f"{country}_radiation_direct_horizontal"
    diffuse_col = f"{country}_radiation_diffuse_horizontal"
    
    if direct_col not in df.columns or diffuse_col not in df.columns:
        print(f"警告: 没有找到{country}的辐射数据")
        return
    
    fig, axes = plt.subplots(len(years), 2, figsize=(16, 6*len(years)))
    if len(years) == 1:
        axes = axes.reshape(1, -1)
    
    for i, year in enumerate(years):
        # 筛选指定年份的数据
        year_data = df[df['utc_timestamp'].dt.year == year].copy()
        
        # 计算月平均辐射
        monthly_data = year_data[['utc_timestamp', direct_col, diffuse_col]].copy()
        monthly_data['month'] = monthly_data['utc_timestamp'].dt.month
        monthly_avg = monthly_data.groupby('month')[[direct_col, diffuse_col]].mean()
        
        # 直接辐射
        axes[i, 0].plot(monthly_avg.index, monthly_avg[direct_col], 
                       linewidth=2, marker='o', markersize=4, color='orange')
        axes[i, 0].set_title(f'{year}年{country}月均直接辐射 (W/m²)', fontsize=12, fontweight='bold')
        axes[i, 0].set_xlabel('月份')
        axes[i, 0].set_ylabel('直接辐射 (W/m²)')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_xticks(range(1, 13))
        axes[i, 0].set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
        
        # 散射辐射
        axes[i, 1].plot(monthly_avg.index, monthly_avg[diffuse_col], 
                       linewidth=2, marker='o', markersize=4, color='skyblue')
        axes[i, 1].set_title(f'{year}年{country}月均散射辐射 (W/m²)', fontsize=12, fontweight='bold')
        axes[i, 1].set_xlabel('月份')
        axes[i, 1].set_ylabel('散射辐射 (W/m²)')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xticks(range(1, 13))
        axes[i, 1].set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{country}_radiation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly交互式版本 - 只显示最近一年
    latest_year = df['utc_timestamp'].dt.year.max()
    year_data = df[df['utc_timestamp'].dt.year == latest_year].copy()
    
    monthly_data = year_data[['utc_timestamp', direct_col, diffuse_col]].copy()
    monthly_data['month'] = monthly_data['utc_timestamp'].dt.month
    monthly_avg = monthly_data.groupby('month')[[direct_col, diffuse_col]].mean()
    
    fig_plotly = make_subplots(
        rows=1, cols=2,
        subplot_titles=('直接辐射 (W/m²)', '散射辐射 (W/m²)')
    )
    
    fig_plotly.add_trace(
        go.Scatter(
            x=monthly_avg.index,
            y=monthly_avg[direct_col],
            mode='lines+markers',
            name='直接辐射',
            line=dict(width=2, color='orange')
        ),
        row=1, col=1
    )
    
    fig_plotly.add_trace(
        go.Scatter(
            x=monthly_avg.index,
            y=monthly_avg[diffuse_col],
            mode='lines+markers',
            name='散射辐射',
            line=dict(width=2, color='skyblue')
        ),
        row=1, col=2
    )
    
    fig_plotly.update_layout(
        title=f'{latest_year}年{country}太阳辐射分析',
        height=600
    )
    
    fig_plotly.write_html(OUTPUT_DIR / f"{country}_radiation_analysis_interactive.html")
    print(f"{country}太阳辐射分析图已保存到 {OUTPUT_DIR} 目录")

def plot_seasonal_patterns(df, country='DE', year=2019):
    """分析季节性模式"""
    print(f"\n生成{country}季节性模式分析图...")
    
    # 检查相关列是否存在
    temp_col = f"{country}_temperature"
    direct_col = f"{country}_radiation_direct_horizontal"
    diffuse_col = f"{country}_radiation_diffuse_horizontal"
    
    if temp_col not in df.columns:
        print(f"警告: 没有找到{country}的温度数据")
        return
    
    # 筛选指定年份的数据
    year_data = df[df['utc_timestamp'].dt.year == year].copy()
    
    # 准备数据
    year_data['hour'] = year_data['utc_timestamp'].dt.hour
    year_data['month'] = year_data['utc_timestamp'].dt.month
    year_data['season'] = year_data['month'].map({
        12: '冬季', 1: '冬季', 2: '冬季',
        3: '春季', 4: '春季', 5: '春季',
        6: '夏季', 7: '夏季', 8: '夏季',
        9: '秋季', 10: '秋季', 11: '秋季'
    })
    
    # 计算季节性平均温度
    seasonal_hourly = year_data.groupby(['season', 'hour'])[temp_col].mean().unstack(level=0)
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 温度季节性模式
    for season in ['春季', '夏季', '秋季', '冬季']:
        if season in seasonal_hourly.columns:
            axes[0].plot(seasonal_hourly.index, seasonal_hourly[season], 
                       label=season, linewidth=2, marker='o', markersize=3)
    
    axes[0].set_title(f'{country}不同季节日内温度模式 (°C)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('小时')
    axes[0].set_ylabel('温度 (°C)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(0, 24, 2))
    
    # 月平均温度
    monthly_temp = year_data.groupby('month')[temp_col].mean()
    axes[1].bar(monthly_temp.index, monthly_temp.values, color='skyblue')
    axes[1].set_title(f'{country}月平均温度 (°C)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('月份')
    axes[1].set_ylabel('温度 (°C)')
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
    axes[1].grid(True, alpha=0.3)
    
    # 如果有辐射数据，绘制辐射图
    if direct_col in year_data.columns and diffuse_col in year_data.columns:
        # 总辐射季节性模式
        year_data['total_radiation'] = year_data[direct_col] + year_data[diffuse_col]
        seasonal_rad = year_data.groupby(['season', 'hour'])['total_radiation'].mean().unstack(level=0)
        
        for season in ['春季', '夏季', '秋季', '冬季']:
            if season in seasonal_rad.columns:
                axes[2].plot(seasonal_rad.index, seasonal_rad[season], 
                           label=season, linewidth=2, marker='o', markersize=3)
        
        axes[2].set_title(f'{country}不同季节日内总辐射模式 (W/m²)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('小时')
        axes[2].set_ylabel('总辐射 (W/m²)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(range(0, 24, 2))
        
        # 月平均总辐射
        monthly_rad = year_data.groupby('month')['total_radiation'].mean()
        axes[3].bar(monthly_rad.index, monthly_rad.values, color='orange')
        axes[3].set_title(f'{country}月平均总辐射 (W/m²)', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('月份')
        axes[3].set_ylabel('总辐射 (W/m²)')
        axes[3].set_xticks(range(1, 13))
        axes[3].set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
        axes[3].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, '无辐射数据', ha='center', va='center', transform=axes[2].transAxes)
        axes[3].text(0.5, 0.5, '无辐射数据', ha='center', va='center', transform=axes[3].transAxes)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{country}_seasonal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly交互式版本
    fig_plotly = make_subplots(
        rows=2, cols=2,
        subplot_titles=('日内温度模式 (°C)', '月平均温度 (°C)', 
                       '日内总辐射模式 (W/m²)', '月平均总辐射 (W/m²)')
    )
    
    # 温度数据
    for season in ['春季', '夏季', '秋季', '冬季']:
        if season in seasonal_hourly.columns:
            fig_plotly.add_trace(
                go.Scatter(
                    x=seasonal_hourly.index,
                    y=seasonal_hourly[season],
                    mode='lines+markers',
                    name=f'{season}温度',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
    
    fig_plotly.add_trace(
        go.Bar(
            x=monthly_temp.index,
            y=monthly_temp.values,
            name='月平均温度',
            marker_color='skyblue'
        ),
        row=1, col=2
    )
    
    # 如果有辐射数据
    if direct_col in year_data.columns and diffuse_col in year_data.columns:
        for season in ['春季', '夏季', '秋季', '冬季']:
            if season in seasonal_rad.columns:
                fig_plotly.add_trace(
                    go.Scatter(
                        x=seasonal_rad.index,
                        y=seasonal_rad[season],
                        mode='lines+markers',
                        name=f'{season}辐射',
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )
        
        fig_plotly.add_trace(
            go.Bar(
                x=monthly_rad.index,
                y=monthly_rad.values,
                name='月平均总辐射',
                marker_color='orange'
            ),
            row=2, col=2
        )
    
    fig_plotly.update_layout(
        title=f'{year}年{country}季节性模式分析',
        height=800,
        showlegend=True
    )
    
    fig_plotly.write_html(OUTPUT_DIR / f"{country}_seasonal_patterns_interactive.html")
    print(f"{country}季节性模式分析图已保存到 {OUTPUT_DIR} 目录")

def plot_temperature_distribution(df, countries=['DE', 'FR', 'GB', 'IT', 'ES']):
    """分析温度分布"""
    print("\n生成温度分布分析图...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, country in enumerate(countries[:6]):
        temp_col = f"{country}_temperature"
        if temp_col in df.columns:
            # 获取最近一年的数据
            latest_year = df['utc_timestamp'].dt.year.max()
            year_data = df[df['utc_timestamp'].dt.year == latest_year][temp_col].dropna()
            
            if len(year_data) > 0:
                axes[i].hist(year_data.values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].axvline(year_data.mean(), color='red', linestyle='--', 
                              label=f'平均值: {year_data.mean():.1f}°C')
                axes[i].axvline(year_data.median(), color='green', linestyle='--', 
                              label=f'中位数: {year_data.median():.1f}°C')
                axes[i].set_title(f'{country} 温度分布 ({latest_year}年)', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('温度 (°C)')
                axes[i].set_ylabel('频次')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, f'无{country}温度数据', 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    # 隐藏多余的子图
    for j in range(len(countries), 6):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temperature_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly交互式版本
    fig_plotly = go.Figure()
    
    for country in countries:
        temp_col = f"{country}_temperature"
        if temp_col in df.columns:
            latest_year = df['utc_timestamp'].dt.year.max()
            year_data = df[df['utc_timestamp'].dt.year == latest_year][temp_col].dropna()
            
            if len(year_data) > 0:
                fig_plotly.add_trace(go.Histogram(
                    x=year_data.values,
                    name=country,
                    opacity=0.7,
                    nbinsx=50
                ))
    
    fig_plotly.update_layout(
        title=f'主要国家温度分布比较 ({latest_year}年)',
        xaxis_title='温度 (°C)',
        yaxis_title='频次',
        barmode='overlay',
        height=600
    )
    
    fig_plotly.write_html(OUTPUT_DIR / "temperature_distribution_interactive.html")
    print(f"温度分布分析图已保存到 {OUTPUT_DIR} 目录")

def plot_correlation_matrix(df, country='DE'):
    """分析变量间的相关性"""
    print(f"\n生成{country}变量相关性矩阵...")
    
    # 收集相关变量
    variables = {}
    
    temp_col = f"{country}_temperature"
    direct_col = f"{country}_radiation_direct_horizontal"
    diffuse_col = f"{country}_radiation_diffuse_horizontal"
    
    if temp_col in df.columns:
        variables['温度'] = temp_col
    if direct_col in df.columns:
        variables['直接辐射'] = direct_col
    if diffuse_col in df.columns:
        variables['散射辐射'] = diffuse_col
    
    if len(variables) < 2:
        print(f"警告: {country}可用变量不足，无法计算相关性")
        return
    
    # 准备数据 - 使用最近一年的数据
    latest_year = df['utc_timestamp'].dt.year.max()
    year_data = df[df['utc_timestamp'].dt.year == latest_year].copy()
    
    corr_data = year_data[list(variables.values())].copy()
    corr_data.columns = list(variables.keys())
    
    # 计算相关性矩阵
    corr_matrix = corr_data.corr()
    
    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    ax.set_title(f'{country}天气变量相关性矩阵 ({latest_year}年)', fontsize=14, fontweight='bold')
    
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
        title=f'{country}天气变量相关性矩阵 ({latest_year}年)',
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
# 天气数据分析报告

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
        if '_' in col and not col.startswith('utc_'):
            country = col.split('_')[0]
            countries.add(country)
    
    report += f"- **国家/地区数量**: {len(countries)}\n"
    report += f"- **国家/地区列表**: {', '.join(sorted(countries))}\n\n"
    
    # 数据类型统计
    temp_cols = [col for col in df.columns if 'temperature' in col]
    direct_cols = [col for col in df.columns if 'radiation_direct_horizontal' in col]
    diffuse_cols = [col for col in df.columns if 'radiation_diffuse_horizontal' in col]
    
    report += f"""
## 数据类型统计
- **温度数据**: {len(temp_cols)} 个国家/地区
- **直接辐射数据**: {len(direct_cols)} 个国家/地区
- **散射辐射数据**: {len(diffuse_cols)} 个国家/地区

## 数据完整性
"""
    
    # 计算数据完整性
    for data_type, cols in [('温度', temp_cols), ('直接辐射', direct_cols), 
                            ('散射辐射', diffuse_cols)]:
        if cols:
            completeness = (df[cols].notna().sum().sum() / (len(df) * len(cols))) * 100
            report += f"- **{data_type}数据完整性**: {completeness:.1f}%\n"
    
    # 示例统计 - 德国数据
    if 'DE_temperature' in df.columns:
        de_temp = df['DE_temperature'].dropna()
        report += f"""
## 示例统计 - 德国数据
- **平均温度**: {de_temp.mean():.2f}°C
- **最高温度**: {de_temp.max():.2f}°C
- **最低温度**: {de_temp.min():.2f}°C
- **温度标准差**: {de_temp.std():.2f}°C
"""
    
    with open(OUTPUT_DIR / 'weather_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"数据摘要报告已保存到 {OUTPUT_DIR / 'weather_analysis_report.md'}")

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
    plot_temperature_comparison(df)
    plot_radiation_analysis(df, 'DE')
    plot_seasonal_patterns(df, 'DE')
    plot_temperature_distribution(df)
    plot_correlation_matrix(df, 'DE')
    
    # 生成摘要报告
    generate_summary_report(df)
    
    print("\n" + "="*50)
    print("所有天气数据可视化图表已生成完成！")
    print(f"请查看 {OUTPUT_DIR} 目录下的文件：")
    print("- PNG格式的静态图表")
    print("- HTML格式的交互式图表")
    print("- Markdown格式的分析报告")
    print("="*50)

if __name__ == "__main__":
    main()
