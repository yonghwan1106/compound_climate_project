"""
전문적인 한국 취약성 지도 생성
실제 행정구역 GeoJSON + Geopandas + Cartopy
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pathlib import Path
import json
import urllib.request
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def download_korea_geojson():
    """한국 시도 경계 GeoJSON 다운로드"""
    # 대한민국 시도 경계 GeoJSON (간소화 버전)
    url = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2018/json/skorea-provinces-2018-geo.json"

    geojson_path = Path("data/korea_provinces.geojson")
    geojson_path.parent.mkdir(parents=True, exist_ok=True)

    if not geojson_path.exists():
        print("Downloading Korea GeoJSON...")
        try:
            urllib.request.urlretrieve(url, geojson_path)
            print("Downloaded successfully!")
        except Exception as e:
            print(f"Download failed: {e}")
            return None

    return geojson_path


def get_vulnerability_data():
    """30개 지역 취약성 데이터"""
    regions = [
        # 서울
        {'name': '서울 강남구', 'name_en': 'Seoul Gangnam', 'province': '서울', 'lat': 37.52, 'lon': 127.05, 'vuln': 0.603, 'risk': 'High'},
        {'name': '서울 서초구', 'name_en': 'Seoul Seocho', 'province': '서울', 'lat': 37.48, 'lon': 127.03, 'vuln': 0.478, 'risk': 'Medium'},
        {'name': '서울 송파구', 'name_en': 'Seoul Songpa', 'province': '서울', 'lat': 37.51, 'lon': 127.11, 'vuln': 0.385, 'risk': 'Low'},
        {'name': '서울 강서구', 'name_en': 'Seoul Gangseo', 'province': '서울', 'lat': 37.55, 'lon': 126.85, 'vuln': 0.342, 'risk': 'Low'},
        {'name': '서울 노원구', 'name_en': 'Seoul Nowon', 'province': '서울', 'lat': 37.65, 'lon': 127.06, 'vuln': 0.328, 'risk': 'Low'},
        # 경기
        {'name': '경기 수원시', 'name_en': 'Suwon', 'province': '경기', 'lat': 37.27, 'lon': 127.02, 'vuln': 0.445, 'risk': 'Medium'},
        {'name': '경기 성남시', 'name_en': 'Seongnam', 'province': '경기', 'lat': 37.42, 'lon': 127.13, 'vuln': 0.412, 'risk': 'Medium'},
        {'name': '경기 고양시', 'name_en': 'Goyang', 'province': '경기', 'lat': 37.66, 'lon': 126.83, 'vuln': 0.356, 'risk': 'Low'},
        {'name': '경기 용인시', 'name_en': 'Yongin', 'province': '경기', 'lat': 37.24, 'lon': 127.18, 'vuln': 0.334, 'risk': 'Low'},
        {'name': '경기 파주시', 'name_en': 'Paju', 'province': '경기', 'lat': 37.76, 'lon': 126.78, 'vuln': 0.298, 'risk': 'Low'},
        # 인천
        {'name': '인천 강화군', 'name_en': 'Ganghwa', 'province': '인천', 'lat': 37.75, 'lon': 126.49, 'vuln': 0.456, 'risk': 'Medium'},
        {'name': '인천 연수구', 'name_en': 'Incheon Yeonsu', 'province': '인천', 'lat': 37.41, 'lon': 126.68, 'vuln': 0.367, 'risk': 'Low'},
        # 부산
        {'name': '부산 해운대구', 'name_en': 'Haeundae', 'province': '부산', 'lat': 35.16, 'lon': 129.16, 'vuln': 0.489, 'risk': 'Medium'},
        {'name': '부산 사하구', 'name_en': 'Busan Saha', 'province': '부산', 'lat': 35.10, 'lon': 128.97, 'vuln': 0.378, 'risk': 'Low'},
        # 대구
        {'name': '대구 수성구', 'name_en': 'Daegu Suseong', 'province': '대구', 'lat': 35.86, 'lon': 128.63, 'vuln': 0.636, 'risk': 'High'},
        {'name': '대구 달서구', 'name_en': 'Daegu Dalseo', 'province': '대구', 'lat': 35.83, 'lon': 128.53, 'vuln': 0.389, 'risk': 'Low'},
        # 광주
        {'name': '광주 북구', 'name_en': 'Gwangju Buk', 'province': '광주', 'lat': 35.17, 'lon': 126.91, 'vuln': 0.423, 'risk': 'Medium'},
        {'name': '광주 서구', 'name_en': 'Gwangju Seo', 'province': '광주', 'lat': 35.15, 'lon': 126.88, 'vuln': 0.356, 'risk': 'Low'},
        # 대전
        {'name': '대전 유성구', 'name_en': 'Yuseong', 'province': '대전', 'lat': 36.36, 'lon': 127.36, 'vuln': 0.401, 'risk': 'Medium'},
        {'name': '대전 서구', 'name_en': 'Daejeon Seo', 'province': '대전', 'lat': 36.35, 'lon': 127.38, 'vuln': 0.345, 'risk': 'Low'},
        # 울산
        {'name': '울산 중구', 'name_en': 'Ulsan Jung', 'province': '울산', 'lat': 35.57, 'lon': 129.33, 'vuln': 0.412, 'risk': 'Medium'},
        {'name': '울산 북구', 'name_en': 'Ulsan Buk', 'province': '울산', 'lat': 35.58, 'lon': 129.36, 'vuln': 0.356, 'risk': 'Low'},
        # 강원
        {'name': '강원 춘천시', 'name_en': 'Chuncheon', 'province': '강원', 'lat': 37.88, 'lon': 127.73, 'vuln': 0.312, 'risk': 'Low'},
        {'name': '강원 원주시', 'name_en': 'Wonju', 'province': '강원', 'lat': 37.34, 'lon': 127.95, 'vuln': 0.298, 'risk': 'Low'},
        # 충청
        {'name': '충북 청주시', 'name_en': 'Cheongju', 'province': '충북', 'lat': 36.64, 'lon': 127.49, 'vuln': 0.334, 'risk': 'Low'},
        {'name': '충남 천안시', 'name_en': 'Cheonan', 'province': '충남', 'lat': 36.81, 'lon': 127.15, 'vuln': 0.312, 'risk': 'Low'},
        # 전라
        {'name': '전북 전주시', 'name_en': 'Jeonju', 'province': '전북', 'lat': 35.82, 'lon': 127.15, 'vuln': 0.356, 'risk': 'Low'},
        {'name': '전남 목포시', 'name_en': 'Mokpo', 'province': '전남', 'lat': 34.81, 'lon': 126.39, 'vuln': 0.378, 'risk': 'Low'},
        # 경상
        {'name': '경북 포항시', 'name_en': 'Pohang', 'province': '경북', 'lat': 36.02, 'lon': 129.37, 'vuln': 0.345, 'risk': 'Low'},
        {'name': '경남 창원시', 'name_en': 'Changwon', 'province': '경남', 'lat': 35.23, 'lon': 128.68, 'vuln': 0.367, 'risk': 'Low'},
    ]
    return pd.DataFrame(regions)


def create_professional_map(gdf_korea, df_vuln, output_path):
    """전문적인 취약성 지도 생성"""

    fig = plt.figure(figsize=(20, 24))

    # 메인 지도 영역
    ax_map = fig.add_axes([0.05, 0.25, 0.70, 0.68])

    # 컬러바 영역
    ax_cbar = fig.add_axes([0.78, 0.35, 0.02, 0.45])

    # 범례 영역
    ax_legend = fig.add_axes([0.76, 0.82, 0.22, 0.12])
    ax_legend.axis('off')

    # 통계 영역
    ax_stats = fig.add_axes([0.05, 0.05, 0.40, 0.17])
    ax_stats.axis('off')

    # 미니 차트 영역
    ax_chart = fig.add_axes([0.50, 0.05, 0.45, 0.17])

    # ========== 메인 지도 ==========
    # 바다 배경
    ax_map.set_facecolor('#cce5ff')

    # 시도 경계 그리기
    gdf_korea.plot(ax=ax_map,
                   color='#f8f9fa',
                   edgecolor='#495057',
                   linewidth=1.5,
                   alpha=0.95)

    # 컬러맵 설정
    cmap = LinearSegmentedColormap.from_list('vulnerability',
        ['#2d6a4f', '#40916c', '#52b788', '#74c69d', '#95d5b2',
         '#b7e4c7', '#d8f3dc', '#fff3cd', '#ffec99', '#ffd43b',
         '#fab005', '#f59f00', '#e67700', '#d9480f', '#c92a2a'])

    # 취약성 점 그리기
    for idx, row in df_vuln.iterrows():
        # 마커 스타일
        if row['risk'] == 'High':
            marker = 's'
            size = 600
            edgecolor = '#7f1d1d'
            linewidth = 4
            zorder = 100
        elif row['risk'] == 'Medium':
            marker = 'o'
            size = 400
            edgecolor = '#92400e'
            linewidth = 3
            zorder = 50
        else:
            marker = 'o'
            size = 280
            edgecolor = '#166534'
            linewidth = 2
            zorder = 10

        # 취약성에 따른 색상
        color = cmap((row['vuln'] - 0.25) / 0.45)

        ax_map.scatter(row['lon'], row['lat'],
                      c=[color], s=size, marker=marker,
                      edgecolors=edgecolor, linewidths=linewidth,
                      alpha=0.9, zorder=zorder)

    # 지역명 라벨 추가
    for idx, row in df_vuln.iterrows():
        # 라벨 스타일
        if row['risk'] == 'High':
            fontsize = 11
            fontweight = 'bold'
            bbox_style = dict(boxstyle='round,pad=0.4', facecolor='#fecaca',
                            edgecolor='#dc2626', linewidth=2, alpha=0.95)
            text_color = '#7f1d1d'
        elif row['risk'] == 'Medium':
            fontsize = 9
            fontweight = 'bold'
            bbox_style = dict(boxstyle='round,pad=0.3', facecolor='#fef3c7',
                            edgecolor='#d97706', linewidth=1.5, alpha=0.9)
            text_color = '#92400e'
        else:
            fontsize = 8
            fontweight = 'normal'
            bbox_style = dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='#6b7280', linewidth=1, alpha=0.85)
            text_color = '#374151'

        # 라벨 위치 조정
        offset_y = 0.08
        if row['risk'] == 'High':
            offset_y = 0.12
            label_text = f"{row['name']}\n({row['vuln']:.3f})"
        elif row['risk'] == 'Medium':
            offset_y = 0.10
            label_text = row['name']
        else:
            label_text = row['name'].split()[-1]

        ax_map.annotate(label_text,
                       xy=(row['lon'], row['lat']),
                       xytext=(row['lon'], row['lat'] + offset_y),
                       fontsize=fontsize, fontweight=fontweight,
                       color=text_color,
                       ha='center', va='bottom',
                       bbox=bbox_style,
                       zorder=200)

    # 지도 범위 설정
    ax_map.set_xlim(125.0, 130.0)
    ax_map.set_ylim(33.0, 38.8)
    ax_map.set_xlabel('경도 (Longitude, °E)', fontsize=14, fontweight='bold')
    ax_map.set_ylabel('위도 (Latitude, °N)', fontsize=14, fontweight='bold')
    ax_map.tick_params(labelsize=11)

    # 그리드
    ax_map.grid(True, linestyle='--', alpha=0.4, color='#6b7280')

    # 제목
    ax_map.set_title('Figure 3: 복합 극한기후 취약성 지수 (Compound Climate Event Vulnerability Index)\n'
                    '30개 시군구 분석 (2000-2023)',
                    fontsize=18, fontweight='bold', pad=20)

    # ========== 컬러바 ==========
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.25, vmax=0.70))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('취약성 지수\n(Vulnerability Index)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # 컬러바에 위험 등급 표시
    cbar.ax.axhline(y=0.40, color='#d97706', linewidth=3, linestyle='-')
    cbar.ax.axhline(y=0.55, color='#dc2626', linewidth=3, linestyle='-')
    cbar.ax.text(1.5, 0.32, 'Low', fontsize=10, va='center', color='#166534', fontweight='bold')
    cbar.ax.text(1.5, 0.47, 'Medium', fontsize=10, va='center', color='#d97706', fontweight='bold')
    cbar.ax.text(1.5, 0.62, 'High', fontsize=10, va='center', color='#dc2626', fontweight='bold')

    # ========== 범례 ==========
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#dc2626',
               markersize=18, markeredgecolor='#7f1d1d', markeredgewidth=3,
               label='High Risk (V ≥ 0.55)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f59e0b',
               markersize=15, markeredgecolor='#92400e', markeredgewidth=2,
               label='Medium Risk (0.40-0.55)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#10b981',
               markersize=12, markeredgecolor='#166534', markeredgewidth=1.5,
               label='Low Risk (V < 0.40)'),
    ]

    ax_legend.legend(handles=legend_elements, loc='center', fontsize=11,
                    title='위험 등급 (Risk Level)', title_fontsize=12,
                    frameon=True, fancybox=True, shadow=True,
                    edgecolor='#374151')

    # ========== 통계 박스 ==========
    stats_text = (
        "■ 분석 개요 (Analysis Overview)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  • 분석 지역: 30개 시군구 (30 Districts)\n"
        f"  • 분석 기간: 2000-2023 (24 years)\n"
        f"  • 취약성 범위: {df_vuln['vuln'].min():.3f} ~ {df_vuln['vuln'].max():.3f}\n"
        f"  • 평균 취약성: {df_vuln['vuln'].mean():.3f} (Std: {df_vuln['vuln'].std():.3f})\n\n"
        "■ 고위험 지역 (High Risk Regions)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  [1] 서울 강남구: 0.603 (도시열섬 + 고밀도)\n"
        "  [2] 대구 수성구: 0.636 (분지지형 + 고령화)\n\n"
        "■ 위험 등급 분포 (Risk Distribution)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  • High (고위험): 2개 지역 (6.7%)\n"
        f"  • Medium (중위험): 8개 지역 (26.7%)\n"
        f"  • Low (저위험): 20개 지역 (66.6%)"
    )

    ax_stats.text(0.02, 0.98, stats_text, transform=ax_stats.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='Malgun Gothic',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f9ff',
                          edgecolor='#0284c7', linewidth=2, alpha=0.95))

    # ========== 미니 차트: 위험등급별 분포 ==========
    risk_counts = df_vuln['risk'].value_counts()
    risk_order = ['Low', 'Medium', 'High']
    counts = [risk_counts.get(r, 0) for r in risk_order]
    colors_bar = ['#10b981', '#f59e0b', '#ef4444']

    bars = ax_chart.bar(risk_order, counts, color=colors_bar, edgecolor='white', linewidth=2)
    ax_chart.set_ylabel('지역 수 (Number of Regions)', fontsize=11)
    ax_chart.set_xlabel('위험 등급 (Risk Level)', fontsize=11)
    ax_chart.set_title('위험 등급별 지역 분포', fontsize=12, fontweight='bold')
    ax_chart.set_ylim(0, 25)

    # 바 위에 값 표시
    for bar, count in zip(bars, counts):
        ax_chart.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{count}개\n({count/30*100:.1f}%)',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax_chart.spines['top'].set_visible(False)
    ax_chart.spines['right'].set_visible(False)
    ax_chart.grid(True, axis='y', alpha=0.3)

    # 저장
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Professional Korea Vulnerability Map Generation")
    print("=" * 60)

    # 출력 디렉토리
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # GeoJSON 다운로드
    geojson_path = download_korea_geojson()

    # 취약성 데이터 로드
    df_vuln = get_vulnerability_data()
    print(f"Loaded {len(df_vuln)} regions")
    print(f"Risk Distribution: {df_vuln['risk'].value_counts().to_dict()}")

    # GeoDataFrame 로드
    if geojson_path and geojson_path.exists():
        gdf_korea = gpd.read_file(geojson_path)
        print(f"Loaded Korea map with {len(gdf_korea)} provinces")
    else:
        print("Using fallback map...")
        gdf_korea = None

    # 전문 지도 생성
    map_path = output_dir / "fig3_vulnerability_professional.png"
    create_professional_map(gdf_korea, df_vuln, str(map_path))

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
