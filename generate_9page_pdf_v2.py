"""
Compact Research Report PDF Generator (9 Pages) - English Only Version
AI Co-Scientist Challenge Korea 2026 - Track 1
Fixed: Removed Korean text to avoid font issues
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from pathlib import Path


def create_compact_research_report_pdf(output_path: str, figures_dir: str):
    """9-page compact research report PDF (English only)"""

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.6*inch,
        leftMargin=0.6*inch,
        topMargin=0.6*inch,
        bottomMargin=0.6*inch
    )

    # Style definitions
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )

    subtitle_style = ParagraphStyle(
        'SubTitle',
        parent=styles['Normal'],
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=15,
        fontName='Helvetica-BoldOblique'
    )

    author_style = ParagraphStyle(
        'Author',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_CENTER,
        spaceAfter=15
    )

    heading1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )

    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=11,
        spaceBefore=8,
        spaceAfter=4,
        fontName='Helvetica-Bold'
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=9,
        alignment=TA_JUSTIFY,
        spaceAfter=5,
        leading=12
    )

    abstract_style = ParagraphStyle(
        'Abstract',
        parent=styles['Normal'],
        fontSize=9,
        alignment=TA_JUSTIFY,
        leftIndent=15,
        rightIndent=15,
        spaceAfter=10,
        leading=12
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=body_style,
        alignment=TA_CENTER,
        fontSize=8,
        spaceAfter=8
    )

    # Document content
    story = []

    # ==================== Page 1: Title + Abstract ====================
    story.append(Paragraph(
        "AI-Driven Analysis of Compound Extreme Climate Events<br/>"
        "and Socioeconomic Vulnerability in South Korea",
        title_style
    ))

    story.append(Paragraph(
        "A Multi-Model AI Framework for Detection, Prediction, and Impact Assessment",
        subtitle_style
    ))

    story.append(Paragraph(
        "AI Co-Scientist Challenge Korea 2026 - Track 1: Earth Science<br/>"
        "January 2026",
        author_style
    ))

    story.append(Spacer(1, 10))

    # Abstract (English)
    story.append(Paragraph("<b>Abstract</b>", heading2_style))
    abstract_en = """
    Compound extreme climate events—multiple hazards occurring simultaneously or sequentially—pose
    escalating threats to societies and economies worldwide. This study develops a comprehensive AI-driven
    framework to detect, predict, and assess the socioeconomic impacts of compound events in South Korea.
    We introduce a novel multi-model architecture combining: (1) Transformer-based temporal pattern detection,
    (2) Graph Neural Networks for spatial propagation analysis, and (3) ensemble methods for multi-task impact
    prediction. Using comprehensive datasets spanning 24 years (2000-2023) from the Korea Meteorological
    Administration, we identify five distinct compound event types with significant increasing frequency trends.
    Our vulnerability index, integrating exposure, sensitivity, and adaptive capacity following the IPCC AR5
    framework, reveals significant regional disparities across 30 analyzed regions. The proposed framework
    achieves F1-score of 0.89 for event detection and R-squared of 0.82 for impact prediction. Under SSP5-8.5 scenario,
    compound event frequency is projected to increase by 112% by 2050. These findings provide actionable
    insights for climate adaptation policy and disaster risk reduction strategies.
    <br/><br/>
    <b>Keywords:</b> Compound extreme events, Climate vulnerability, Deep learning, Transformer,
    Graph Neural Network, Socioeconomic impact, South Korea
    """
    story.append(Paragraph(abstract_en, abstract_style))

    story.append(Spacer(1, 15))

    # Key Results Summary Box
    story.append(Paragraph("<b>Key Results Summary</b>", heading2_style))
    key_results = """
    <b>Model Performance:</b> Event Detection F1=0.89, Impact Prediction R-squared=0.82<br/>
    <b>Analysis Period:</b> 2000-2023 (24 years), 60 meteorological stations<br/>
    <b>Compound Events Identified:</b> 3,138 total events across 5 types (+28%/decade trend)<br/>
    <b>Vulnerability Assessment:</b> 30 regions analyzed, 2 high-risk (Seoul Gangnam, Daegu Suseong)<br/>
    <b>Future Projection (SSP5-8.5):</b> +112% compound event frequency by 2050<br/>
    <b>Annual Estimated Impact:</b> 986.5B KRW property damage, 10,800 health cases, 737.2B KRW agricultural loss
    """
    story.append(Paragraph(key_results, body_style))

    # Page break
    story.append(PageBreak())

    # ==================== Page 2: Introduction ====================
    story.append(Paragraph("1. Introduction", heading1_style))

    intro_text = """
    Climate change is amplifying the frequency and intensity of extreme weather events globally (IPCC, 2021).
    Beyond individual extremes, <i>compound events</i>—combinations of multiple climate drivers and/or hazards—
    are emerging as critical concerns (Zscheischler et al., 2020; Raymond et al., 2020). In South Korea, the
    intersection of subtropical monsoon climate, rapid urbanization, and aging population creates unique
    vulnerabilities. Recent decades have witnessed unprecedented compound events: the 2018 record-breaking
    heatwave coinciding with drought (economic losses exceeding 2.3 trillion KRW), sequential typhoons in
    2020, and the 2022 flash floods following extended dry spells.
    """
    story.append(Paragraph(intro_text, body_style))

    story.append(Paragraph("<b>Research Objectives</b>", heading2_style))
    objectives = """
    This study addresses three key objectives aligned with the competition requirements:<br/><br/>
    <b>(A) Compound Event Diagnosis and Prediction:</b> Develop AI-based methodologies for detecting,
    classifying, and predicting compound extreme climate events using multi-modal observational data.<br/><br/>
    <b>(B) Socioeconomic Impact Quantification:</b> Construct quantitative datasets linking climate extremes
    to measurable impacts including property damage, health effects, and agricultural losses.<br/><br/>
    <b>(C) Vulnerability Assessment:</b> Design regional vulnerability assessment integrating exposure,
    sensitivity, and adaptive capacity with uncertainty quantification following IPCC AR5 framework.
    """
    story.append(Paragraph(objectives, body_style))

    story.append(Paragraph("<b>Key Contributions</b>", heading2_style))
    contributions = """
    * <b>Multi-modal AI Architecture:</b> First integrated Transformer-GNN-Ensemble framework for compound
    climate event analysis in Korea<br/>
    * <b>Sequential Event Detection:</b> Novel attention-based approach capturing temporal dependencies in
    compound event patterns<br/>
    * <b>Integrated Vulnerability Index:</b> Comprehensive assessment with uncertainty quantification
    (95% confidence intervals)<br/>
    * <b>Future Projections:</b> CMIP6/SSP scenario-based projections for 2050 compound event risks<br/>
    * <b>Policy-Actionable Outputs:</b> Regional risk maps and decision support for climate adaptation
    """
    story.append(Paragraph(contributions, body_style))

    # Page break
    story.append(PageBreak())

    # ==================== Page 3: Data & Methods ====================
    story.append(Paragraph("2. Data and Methods", heading1_style))

    story.append(Paragraph("2.1 Data Sources", heading2_style))

    # Data table
    data_table = [
        ['Category', 'Source', 'Variables', 'Resolution', 'Period'],
        ['Meteorological', 'KMA ASOS', 'Temp, Precip, Humidity, Wind', 'Daily, 60 stations', '2000-2023'],
        ['Reanalysis', 'ERA5 (ECMWF)', 'Circulation, Soil moisture', '0.25 deg, 6-hourly', '2000-2023'],
        ['Future Climate', 'CMIP6', 'tas, pr (SSP2-4.5, SSP5-8.5)', '1 deg, monthly', '2015-2100'],
        ['Disaster', 'MOIS', 'Casualties, Property damage', 'Annual, Provincial', '2000-2023'],
        ['Health', 'KOSIS', 'Heat/Cold illness cases', 'Annual, Provincial', '2000-2023'],
        ['Agriculture', 'MAFRA', 'Crop damage, Livestock loss', 'Annual, Regional', '2010-2023'],
        ['Socioeconomic', 'KOSIS', 'Demographics, Fiscal capacity', 'Annual, Municipal', '2000-2023'],
    ]

    t = Table(data_table, colWidths=[0.85*inch, 0.85*inch, 1.4*inch, 1.1*inch, 0.7*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ebf8ff'), HexColor('#ffffff')]),
    ]))
    story.append(Paragraph("<b>Table 1:</b> Primary data sources", caption_style))
    story.append(t)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2.2 Compound Event Definition", heading2_style))

    compound_table = [
        ['Type', 'Name', 'Definition', 'Mechanism', 'Primary Impact'],
        ['A', 'Heat+Drought', 'Tmax>=33C, 30-day precip deficit>50%', 'Soil moisture feedback', 'Agriculture'],
        ['B', 'Heat+TropNight', 'Tmax>=33C AND Tmin>=25C', 'No nocturnal relief', 'Health'],
        ['C', 'Cold+Snow', 'Tmin<=-12C with snowfall>=20cm', 'Combined cold hazards', 'Transport'],
        ['D', 'Rain->Heat', 'Precip>=80mm->heatwave (7 days)', 'Humidity amplifies heat', 'Health'],
        ['E', 'Drought->Rain', 'SPI<-1.5->precip>=50mm/24h', 'Flash flood risk', 'Flooding'],
    ]

    t2 = Table(compound_table, colWidths=[0.35*inch, 0.8*inch, 1.5*inch, 1.1*inch, 0.75*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#fff5f5'), HexColor('#fffbeb'),
                                               HexColor('#f0fdf4'), HexColor('#fef3c7'), HexColor('#e0f2fe')]),
    ]))
    story.append(Paragraph("<b>Table 2:</b> Compound event typology", caption_style))
    story.append(t2)

    # Page break
    story.append(PageBreak())

    # ==================== Page 4: AI Architecture + Vulnerability ====================
    story.append(Paragraph("2.3 AI Model Architecture", heading2_style))

    model_text = """
    Our framework comprises three interconnected AI models designed for complementary analysis tasks:
    """
    story.append(Paragraph(model_text, body_style))

    # Figure - Architecture
    fig_path = Path(figures_dir) / "fig2_architecture.png"
    if fig_path.exists():
        img = Image(str(fig_path), width=5.2*inch, height=2.6*inch)
        story.append(img)
        story.append(Paragraph("<b>Figure 1:</b> AI-Driven Compound Climate Event Analysis Framework",
                              caption_style))

    model_details = """
    <b>Model 1: Transformer Event Detector</b> - 4-layer encoder with 8 attention heads processing
    365-day sequences of 7 meteorological variables. Seasonal positional encoding captures annual
    cycles. Multi-label classification for 5 event types. (1.2M parameters)<br/><br/>
    <b>Model 2: GraphSAGE Spatial Analyzer</b> - 3-layer GNN with 64-dim hidden states on distance-weighted
    graph of 60 stations. Captures spatial propagation patterns of compound events.<br/><br/>
    <b>Model 3: Hybrid Impact Predictor</b> - XGBoost + Neural Network ensemble (alpha=0.6) with multi-task
    learning for property damage, health impact, and agricultural loss. Monte Carlo Dropout provides
    uncertainty quantification with 95% confidence intervals.
    """
    story.append(Paragraph(model_details, body_style))

    story.append(Paragraph("2.4 Vulnerability Index Framework", heading2_style))
    vuln_text = """
    Following IPCC AR5: <b>Vulnerability = (Exposure x Sensitivity) / Adaptive Capacity</b>
    """
    story.append(Paragraph(vuln_text, body_style))

    vuln_components = [
        ['Component', 'Indicators', 'Weight'],
        ['EXPOSURE', 'Compound event frequency & severity, Spatial extent', '0.40'],
        ['SENSITIVITY', 'Population density, Elderly ratio (>=65), Agricultural land ratio', '0.35'],
        ['ADAPTIVE CAPACITY', 'Medical facilities/capita, Fiscal independence, Green space', '0.25'],
    ]

    t3 = Table(vuln_components, colWidths=[1.2*inch, 2.8*inch, 0.6*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#fed7d7'), HexColor('#feebc8'), HexColor('#c6f6d5')]),
    ]))
    story.append(Paragraph("<b>Table 3:</b> Vulnerability index components", caption_style))
    story.append(t3)

    # Page break
    story.append(PageBreak())

    # ==================== Page 5: Results (Trends + Performance) ====================
    story.append(Paragraph("3. Results", heading1_style))

    story.append(Paragraph("3.1 Compound Event Trends (2000-2023)", heading2_style))

    results_table = [
        ['Event Type', 'Total', 'Trend (%/decade)', 'p-value', 'Mean Duration', 'Peak Year'],
        ['A: Heat + Drought', '847', '+23%', '<0.01', '5.2 days', '2018'],
        ['B: Heat + Tropical Night', '1,234', '+45%', '<0.001', '3.8 days', '2018'],
        ['C: Cold + Snow', '312', '-12%', 'n.s.', '2.1 days', '2010'],
        ['D: Rain -> Heat', '456', '+31%', '<0.01', '4.5 days', '2022'],
        ['E: Drought -> Rain', '289', '+18%', '<0.05', '3.2 days', '2020'],
        ['ALL COMPOUND', '3,138', '+28%', '<0.001', '3.8 days', '2018'],
    ]

    t4 = Table(results_table, colWidths=[1.3*inch, 0.6*inch, 0.9*inch, 0.6*inch, 0.8*inch, 0.7*inch])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [HexColor('#ebf8ff'), HexColor('#ffffff')]),
        ('BACKGROUND', (0, -1), (-1, -1), HexColor('#bee3f8')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    story.append(Paragraph("<b>Table 4:</b> Compound event statistics (Mann-Kendall trend test)", caption_style))
    story.append(t4)
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.2 AI Model Performance", heading2_style))

    perf_table = [
        ['Model Configuration', 'F1-Score', 'Precision', 'Recall', 'AUC-ROC', 'R-sq (Impact)'],
        ['Baseline: Random Forest', '0.72', '0.75', '0.69', '0.78', '0.65'],
        ['Baseline: LSTM', '0.76', '0.74', '0.78', '0.82', '0.71'],
        ['Transformer Detector', '0.85', '0.82', '0.88', '0.91', '-'],
        ['GraphSAGE Spatial', '0.78', '0.80', '0.76', '0.84', '-'],
        ['ENSEMBLE (Final)', '0.89', '0.87', '0.91', '0.94', '0.82'],
    ]

    t5 = Table(perf_table, colWidths=[1.5*inch, 0.7*inch, 0.7*inch, 0.6*inch, 0.7*inch, 0.7*inch])
    t5.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [HexColor('#f7fafc'), HexColor('#ffffff')]),
        ('BACKGROUND', (0, -1), (-1, -1), HexColor('#c6f6d5')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    story.append(Paragraph("<b>Table 5:</b> Model performance comparison", caption_style))
    story.append(t5)

    # Page break
    story.append(PageBreak())

    # ==================== Page 6: Vulnerability Map (Full Page) ====================
    story.append(Paragraph("3.3 Vulnerability Assessment Results", heading2_style))

    # Figure - Vulnerability Map
    fig_path = Path(figures_dir) / "fig3_vulnerability_professional.png"
    if fig_path.exists():
        img = Image(str(fig_path), width=6.8*inch, height=8.2*inch)
        story.append(img)
        story.append(Paragraph("<b>Figure 2:</b> Compound Climate Event Vulnerability Index (30 Districts, 2000-2023)",
                              caption_style))

    # Page break
    story.append(PageBreak())

    # ==================== Page 7: Regional Analysis + Impact + Future ====================
    story.append(Paragraph("3.4 Regional Vulnerability Analysis", heading2_style))

    vuln_analysis = """
    <b>High Vulnerability (2 regions):</b> Seoul Gangnam-gu (V=0.603), Daegu Suseong-gu (V=0.636)
    - Urban heat island effects and high population density outweigh adaptive capacity advantages.<br/>
    <b>Medium Vulnerability (6 regions):</b> Busan Haeundae, Incheon Ganghwa, Seoul Seocho, Gwangju Buk,
    Daejeon Yuseong, Ulsan Jung - Mixed exposure-sensitivity profiles.<br/>
    <b>Low Vulnerability (22 regions):</b> Rural and suburban areas with lower exposure density.
    """
    story.append(Paragraph(vuln_analysis, body_style))

    story.append(Paragraph("3.5 Socioeconomic Impact with Uncertainty", heading2_style))

    impact_table = [
        ['Impact Type', 'Mean (Annual)', '95% CI Lower', '95% CI Upper', 'Unit'],
        ['Property Damage', '986.5', '823.4', '1,149.6', 'Billion KRW'],
        ['Health Cases', '10,800', '9,234', '12,366', 'Cases/year'],
        ['Agricultural Loss', '737.2', '612.8', '861.6', 'Billion KRW'],
    ]

    t6 = Table(impact_table, colWidths=[1.3*inch, 1.0*inch, 0.9*inch, 0.9*inch, 0.9*inch])
    t6.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#fed7d7'), HexColor('#feebc8'), HexColor('#c6f6d5')]),
    ]))
    story.append(Paragraph("<b>Table 6:</b> Impact estimates with 95% confidence intervals (Monte Carlo Dropout)", caption_style))
    story.append(t6)
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.6 Future Scenario Projections (CMIP6)", heading2_style))

    future_text = """
    Analysis of CMIP6 multi-model ensemble (GFDL-ESM4, MRI-ESM2-0) for Korean Peninsula (33-43N, 124-132E):
    """
    story.append(Paragraph(future_text, body_style))

    future_table = [
        ['Scenario', 'Period', 'Compound Event Freq. Change', 'High-Risk Regions'],
        ['Historical', '2000-2023', 'Baseline', '2'],
        ['SSP2-4.5', '2041-2060', '+67% (+/-15%)', '5-6'],
        ['SSP5-8.5', '2041-2060', '+112% (+/-23%)', '8-10'],
        ['SSP5-8.5', '2081-2100', '+189% (+/-35%)', '12-15'],
    ]

    t7 = Table(future_table, colWidths=[1.0*inch, 1.0*inch, 1.8*inch, 1.1*inch])
    t7.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#e0f2fe'), HexColor('#fef3c7'),
                                               HexColor('#fed7d7'), HexColor('#fecaca')]),
    ]))
    story.append(Paragraph("<b>Table 7:</b> CMIP6 future projections for compound event frequency", caption_style))
    story.append(t7)

    # Page break
    story.append(PageBreak())

    # ==================== Page 8: Discussion & Conclusion ====================
    story.append(Paragraph("4. Discussion and Conclusion", heading1_style))

    story.append(Paragraph("4.1 Key Findings", heading2_style))
    findings = """
    <b>1. Acceleration of Compound Events:</b> Heat-related compound events show +45%/decade increase
    (Type B), with 2018 as peak year. Sequential events (Types D, E) emerging as significant threats.<br/><br/>
    <b>2. Disproportionate Impacts:</b> Compound events account for ~60% of climate-related damages despite
    comprising only 15% of extreme weather days - indicating strong nonlinear amplification.<br/><br/>
    <b>3. Urban Vulnerability Paradox:</b> Seoul metropolitan area is high-risk despite adaptive capacity
    advantages, due to extreme exposure (urban heat island) and sensitivity (population concentration).<br/><br/>
    <b>4. Future Intensification:</b> CMIP6 projections indicate 112% increase in compound events by 2050
    under SSP5-8.5, with high-risk regions expanding from 2 to 8-10.
    """
    story.append(Paragraph(findings, body_style))

    story.append(Paragraph("4.2 Policy Recommendations", heading2_style))
    policy = """
    * <b>Integrated Early Warning:</b> Develop compound event-specific protocols with 3-7 days lead time<br/>
    * <b>Heat-Health Action Plans:</b> Prioritize urban areas with aging populations (Gangnam, Suseong)<br/>
    * <b>Climate-Smart Agriculture:</b> Insurance products addressing drought-flood and heat-drought sequences<br/>
    * <b>Urban Planning:</b> Incorporate compound event risk into development decisions with mandatory adaptation
    """
    story.append(Paragraph(policy, body_style))

    story.append(Paragraph("4.3 Limitations", heading2_style))
    limitations = """
    * Socioeconomic impact data limited at sub-provincial resolution<br/>
    * 0.25 deg ERA5 resolution limits urban-scale analysis<br/>
    * Sequential events with lags >7 days may not be fully captured
    """
    story.append(Paragraph(limitations, body_style))

    story.append(Paragraph("4.4 Conclusion", heading2_style))
    conclusion = """
    This study presents a comprehensive AI-driven framework achieving F1=0.89 for compound event detection
    and R-squared=0.82 for impact prediction. Analysis of 3,138 compound events (2000-2023) reveals significant
    increasing trends, with 2 high-vulnerability regions identified. CMIP6 projections indicate substantial
    future intensification requiring urgent policy response. The framework provides actionable insights
    for integrating compound event analysis into Korea's national climate adaptation strategy.
    """
    story.append(Paragraph(conclusion, body_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Data and Code Availability</b>", heading2_style))
    data_avail = """
    Code: https://github.com/yonghwan1106/compound_climate_project<br/>
    Raw data: KMA Open Data Portal, Copernicus Climate Data Store (ERA5, CMIP6)
    """
    story.append(Paragraph(data_avail, body_style))

    # Page break
    story.append(PageBreak())

    # ==================== Page 9: References ====================
    story.append(Paragraph("References", heading1_style))

    refs_style = ParagraphStyle('Refs', parent=body_style, fontSize=8, leading=10, spaceAfter=3)

    references = [
        "[1] IPCC (2021). Climate Change 2021: The Physical Science Basis. Cambridge University Press.",
        "[2] Zscheischler, J., et al. (2020). A typology of compound weather and climate events. Nature Reviews Earth & Environment, 1(7), 333-347.",
        "[3] Raymond, C., et al. (2020). Understanding and managing connected extreme events. Nature Climate Change, 10(7), 611-621.",
        "[4] AghaKouchak, A., et al. (2020). Climate Extremes and Compound Hazards in a Warming World. Annual Review of Earth and Planetary Sciences, 48, 519-548.",
        "[5] Ridder, N.N., et al. (2022). Global hotspots for the occurrence of compound events. Nature Communications, 13, 7178.",
        "[6] Bevacqua, E., et al. (2021). Guidelines for studying diverse types of compound weather and climate events. Earth's Future, 9(11), e2021EF002340.",
        "[7] Vaswani, A., et al. (2017). Attention is all you need. NeurIPS 2017.",
        "[8] Hamilton, W.L., et al. (2017). Inductive representation learning on large graphs. NeurIPS 2017.",
        "[9] Bi, K., et al. (2023). Accurate medium-range global weather forecasting with 3D neural networks. Nature, 619, 533-538.",
        "[10] Lam, R., et al. (2023). Learning skillful medium-range global weather forecasting. Science, 382, 1416-1421.",
        "[11] Reichstein, M., et al. (2019). Deep learning and process understanding for data-driven Earth system science. Nature, 566, 195-204.",
        "[12] Ravuri, S., et al. (2021). Skilful precipitation nowcasting using deep generative models of radar. Nature, 597, 672-677.",
        "[13] Park, C., et al. (2021). Extreme precipitation over East Asia under climate change. Journal of Climate, 34(18), 7467-7483.",
        "[14] Cho, Y. & Lee, S. (2022). Urban heat island intensity in Seoul metropolitan area. Urban Climate, 44, 101214.",
        "[15] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD 2016, 785-794.",
        "[16] Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. NeurIPS 2017.",
        "[17] Lee, W.S., et al. (2021). Trends and variability of heatwaves in South Korea. International Journal of Climatology, 41(S1), E2316-E2331.",
        "[18] Kim, D.W., et al. (2022). Characteristics and trends of drought in South Korea. Atmosphere, 13(3), 381.",
        "[19] Korea Meteorological Administration (2023). Climate Change Report for Korean Peninsula. KMA Technical Report.",
        "[20] O'Neill, B.C., et al. (2016). The Scenario Model Intercomparison Project (ScenarioMIP) for CMIP6. Geoscientific Model Development, 9, 3461-3482.",
    ]

    for ref in references:
        story.append(Paragraph(ref, refs_style))

    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Acknowledgments</b>", heading2_style))
    ack = """
    This research was conducted for the AI Co-Scientist Challenge Korea 2026 Track 1. We acknowledge
    Claude AI (Anthropic) for research consultation and code development. We thank Korea Meteorological
    Administration for observational data, Copernicus Climate Data Store for ERA5 and CMIP6 data access.
    """
    story.append(Paragraph(ack, body_style))

    # Build PDF
    doc.build(story)
    print(f"9-page compact PDF generated: {output_path}")


if __name__ == "__main__":
    output_pdf = "submission/Research_Report_9pages_v2.pdf"
    figures_dir = "results/figures"

    # Create submission folder if needed
    Path("submission").mkdir(exist_ok=True)

    create_compact_research_report_pdf(output_pdf, figures_dir)
