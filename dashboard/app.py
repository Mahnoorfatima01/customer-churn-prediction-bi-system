import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from datetime import datetime
import os

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="ChurnIQ | Customer Retention Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# DESIGN SYSTEM — CUSTOM CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background-color: #F7F8FA;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
        html, body {
        font-family: 'Inter', sans-serif !important;
    }
    p, .stMarkdown, .stMarkdown p {
        color: #1A1D29;
    }
    #MainMenu, footer {visibility: hidden;}
header {background: transparent !important;}
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1300px;}

    .spacer-sm {height: 8px;}
    .spacer-md {height: 18px;}
    .spacer-lg {height: 28px;}

    section[data-testid="stSidebar"] {background-color: #12172B; border-right: 1px solid #1F2540;}
    section[data-testid="stSidebar"] * {color: #E5E7EB !important; font-family: 'Inter', sans-serif !important;}
    section[data-testid="stSidebar"] .stRadio > label {display: none;}
    section[data-testid="stSidebar"] div[role="radiogroup"] {gap: 2px;}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: transparent; border-radius: 8px; padding: 10px 12px;
        margin-bottom: 2px; transition: background-color 0.15s ease; width: 100%;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {background-color: #1E2440;}
    
        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
        padding-left: 0 !important;
    } 
    .kpi-card {
        background: #FFFFFF; border-radius: 14px; padding: 20px 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        border: 1px solid #EDEEF2; height: 148px; display: flex; flex-direction: column;
        justify-content: space-between;
    }
    .kpi-label {font-size: 12px; font-weight: 600; letter-spacing: 0.06em; color: #8B8FA3; text-transform: uppercase;}
    .kpi-value {font-size: 30px; font-weight: 700; color: #12172B; line-height: 1.1;}
    .kpi-desc {font-size: 12.5px; color: #6B7080;}
    .kpi-badge {display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; width: fit-content;}
    .badge-red {background: #FDECEC; color: #C0392B;}
    .badge-amber {background: #FEF6E7; color: #B8860B;}
    .badge-green {background: #E9F9EF; color: #1E8449;}
    .badge-blue {background: #EAF1FF; color: #2E5FC1;}

    .insight-card {
        background: #FFFFFF; border-radius: 12px; padding: 16px 18px;
        border-left: 4px solid #4C6FFF; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        height: 110px; display: flex; flex-direction: column; justify-content: center;
    }
    .insight-tag {font-size: 11px; font-weight: 700; letter-spacing: 0.05em; color: #4C6FFF; text-transform: uppercase;}
    .insight-text {font-size: 13.5px; color: #2A2E3F; margin-top: 6px; line-height: 1.4;}

    .section-title {font-size: 19px; font-weight: 700; color: #12172B; margin-bottom: 2px;}
    .section-subtitle {font-size: 13px; color: #8B8FA3; margin-bottom: 14px;}

    .risk-row {display: flex; align-items: center; margin-bottom: 10px;}
    .risk-label {width: 90px; font-size: 12px; font-weight: 600; color: #4A4E5C;}
    .risk-bar-bg {flex: 1; background: #EEF0F4; border-radius: 6px; height: 10px; overflow: hidden;}
    .risk-bar-fill {height: 100%; border-radius: 6px;}
    .risk-count {width: 50px; text-align: right; font-size: 12px; font-weight: 700; color: #12172B;}

    .action-card {background: #FFFFFF; border-radius: 12px; padding: 18px; border: 1px solid #EDEEF2; height: 140px;}
    .action-num {font-size: 22px; font-weight: 800; color: #D6D9E3;}
    .action-title {font-size: 14px; font-weight: 700; color: #12172B; margin: 4px 0 6px 0;}
    .action-desc {font-size: 13px; color: #6B7080; line-height: 1.4;}

    table {
        width: 100%; border-collapse: collapse; font-size: 13.5px;
        background: #FFFFFF; border-radius: 10px; overflow: hidden;
    }
    table thead th {
        background: #F7F8FA; color: #6B7080; font-weight: 600; font-size: 11.5px;
        letter-spacing: 0.04em; text-transform: uppercase; text-align: left;
        padding: 10px 14px; border-bottom: 1px solid #EDEEF2;
    }
    table tbody td {padding: 11px 14px; border-bottom: 1px solid #F2F3F6; color: #2A2E3F;}
    table tbody tr:hover {background: #FAFBFD;}
    .risk-pill {padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700;}

    .stTabs [data-baseweb="tab-list"] {gap: 4px; border-bottom: 1px solid #EDEEF2;}
    .stTabs [data-baseweb="tab"] {
        height: 38px; font-size: 13.5px; font-weight: 600; color: #8B8FA3;
        border-radius: 8px 8px 0 0; padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {color: #4C6FFF !important; border-bottom: 2px solid #4C6FFF;}

    .stSelectbox > div > div, .stNumberInput > div > div, .stMultiSelect > div > div {
        background-color: #FAFBFD; border: 1px solid #E5E7EE; border-radius: 8px;
    }
    label {font-size: 12.5px !important; font-weight: 600 !important; color: #4A4E5C !important;}

    span[data-baseweb="tag"] {background-color: #EAF1FF !important; color: #2E5FC1 !important;}
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA & MODEL (unchanged logic)
# ============================================
@st.cache_resource
def load_artifacts():
    with open('models/churn_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/telco_churn_processed.csv')
    predictions = pd.read_csv('data/processed/model_predictions.csv')
    comparison = pd.read_csv('data/processed/model_comparison.csv')
    return df, predictions, comparison

artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
num_cols = artifacts['num_cols']
cat_cols = artifacts['cat_cols']
feature_columns = artifacts['feature_columns']
risk_thresholds = artifacts['risk_thresholds']

df, predictions_df, comparison_df = load_data()

try:
    data_updated = datetime.fromtimestamp(
        os.path.getmtime('data/processed/telco_churn_processed.csv')
    ).strftime('%b %d, %Y')
except Exception:
    data_updated = "Unavailable"

best_model_name = comparison_df.sort_values('ROC-AUC', ascending=False).iloc[0]['Model']

# ============================================
# HELPERS (unchanged ML logic)
# ============================================
def prepare_and_predict(input_dict):
    input_df = pd.DataFrame([input_dict])

    def tenure_group(t):
        if t <= 12: return 'New (0-12mo)'
        elif t <= 24: return 'Established (12-24mo)'
        elif t <= 48: return 'Loyal (24-48mo)'
        else: return 'Very Loyal (48mo+)'
    input_df['Tenure_Group'] = input_df['tenure'].apply(tenure_group)

    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies']
    input_df['Service_Count'] = (input_df[service_cols] == 'Yes').sum(axis=1)
    input_df['Total_Services'] = input_df['Service_Count'] \
        + (input_df['PhoneService'] == 'Yes').astype(int) \
        + (input_df['InternetService'] != 'No').astype(int)
    input_df['Avg_Monthly_Spend'] = input_df['MonthlyCharges']
    input_df['Is_New_Customer'] = (input_df['tenure'] <= 6).astype(int)

    input_encoded = pd.get_dummies(input_df, columns=cat_cols)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

    prob = model.predict_proba(input_encoded)[0, 1]
    if prob < risk_thresholds['LOW']:
        risk = 'LOW'
    elif prob < risk_thresholds['MEDIUM']:
        risk = 'MEDIUM'
    else:
        risk = 'HIGH'

    coef_series = pd.Series(model.coef_[0], index=feature_columns)
    contribution = coef_series * input_encoded.iloc[0]
    top_factors = contribution[contribution > 0].sort_values(ascending=False).head(3).index.tolist()

    return prob, risk, top_factors

def recommend_action(risk_level, factors):
    if risk_level == 'LOW':
        return "No action needed — monitor periodically."
    actions = []
    if any('Contract' in f or f == 'tenure' for f in factors):
        actions.append("Offer a discounted longer-term contract upgrade")
    if any('InternetService_Fiber' in f or f == 'MonthlyCharges' for f in factors):
        actions.append("Review pricing/package value for this customer's service tier")
    if any('PaymentMethod_Electronic check' in f for f in factors):
        actions.append("Encourage switch to automatic payment")
    if any('TechSupport' in f or 'OnlineSecurity' in f for f in factors):
        actions.append("Proactive customer support / service check-in")
    if any('New_Customer' in f or 'Tenure_Group_New' in f for f in factors):
        actions.append("Early-stage retention campaign / onboarding follow-up")
    if not actions:
        actions.append("General retention outreach recommended")
    urgency = "URGENT: " if risk_level == 'HIGH' else ""
    return urgency + "; ".join(actions)

def risk_pill_html(risk_level):
    colors = {'HIGH': ('#FDECEC', '#C0392B'), 'MEDIUM': ('#FEF6E7', '#B8860B'), 'LOW': ('#E9F9EF', '#1E8449')}
    bg, fg = colors.get(risk_level, ('#EEE', '#555'))
    return f'<span class="risk-pill" style="background:{bg};color:{fg};">{risk_level}</span>'

def style_chart(fig, height=320):
    fig.update_layout(
        height=height,
        margin=dict(t=20, b=10, l=10, r=10),
        font=dict(family="Inter, sans-serif", color="#4A4E5C", size=12),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(gridcolor="#F0F1F5", showline=True, linecolor="#E5E7EE"),
        yaxis=dict(gridcolor="#F0F1F5", showline=True, linecolor="#E5E7EE"),
        legend=dict(font=dict(size=11))
    )
    return fig

def spacer(size="md"):
    st.markdown(f'<div class="spacer-{size}"></div>', unsafe_allow_html=True)

CHURN_COLOR = {'No': '#2ECC71', 'Yes': '#E74C3C'}
PLOTLY_TEMPLATE = "plotly_white"

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("""
        <div style="padding: 6px 0 18px 0;">
            <span style="font-size:22px; font-weight:800; color:#FFFFFF;">◆ ChurnIQ</span><br>
            <span style="font-size:12px; color:#9AA0B4;">Customer Intelligence</span>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "▣  Executive Overview",
        "◉  Customer Analytics",
        "◔  Churn Intelligence",
        "⌁  Customer Prediction",
        "▤  Model Performance",
        "◈  Explainability",
        "★  Retention Strategy"
    ], label_visibility="collapsed")
    page = page.split("  ", 1)[1]

    st.markdown("<hr style='border-color:#232948; margin: 20px 0 14px 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="font-size:11px; font-weight:700; letter-spacing:0.06em; color:#7A80A0; margin-bottom:8px;">
            SYSTEM STATUS
        </div>
        <div style="font-size:12px; color:#D5D8E5; margin-bottom:4px;">🟢 Model Ready ({best_model_name})</div>
        <div style="font-size:12px; color:#D5D8E5;">🟢 Data Loaded ({len(df):,} records)</div>
    """, unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("""
        <div style="font-size:26px; font-weight:800; color:#12172B;">◆ ChurnIQ</div>
        <div style="font-size:13px; color:#8B8FA3; margin-top:-4px;">Customer Retention Intelligence Platform</div>
    """, unsafe_allow_html=True)
with h2:
    st.markdown(f"""
        <div style="text-align:right; font-size:12px; color:#8B8FA3; margin-top:10px;">
            Data updated: <b style="color:#2A2E3F;">{data_updated}</b><br>
            Model: <b style="color:#2A2E3F;">{best_model_name}</b>
        </div>
    """, unsafe_allow_html=True)
st.markdown("<hr style='margin: 10px 0 20px 0; border-color:#EDEEF2;'>", unsafe_allow_html=True)

# ============================================
# PAGE 1: EXECUTIVE OVERVIEW
# ============================================
if page == "Executive Overview":
    st.markdown('<div class="section-title">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Monitor customer health, identify churn risk, and prioritize retention opportunities.</div>', unsafe_allow_html=True)

    total_customers = len(df)
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    prob_cols = [c for c in predictions_df.columns if c.endswith('_Prob')]
    lr_prob_col = [c for c in prob_cols if 'Logistic Regression' in c]
    prob_series = predictions_df[lr_prob_col[0]] if lr_prob_col else predictions_df[prob_cols[0]]

    low_count = (prob_series < risk_thresholds['LOW']).sum()
    med_count = ((prob_series >= risk_thresholds['LOW']) & (prob_series < risk_thresholds['MEDIUM'])).sum()
    high_count = (prob_series >= risk_thresholds['MEDIUM']).sum()
    revenue_at_risk = df.loc[df['Churn'] == 'Yes', 'MonthlyCharges'].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card"><div><div class="kpi-label">Total Customers</div>
            <div class="kpi-value">{total_customers:,}</div></div>
            <div><div class="kpi-desc">Full customer base</div>
            <span class="kpi-badge badge-blue">Dataset</span></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card"><div><div class="kpi-label">Churn Rate</div>
            <div class="kpi-value">{churn_rate:.2f}%</div></div>
            <div><div class="kpi-desc">Historical churn observed</div>
            <span class="kpi-badge badge-red">Attrition</span></div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card"><div><div class="kpi-label">High-Risk Customers</div>
            <div class="kpi-value">{high_count:,}</div></div>
            <div><div class="kpi-desc">Customers requiring attention</div>
            <span class="kpi-badge badge-amber">High Risk</span></div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card"><div><div class="kpi-label">Monthly Revenue at Risk</div>
            <div class="kpi-value">${revenue_at_risk:,.0f}</div></div>
            <div><div class="kpi-desc">Billed to churned customers</div>
            <span class="kpi-badge badge-red">Revenue</span></div></div>""", unsafe_allow_html=True)

    spacer("lg")
    st.markdown('<div class="section-title" style="font-size:16px;">Executive Insights</div>', unsafe_allow_html=True)
    mtm_churn = df[df['Contract'] == 'Month-to-month']['Churn'].eq('Yes').mean() * 100
    long_tenure_churn = df[df['tenure'] > 48]['Churn'].eq('Yes').mean() * 100

    i1, i2, i3, i4 = st.columns(4)
    with i1:
        st.markdown(f"""<div class="insight-card"><div class="insight-tag">High Risk</div>
            <div class="insight-text">{high_count:,} customers currently classified as high risk require attention.</div></div>""", unsafe_allow_html=True)
    with i2:
        st.markdown(f"""<div class="insight-card"><div class="insight-tag">Churn Driver</div>
            <div class="insight-text">Month-to-month contracts show a {mtm_churn:.0f}% churn rate, the highest of any contract type.</div></div>""", unsafe_allow_html=True)
    with i3:
        st.markdown(f"""<div class="insight-card"><div class="insight-tag">Customer Health</div>
            <div class="insight-text">Customers with 48+ months tenure churn at just {long_tenure_churn:.1f}%.</div></div>""", unsafe_allow_html=True)
    with i4:
        st.markdown(f"""<div class="insight-card"><div class="insight-tag">Revenue Exposure</div>
            <div class="insight-text">${revenue_at_risk:,.0f} in monthly charges is associated with churned customers.</div></div>""", unsafe_allow_html=True)

    spacer("lg")
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown('<div class="section-title" style="font-size:16px;">Customer Churn Distribution</div>', unsafe_allow_html=True)
        churn_counts = df['Churn'].value_counts().reset_index()
        churn_counts.columns = ['Churn', 'Count']
        fig = px.pie(churn_counts, names='Churn', values='Count', color='Churn',
                     color_discrete_map=CHURN_COLOR, hole=0.6)
        st.plotly_chart(style_chart(fig, height=280), use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title" style="font-size:16px;">Customer Health</div>', unsafe_allow_html=True)
        total_scored = low_count + med_count + high_count
        for label, count, color in [('LOW RISK', low_count, '#2ECC71'),
                                      ('MEDIUM RISK', med_count, '#F1C40F'),
                                      ('HIGH RISK', high_count, '#E74C3C')]:
            pct = (count / total_scored * 100) if total_scored else 0
            st.markdown(f"""
                <div class="risk-row">
                    <div class="risk-label">{label}</div>
                    <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:{pct}%; background:{color};"></div></div>
                    <div class="risk-count">{count}</div>
                </div>""", unsafe_allow_html=True)
        high_pct = (high_count / total_scored * 100) if total_scored else 0
        st.markdown(f'<div style="font-size:13px; color:#6B7080; margin-top:10px;">Approximately {high_pct:.1f}% of scored customers are currently classified as high risk.</div>', unsafe_allow_html=True)

    spacer("lg")
    st.markdown('<div class="section-title" style="font-size:16px;">What is happening?</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        st.markdown('<div style="font-size:13px; font-weight:600; color:#4A4E5C; margin-bottom:6px;">Churn by Contract</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='Contract', color='Churn', barmode='group', color_discrete_map=CHURN_COLOR)
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with a2:
        st.markdown('<div style="font-size:13px; font-weight:600; color:#4A4E5C; margin-bottom:6px;">Churn by Tenure</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='tenure', color='Churn', nbins=30, color_discrete_map=CHURN_COLOR)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown('<div class="section-title" style="font-size:16px;">Why is it happening? — Top Churn Drivers</div>', unsafe_allow_html=True)
    coef_df = pd.DataFrame({'Feature': feature_columns, 'Coefficient': model.coef_[0]}) \
        .sort_values('Coefficient', key=abs, ascending=False).head(10).sort_values('Coefficient')
    fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                 color='Coefficient', color_continuous_scale=['#2ECC71', '#EEEEEE', '#E74C3C'])
    fig = style_chart(fig, height=350)
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    spacer("md")
    st.markdown('<div class="section-title" style="font-size:16px;">Customers Requiring Attention</div>', unsafe_allow_html=True)
    if lr_prob_col:
        risk_table = predictions_df[['CustomerID', lr_prob_col[0]]].copy()
        risk_table.columns = ['Customer ID', 'Churn Probability']
        risk_table['Risk'] = risk_table['Churn Probability'].apply(
            lambda p: 'LOW' if p < risk_thresholds['LOW'] else ('MEDIUM' if p < risk_thresholds['MEDIUM'] else 'HIGH'))
        risk_table = risk_table.sort_values('Churn Probability', ascending=False).head(10)
        risk_table['Churn Probability'] = (risk_table['Churn Probability'] * 100).round(1).astype(str) + '%'
        risk_table['Risk'] = risk_table['Risk'].apply(risk_pill_html)
        st.write(risk_table.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.caption("View all high-risk customers in the Retention Strategy page →")

    spacer("lg")
    st.markdown('<div class="section-title" style="font-size:16px;">Recommended Actions</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""<div class="action-card"><div class="action-num">01</div>
            <div class="action-title">RETENTION CAMPAIGN</div>
            <div class="action-desc">Target high-risk customers with short tenure through early engagement offers.</div></div>""", unsafe_allow_html=True)
    with r2:
        st.markdown("""<div class="action-card"><div class="action-num">02</div>
            <div class="action-title">CONTRACT OPTIMIZATION</div>
            <div class="action-desc">Encourage month-to-month customers to move to annual or two-year plans.</div></div>""", unsafe_allow_html=True)
    with r3:
        st.markdown("""<div class="action-card"><div class="action-num">03</div>
            <div class="action-title">CUSTOMER SUPPORT</div>
            <div class="action-desc">Prioritize proactive support outreach for customers without Tech Support.</div></div>""", unsafe_allow_html=True)

# ============================================
# PAGE 2: CUSTOMER ANALYTICS
# ============================================
elif page == "Customer Analytics":
    st.markdown('<div class="section-title">Customer Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Understand who your customers are and how their behavior relates to churn.</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        contract_filter = st.multiselect("Contract", df['Contract'].unique().tolist(), default=df['Contract'].unique().tolist())
    with f2:
        payment_filter = st.multiselect("Payment Method", df['PaymentMethod'].unique().tolist(), default=df['PaymentMethod'].unique().tolist())
    with f3:
        internet_filter = st.multiselect("Internet Service", df['InternetService'].unique().tolist(), default=df['InternetService'].unique().tolist())

    fdf = df[df['Contract'].isin(contract_filter) & df['PaymentMethod'].isin(payment_filter) & df['InternetService'].isin(internet_filter)]

    spacer("sm")
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"""<div class="kpi-card" style="height:110px;"><div class="kpi-label">Filtered Customers</div><div class="kpi-value">{len(fdf):,}</div></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="kpi-card" style="height:110px;"><div class="kpi-label">Avg Monthly Charges</div><div class="kpi-value">${fdf['MonthlyCharges'].mean():.0f}</div></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="kpi-card" style="height:110px;"><div class="kpi-label">Avg Tenure</div><div class="kpi-value">{fdf['tenure'].mean():.0f} mo</div></div>""", unsafe_allow_html=True)

    spacer("md")
    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Services", "Financial", "Behavior"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(fdf, x='gender', color='Churn', barmode='group', color_discrete_map=CHURN_COLOR)
            st.plotly_chart(style_chart(fig), use_container_width=True)
        with c2:
            fig = px.histogram(fdf, x='SeniorCitizen', color='Churn', barmode='group', color_discrete_map=CHURN_COLOR)
            st.plotly_chart(style_chart(fig), use_container_width=True)
    with tab2:
        fig = px.histogram(fdf, x='Total_Services', color='Churn', barmode='group', color_discrete_map=CHURN_COLOR)
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with tab3:
        fig = px.histogram(fdf, x='MonthlyCharges', color='Churn', nbins=30, color_discrete_map=CHURN_COLOR)
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with tab4:
        fig = px.histogram(fdf, x='tenure', color='Churn', nbins=30, color_discrete_map=CHURN_COLOR)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with st.expander("View detailed customer table"):
        st.dataframe(fdf.head(200), use_container_width=True)

# ============================================
# PAGE 3: CHURN INTELLIGENCE
# ============================================
elif page == "Churn Intelligence":
    st.markdown('<div class="section-title">Churn Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Identify patterns behind customer attrition.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('<div style="font-size:13px; font-weight:600; color:#4A4E5C; margin-bottom:6px;">Churn by Contract</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='Contract', color='Churn', barmode='group', color_discrete_map=CHURN_COLOR)
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with c2:
        mtm = df[df['Contract'] == 'Month-to-month']['Churn'].eq('Yes').mean() * 100
        st.markdown(f"""<div class="insight-card"><div class="insight-tag">Insight</div>
            <div class="insight-text">Month-to-month contracts churn at {mtm:.0f}%, far higher than longer commitments.</div></div>""", unsafe_allow_html=True)

    spacer("md")
    c3, c4 = st.columns([2, 1])
    with c3:
        st.markdown('<div style="font-size:13px; font-weight:600; color:#4A4E5C; margin-bottom:6px;">Churn by Payment Method</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='PaymentMethod', color='Churn', barmode='group', color_discrete_map=CHURN_COLOR)
        fig.update_xaxes(tickangle=15)
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with c4:
        ec = df[df['PaymentMethod'] == 'Electronic check']['Churn'].eq('Yes').mean() * 100
        st.markdown(f"""<div class="insight-card"><div class="insight-tag">Insight</div>
            <div class="insight-text">Electronic check users churn at {ec:.0f}%, the highest of any payment method.</div></div>""", unsafe_allow_html=True)

    spacer("md")
    c5, c6 = st.columns([2, 1])
    with c5:
        st.markdown('<div style="font-size:13px; font-weight:600; color:#4A4E5C; margin-bottom:6px;">Churn by Tech Support</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='TechSupport', color='Churn', barmode='group', color_discrete_map=CHURN_COLOR)
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with c6:
        ts = df[df['TechSupport'] == 'No']['Churn'].eq('Yes').mean() * 100
        st.markdown(f"""<div class="insight-card"><div class="insight-tag">Insight</div>
            <div class="insight-text">Customers without Tech Support churn at {ts:.0f}%, roughly double those with it.</div></div>""", unsafe_allow_html=True)

    spacer("lg")
    st.markdown('<div class="section-title" style="font-size:16px;">Churn by Monthly Charges</div>', unsafe_allow_html=True)
    fig = px.histogram(df, x='MonthlyCharges', color='Churn', nbins=30, color_discrete_map=CHURN_COLOR)
    st.plotly_chart(style_chart(fig), use_container_width=True)

# ============================================
# PAGE 4: CUSTOMER PREDICTION
# ============================================
elif page == "Customer Prediction":
    st.markdown('<div class="section-title">Customer Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Enter customer information to estimate churn probability.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Single Customer", "Batch Upload (CSV)"])

    with tab1:
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("Gender", ['Female', 'Male'])
                SeniorCitizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
                Partner = st.selectbox("Partner", ['No', 'Yes'])
                Dependents = st.selectbox("Dependents", ['No', 'Yes'])
                tenure = st.number_input("Tenure (months)", 0, 100, 12)
            with col2:
                PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
                MultipleLines = st.selectbox("Multiple Lines", ['No', 'Yes'])
                InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
                OnlineSecurity = st.selectbox("Online Security", ['No', 'Yes'])
                OnlineBackup = st.selectbox("Online Backup", ['No', 'Yes'])
                DeviceProtection = st.selectbox("Device Protection", ['No', 'Yes'])
            with col3:
                TechSupport = st.selectbox("Tech Support", ['No', 'Yes'])
                StreamingTV = st.selectbox("Streaming TV", ['No', 'Yes'])
                StreamingMovies = st.selectbox("Streaming Movies", ['No', 'Yes'])
                Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
                PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
                PaymentMethod = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
                MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)

            predict_clicked = st.button("Predict Churn Risk", type="primary")

        if predict_clicked:
            input_dict = {
                'gender': gender, 'SeniorCitizen': SeniorCitizen, 'Partner': Partner,
                'Dependents': Dependents, 'tenure': tenure, 'PhoneService': PhoneService,
                'MultipleLines': MultipleLines, 'InternetService': InternetService,
                'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup,
                'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport,
                'StreamingTV': StreamingTV, 'StreamingMovies': StreamingMovies,
                'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
                'PaymentMethod': PaymentMethod, 'MonthlyCharges': MonthlyCharges
            }
            prob, risk, factors = prepare_and_predict(input_dict)
            action = recommend_action(risk, factors)
            risk_colors = {'HIGH': '#E74C3C', 'MEDIUM': '#F1C40F', 'LOW': '#2ECC71'}

            spacer("md")
            rc1, rc2 = st.columns([1, 2])
            with rc1:
                st.markdown(f"""
                    <div class="kpi-card" style="height:180px; text-align:center; align-items:center;">
                        <div class="kpi-label">CHURN PROBABILITY</div>
                        <div class="kpi-value" style="font-size:44px; color:{risk_colors[risk]};">{prob*100:.1f}%</div>
                        <span class="kpi-badge" style="background:{risk_colors[risk]}22; color:{risk_colors[risk]};">{risk} RISK</span>
                    </div>""", unsafe_allow_html=True)
            with rc2:
                factors_html = "".join([f"<li>{f}</li>" for f in factors]) if factors else "<li>No significant risk factors identified</li>"
                st.markdown(f"""
                    <div class="kpi-card" style="height:180px;">
                        <div class="kpi-label">WHY THIS CUSTOMER IS AT RISK</div>
                        <ul style="margin-top:8px; font-size:14px; color:#2A2E3F;">{factors_html}</ul>
                        <div class="kpi-label" style="margin-top:14px;">RECOMMENDED ACTION</div>
                        <div style="font-size:14px; color:#2A2E3F; margin-top:4px;">{action}</div>
                    </div>""", unsafe_allow_html=True)

    with tab2:
        st.write("Upload a CSV with the same columns as the original dataset for batch predictions.")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            results = []
            for _, row in batch_df.iterrows():
                try:
                    input_dict = row.to_dict()
                    prob, risk, factors = prepare_and_predict(input_dict)
                    results.append({
                        'CustomerID': row.get('customerID', 'N/A'),
                        'Churn_Probability': round(prob, 4),
                        'Risk_Level': risk,
                        'Top_Risk_Factors': ", ".join(factors),
                        'Recommended_Action': recommend_action(risk, factors)
                    })
                except Exception as e:
                    st.warning(f"Skipped a row due to error: {e}")
            if results:
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions CSV", csv, "batch_predictions.csv", "text/csv")

# ============================================
# PAGE 5: MODEL PERFORMANCE
# ============================================
elif page == "Model Performance":
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Compare candidate models and evaluate predictive reliability.</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="insight-card" style="border-left-color:#2ECC71; height:auto; padding:14px 18px;">
        <div class="insight-tag" style="color:#1E8449;">Selected Model</div>
        <div class="insight-text"><b>{best_model_name}</b> — highest ROC-AUC among evaluated models.</div></div>""", unsafe_allow_html=True)

    spacer("md")
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Confusion Matrix", "ROC Curve", "Precision-Recall"])

    with tab1:
        st.dataframe(comparison_df.style.highlight_max(subset=['Accuracy','Precision','Recall','F1-Score','ROC-AUC','PR-AUC'], color='#E9F9EF'), use_container_width=True)
        fig = px.bar(comparison_df, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], barmode='group')
        st.plotly_chart(style_chart(fig, height=380), use_container_width=True)

    model_names = [c.replace('_Pred', '') for c in predictions_df.columns if c.endswith('_Pred')]
    selected_model = st.selectbox("Select model for detailed view", model_names, key="perf_model_select")
    y_true = predictions_df['Actual_Churn']
    y_pred = predictions_df[f'{selected_model}_Pred']
    y_prob = predictions_df[f'{selected_model}_Prob']

    with tab2:
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                         labels=dict(x="Predicted", y="Actual"), x=['No Churn', 'Churn'], y=['No Churn', 'Churn'])
        st.plotly_chart(style_chart(fig, height=400), use_container_width=True)

    with tab3:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={auc:.3f})', line=dict(color='#4C6FFF')))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Guess', line=dict(dash='dash', color='#B0B4C0')))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(style_chart(fig, height=400), use_container_width=True)

    with tab4:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, name=f'PR (AUC={pr_auc:.3f})', line=dict(color='#4C6FFF')))
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(style_chart(fig, height=400), use_container_width=True)

# ============================================
# PAGE 6: EXPLAINABILITY
# ============================================
elif page == "Explainability":
    st.markdown('<div class="section-title">Model Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Understand which factors influence churn predictions.</div>', unsafe_allow_html=True)

    coef_df = pd.DataFrame({'Feature': feature_columns, 'Coefficient': model.coef_[0]}) \
        .sort_values('Coefficient', key=abs, ascending=False).head(15)

    fig = px.bar(coef_df.sort_values('Coefficient'), x='Coefficient', y='Feature', orientation='h',
                 color='Coefficient', color_continuous_scale=['#2ECC71', '#EEEEEE', '#E74C3C'])
    fig = style_chart(fig, height=450)
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How to interpret this"):
        st.write("""
        Each bar shows how strongly a feature pushes predictions toward churn (red, positive) or away from churn (green, negative),
        based on the trained Logistic Regression model's coefficients. For example, a feature like *Contract_Two year* having a
        strong negative coefficient means having a two-year contract meaningfully reduces predicted churn risk, holding other
        factors constant.
        """)

# ============================================
# PAGE 7: RETENTION STRATEGY
# ============================================
elif page == "Retention Strategy":
    st.markdown('<div class="section-title">Retention Strategy</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Turn churn predictions into targeted customer actions.</div>', unsafe_allow_html=True)

    prob_col = [c for c in predictions_df.columns if 'Logistic Regression_Prob' in c]
    if prob_col:
        pred_view = predictions_df[['CustomerID', 'Actual_Churn', prob_col[0]]].copy()
        pred_view.columns = ['CustomerID', 'Actual_Churn', 'Churn_Probability']
        pred_view['Risk_Level'] = pred_view['Churn_Probability'].apply(
            lambda p: 'LOW' if p < risk_thresholds['LOW'] else ('MEDIUM' if p < risk_thresholds['MEDIUM'] else 'HIGH'))

        high_n = (pred_view['Risk_Level'] == 'HIGH').sum()
        med_n = (pred_view['Risk_Level'] == 'MEDIUM').sum()
        low_n = (pred_view['Risk_Level'] == 'LOW').sum()

        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(f"""<div class="kpi-card" style="height:110px;"><div class="kpi-label" style="color:#C0392B;">HIGH PRIORITY</div>
                <div class="kpi-value">{high_n}</div><div class="kpi-desc">Requiring immediate attention</div></div>""", unsafe_allow_html=True)
        with p2:
            st.markdown(f"""<div class="kpi-card" style="height:110px;"><div class="kpi-label" style="color:#B8860B;">MEDIUM PRIORITY</div>
                <div class="kpi-value">{med_n}</div><div class="kpi-desc">Showing warning signals</div></div>""", unsafe_allow_html=True)
        with p3:
            st.markdown(f"""<div class="kpi-card" style="height:110px;"><div class="kpi-label" style="color:#1E8449;">LOW PRIORITY</div>
                <div class="kpi-value">{low_n}</div><div class="kpi-desc">Relatively low churn risk</div></div>""", unsafe_allow_html=True)

        spacer("lg")
        risk_filter = st.multiselect("Filter by Risk Level", ['LOW', 'MEDIUM', 'HIGH'], default=['HIGH'])
        filtered = pred_view[pred_view['Risk_Level'].isin(risk_filter)].sort_values('Churn_Probability', ascending=False)
        display_df = filtered.copy()
        display_df['Churn_Probability'] = (display_df['Churn_Probability'] * 100).round(1).astype(str) + '%'
        display_df['Risk_Level'] = display_df['Risk_Level'].apply(risk_pill_html)
        st.write(display_df.head(50).to_html(escape=False, index=False), unsafe_allow_html=True)

        st.caption(f"Showing {min(len(filtered), 50)} of {len(filtered)} customers. Risk thresholds are configurable "
                   f"(currently LOW < {risk_thresholds['LOW']}, MEDIUM < {risk_thresholds['MEDIUM']}).")
        st.warning("These recommendations are pattern-based suggestions derived from historical data. They do not guarantee retention outcomes.")
    else:
        st.error("Logistic Regression predictions not found in saved results.")