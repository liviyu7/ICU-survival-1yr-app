import pathlib
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 加载预训练模型
@st.cache_resource
def load_model():
    current_dir = pathlib.Path(__file__).parent
    model_path = current_dir / "optimized_rsf.pkl"
    return joblib.load(model_path)

model = load_model()

# 网页标题
st.title("ICU老年患者1年生存率预测系统")
st.markdown("""
**临床指导说明**：  
请输入您的临床特征，系统将自动预测您的1年生存风险。  
高风险患者（>70%）建议加强监护和定期随访。
""")

# 侧边栏输入界面
with st.sidebar:
    st.header("患者特征输入")

    with st.expander("基本信息"):
        gender = st.selectbox("性别",options=[0, 1],help="0: 女性, 1:男性")
        raw_age = st.selectbox("年龄分组", options=[0, 1, 2, 3],help="0: 61-70岁, 1: 71-80岁, 2: 81-90岁, 3: ≥91岁")

    # 二分类特征分组显示
    with st.expander("躯体健康指标"):
        malnutrition = st.checkbox("营养不良")
        mobility = st.checkbox("行动障碍")
        copd = st.checkbox("慢性阻塞性肺病")
        chronic_renal_failure = st.checkbox("慢性肾衰竭")
        abnormal_liver_function = st.checkbox("肝功能异常")
        pressure_ulcer = st.checkbox("压疮")
        history_of_falls = st.checkbox("跌倒史")
        sarcopenia = st.checkbox("肌肉减少症")
        hearing_impairment = st.checkbox("听力障碍")
        osteoporosis = st.checkbox("骨质疏松症")
        degenerative_joint_disease = st.checkbox("关节退行性疾病")
        coronary_atherosclerosis = st.checkbox("冠状动脉粥样硬化")
        dysphagia = st.checkbox("吞咽困难")
        hypertension = st.checkbox("高血压")

    with st.expander("社会支持"):
        insurance = st.selectbox("保险状况", options=[0, 1, 2, 3],help="0: 其他, 1: 基本医保, 2: 全面医保, 3: 个人保险")
        marital_status = st.selectbox("婚姻状况",options=[0, 1],help="0: 未婚/离异/丧偶/独居, 1:已婚")

    with st.expander("心理健康指标"):
        sleep_disorder = st.checkbox("睡眠障碍")
        anxiety = st.checkbox("焦虑")
        depression = st.checkbox("抑郁")
        delirium = st.checkbox("谵妄")

    with st.expander("药物使用情况"):
        drug_renal_failure = st.checkbox("使用肾功能不全药物")
        drug_antithrombotic = st.checkbox("使用抗血栓药物")
        drug_cardiovascular = st.checkbox("使用心血管药物")
        drug_chronic_pain = st.checkbox("使用慢性疼痛药物")
        drug_diabetes = st.checkbox("使用糖尿病药物")
        drug_urinary_incontinence = st.checkbox("使用尿失禁药物")

# 构建输入数据框
input_data = pd.DataFrame({
    "gender": [int(gender)],
    "raw_age": [raw_age],
    "marital_status": [int(marital_status)],
    "insurance": [insurance],
    "malnutrition": [int(malnutrition)],
    "mobility": [int(mobility)],
    "abnormal_liver_function": [int(abnormal_liver_function)],
    "dysphagia": [int(dysphagia)],
    "chronic_renal_failure": [int(chronic_renal_failure)],
    "sarcopenia": [int(sarcopenia)],
    "hypertension": [int(hypertension)],
    "coronary_atherosclerosis": [int(coronary_atherosclerosis)],
    "history_of_falls": [int(history_of_falls)],
    "copd": [int(copd)],
    "pressure_ulcer": [int(pressure_ulcer)],
    "osteoporosis": [int(osteoporosis)],
    "degenerative_joint_disease": [int(degenerative_joint_disease)],
    "sleep_disorder": [int(sleep_disorder)],
    "anxiety": [int(anxiety)],
    "delirium": [int(delirium)],
    "depression": [int(depression)],
    "hearing_impairment": [int(hearing_impairment)],
    "drug_urinary_incontinence": [int(drug_urinary_incontinence)],
    "drug_cardiovascular": [int(drug_cardiovascular)],
    "drug_antithrombotic": [int(drug_antithrombotic)],
    "drug_renal_failure": [int(drug_renal_failure)],
    "drug_chronic_pain": [int(drug_chronic_pain)],
    "drug_diabetes": [int(drug_diabetes)]
})

# 使用训练时的特征顺序排列预测输入
correct_order = model.feature_names_in_
input_data = input_data[correct_order]

# 预测按钮
if st.button("开始预测"):
    try:
        # 获取预测结果
        cum_hazard_fns = model.predict_cumulative_hazard_function(input_data)
        max_time = 365
        risk_score = 1 - np.exp(-cum_hazard_fns[0](max_time))

        # 结果可视化
        st.subheader("预测结果")

        # 进度条显示风险等级
        risk_percent = risk_score * 100
        st.progress(int(risk_percent))

        # 颜色标注风险等级
        if risk_percent < 30:
            risk_level = "低风险"
            color = "green"
        elif 30 <= risk_percent < 70:
            risk_level = "中风险"
            color = "orange"
        else:
            risk_level = "高风险"
            color = "red"

        st.markdown(f"""
        <div style='border-radius: 10px; padding: 20px; background-color: #f0f2f6;'>
            <h3 style='color: {color};'>1年死亡风险: <b>{risk_percent:.1f}%</b> ({risk_level})</h3>
        </div>
        """, unsafe_allow_html=True)

        # 临床建议
        st.subheader("临床建议")
        if risk_level == "高风险":
            st.warning("""
            **建议措施**：
            1. 加强生命体征监测（每小时）
            2. 启动多学科会诊
            3. 家属风险告知
            4. 考虑转入ICU监护
            """)
        elif risk_level == "中风险":
            st.info("""
            **建议措施**：
            1. 每日两次生命体征监测
            2. 营养支持治疗
            3. 预防跌倒措施
            4. 每周随访评估
            """)
        else:
            st.success("""
            **建议措施**：
            1. 常规护理监测
            2. 健康教育指导
            3. 每月门诊随访
            """)

    except Exception as e:
        st.error(f"预测错误: {str(e)}")

# 运行说明
st.markdown("---")
with st.expander("使用说明"):
    st.write("""
    1. 在左侧边栏输入/选择患者特征
    2. 点击【开始预测】按钮获取结果
    3. 根据风险等级查看临床建议
    4. 高风险患者建议打印报告留存
    """)

