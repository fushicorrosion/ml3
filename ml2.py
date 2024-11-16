import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# 定义机器学习模型字典
models = {
    '线性回归': LinearRegression,
    '随机森林回归': RandomForestRegressor,
    '支持向量回归': SVR,
    '决策树回归': DecisionTreeRegressor,
    'K近邻回归': KNeighborsRegressor
}

# 侧边栏选择
st.sidebar.title("机器学习应用")
app_mode = st.sidebar.selectbox("选择操作", ["训练模型", "进行预测"])

# 初始化 session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'label' not in st.session_state:
    st.session_state.label = None
if 'data' not in st.session_state:
    st.session_state.data = None

if app_mode == "训练模型":
    # 数据上传
    st.subheader("上传数据集")
    uploaded_file = st.file_uploader("选择Excel文件", type=["xlsx", "xls"])

    if uploaded_file:
        st.session_state.data = pd.read_excel(uploaded_file)
        st.write("数据预览：", st.session_state.data.head())

        # 选择特征列和标签列
        st.subheader("选择特征和标签")
        st.session_state.features = st.multiselect("特征列", st.session_state.data.columns.tolist())
        st.session_state.label = st.selectbox("标签列", st.session_state.data.columns.tolist())

        # 选择机器学习算法
        st.subheader("选择机器学习算法")
        algorithm = st.selectbox("选择算法", list(models.keys()))

        # 超参数设置
        st.subheader("超参数设置")
        if algorithm == "随机森林回归":
            n_estimators = st.sidebar.slider("树的数量", 1, 100, 10)
            max_depth = st.sidebar.slider("树的最大深度", 1, 20, 5)
        elif algorithm == "支持向量回归":
            C = st.sidebar.slider("惩罚参数 C", 0.01, 100.0, 1.0)
            epsilon = st.sidebar.slider("epsilon", 0.0, 1.0, 0.1)
        elif algorithm == "K近邻回归":
            n_neighbors = st.sidebar.slider("邻居数量", 1, 20, 5)

        # 划分训练集和测试集
        test_size = st.sidebar.slider("测试集比例", 0.1, 0.9, 0.2)

        if st.button("训练模型"):
            X = st.session_state.data[st.session_state.features]
            y = st.session_state.data[st.session_state.label]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # 初始化模型
            model_class = models[algorithm]
            if algorithm == "随机森林回归":
                st.session_state.model = model_class(n_estimators=n_estimators, max_depth=max_depth)
            elif algorithm == "支持向量回归":
                st.session_state.model = model_class(C=C, epsilon=epsilon)
            elif algorithm == "K近邻回归":
                st.session_state.model = model_class(n_neighbors=n_neighbors)
            else:
                st.session_state.model = model_class()

            # 训练模型
            st.session_state.model.fit(X_train, y_train)

            # 预测
            y_train_pred = st.session_state.model.predict(X_train)
            y_test_pred = st.session_state.model.predict(X_test)

            # 绘制散点图
            train_df = pd.DataFrame({"真实值": y_train, "预测值": y_train_pred})
            test_df = pd.DataFrame({"真实值": y_test, "预测值": y_test_pred})

            fig = px.scatter(train_df, x="真实值", y="预测值", title="训练集预测效果", color_discrete_sequence=["blue"])
            fig.add_scatter(x=test_df["真实值"], y=test_df["预测值"], mode='markers', name='测试集', marker=dict(color='red'))
            st.plotly_chart(fig)

elif app_mode == "进行预测":
    st.subheader("输入参数进行预测")

    if st.session_state.model is None:
        st.warning("请先训练模型以便进行预测。")
    else:
        # 输入框
        input_data = {}
        for feature in st.session_state.features:
            input_data[feature] = st.number_input(f"输入 {feature}", format="%.4f")

        if st.button("进行预测"):
            input_df = pd.DataFrame([input_data])
            prediction = st.session_state.model.predict(input_df)
            st.write("预测结果：", prediction[0])
