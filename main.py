import pandas as pd
import streamlit as st
from get_datasource.machine_predict import  aircrft_aging_predict
from get_datasource.get_datasources_class import Createtable

if __name__ == '__main__':
    st.title("机龄对成本影响的研究")

    st.header("一、请输入机型参数：")
    a = Createtable()
    # # 显示所有列
    # pd.set_option('display.max_columns', None)
    available_aircft_type_spd_list,available_aircft_type_spd_num,A= a.concat_data()
    # print(A)

    st.write('支持预测的规划部机型数量:',available_aircft_type_spd_num)
    pred_aircft_type_spd = st.selectbox(
        label="可预测规划部机型:",
        options=available_aircft_type_spd_list
    )
    list_of_pred_aircft_type_spd = pred_aircft_type_spd.split() if pred_aircft_type_spd else ['A320']

    pred_aircft_aging_y = st.slider("预估机龄", 0,30,10)
    pred_aircft_aging_y = int(pred_aircft_aging_y)

    pred_flight_hours = st.slider("预估飞行小时", 0.00,20.00,2.00)

    st.header("二、请输入模型参数：")
    linearregression_n = st.text_input("线性回归模型的多项式特征阶数", 2)
    linearregression_n = int(linearregression_n)

    validate_size_in = st.slider("测试集占比", 0.00,1.00,0.80)
    test_size_in = 1-validate_size_in

    random_state_in = st.text_input("随机训练次数上限", 5)
    random_state_in = int(random_state_in)

    # pred_aircft_type_spd = st.text_input("",'A320NEO')

    (valid_rows_num,y_pred_test_fuel_consumption,y_pred_test_maintenance_cost,y_pred_test_aircft_engi_deprn_cost,
    pred_test_model_intercept,pred_test_model_coef,mse_validate,r2_validate)\
        =aircrft_aging_predict(linearregression_n,test_size_in,random_state_in,
                               list_of_pred_aircft_type_spd,A,pred_aircft_aging_y,pred_flight_hours)

    st.header("三、预测结果展示：")
    st.write("该机型本次测算依据的历史数据量: ", valid_rows_num)
    st.write("预估该航班消耗燃油重量(吨): ", y_pred_test_fuel_consumption)
    st.write("预估分摊在该航班的维修费用(元):", y_pred_test_maintenance_cost)
    st.write("预估分摊在该航班的折旧成本(元):", y_pred_test_aircft_engi_deprn_cost)

    st.header("四、模型准确度评估及关键参数展示：")
    st.write("pred_validate均方误差（MSE）: ", mse_validate)
    st.write("pred_validate决定系数（R^2）: ", r2_validate)
    st.write("pred_test截距（Intercept）: ", pd.DataFrame(pred_test_model_intercept,index=['油耗','维修','折旧'],columns=['截距']))
    st.write("pred_test系数（Coefficients）: ", pd.DataFrame(pred_test_model_coef,index=['油耗','维修','折旧']))