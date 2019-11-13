import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import datetime
import keras
from keras.layers import GRU, LSTM, Input, Dense, concatenate, Reshape, Dropout, average, BatchNormalization, add, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_absolute_percentage_error, mean_squared_error
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = './selected_data_ISONE.csv'
data = pd.read_csv(path)

# 构建特征
d_max_daily = data.groupby('date').demand.max().to_numpy()
d_min_daily = data.groupby('date').demand.min().to_numpy()

loads = data['demand'].to_numpy().reshape(-1, 1)

d_max = np.zeros(len(loads))
d_min = np.zeros(len(loads))
for i in range(len(loads)):
    n_day = int(i / 24)
    d_max[i] = d_max_daily[n_day]
    d_min[i] = d_min_daily[n_day]

# 标准化
ss_load = StandardScaler()
loads = ss_load.fit_transform(loads)
d_max = ss_load.transform(d_max.reshape(-1, 1))
d_min = ss_load.transform(d_min.reshape(-1, 1))
# TODO
d_max = d_max + 0.01*d_max
d_min = d_min - 0.01*d_min
temperatures = data['temperature'].to_numpy().reshape(-1, 1)
ss_temperature = StandardScaler()
temperatures = ss_temperature.fit_transform(temperatures)

# 构建工作日特征
data['week_day'] = data['weekday'].map(lambda x: 0 if x < 6 else 1)
weekdays = data['week_day'].to_numpy()

# 构建节日和季节特征
iter_date = datetime.date(2003, 3, 1)
seasons = np.zeros((24 * 4324,))  # 4324为数据的总天数
festivals = np.zeros((24 * 4324,))
for index in range(4324):
    month = iter_date.month
    day = iter_date.day
    for j in range(24):
        if (month == 4) | (month == 5) | ((month == 3) and (day > 7)) | ((month == 6) and (day < 8)):
            seasons[index * 24 + j] = 0
        elif (month == 7) | (month == 8) | ((month == 6) and (day > 7)) | ((month == 9) and (day < 8)):
            seasons[index * 24 + j] = 1
        elif (month == 10) | (month == 11) | ((month == 9) and (day > 7)) | ((month == 12) and (day < 8)):
            seasons[index * 24 + j] = 2
        elif (month == 1) | (month == 2) | ((month == 12) and (day > 7)) | ((month == 3) and (day < 8)):
            seasons[index * 24 + j] = 3

        if (month == 7) and (day == 4):
            festivals[index * 24 + j] = 1
        if (month == 11) and (iter_date.weekday() == 4) and (day + 7 > 30):
            festivals[index * 24 + j] = 1
        if (month == 12) and (day == 25):
            festivals[index * 24 + j] = 1
    iter_date = iter_date + datetime.timedelta(1)


# 训练和测试数据
# 训练集从2003年开始
# 测试集从2008年开始
num_pre_day = 84
train_day = 1683
test_day = 365
all_day = 84+1683+365


# 把过去30天作为历史数据
def data_split(load, weekday, season, festival, temperature, d_max, d_min, num_pre_day=num_pre_day, num_trainday=train_day,
               all_day=all_day, validation_split=0.1):
    # 除去84天的数据
    num_sample = (all_day - num_pre_day)*24

    x_load_pre_24h = []
    x_load_pre_7day = []
    t_temp_pre_7day = []
    x_load_pre_2months = []
    t_temp_pre_2months = []
    x_load_pre_3months = []
    t_temp_pre_3months = []
    t_temp = []
    w_weekday = []
    s_season = []
    f_festival = []
    x_load_max = []
    x_load_min = []
    y = []
    for i in range(num_pre_day * 24, (train_day+test_day+num_pre_day) * 24):
        # 提前24小时的数据--这部分可以用来GRU
        x_load_pre_24h.append(load[i-24:i])
        # 提前7天的数据
        index_x_pre_7day = [i - 24, i - 48, i - 72, i - 96, i - 120, i - 144, i - 168]
        x_load_pre_7day.append(load[index_x_pre_7day])
        t_temp_pre_7day.append(temperature[index_x_pre_7day])
        # 提前两个月的数据
        index_x_pre_2months = [i - 168, i - 336, i - 504, i - 672, i - 840, i - 1008, i - 1176, i - 1344]
        x_load_pre_2months.append(load[index_x_pre_2months])
        t_temp_pre_2months.append(temperature[index_x_pre_2months])
        # 提前三个月的数据
        index_x_pre_3months = [i - 672, i - 1344, i - 2016]
        x_load_pre_3months.append(load[index_x_pre_3months])
        t_temp_pre_3months.append(temperature[index_x_pre_3months])

        # 得到当前小时下的温度，季节，工作日和假日
        t_temp.append(temperature[i])

        season_onehot = np.zeros(4)
        season_onehot[int(season[i])] = 1
        s_season.append(season_onehot)

        weekday_onehot = np.zeros(2)
        weekday_onehot[int(weekday[i])] = 1
        w_weekday.append(weekday_onehot)

        festival_onehot = np.zeros(2)
        festival_onehot[int(festival[i])] = 1
        f_festival.append(festival_onehot)

        x_load_max.append(d_max[i])
        x_load_min.append(d_min[i])
        y.append(load[i])
    # 该时刻前24个小时的demand
    x_d_pre_24h = np.array(x_load_pre_24h).squeeze()
    # 该时刻前7天内对应该时刻的demand和temperature
    x_d_pre_7day = np.array(x_load_pre_7day).squeeze()
    x_t_pre_7day = np.array(t_temp_pre_7day).squeeze()

    # 该时刻前8周内对应该时刻的demand和temperature
    x_d_pre_2months = np.array(x_load_pre_2months).squeeze()
    x_t_pre_2months = np.array(t_temp_pre_2months).squeeze()

    # 该时刻前3个月内对应该时刻的demand和temperature
    x_d_pre_3month = np.array(x_load_pre_3months).squeeze()
    x_t_pre_3month = np.array(t_temp_pre_3months).squeeze()

    # 该时刻内对应的的temperature特征
    x_temperature = np.array(t_temp).squeeze()

    # 该时刻内对应的季节，工作日，节日特征
    x_season_onehot = np.array(s_season).squeeze()
    x_weekday_onehot = np.array(w_weekday).squeeze()
    x_festival_onehot = np.array(f_festival).squeeze()

    x_max_daily_fill = np.array(x_load_max)
    x_min_daily_fill = np.array(x_load_min)

    # 31608小时的demand
    y_1 = np.array(y).squeeze()

    x_train = []
    y_train = []
    x_val, y_val = [], []
    x_test, y_test = [], []

    num_train = num_trainday * 24
    num_val = int(num_trainday * validation_split)

    for i in range(24):
        # TODO 应该改成i:
        x_train.append(
            [x_d_pre_24h[i:num_train:24, i:], x_d_pre_7day[i:num_train:24, :], x_t_pre_7day[i:num_train:24, :],
             x_d_pre_2months[i:num_train:24, :], x_t_pre_2months[i:num_train:24, :], x_d_pre_3month[i:num_train:24, :],
             x_t_pre_3month[i:num_train:24, :], x_temperature[i:num_train:24], x_max_daily_fill[i:num_train:24],
             x_min_daily_fill[i:num_train:24], x_season_onehot[i:num_train:24, :],
             x_weekday_onehot[i:num_train:24, :],
             x_festival_onehot[i:num_train:24, :]])
        x_val.append([x_d_pre_24h[num_train - num_val + i:num_train:24, i:],
                      x_d_pre_7day[num_train - num_val + i:num_train:24, :],
                      x_t_pre_7day[num_train - num_val + i:num_train:24, :],
                      x_d_pre_2months[num_train - num_val + i:num_train:24, :],
                      x_t_pre_2months[num_train - num_val + i:num_train:24, :],
                      x_d_pre_3month[num_train - num_val + i:num_train:24, :],
                      x_t_pre_3month[num_train - num_val + i:num_train:24, :],
                      x_temperature[num_train - num_val + i:num_train:24],
                      x_max_daily_fill[num_train - num_val + i:num_train:24],
                      x_min_daily_fill[num_train - num_val + i:num_train:24],
                      x_season_onehot[num_train - num_val + i:num_train:24, :],
                      x_weekday_onehot[num_train - num_val + i:num_train:24, :],
                      x_festival_onehot[num_train - num_val + i:num_train:24, :]])
        x_test.append([x_d_pre_24h[num_train + i:num_sample:24, i:], x_d_pre_7day[num_train + i:num_sample:24, :],
                       x_t_pre_7day[num_train + i:num_sample:24, :], x_d_pre_2months[num_train + i:num_sample:24, :],
                       x_t_pre_2months[num_train + i:num_sample:24, :], x_d_pre_3month[num_train + i:num_sample:24, :],
                       x_t_pre_3month[num_train + i:num_sample:24, :], x_temperature[num_train + i:num_sample:24],
                       x_max_daily_fill[num_train + i:num_sample:24], x_min_daily_fill[num_train + i:num_sample:24],
                       x_season_onehot[num_train + i:num_sample:24, :],
                       x_weekday_onehot[num_train + i:num_sample:24, :],
                       x_festival_onehot[num_train + i:num_sample:24, :]])
        y_train.append(y_1[i:num_train:24])
        y_val.append(y_1[num_train-num_val+i:num_train:24])
        y_test.append(y_1[num_train + i:num_sample:24])
    return x_train, y_train, x_val, y_val, x_test, y_test


x_train, y_train, x_val, y_val, x_test, y_test = data_split(loads, weekdays, seasons, festivals, temperatures, d_max, d_min)


def get_x_y(x, y):
    x_new = []
    y_new = []
    for i in range(24):
        # 该时刻前7天内对应该时刻的demand
        x_new.append(x[i][1])
        # 该时刻前8周内对应该时刻的demand
        x_new.append(x[i][3])
        # 该时刻前3个月内对应该时刻的demand
        x_new.append(x[i][5])
        # 该时刻前24小时的demand
        x_new.append(x[i][0])
        x_new.append(x[i][2])
        x_new.append(x[i][4])
        x_new.append(x[i][6])
        # 该时刻的温度
        x_new.append(x[i][7])
        y_new.append(y[i])
    x_new = x_new + [x[0][8], x[0][9], x[0][10], x[0][11], x[0][12]]
    y_new = [np.squeeze(np.array(y_new)).transpose()]
    return x_new, y_new


x_train_fit, y_train_fit = get_x_y(x_train, y_train)
x_val_fit, y_val_fit = get_x_y(x_val, y_val)
x_test_pred, y_test_pred = get_x_y(x_test, y_test)
# print(x_train_fit[0].shape)
# print(y_train_fit[0].shape)
# print(x_test_pred[0].shape)
# print(y_test_pred[0].shape)


def get_input(hour):
    input_d_7day = Input(shape=(7,), name='input' + str(hour) + '_D_7day')
    input_d_2months = Input(shape=(8,), name='input' + str(hour) + '_D_2months')
    input_d_3months = Input(shape=(3,), name='input' + str(hour) + '_D_3months')
    input_d_24h = Input(shape=(24 - hour + 1,), name='input' + str(hour) + '_D_24h')

    input_t_7day = Input(shape=(7,), name='input' + str(hour) + '_T_7day')
    input_t_2months = Input(shape=(8,), name='input' + str(hour) + '_T_2months')
    input_t_3months = Input(shape=(3,), name='input' + str(hour) + '_T_3months')

    input_t = Input(shape=(1,))

    return (input_d_7day, input_d_2months, input_d_3months, input_d_24h, input_t_7day, input_t_2months, input_t_3months,
            input_t)


input1_Dd, input1_Dw, input1_Dm, input1_Dr, input1_Td, input1_Tw, input1_Tm, input1_T = get_input(1)
input2_Dd, input2_Dw, input2_Dm, input2_Dr, input2_Td, input2_Tw, input2_Tm, input2_T = get_input(2)
input3_Dd, input3_Dw, input3_Dm, input3_Dr, input3_Td, input3_Tw, input3_Tm, input3_T = get_input(3)
input4_Dd, input4_Dw, input4_Dm, input4_Dr, input4_Td, input4_Tw, input4_Tm, input4_T = get_input(4)
input5_Dd, input5_Dw, input5_Dm, input5_Dr, input5_Td, input5_Tw, input5_Tm, input5_T = get_input(5)
input6_Dd, input6_Dw, input6_Dm, input6_Dr, input6_Td, input6_Tw, input6_Tm, input6_T = get_input(6)
input7_Dd, input7_Dw, input7_Dm, input7_Dr, input7_Td, input7_Tw, input7_Tm, input7_T = get_input(7)
input8_Dd, input8_Dw, input8_Dm, input8_Dr, input8_Td, input8_Tw, input8_Tm, input8_T = get_input(8)
input9_Dd, input9_Dw, input9_Dm, input9_Dr, input9_Td, input9_Tw, input9_Tm, input9_T = get_input(9)
input10_Dd, input10_Dw, input10_Dm, input10_Dr, input10_Td, input10_Tw, input10_Tm, input10_T = get_input(10)
input11_Dd, input11_Dw, input11_Dm, input11_Dr, input11_Td, input11_Tw, input11_Tm, input11_T = get_input(11)
input12_Dd, input12_Dw, input12_Dm, input12_Dr, input12_Td, input12_Tw, input12_Tm, input12_T = get_input(12)
input13_Dd, input13_Dw, input13_Dm, input13_Dr, input13_Td, input13_Tw, input13_Tm, input13_T = get_input(13)
input14_Dd, input14_Dw, input14_Dm, input14_Dr, input14_Td, input14_Tw, input14_Tm, input14_T = get_input(14)
input15_Dd, input15_Dw, input15_Dm, input15_Dr, input15_Td, input15_Tw, input15_Tm, input15_T = get_input(15)
input16_Dd, input16_Dw, input16_Dm, input16_Dr, input16_Td, input16_Tw, input16_Tm, input16_T = get_input(16)
input17_Dd, input17_Dw, input17_Dm, input17_Dr, input17_Td, input17_Tw, input17_Tm, input17_T = get_input(17)
input18_Dd, input18_Dw, input18_Dm, input18_Dr, input18_Td, input18_Tw, input18_Tm, input18_T = get_input(18)
input19_Dd, input19_Dw, input19_Dm, input19_Dr, input19_Td, input19_Tw, input19_Tm, input19_T = get_input(19)
input20_Dd, input20_Dw, input20_Dm, input20_Dr, input20_Td, input20_Tw, input20_Tm, input20_T = get_input(20)
input21_Dd, input21_Dw, input21_Dm, input21_Dr, input21_Td, input21_Tw, input21_Tm, input21_T = get_input(21)
input22_Dd, input22_Dw, input22_Dm, input22_Dr, input22_Td, input22_Tw, input22_Tm, input22_T = get_input(22)
input23_Dd, input23_Dw, input23_Dm, input23_Dr, input23_Td, input23_Tw, input23_Tm, input23_T = get_input(23)
input24_Dd, input24_Dw, input24_Dm, input24_Dr, input24_Td, input24_Tw, input24_Tm, input24_T = get_input(24)
input_D_max = Input(shape=(1,), name='input_D_max')
input_D_min = Input(shape=(1,), name='input_D_min')
input_season = Input(shape=(4,), name='input_season')
input_weekday = Input(shape=(2,), name='input_weekday')
input_festival = Input(shape=(2,), name='input_festival')


# gru rmse=832.9120905673743 mape  4.6
def gru_model(input_dr, output_pre=[]):
    if len(output_pre) == 0:
        input_dr = Reshape(target_shape=(input_dr.shape[1], 1))(input_dr)
        gru = LSTM(units=30)(input_dr)
    else:
        concat_gru = concatenate([input_dr]+output_pre)
        concat_gru_reshape = Reshape(target_shape=(concat_gru.shape[1], 1))(concat_gru)
        gru = LSTM(units=30)(concat_gru_reshape)

    # 改变MC DROPOUT
    dense_gru = Dense(20, activation='selu')(gru)
    # gru_batch = BatchNormalization()(dense_gru)
    gru_dropout = Dropout(rate=0.2)(dense_gru, training=False)
    dense_output = Dense(10, activation='selu')(gru_dropout)
    # dense_batch = BatchNormalization()(dense_output)
    output = Dense(units=1, activation='selu')(dense_output)
    output_pre_new = output_pre+[output]
    return output, output_pre_new


gru_output1, gru_output_pre1 = gru_model(input1_Dr)
gru_output2, gru_output_pre2 = gru_model(input2_Dr, gru_output_pre1)
gru_output3, gru_output_pre3 = gru_model(input3_Dr, gru_output_pre2)
gru_output4, gru_output_pre4 = gru_model(input4_Dr, gru_output_pre3)
gru_output5, gru_output_pre5 = gru_model(input5_Dr, gru_output_pre4)
gru_output6, gru_output_pre6 = gru_model(input6_Dr, gru_output_pre5)
gru_output7, gru_output_pre7 = gru_model(input7_Dr, gru_output_pre6)
gru_output8, gru_output_pre8 = gru_model(input8_Dr, gru_output_pre7)
gru_output9, gru_output_pre9 = gru_model(input9_Dr, gru_output_pre8)
gru_output10, gru_output_pre10 = gru_model(input10_Dr, gru_output_pre9)
gru_output11, gru_output_pre11 = gru_model(input11_Dr, gru_output_pre10)
gru_output12, gru_output_pre12 = gru_model(input12_Dr, gru_output_pre11)
gru_output13, gru_output_pre13 = gru_model(input13_Dr, gru_output_pre12)
gru_output14, gru_output_pre14 = gru_model(input14_Dr, gru_output_pre13)
gru_output15, gru_output_pre15 = gru_model(input15_Dr, gru_output_pre14)
gru_output16, gru_output_pre16 = gru_model(input16_Dr, gru_output_pre15)
gru_output17, gru_output_pre17 = gru_model(input17_Dr, gru_output_pre16)
gru_output18, gru_output_pre18 = gru_model(input18_Dr, gru_output_pre17)
gru_output19, gru_output_pre19 = gru_model(input19_Dr, gru_output_pre18)
gru_output20, gru_output_pre20 = gru_model(input20_Dr, gru_output_pre19)
gru_output21, gru_output_pre21 = gru_model(input21_Dr, gru_output_pre20)
gru_output22, gru_output_pre22 = gru_model(input22_Dr, gru_output_pre21)
gru_output23, gru_output_pre23 = gru_model(input23_Dr, gru_output_pre22)
gru_output24, gru_output_pre24 = gru_model(input24_Dr, gru_output_pre23)

# gru输出的24小时(最后输出)
gru_output_pre = concatenate(gru_output_pre24)


def get_basic_structure(hour, input_d_7day, input_d_8week, input_d_3months, input_d_24h, input_t_7day, input_t_8week,
                        input_t_3months, input_t, output_pre=[]):

    num_dense = 10
    dense_Dd = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal')(input_d_7day)
    dense_Dw = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal')(input_d_8week)
    dense_Dm = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal')(input_d_3months)
    dense_Td = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal')(input_t_7day)
    dense_Tw = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal')(input_t_8week)
    dense_Tm = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal')(input_t_3months)

    concat_d = concatenate([dense_Dd, dense_Td])
    dense_d = Dense(num_dense, kernel_initializer='lecun_normal')(concat_d)
    dense_d_batch = BatchNormalization()(dense_d)
    activation_d = LeakyReLU()(dense_d_batch)
    dropout_d = Dropout(rate=0.5)(activation_d)

    concat_w = concatenate([dense_Dw, dense_Tw])
    dense_w = Dense(num_dense, kernel_initializer='lecun_normal')(concat_w)
    dense_w_batch = BatchNormalization()(dense_w)
    activation_w = LeakyReLU()(dense_w_batch)
    dropout_w = Dropout(rate=0.5)(activation_w)

    concat_m = concatenate([dense_Dm, dense_Tm])
    dense_m = Dense(num_dense, kernel_initializer='lecun_normal')(concat_m)
    dense_m_batch = BatchNormalization()(dense_m)
    activation_m = LeakyReLU()(dense_m_batch)
    dropout_m = Dropout(rate=0.5)(activation_m)

    concat_date_info = concatenate([input_season, input_weekday])
    dense_concat_date_info_1 = Dense(5, activation='selu', kernel_initializer='lecun_normal')(concat_date_info)
    dense_concat_date_info_2 = Dense(5, activation='selu', kernel_initializer='lecun_normal')(concat_date_info)

    concat_FC2 = concatenate([dropout_d, dropout_w, dropout_m, dense_concat_date_info_1, input_festival])
    dense_FC2 = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal')(concat_FC2)
    # 这个的功能是某个时刻将前24个小时（例如20小时，不满24）与预测的小时结合得到新的24小时
    if output_pre == []:
        dense_Dr = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal')(input_d_24h)
    else:
        concat_Dr = concatenate([input_d_24h] + output_pre)
        dense_Dr = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal')(concat_Dr)
    dense_FC1 = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal')(
        concatenate([dense_Dr, dense_concat_date_info_2]))

    concat = concatenate([dense_FC2, dense_FC1, input_t])
    dense_pre_output = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal')(concat)

    output = Dense(1, activation='linear', name='output' + str(hour), kernel_initializer='lecun_normal')(
        dense_pre_output)

    output_pre_new = output_pre + [output]
    # output.shape = (？, 1)代表了这个时刻的预测负荷
    # output_pre_new将预测时刻一个加了起来，最后的output_pre24包含了总预测输出负荷
    return (output, output_pre_new)


output1, output_pre1 = get_basic_structure(1, input1_Dd, input1_Dw, input1_Dm, input1_Dr, input1_Td, input1_Tw,
                                           input1_Tm, input1_T)
output2, output_pre2 = get_basic_structure(2, input2_Dd, input2_Dw, input2_Dm, input2_Dr, input2_Td, input2_Tw,
                                           input2_Tm, input2_T, gru_output_pre1)
output3, output_pre3 = get_basic_structure(3, input3_Dd, input3_Dw, input3_Dm, input3_Dr, input3_Td, input3_Tw,
                                           input3_Tm, input3_T, gru_output_pre2)
output4, output_pre4 = get_basic_structure(4, input4_Dd, input4_Dw, input4_Dm, input4_Dr, input4_Td, input4_Tw,
                                           input4_Tm, input4_T, gru_output_pre3)
output5, output_pre5 = get_basic_structure(5, input5_Dd, input5_Dw, input5_Dm, input5_Dr, input5_Td, input5_Tw,
                                           input5_Tm, input5_T, gru_output_pre4)
output6, output_pre6 = get_basic_structure(6, input6_Dd, input6_Dw, input6_Dm, input6_Dr, input6_Td, input6_Tw,
                                           input6_Tm, input6_T, gru_output_pre5)
output7, output_pre7 = get_basic_structure(7, input7_Dd, input7_Dw, input7_Dm, input7_Dr, input7_Td, input7_Tw,
                                           input7_Tm, input7_T, gru_output_pre6)
output8, output_pre8 = get_basic_structure(8, input8_Dd, input8_Dw, input8_Dm, input8_Dr, input8_Td, input8_Tw,
                                           input8_Tm, input8_T, gru_output_pre7)
output9, output_pre9 = get_basic_structure(9, input9_Dd, input9_Dw, input9_Dm, input9_Dr, input9_Td, input9_Tw,
                                           input9_Tm, input9_T, gru_output_pre8)
output10, output_pre10 = get_basic_structure(10, input10_Dd, input10_Dw, input10_Dm, input10_Dr, input10_Td, input10_Tw,
                                             input10_Tm, input10_T, gru_output_pre9)
output11, output_pre11 = get_basic_structure(11, input11_Dd, input11_Dw, input11_Dm, input11_Dr, input11_Td, input11_Tw,
                                             input11_Tm, input11_T, gru_output_pre10)
output12, output_pre12 = get_basic_structure(12, input12_Dd, input12_Dw, input12_Dm, input12_Dr, input12_Td, input12_Tw,
                                             input12_Tm, input12_T, gru_output_pre11)
output13, output_pre13 = get_basic_structure(13, input13_Dd, input13_Dw, input13_Dm, input13_Dr, input13_Td, input13_Tw,
                                             input13_Tm, input13_T, gru_output_pre12)
output14, output_pre14 = get_basic_structure(14, input14_Dd, input14_Dw, input14_Dm, input14_Dr, input14_Td, input14_Tw,
                                             input14_Tm, input14_T, gru_output_pre13)
output15, output_pre15 = get_basic_structure(15, input15_Dd, input15_Dw, input15_Dm, input15_Dr, input15_Td, input15_Tw,
                                             input15_Tm, input15_T, gru_output_pre14)
output16, output_pre16 = get_basic_structure(16, input16_Dd, input16_Dw, input16_Dm, input16_Dr, input16_Td, input16_Tw,
                                             input16_Tm, input16_T, gru_output_pre15)
output17, output_pre17 = get_basic_structure(17, input17_Dd, input17_Dw, input17_Dm, input17_Dr, input17_Td, input17_Tw,
                                             input17_Tm, input17_T, gru_output_pre16)
output18, output_pre18 = get_basic_structure(18, input18_Dd, input18_Dw, input18_Dm, input18_Dr, input18_Td, input18_Tw,
                                             input18_Tm, input18_T, gru_output_pre17)
output19, output_pre19 = get_basic_structure(19, input19_Dd, input19_Dw, input19_Dm, input19_Dr, input19_Td, input19_Tw,
                                             input19_Tm, input19_T, gru_output_pre18)
output20, output_pre20 = get_basic_structure(20, input20_Dd, input20_Dw, input20_Dm, input20_Dr, input20_Td, input20_Tw,
                                             input20_Tm, input20_T, gru_output_pre19)
output21, output_pre21 = get_basic_structure(21, input21_Dd, input21_Dw, input21_Dm, input21_Dr, input21_Td, input21_Tw,
                                             input21_Tm, input21_T, gru_output_pre20)
output22, output_pre22 = get_basic_structure(22, input22_Dd, input22_Dw, input22_Dm, input22_Dr, input22_Td, input22_Tw,
                                             input22_Tm, input22_T, gru_output_pre21)
output23, output_pre23 = get_basic_structure(23, input23_Dd, input23_Dw, input23_Dm, input23_Dr, input23_Td, input23_Tw,
                                             input23_Tm, input23_T, gru_output_pre22)
output24, output_pre24 = get_basic_structure(24, input24_Dd, input24_Dw, input24_Dm, input24_Dr, input24_Td, input24_Tw,
                                             input24_Tm, input24_T, gru_output_pre23)

# basic structure输出的24小时(最后输出)
gru_basic_output_pre = concatenate(output_pre24)


def penalized_loss(y_true, y_pred):
    beta = 0.1
    loss1 = mean_absolute_percentage_error(y_true, y_pred)
    loss2 = K.mean(K.maximum(K.max(y_pred, axis=1) - input_D_max, 0.), axis=-1)
    loss3 = K.mean(K.maximum(input_D_min - K.min(y_pred, axis=1), 0.), axis=-1)
    return loss1 + beta * (loss2 + loss3)


def get_model():
    model = Model(inputs=[input1_Dd, input1_Dw, input1_Dm, input1_Dr, input1_Td, input1_Tw, input1_Tm, input1_T,
                          input2_Dd, input2_Dw, input2_Dm, input2_Dr, input2_Td, input2_Tw, input2_Tm, input2_T,
                          input3_Dd, input3_Dw, input3_Dm, input3_Dr, input3_Td, input3_Tw, input3_Tm, input3_T,
                          input4_Dd, input4_Dw, input4_Dm, input4_Dr, input4_Td, input4_Tw, input4_Tm, input4_T,
                          input5_Dd, input5_Dw, input5_Dm, input5_Dr, input5_Td, input5_Tw, input5_Tm, input5_T,
                          input6_Dd, input6_Dw, input6_Dm, input6_Dr, input6_Td, input6_Tw, input6_Tm, input6_T,
                          input7_Dd, input7_Dw, input7_Dm, input7_Dr, input7_Td, input7_Tw, input7_Tm, input7_T,
                          input8_Dd, input8_Dw, input8_Dm, input8_Dr, input8_Td, input8_Tw, input8_Tm, input8_T,
                          input9_Dd, input9_Dw, input9_Dm, input9_Dr, input9_Td, input9_Tw, input9_Tm, input9_T,
                          input10_Dd, input10_Dw, input10_Dm, input10_Dr, input10_Td, input10_Tw, input10_Tm, input10_T,
                          input11_Dd, input11_Dw, input11_Dm, input11_Dr, input11_Td, input11_Tw, input11_Tm, input11_T,
                          input12_Dd, input12_Dw, input12_Dm, input12_Dr, input12_Td, input12_Tw, input12_Tm, input12_T,
                          input13_Dd, input13_Dw, input13_Dm, input13_Dr, input13_Td, input13_Tw, input13_Tm, input13_T,
                          input14_Dd, input14_Dw, input14_Dm, input14_Dr, input14_Td, input14_Tw, input14_Tm, input14_T,
                          input15_Dd, input15_Dw, input15_Dm, input15_Dr, input15_Td, input15_Tw, input15_Tm, input15_T,
                          input16_Dd, input16_Dw, input16_Dm, input16_Dr, input16_Td, input16_Tw, input16_Tm, input16_T,
                          input17_Dd, input17_Dw, input17_Dm, input17_Dr, input17_Td, input17_Tw, input17_Tm, input17_T,
                          input18_Dd, input18_Dw, input18_Dm, input18_Dr, input18_Td, input18_Tw, input18_Tm, input18_T,
                          input19_Dd, input19_Dw, input19_Dm, input19_Dr, input19_Td, input19_Tw, input19_Tm, input19_T,
                          input20_Dd, input20_Dw, input20_Dm, input20_Dr, input20_Td, input20_Tw, input20_Tm, input20_T,
                          input21_Dd, input21_Dw, input21_Dm, input21_Dr, input21_Td, input21_Tw, input21_Tm, input21_T,
                          input22_Dd, input22_Dw, input22_Dm, input22_Dr, input22_Td, input22_Tw, input22_Tm, input22_T,
                          input23_Dd, input23_Dw, input23_Dm, input23_Dr, input23_Td, input23_Tw, input23_Tm, input23_T,
                          input24_Dd, input24_Dw, input24_Dm, input24_Dr, input24_Td, input24_Tw, input24_Tm, input24_T,
                          input_D_max, input_D_min, input_season, input_weekday, input_festival
                          ], outputs=[gru_basic_output_pre])
    return model


model = get_model()
model.compile(optimizer='adam', loss=penalized_loss)

floder_gru_basic = 'gru_basic/'
floder_gru_basic_resnet = 'gru_basic_resnet/'
floder_basic_resnet = 'basic_resnet/'
now_floder = floder_gru_basic
num_models = 1
num_snapshot = 3
train = 1
epoch_1 = 40
epoch_2 = 5
epoch_3 = 5
start = time.time()
if train:
    for i in range(num_models):
        file_path_1 = now_floder + str(i+1) + "_1test.hdf5"
        checkpoint = ModelCheckpoint(file_path_1, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=True)
        callback_list = [checkpoint]
        history1 = model.fit(x_train_fit, y_train_fit, epochs=epoch_1, batch_size=128, validation_data=(x_val_fit, y_val_fit),
                             verbose=2, callbacks=callback_list)

        file_path_2 = now_floder + str(i+1) + "_2test.hdf5"
        checkpoint = ModelCheckpoint(file_path_2, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=True)
        callback_list = [checkpoint]
        history2 = model.fit(x_train_fit, y_train_fit, epochs=epoch_2, batch_size=128,
                             validation_data=(x_val_fit, y_val_fit), verbose=2, callbacks=callback_list)

        file_path_3 = now_floder + str(i+1) + "_3test.hdf5"
        checkpoint = ModelCheckpoint(file_path_3, monitor='val_loss', save_best_only=True, mode='min',save_weights_only=True)
        callback_list = [checkpoint]
        history3 = model.fit(x_train_fit, y_train_fit, epochs=epoch_3, batch_size=128,
                             validation_data=(x_val_fit, y_val_fit), verbose=2, callbacks=callback_list)
        val_loss_1 = history1.history['val_loss']
        val_loss_2 = history2.history['val_loss']
        val_loss_3 = history3.history['val_loss']
        val_loss = val_loss_1+val_loss_2+val_loss_3
        plt.figure()
        plt.plot(val_loss)
        plt.show()


end = time.time()
print('time = ', end - start)

# 最好的模型
predict = np.zeros(shape=(num_models*num_snapshot, 24*test_day))
for i in range(num_models):
    for j in range(num_snapshot):
        model.load_weights(now_floder + str(i+1) + "_" + str(j+1) + "test.hdf5")
        pred = model.predict(x_test_pred)
        predict[i*num_models+j, :] = pred.reshape((24*test_day,))

predict = predict.mean(axis=0)
print(predict.shape)
predict_end = np.array(predict).reshape(-1, 1)
y_test_end = np.array(y_test_pred).squeeze().reshape(-1, 1)
y_test_reverse = ss_load.inverse_transform(y_test_end)
predict_reverse = ss_load.inverse_transform(predict_end)
rmse = np.sqrt(np.mean(np.square(y_test_reverse - predict_reverse)))
print('rmse = ', rmse)
mape = np.round(np.mean(np.divide(np.abs(y_test_reverse-predict_reverse), y_test_reverse)), 5)
print('mape = ', mape*100)

plt.figure(1)
plt.plot(y_test_reverse[7200:8500], c='g', label='true')
plt.plot(predict_reverse[7200:8500], c='r', label='predict')
plt.legend()
plt.title('Predict and True' + '_mape = ' + str(mape))
plt.savefig('gru_basic Predict and True' + '_mape = ' + str(mape)+'.png')
plt.show()

difference_value = y_test_reverse - predict_reverse
plt.figure(2)
plt.plot(difference_value[7200:8500], c='b')
plt.title('Difference Value' + '_mape = ' + str(mape))
plt.show()
