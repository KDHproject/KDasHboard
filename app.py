import math
import numpy as np
import pandas as pd
import streamlit as st
import openpyxl
import matplotlib.pyplot as plt
import time


### config


### 사용자 정의 함수 선언
def make_binary_matrix(matrix, threshold):
    # 임계값 이하의 원소들을 0으로 설정
    binary_matrix = matrix.apply(lambda x: np.where(x > threshold, 1, 0))
    return binary_matrix

def filter_matrix(matrix, threshold):
    # 임계값 이하의 원소들을 0으로 설정
    filtered_matrix = matrix.where(matrix > threshold, 0)
    return filtered_matrix

# 임계 값을 0-1까지로, 25%로 x축을 한정해서 시각화, 최대 변화율 지점의 x축 값 찾기
def threshold_count(matrix):
    L = matrix
    element_counts = []

    # 임계값 생성
    threshold_values = np.linspace(0, 1, 1000)[:250]

    # 각 임계값에 대해 생존값 계산
    for threshold in threshold_values:
        thresholded_matrix = filter_matrix(L, threshold)
        thresholded_matrix = thresholded_matrix.copy().to_numpy()

        np.fill_diagonal(thresholded_matrix, 0)  # 대각선 원소는 0으로 설정
        count = (thresholded_matrix >= threshold).sum().sum()
        element_counts.append(count)

    # 최대 변화율(절대값) 찾기
    max_change = 0
    max_change_index = 0
    for i in range(1, len(element_counts)):
        change = abs(element_counts[i - 1] - element_counts[i])
        if change > max_change:
            max_change = change
            max_change_index = i
    
    # df_graph = pd.DataFrame({'x': threshold_values, 'y': element_counts})
    max_change_threshold = threshold_values[max_change_index]

    # 그래프 그리기
    fig, ax = plt.subplots() # 수정된 부분
    ax.plot(threshold_values, element_counts)
    ax.set_xlabel('Threshold Value') # ax를 사용하여 라벨 설정
    ax.set_ylabel('Number of Elements >= Threshold')
    ax.set_title('Number of Elements Greater than or Equal to Threshold in a Matrix')

    # 최대 변화율 지점 표시
    ax.plot(max_change_threshold, element_counts[max_change_index], 'ro') # ax를 사용하여 데이터 표시

    ax.grid(True)
    st.pyplot(fig) # 수정된 부분
    st.write(f'생존율 급감 구간의 임계 값 : {max_change_threshold}')

    return plt.show()

@st.cache_data()
def get_submatrix_withlabel(df, start_row, start_col, end_row, end_col, first_index_of_df, numberoflabel = 2):
    row_indexs = list(range(first_index_of_df[0]-numberoflabel, first_index_of_df[0])) + list(range(start_row, end_row+1))
    col_indexs = list(range(first_index_of_df[1]-numberoflabel, first_index_of_df[1])) + list(range(start_col, end_col+1))
    # print(row_indexs)
    # print(col_indexs)

    submatrix_withlabel = df.iloc[row_indexs, col_indexs]
    return submatrix_withlabel


def get_mid_ID_idx(df, first_idx):
    matrix_X = df.iloc[first_idx[0]:, first_idx[1]:].astype(float)
    row_cnt, col_cnt, row_sum, col_sum = 0, 0, 0, 0
    for v in matrix_X.iloc[0]:
        if abs(row_sum - v) < 0.001:
            if v == 0:
                continue
            else: break
        row_cnt += 1
        row_sum += v
    for v in matrix_X.iloc[:, 0]:
        print(f'gap: {col_sum-v}, sum: {col_sum}, value: {v}')
        if abs(col_sum - v) < 0.001:
            if v == 0:
                continue
            else: break
        col_cnt += 1
        col_sum += v
    
    if row_cnt == col_cnt:
        size = row_cnt
    else:
        size = max(row_cnt, col_cnt)

    return (first_idx[0]+size, first_idx[1]+size)

def insert_row_and_col(df, first_idx, mid_ID_idx, code, name, num_of_label):
    df_editing = df.copy()
    df_editing.insert(loc=mid_ID_idx[1], column='a', value=np.nan, allow_duplicates=True)
    df_editing.iloc[first_idx[0]-num_of_label, mid_ID_idx[1]] = code
    df_editing.iloc[first_idx[0]-num_of_label+1, mid_ID_idx[1]] = name
    df_editing.iloc[first_idx[0]:, mid_ID_idx[1]] = 0
    df_editing.columns = range(df_editing.shape[1])
    df_editing = df_editing.T   
    df_editing.insert(loc=mid_ID_idx[0], column='a', value=np.nan, allow_duplicates=True)
    df_editing.iloc[first_idx[1]-num_of_label, mid_ID_idx[0]] = code
    df_editing.iloc[first_idx[1]-num_of_label+1, mid_ID_idx[0]] = name
    df_editing.iloc[first_idx[1]:, mid_ID_idx[0]] = 0
    df_editing.columns = range(df_editing.shape[1])
    df_editing = df_editing.T
    df_inserted = df_editing.copy()
    mid_ID_idx = (mid_ID_idx[0]+1, mid_ID_idx[1]+1)
    msg = f'A new row and column (Code: {code}, Name: {name}) have been inserted.'

    return df_inserted, mid_ID_idx, msg

def transfer_to_new_sector(df, first_idx, origin_code, target_code, ratio, code_label = 2):
    df_editing = df.copy()
    target_idx = df_editing.index[df_editing[first_idx[1]-code_label] == target_code].tolist()
    if len(target_idx) == 1:
        target_idx = target_idx[0]
    else:
        msg = 'ERROR: target code is not unique.'
        return df_editing, msg
    origin_idx = df_editing.index[df_editing[first_idx[1]-code_label] == origin_code].tolist()
    if len(origin_idx) == 1:
        origin_idx = origin_idx[0]
    else:
        msg = 'ERROR: origin code is not unique.'
        return df_editing, msg
    df_editing.iloc[first_idx[0]:, first_idx[1]:] = df_editing.iloc[first_idx[0]:, first_idx[1]:].apply(pd.to_numeric, errors='coerce')
    origin_idx = (origin_idx, origin_idx-first_idx[0]+first_idx[1])
    target_idx = (target_idx, target_idx-first_idx[0]+first_idx[1])
    df_editing.iloc[target_idx[0] ,first_idx[1]:] += df_editing.iloc[origin_idx[0] ,first_idx[1]:] * ratio
    df_editing.iloc[origin_idx[0] ,first_idx[1]:] = df_editing.iloc[origin_idx[0] ,first_idx[1]:] * (1-ratio)
    df_editing.iloc[first_idx[0]: ,target_idx[1]] += df_editing.iloc[first_idx[0]: ,origin_idx[1]] * ratio
    df_editing.iloc[first_idx[0]: ,origin_idx[1]] = df_editing.iloc[first_idx[0]: ,origin_idx[1]] * (1-ratio)

    msg = f'{ratio*100}% of {origin_code} has been moved to {target_code}.'
    return df_editing, msg

def remove_zero_series(df, first_idx, mid_ID_idx):
    df_editing = df.copy()
    df_test = df_editing.copy()
    df_test = df_editing.iloc[first_idx[0]:, first_idx[1]:].apply(pd.to_numeric, errors='coerce')
    zero_row_indices = df_test.index[(df_test == 0).all(axis=1)].tolist()
    zero_row_indices = [item for item in zero_row_indices if item >first_idx[0] and item <mid_ID_idx[0]]
    zero_col_indices = list(map(lambda x: x - first_idx[0] + first_idx[1], zero_row_indices))
    df_editing.drop(zero_row_indices, inplace=True)
    df_editing.drop(zero_col_indices, inplace=True, axis=1)
    df_editing.columns = range(df_editing.shape[1])
    df_editing.index = range(df_editing.shape[0])
    count = len(zero_col_indices)
    msg = f'{count}개의 행(열)이 삭제되었습니다.'
    mid_ID_idx = (mid_ID_idx[0] - count, mid_ID_idx[1] - count)
    return df_editing, msg, mid_ID_idx

@st.cache_data()
def load_data(file):
    st.session_state['df'] = pd.read_excel(file, header=None)
    return st.session_state['df']



### Streamlit 구현
def main():
    st.title("DasHboard beta 1.1")
    mode = st.radio('모드 선택', ['Korea', 'China', 'Manual'])
    if mode == 'Korea':
        first_idx = (6,2)
        number_of_label = 2
    elif mode == 'China':
        first_idx = (6,2)
        number_of_label = 1
    else:
        first_idx = 0
        number_of_label = 2

    # 파일 업로드 섹션
    st.session_state['uploaded_file'] = st.file_uploader("여기에 파일을 드래그하거나 클릭하여 업로드하세요.", type=['xls', 'xlsx'])
    if 'df' not in st.session_state:
        if st.session_state['uploaded_file']:
            st.session_state['df'] = load_data(st.session_state.uploaded_file)
            st.session_state['mid_ID_idx'] = get_mid_ID_idx(st.session_state['df'], first_idx)
            st.session_state['df'].iloc[first_idx[0]:, first_idx[1]:] = st.session_state['df'].iloc[first_idx[0]:, first_idx[1]:].apply(pd.to_numeric, errors='coerce')

    if 'df' in st.session_state:
        matrix_X = get_submatrix_withlabel(st.session_state['df'], first_idx[0], first_idx[1], st.session_state['mid_ID_idx'][0], st.session_state['mid_ID_idx'][1], first_idx, numberoflabel=number_of_label)
        matrix_R = get_submatrix_withlabel(st.session_state['df'], st.session_state['mid_ID_idx'][0], first_idx[1], st.session_state['df'].shape[0]-1, st.session_state['mid_ID_idx'][1], first_idx, numberoflabel=number_of_label)
        matrix_C = get_submatrix_withlabel(st.session_state['df'], first_idx[0], st.session_state['mid_ID_idx'][1], st.session_state['mid_ID_idx'][0], st.session_state['df'].shape[1]-1, first_idx, numberoflabel=number_of_label)

        # 원본 부분 header 표시
        st.header('최초 업로드 된 Excel파일 입니다.')
        # 데이터프레임 표시 
        tab1, tab2, tab3, tab4 = st.tabs(['원본 Excel', 'original_matrix_X', 'original_matrix_R', 'original_matrix_C'])

        with tab1:
            st.subheader('최초 업로드 된 원본 Excel파일 입니다.')
            st.write(st.session_state['df'])

        with tab2:
            st.subheader('최초 업로드 된 원본 Excel파일의 matrix_X 입니다.')
            st.write(matrix_X)

        with tab3:
            st.subheader('최초 업로드 된 원본 Excel파일의 matrix_R 입니다.')
            st.write(matrix_R)

        with tab4:
            st.subheader('최초 업로드 된 원본 Excel파일의 matrix_C 입니다.')
            st.write(matrix_C)

        if 'df_editing' not in st.session_state:
            st.session_state['df_editing'] = st.session_state['df'].copy()
    
    if 'data_editing_log' not in st.session_state:
        st.session_state['data_editing_log'] = ''

    if 'df_editing' in st.session_state:
        st.header("DataFrame을 수정합니다.")
        col1, col2, col3 = st.columns(3)
        with col1:
            new_code = st.text_input('새로 삽입할 산업의 code를 입력하세요')
        with col2:
            name = st.text_input('새로 삽입할 산업의 이름을 입력하세요')
        with col3:
            if st.button('Insert'):
                result = insert_row_and_col(st.session_state['df_editing'], first_idx, st.session_state['mid_ID_idx'], new_code, name, number_of_label)
                st.session_state['df_editing'], st.session_state['mid_ID_idx'] = result[0:2]
                st.session_state['data_editing_log'] += (result[2] + '\n\n')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            origin_code = st.text_input('from')
        with col2:
            target_code = st.text_input('to')
        with col3:
            alpha = st.number_input('alpha value', 0.000, 1.000, step=None)
        with col4:
            if st.button('Edit Data'):
                result = transfer_to_new_sector(st.session_state['df_editing'], first_idx, origin_code, target_code, alpha)
                st.session_state['df_editing'] = result[0]
                st.session_state['data_editing_log'] += (result[1] + '\n\n')
        col1, col3 = st.columns(2)
        with col1:
            if st.button('0인 행(열) 삭제'):
                result = remove_zero_series(st.session_state['df_editing'], first_idx, st.session_state['mid_ID_idx'])
                st.session_state['df_editing'] = result[0]
                st.session_state['data_editing_log'] += (result[1] + '\n\n')
                st.session_state['mid_ID_idx'] = result[2]
        # with col2:
        #     if st.button('처음부터 다시 수정'):
        #         del st.session_state['df_editing']
        #         st.rerun()
        with col3:
            if st.button('적용'):
                st.session_state['df_edited'] = st.session_state['df_editing'].copy()
        st.write(st.session_state['df_editing'])

    if 'df_edited' in st.session_state:
        st.header('수정 된 Excel파일 입니다.')
        edited_matrix_X = get_submatrix_withlabel(st.session_state['df_edited'], first_idx[0],first_idx[1], st.session_state['mid_ID_idx'][0], st.session_state['mid_ID_idx'][1], first_idx, numberoflabel = 2)
        edited_matrix_R = get_submatrix_withlabel(st.session_state['df_edited'], st.session_state['mid_ID_idx'][0],first_idx[1], st.session_state['df_edited'].shape[0]-1, st.session_state['mid_ID_idx'][1], first_idx, numberoflabel = 2)
        edited_matrix_C = get_submatrix_withlabel(st.session_state['df_edited'], first_idx[0], st.session_state['mid_ID_idx'][1], st.session_state['mid_ID_idx'][0], st.session_state['df_edited'].shape[1]-1, first_idx, numberoflabel = 2)

        # 데이터프레임 표시
        tab1, tab2, tab3, tab4 = st.tabs(['edited Excel', 'edited matrix X', 'edited matrix R', 'edited matrix C'])

        with tab1:
            st.subheader('수정 된 Excel파일 입니다.')
            st.write(st.session_state['df_edited'])

        with tab2:
            st.subheader('최초 업로드 된 원본 Excel파일의 matrix_X 입니다.')
            st.write(edited_matrix_X)

        with tab3:
            st.subheader('최초 업로드 된 원본 Excel파일의 matrix_R 입니다.')
            st.write(edited_matrix_R)

        with tab4:
            st.subheader('최초 업로드 된 원본 Excel파일의 matrix_C 입니다.')
            st.write(edited_matrix_C)
        st.header("DataFrame을 임계값을 기준으로 filtering 합니다.")
        st.subheader('threshold에 따른 생존비율 그래프')

    if 'df_for_leontief' not in st.session_state:
        if 'df_edited' in st.session_state:
            st.session_state['df_for_leontief'] = edited_matrix_X.iloc[:-1, :-1].copy()
            st.session_state['df_for_leontief'].index = range(st.session_state['df_for_leontief'].shape[0])
            st.session_state['df_for_leontief'].columns = range(st.session_state['df_for_leontief'].shape[1])
            st.session_state['normalization_denominator'] = st.session_state['df_edited'].iloc[st.session_state['df_edited'].shape[0]-1, first_idx[1]:st.session_state['mid_ID_idx'][1]]
            st.session_state['normalization_denominator'] = pd.to_numeric(st.session_state['normalization_denominator'])
            st.session_state['normalization_denominator_replaced'] = st.session_state['normalization_denominator'].replace(0, np.finfo(float).eps)
        
    if 'df_for_leontief' in st.session_state:
        st.session_state['df_for_leontief_without_label'] = st.session_state['df_for_leontief'].iloc[2:, 2:].copy()
        st.session_state['df_for_leontief_with_label'] = st.session_state['df_for_leontief'].copy()
        tmp = st.session_state['df_for_leontief_without_label'].copy()
        tmp = tmp.apply(pd.to_numeric, errors='coerce')
        tmp = tmp.divide(st.session_state['normalization_denominator_replaced'], axis=1)
        st.session_state['df_for_leontief_with_label'].iloc[2:, 2:] = tmp
        st.session_state['df_normalized_with_label'] = st.session_state['df_for_leontief_with_label'].copy()
        unit_matrix = np.eye(tmp.shape[0])
        subtracted_matrix = unit_matrix - tmp
        leontief = np.linalg.inv(subtracted_matrix.values)
        leontief = pd.DataFrame(leontief)
        st.session_state['df_for_leontief_with_label'].iloc[2:, 2:] = leontief
        threshold_count(st.session_state['df_for_leontief_with_label'].iloc[2:, 2:])

        st.subheader('Leontief 과정 matrices')
        col1, col2, col3, col4 = st.tabs(['edited', 'normailization denominator', 'normalized', 'leontief inverse'])
        with col1:
            st.write(st.session_state['df_for_leontief'])
        with col2:
            st.write(st.session_state['normalization_denominator'])
        with col3:
            st.write(st.session_state['df_normalized_with_label'])
        with col4:
            st.write(st.session_state['df_for_leontief_with_label'])

        col1, col2= st.columns(2)
        with col1:
            threshold = st.number_input('threshold를 입력하세요', 0.000, 1.000, step=None)
        with col2:
            if st.button('Apply threshold'):
                st.session_state.threshold = threshold

    if 'threshold' in st.session_state:
        # binary matrix 생성
        binary_matrix = make_binary_matrix(st.session_state['df_for_leontief_with_label'].iloc[2:, 2:].apply(pd.to_numeric, errors='coerce'), st.session_state.threshold)
        df_edited_filtered = st.session_state['df_for_leontief']
        df_edited_filtered.iloc[2:, 2:] = df_edited_filtered.iloc[2:, 2:].apply(pd.to_numeric, errors='coerce')*binary_matrix
        df_normalized_matrix_filtered = st.session_state['df_normalized_with_label']
        df_normalized_matrix_filtered.iloc[2:, 2:] = st.session_state['df_normalized_with_label'].iloc[2:, 2:].apply(pd.to_numeric, errors='coerce')*binary_matrix
        leontief_inverse_filtered = st.session_state['df_for_leontief_with_label']
        leontief_inverse_filtered.iloc[2:, 2:] = st.session_state['df_for_leontief_with_label'].iloc[2:, 2:].apply(pd.to_numeric, errors='coerce')*binary_matrix
        st.subheader('Filtered matrices')
        col1, col2, col3, col4 = st.tabs(['binary_matrix', 'normailization denominator', 'normalized', 'leontief inverse'])
        with col1:
            st.write(binary_matrix)
        with col2:
            st.write(df_edited_filtered)
        with col3:
            st.write(df_normalized_matrix_filtered)
        with col4:
            st.write(leontief_inverse_filtered)
    with st.expander('수정내역 보기'):
        st.write('수정내역')
        st.write(st.session_state['data_editing_log'])
if __name__ == "__main__":
    main()