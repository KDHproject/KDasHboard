import math
import numpy as np
import pandas as pd
import streamlit as st
import openpyxl
import matplotlib.pyplot as plt


### config
# st.session_state.number_of_label = None




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


def get_submatrix_withlabel(df, start_row, start_col, end_row, end_col, first_index_of_df, numberoflabel = 2):
    row_indexs = list(range(first_index_of_df[0]-numberoflabel, first_index_of_df[0])) + list(range(start_row, end_row+1))
    col_indexs = list(range(first_index_of_df[1]-numberoflabel, first_index_of_df[1])) + list(range(start_col, end_col+1))
    print(row_indexs)
    print(col_indexs)

    submatrix_withlabel = df.iloc[row_indexs, col_indexs]
    return submatrix_withlabel



def create_thresholded_matrix(data_frame, first_row_index, first_col_index, size, sum_row_index):
    """
    주어진 데이터 프레임의 특정 부분으로부터 가중치 matrix를 생성하는 함수

    Args:
    data_frame (pd.DataFrame): 데이터를 포함하고 있는 데이터 프레임.
    first_row_index (int): 추출할 행렬의 시작 행 인덱스.
    first_col_index (int): 추출할 행렬의 시작 열 인덱스.
    size (int): 추출할 행렬의 크기.

    Returns:
    가중치 이상인 값의 정규화된 행렬
    """
    # 주어진 인덱스와 크기에 따라 데이터 프레임에서 행렬 추출
    matrix = data_frame.iloc[first_row_index:first_row_index+size, first_col_index:first_col_index+size]

    # 열 합계 계산
    normalization_denominator = data_frame.iloc[sum_row_index, first_col_index:first_col_index+size]

    # 0인 합계를 작은 양수로 대체
    normalization_denominator_replaced = normalization_denominator.replace(0, np.finfo(float).eps)

    # 정규화된 행렬 계산
    normalized_matrix = matrix.divide(normalization_denominator_replaced, axis=1)
    normalized_matrix = normalized_matrix.astype(float)

    # 단위 행렬 생성
    unit_matrix = np.eye(normalized_matrix.shape[0])

    # 단위 행렬과 정규화된 행렬의 차 계산
    st.session_statesubtracted_matrix = unit_matrix - normalized_matrix
    L = np.linalg.inv(st.session_statesubtracted_matrix.values)
    L = pd.DataFrame(L)
    # print(L)

    threshold_count(L) # threshold에 따른 생존값 시각화

    # 임계값 설정
    threshold = float(input('input threshold :')) # np.sum(L)/(L.shape[0]**2)

    # 임계값을 넘는 원소들로 이루어진 행렬 생성
    thresholded_matrix = filter_matrix(L, threshold)
    thresholded_matrix = thresholded_matrix.to_numpy()

    np.fill_diagonal(thresholded_matrix, 0) # 대각선 원소는 0으로 설정

    # 0 초과하는 값의 개수 계산
    count_over_zero = np.sum(thresholded_matrix > 0)

    # 전체 Matrix의 크기 계산
    total_elements = matrix.size

    # 비율 계산
    ratio_over_zero = count_over_zero / total_elements

    print(f"Size of matrix: {total_elements}")
    print(f"생존 값 개수: {count_over_zero}")
    print(f"생존 값의 비율: {round(ratio_over_zero,3)}")

    return thresholded_matrix

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


@st.cache_data()
def load_data(file):
    df = pd.read_excel(file, header=None)
    return df



### Streamlit 구현
def main():
    st.title("DasHboard beta0")
    mode = st.radio('모드 선택', ['Korea', 'China', 'Manual'])
    if mode == 'Korea':
        first_idx = (6,2)
        number_of_label = 2
    elif mode == 'China':
        first_idx = (6,2)
        number_of_label = 1
    else:
        first_idx = 'd'

    if 'st.session_state.df_edited' not in st.session_state:
        st.session_state.df_edited = None
    # 파일 업로드 섹션
    uploaded_file = st.file_uploader("여기에 파일을 드래그하거나 클릭하여 업로드하세요.", type=['xls', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # 파일을 데이터프레임으로 읽기
            df = load_data(uploaded_file)

            mid_ID_idx = get_mid_ID_idx(df, first_idx)
            # matrix_X,  _, mid_ID_idx, __ = get_matrix_x(df)
            df.iloc[first_idx[0]:, first_idx[1]:] = df.iloc[first_idx[0]:, first_idx[1]:].astype(float)
            matrix_X = get_submatrix_withlabel(df, first_idx[0], first_idx[1], mid_ID_idx[0], mid_ID_idx[1], first_idx, numberoflabel=number_of_label)
            matrix_R = get_submatrix_withlabel(df, mid_ID_idx[0], first_idx[1], df.shape[0]-1, mid_ID_idx[1], first_idx, numberoflabel=number_of_label)
            matrix_C = get_submatrix_withlabel(df, first_idx[0], mid_ID_idx[1], mid_ID_idx[0], df.shape[1]-1, first_idx, numberoflabel=number_of_label)


            # 원본 부분 header 표시
            st.header('최초 업로드 된 Excel파일 입니다.')
            # 데이터프레임 표시 
            tab1, tab2, tab3, tab4 = st.tabs(['원본 Excel', 'original_matrix_X', 'original_matrix_R', 'original_matrix_C'])

            with tab1:
                st.subheader('최초 업로드 된 원본 Excel파일 입니다.')
                st.write(df)

            with tab2:
                st.subheader('최초 업로드 된 원본 Excel파일의 matrix_X 입니다.')
                st.write(matrix_X)

            with tab3:
                st.subheader('최초 업로드 된 원본 Excel파일의 matrix_R 입니다.')
                st.write(matrix_R)

            with tab4:
                st.subheader('최초 업로드 된 원본 Excel파일의 matrix_C 입니다.')
                st.write(matrix_C)  

            if 'data_modification_log' not in st.session_state:
                st.session_state.data_modification_log = ""
            

            # new_row, new_col = map(lambda size: pd.Series([0]*size), [df.shape[1], df.shape[0]])
            new_row = pd.Series([0]*df.shape[1])
            new_col = pd.Series([0]*(df.shape[0]+1))

            st.header("DataFrame을 수정합니다.")
            if st.button('수정내역을 삭제하고 처음부터 다시 수정합니다.'):
                st.session_state.df_editing = pd.concat([df.loc[:mid_ID_idx[0]-1], pd.DataFrame([new_row]), df.loc[mid_ID_idx[0]:]], ignore_index=True).reset_index(drop=True)
                st.session_state.df_editing = pd.concat([st.session_state.df_editing.loc[:, :mid_ID_idx[1]-1], pd.DataFrame([new_col]).T, st.session_state.df_editing.loc[:, mid_ID_idx[1]:]], axis = 1, ignore_index=True).reset_index(drop=True)
                st.session_state.data_modification_log = ""
                st.session_state.df_edited = None


            # 편집을 위해 DataFrame의 사본을 생성합니다.
            if 'df_editing' not in st.session_state:
                st.session_state.df_editing = pd.concat([df.loc[:mid_ID_idx[0]-1], pd.DataFrame([new_row]), df.loc[mid_ID_idx[0]:]], ignore_index=True).reset_index(drop=True)
                st.session_state.df_editing = pd.concat([st.session_state.df_editing.loc[:, :mid_ID_idx[1]-1], pd.DataFrame([new_col]).T, st.session_state.df_editing.loc[:, mid_ID_idx[1]:]], axis = 1, ignore_index=True).reset_index(drop=True)

            st.session_state.df_editing.loc[mid_ID_idx[0],first_idx[1]-2:first_idx[1]-1] = ['1000', 'new'] # 새로 삽입한 행에 label 추가
            st.session_state.df_editing.loc[first_idx[0]-2:first_idx[0]-1,mid_ID_idx[1]] = ['1000', 'new'] # 새로 십입한 열에 label 추가

        
            col1, col2, col3 = st.columns(3)
            with col1:
                # st.header("산업의 index를 입력하세요")
                step = st.number_input('산업의 index를 입력하세요', value=1)
            with col2:
                # st.header(" 입력하세요")
                alpha = st.number_input('alpha값을 입력하세요', 0.000, 1.000, step=None)
            with col3:
                if st.button('Edit Data'):
                    stepRowIndex = step + first_idx[0]-1
                    stepColIndex = step + first_idx[1]-1
                    # st.write(st.session_state.df_editing.loc[stepRowIndex,first_idx[1]:])

                    # 조정이 필요한 인덱스의 행, 열에 alpha 값을 곱한 후 새로운 행, 열에 추가
                    st.session_state.df_editing.loc[mid_ID_idx[0],first_idx[1]:] += st.session_state.df_editing.loc[stepRowIndex,first_idx[1]:].astype(float) * alpha
                    st.session_state.df_editing.loc[first_idx[0]:,mid_ID_idx[1]] += st.session_state.df_editing.loc[first_idx[0]:, stepColIndex].astype(float) * alpha

                    # 원래 값에 (1-a) 부분으로 채워주기
                    st.session_state.df_editing.loc[stepRowIndex,first_idx[1]:] = st.session_state.df_editing.loc[stepRowIndex,first_idx[1]:].astype(float) * (1-alpha)
                    st.session_state.df_editing.loc[first_idx[0]:, stepColIndex] = st.session_state.df_editing.loc[first_idx[0]:,stepColIndex].astype(float) * (1-alpha)                    
                    data_modification = f"{step}산업으로부터 {alpha}만큼을 추출하였습니다."
                    st.session_state.data_modification_log += data_modification + "\n\n"
                if st.button('적용 Data'):
                    st.session_state.df_edited = st.session_state.df_editing
                    st.session_state.edited_mid_ID_idx = mid_ID_idx[0]+1, mid_ID_idx[1]+1
            st.write(st.session_state.df_editing)
            st.write(st.session_state.data_modification_log) 

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
    
        if st.session_state.df_edited is not None:
            st.header('수정 된 Excel파일 입니다.')
            edited_matrix_X = get_submatrix_withlabel(st.session_state.df_edited, first_idx[0],first_idx[1], st.session_state.edited_mid_ID_idx[0], st.session_state.edited_mid_ID_idx[1], first_idx, numberoflabel = 2)
            edited_matrix_R = get_submatrix_withlabel(st.session_state.df_edited, st.session_state.edited_mid_ID_idx[0],first_idx[1], st.session_state.df_edited.shape[0]-1, st.session_state.edited_mid_ID_idx[1], first_idx, numberoflabel = 2)
            edited_matrix_C = get_submatrix_withlabel(st.session_state.df_edited, first_idx[0], st.session_state.edited_mid_ID_idx[1], st.session_state.edited_mid_ID_idx[0], st.session_state.df_edited.shape[1]-1, first_idx, numberoflabel = 2)

            # 데이터프레임 표시
            tab1, tab2, tab3, tab4 = st.tabs(['원본 Excel', 'edited_matrix_X', 'edited_matrix_R', 'edited_matrix_C'])

            with tab1:
                st.subheader('수정 된 원본 Excel파일 입니다.')
                st.write(st.session_state.df_edited)

            with tab2:
                st.subheader('최초 업로드 된 원본 Excel파일의 matrix_X 입니다.')
                st.write(edited_matrix_X)

            with tab3:
                st.subheader('최초 업로드 된 원본 Excel파일의 matrix_R 입니다.')
                st.write(edited_matrix_R)

            with tab4:
                st.subheader('최초 업로드 된 원본 Excel파일의 matrix_C 입니다.')
                st.write(edited_matrix_C)  

            st.session_state.filtered_df = None
            


            st.header("DataFrame을 임계값을 기준으로 filtering 합니다.")
            st.subheader('threshold에 따른 생존비율 그래프')
            # df_normalized = st.session_state.df_edited.iloc[]
            st.session_state.filtered_df = None


            # 편집을 위해 DataFrame의 사본을 생성합니다.
            if 'df_normalized' not in st.session_state:
                # 정규화된 행렬을 생성합니다.
                st.session_state.df_normalized_matrix = st.session_state.df_edited.iloc[first_idx[0]:st.session_state.edited_mid_ID_idx[0], first_idx[1]:st.session_state.edited_mid_ID_idx[1]]
                st.session_state.df_normalized_matrix = st.session_state.df_normalized_matrix.apply(pd.to_numeric, errors='coerce')
                st.session_state.x = st.session_state.df_normalized_matrix.apply(pd.to_numeric, errors='coerce')
                normalization_denominator = st.session_state.df_edited.iloc[st.session_state.df_edited.shape[0]-1, first_idx[1]:st.session_state.edited_mid_ID_idx[1]]
                normalization_denominator = pd.to_numeric(normalization_denominator)
                

                # 0인 합계를 작은 양수로 대체
                normalization_denominator_replaced = normalization_denominator.replace(0, np.finfo(float).eps)
                # normalization_denominator_replaced = normalization_denominator.replace(None, np.finfo(float).eps)
                
                # 정규화 계산
                # normalized_matrix = normalized_matrix.astype(float)
                print(st.session_state.df_normalized_matrix.shape)
                print(len(normalization_denominator_replaced))
                st.session_state.df_normalized_matrix = st.session_state.df_normalized_matrix.divide(normalization_denominator_replaced, axis=1)
                    # 단위 행렬 생성
                
                unit_matrix = np.eye(st.session_state.df_normalized_matrix.shape[0])

                    # 단위 행렬과 정규화된 행렬의 차 계산
                st.session_state.subtracted_matrix = unit_matrix - st.session_state.df_normalized_matrix
                st.session_state.leontief_inverse = np.linalg.inv(st.session_state.subtracted_matrix.values)
                st.session_state.leontief_inverse = pd.DataFrame(st.session_state.leontief_inverse)

            threshold_count(st.session_state.leontief_inverse)

            st.subheader('Leontief 과정 matrices')
            col1, col2, col3, col4 = st.tabs(['edited', 'normailization denominator', 'normalized', 'leontief inverse'])
            with col1:
                st.write(st.session_state.x)
            with col2:
                st.write(normalization_denominator)
            with col3:
                st.write(st.session_state.df_normalized_matrix)
            with col4:
                st.write(st.session_state.leontief_inverse)




            col1, col2= st.columns(2)
            with col1:
                # st.header("산업의 index를 입력하세요")
                threshold = st.number_input('threshold를 입력하세요', 0.000, 1.000, step=None)
            with col2:
                if st.button('Apply threshold'):
                    st.session_state.threshold = threshold

            if 'threshold' in st.session_state:
                # binary matrix 생성
                binary_matrix = make_binary_matrix(st.session_state.leontief_inverse, st.session_state.threshold)
                df_edited_filtered = st.session_state.x * binary_matrix
                df_normalized_matrix_filtered = st.session_state.df_normalized_matrix * binary_matrix
                leontief_inverse_filtered = st.session_state.leontief_inverse * binary_matrix

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


            # if st.button('filtering을 삭제하고 처음부터 다시 수정합니다.'):
            #     st.session_state.df_normalized = st.session_state.df_edited
            #     st.session_state.filtered_df = None
if __name__ == "__main__":
    main()