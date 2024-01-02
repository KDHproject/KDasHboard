import math
import pandas as pd
import streamlit as st
import openpyxl

number_of_label = 2

def check_matrix_sum(df, k):
    """
    정방행렬 DataFrame에서 마지막 행과 열이 나머지 데이터 행과 열들의 합계와 일치하는지 검사합니다.
    첫 번째부터 k개의 행과 열은 라벨로 간주하고, 연산에서 제외합니다.
    합계가 일치하지 않는 경우, 차이를 출력합니다.

    :param df: 검사할 정방행렬 (Pandas DataFrame)
    :param k: 라벨로 간주되는 행과 열의 개수
    :return: 마지막 행과 열이 나머지 데이터 행과 열들의 합계와 일치하면 True, 그렇지 않으면 False
    """
    # 데이터 부분만 선택
    data_df = df.iloc[k:-1, k:-1]

    # 마지막 데이터 행과 열의 합계 계산
    row_sum = data_df.sum(axis=1)
    col_sum = data_df.sum(axis=0)

    # 마지막 행과 열의 값 (라벨 제외)
    last_row = df.iloc[-1, k:-1]
    last_col = df.iloc[k:-1, -1]

    # 합계와 마지막 행/열이 일치하는지 검사
    row_match = all(last_row == col_sum)
    col_match = all(last_col == row_sum)

    if not row_match or not col_match:
        print("합계가 일치하지 않습니다.")
        if not row_match:
            print("행 합계 불일치:", last_row - col_sum)
        if not col_match:
            print("열 합계 불일치:", last_col - row_sum)
        return False
    print('정상')
    return True

def get_matrix_x(DataFrame):
    # 1. 행과 열의 개수를 가져오기
    num_rows, num_cols = DataFrame.shape

    # 2. 초기 인덱스 설정
    sum_row_index, sum_col_index = num_rows - 1, num_cols - 1

    # 3. [중간투입계, 중간수요계]의 좌표 찾기
    while math.isnan(DataFrame.iloc[sum_row_index, num_cols - 1]):
        sum_row_index -= 1

    while math.isnan(DataFrame.iloc[num_rows - 1, sum_col_index]):
        sum_col_index -= 1

    # 4. 좌표 반환
    mid_ID_idx = (sum_row_index, sum_col_index)

    sum = float(DataFrame.iloc[mid_ID_idx])
    sum2zero_row = sum
    sum2zero_col = sum
    cnt_row = 1
    cnt_col = 1

    # 행에 대한 sum2zero 계산
    while True:
        sum2zero_row -= float(DataFrame.iloc[mid_ID_idx[0], mid_ID_idx[1] - cnt_row])
        if sum2zero_row <= 0.1:
            break
        cnt_row += 1

    # 열에 대한 sum2zero 계산
    while True:
        sum2zero_col -= float(DataFrame.iloc[mid_ID_idx[0] - cnt_col, mid_ID_idx[1]])
        if sum2zero_col <= 0.1:
            break
        cnt_col += 1

    max_count = max(cnt_row, cnt_col)
    first_index = (mid_ID_idx[0]-max_count, mid_ID_idx[1]-max_count)
    size = max_count
    print(f'size를 찾았습니다. 확인한 값은 {mid_ID_idx, cnt_row, cnt_col, size}입니다.')
    matrixX = DataFrame.iloc[first_index[0]-number_of_label:mid_ID_idx[0]+1, first_index[1]-number_of_label:mid_ID_idx[1]+1]
    check_matrix_sum(matrixX, number_of_label)
    
    return matrixX



def main():
    st.title("DasHboard beta0")
    # 파일 업로드 섹션
    uploaded_file = st.file_uploader("여기에 파일을 드래그하거나 클릭하여 업로드하세요.", type=['xls', 'xlsx'])
    
    if uploaded_file is not None:
        try:


            # 파일을 데이터프레임으로 읽기
            df = pd.read_excel(uploaded_file, header=None)
            matrix_X = get_matrix_x(df)
            # 데이터프레임 표시
            tab1, tab2 = st.tabs(['원본 Excel', 'matrix_X'])

            with tab1:
                st.header('최초 업로드 된 원본 Excel파일 입니다.')
                st.write(df)

            with tab2:
                st.header('최초 업로드 된 원본 Excel파일의 matrix_X 입니다.')
                st.write(matrix_X)
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")


    st.header("DataFrame을 수정합니다.")

    col1, col2, col3 = st.columns(3)

    with col1:
        # st.header("산업의 index를 입력하세요")
        idx = st.number_input('산업의 index를 입력하세요', step=1.0)

    with col2:
        # st.header("alpha값을 입력하세요.")
        alpha = st.number_input('alpha값을 입력하세요.', step=0.01)

    with col3:
        if st.button('Say hello'):
            st.write('Why hello there')

if __name__ == "__main__":
    main()