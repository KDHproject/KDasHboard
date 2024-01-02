import math
import pandas as pd
import streamlit as st
import openpyxl


number_of_label = 2


def upload_and_process_file():
    """
    파일을 업로드하고, 업로드된 파일을 DataFrame으로 변환한 다음, get_matrix_x 함수를 호출합니다.

    :return: 업로드된 파일의 DataFrame과 matrix_X, 파일이 업로드되지 않았다면 두 값 모두 None을 반환
    """
    # 파일 업로드
    uploaded_file = st.file_uploader("여기에 파일을 드래그하거나 클릭하여 업로드하세요.", type=['xls', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # 파일을 데이터프레임으로 읽기
            df = pd.read_excel(uploaded_file, header=None)
            matrix_X = get_matrix_x(df)
            return df, matrix_X
        except Exception as e:
            st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")
            return None, None
    else:
        return None, None


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
    
    return matrixX, first_index, mid_ID_idx


@st.cache_data()
def load_data(file):
    df = pd.read_excel(file, header=None)
    return df


def main():
    st.title("DasHboard beta0")
    # 파일 업로드 섹션
    uploaded_file = st.file_uploader("여기에 파일을 드래그하거나 클릭하여 업로드하세요.", type=['xls', 'xlsx'])

    if uploaded_file is not None:
        try:
            # 파일을 데이터프레임으로 읽기
            df = load_data(uploaded_file) 
            matrix_X,  first_idx, mid_ID_idx= get_matrix_x(df)
            # 데이터프레임 표시
            tab1, tab2, tab3, tab4 = st.tabs(['원본 Excel', 'matrix_X', 'matrix_R', 'matrix_C'])

            with tab1:
                st.header('최초 업로드 된 원본 Excel파일 입니다.')
                st.write(df)

            with tab2:
                st.header('최초 업로드 된 원본 Excel파일의 matrix_X 입니다.')
                st.write(matrix_X)

            with tab3:
                st.header('최초 업로드 된 원본 Excel파일의 matrix_R 입니다.')
                st.write(matrix_X)

            with tab4:
                st.header('최초 업로드 된 원본 Excel파일의 matrix_C 입니다.')
                st.write(matrix_X)  

            if 'data_modification_log' not in st.session_state:
                st.session_state.data_modification_log = ""

            st.header("DataFrame을 수정합니다.")


            new_row, new_col = map(lambda size: pd.Series([0]*size), [df.shape[1], df.shape[0]])



            # 편집을 위해 DataFrame의 사본을 생성합니다.
            if 'df_edited' not in st.session_state:
                st.session_state.df_edited = pd.concat([df.loc[:mid_ID_idx[0]-1], pd.DataFrame([new_row]), df.loc[mid_ID_idx[0]:]], ignore_index=True).reset_index(drop=True)
                st.session_state.df_edited = pd.concat([st.session_state.df_edited.loc[:, :mid_ID_idx[1]-1], pd.DataFrame([new_col]).T, st.session_state.df_edited.loc[:, mid_ID_idx[1]:]], axis = 1, ignore_index=True).reset_index(drop=True)

            st.session_state.df_edited.loc[mid_ID_idx[0],first_idx[1]-2:first_idx[1]-1] = ['1000', 'new'] # 새로 삽입한 행에 label 추가
            st.session_state.df_edited.loc[first_idx[0]-2:first_idx[0]-1,mid_ID_idx[1]] = ['1000', 'new'] # 새로 십입한 열에 label 추가


            col1, col2, col3 = st.columns(3)
            
            with col1:
                # st.header("산업의 index를 입력하세요")
                step = st.number_input('산업의 index를 입력하세요', value=1)
            with col2:
                # st.header(" 입력하세요")
                alpha = st.number_input('alpha값을 입력하세요', 0.000, 1.000, step=0.001)
            with col3:
                if st.button('Edit Data'):
                    stepRowIndex = step + first_idx[0]-1
                    stepColIndex = step + first_idx[1]-1
                    # st.write(st.session_state.df_edited.loc[stepRowIndex,first_idx[1]:])

                    # 조정이 필요한 인덱스의 행, 열에 alpha 값을 곱한 후 새로운 행, 열에 추가
                    st.session_state.df_edited.loc[mid_ID_idx[0],first_idx[1]:] += st.session_state.df_edited.loc[stepRowIndex,first_idx[1]:].astype(float) * alpha
                    st.session_state.df_edited.loc[first_idx[0]:,mid_ID_idx[1]] += st.session_state.df_edited.loc[first_idx[0]:, stepColIndex].astype(float) * alpha

                    # 원래 값에 (1-a) 부분으로 채워주기
                    st.session_state.df_edited.loc[stepRowIndex,first_idx[1]:] = st.session_state.df_edited.loc[stepRowIndex,first_idx[1]:].astype(float) * (1-alpha)
                    st.session_state.df_edited.loc[first_idx[0]:, stepColIndex] = st.session_state.df_edited.loc[first_idx[0]:,stepColIndex].astype(float) * (1-alpha)                    
                    data_modification = f"{step}산업으로부터 {alpha}만큼을 추출하였습니다."
                    st.session_state.data_modification_log += data_modification + "\n\n"
                if st.button('적용 Data'):
                    st.write('Why hello there')
            st.write(st.session_state.df_edited)
            st.write(st.session_state.data_modification_log)              
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
    


if __name__ == "__main__":
    main()