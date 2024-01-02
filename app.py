import pandas as pd
import streamlit as st


def main():
    st.title("DasHboard beta")

    # 파일 업로드 섹션
    uploaded_file = st.file_uploader("여기에 파일을 드래그하거나 클릭하여 업로드하세요.", type=['xls', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # 파일을 데이터프레임으로 읽기
            df = pd.read_excel(uploaded_file)
            # 데이터프레임 표시
            st.write(df)
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()