import os
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from pathlib import Path
import json

from retrieval import retrieve, improved_retrieval

# Hàm để lấy tất cả các ảnh từ thư mục
def get_image_paths(folder_path, extensions=['.jpg', '.jpeg', '.png']):
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path)
            if os.path.splitext(file)[1].lower() in extensions]

# Hàm để chia danh sách ảnh thành các trang
def paginate(items, items_per_page):
    page_number = st.session_state.page_number
    start_idx = page_number * items_per_page
    end_idx = start_idx + items_per_page
    return items[start_idx:end_idx]


# Set page configuration
st.set_page_config(page_title="Fashion Image Retrieval", layout="wide")

# sidebar
with st.sidebar:
    dataset_option = st.radio(
        'Tập dữ liệu',
        ['FashionIQ']
    )

    model_option = st.radio(
        'Mô hình',
        ['ResNet50', 'ResNet50 + SVM']
    )

    results_number = st.selectbox(
        'Số lượng kết quả trả về',
        [10, 100, 1000, 10000]
    )
    
    dress_type = st.selectbox("Loại trang phục", ['shirt', 'dress', 'toptee'])
    
    query_image = st.file_uploader("Chọn ảnh truy vấn", type=["jpg", "jpeg", "png"])
    
    if query_image:
        image_name = query_image.name[:-4]
        image = Image.open(query_image)
        st.image(image, caption="Ảnh truy vấn", width=200)
        
    run_retrieval = st.button('TRUY VẤN')
    
    
# Main Page
st.title('👗🔍 Fashion Image Retrieval')

# Khởi tạo các giá trị
if "page_number" not in st.session_state:
    st.session_state.page_number = 0
    
if "results" not in st.session_state:
    st.session_state.results = None
    
# Số ảnh trên mỗi trang
items_per_page = 50

# Tính tổng số trang
total_pages = (results_number - 1) // items_per_page + 1


if run_retrieval:
    if (query_image) and (dress_type):
        
        with st.spinner("Đang thực hiện truy vấn..."):
            if dress_type == 'dress':
                dresstype = 0
            elif dress_type == 'shirt':
                dresstype = 1
            else:
                dresstype = 2
            
            if model_option == 'ResNet50':
                results = retrieve(image_name, dresstype, results_number)
            else:
                results = improved_retrieval(image_name, dresstype, results_number)
            
            st.session_state.results = results
            st.session_state.page_number = 0
            
    else:
        st.error("Cần cung cấp ảnh truy vấn và loại trang phục tương ứng!")
            

# Phân trang và hiển thị ảnh
if st.session_state.results:
    total_pages = (len(st.session_state.results['retrieval_results']) - 1) // items_per_page + 1
    
    paginated_images = paginate(st.session_state.results['retrieval_results'], items_per_page)

    # Hiển thị các ảnh
    cols = st.columns(5)
    for idx, img_path in enumerate(paginated_images):
        with cols[idx % 5]:
            st.image(img_path, use_column_width=True)
    
    col1, col2 = st.columns([7, 3])
    with col2:
        col3, col4, col5 = st.columns([1, 2, 1])

        with col3:
            if st.button('◀️', key="prev", help="trước"):
                if st.session_state.page_number > 0:
                    st.session_state.page_number -= 1

        with col4:
            st.markdown(f"**Trang {st.session_state.page_number + 1} / {total_pages}**")

        with col5:
            if st.button('▶️', key="next", help="sau"):
                if st.session_state.page_number < total_pages - 1:
                    st.session_state.page_number += 1            
