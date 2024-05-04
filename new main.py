import pickle

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# 基于用户的推荐(用户评分)的书籍-评分表
books_ratings_ = pickle.load(open('books_ratings_.pkl', 'rb'))
# 基于物品的推荐(作者&出版社)的书籍-评分表
_books_ratings_ = pickle.load(open('_books_ratings_.pkl', 'rb'))


# 基于用户的相似度矩阵df
cosine_sim_df = pickle.load(open('sim.pkl', 'rb'))
# 基于物品的相似度矩阵
cosine_simm = pickle.load(open('simm.pkl', 'rb'))

# streamlit screen

st.title('正在读什么?')
title1 = st.sidebar.title("个性化图书推荐系统")

books_titles = books_ratings_['Book-Title'].unique()[1:]
book = st.sidebar.selectbox('选择您当前正在或者要阅读的书籍', books_titles)

alg = st.sidebar.selectbox('推荐依据为', ('作者 & 出版社',
                                          '根据投票选出最相似的书籍推荐',
                                          '根据投票选出最不相似的书籍推荐')
                           )

title2 = st.sidebar.title("Made by wyx, wb, szc")
title3 = st.sidebar.text("对推荐结果进行反馈和评价请联系qq：1095664018"
                         "期待你的反馈~")

# 为侧边栏中的下拉框选择框进行数据编辑和显示，以选取最近阅读的书籍

dff = books_ratings_[['Book-Title', 'Image-URL-M']]
df1 = dff.drop_duplicates()

df2 = books_ratings_[['ISBN', 'Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication']]
df3 = df2.drop_duplicates()

dd1 = df3[df3['Book-Title'] == book]['Book-Author'].tolist()
dd2 = df3[df3['Book-Title'] == book]['Publisher'].tolist()
dd3 = df3[df3['Book-Title'] == book]['Year-Of-Publication'].tolist()
dd4 = df3[df3['Book-Title'] == book]['ISBN'].tolist()

y = dd1[0]
z = dd2[0]
t = dd3[0]
w = dd4[0]

S = df1[df1['Book-Title'] == book]['Image-URL-M'].to_string(index=False).lstrip()

aut = books_ratings_[['Book-Title', 'Book-Author']]
auth = aut.drop_duplicates()
aut_ = books_ratings_[['Book-Title', 'Book-Rating']]
auth_ = books_ratings_.groupby(['Book-Title']).mean()
author = pd.merge(auth, auth_, on='Book-Title')
authors = author.drop_duplicates(subset='Book-Title', keep="first")


# round 函数有时可能无法返回符合预期的有效数字，因此，定义了一个 truncate 函数
def truncate(m, decimals=0):
    multiplier = 10 ** decimals
    return int(m * multiplier) / multiplier


rate = authors[authors['Book-Title'] == book]['Book-Rating'].to_string(index=False).lstrip()
rate = float(rate)
rate = truncate(rate, 2)

st.write(f'## {book}')
col1, mid, col2, mid, col3 = st.columns([10, 1, 10, 1, 10])
with col1:
    st.image(S, width=139)
with col2:
    st.write(y)
    st.write(z)
    st.text(t)
    st.text(w)
with col3:
    st.write('Rating:')
    st.success(rate)


# 定义find_similar_books函数，(使用余弦相似度方法计算书籍之间的相似度)该函数返回与选取的书籍最相似的5本。
def find_similar_books(x):
    selected_book = [x]
    books_summed = np.sum(cosine_sim_df[selected_book], axis=1)
    books_summed = books_summed.sort_values(ascending=False)
    ranked_books = books_summed.index
    ranked_books = ranked_books.tolist()
    ranked_books_5 = [ranked_books[1], ranked_books[2], ranked_books[3], ranked_books[4], ranked_books[5]]
    return ranked_books_5


# 该函数返回与选取书籍最不相似的5本。
def find_nonsimilar_books(x):
    selected_book = [x]
    books_summed = np.sum(cosine_sim_df[selected_book], axis=1)
    books_summed = books_summed.sort_values(ascending=True)
    ranked_books = books_summed.index
    ranked_books = ranked_books.tolist()
    ranked_books_5 = [ranked_books[0], ranked_books[1], ranked_books[2], ranked_books[3], ranked_books[4]]
    return ranked_books_5


_books_ratings_.reset_index(drop=True, inplace=True)
indices = pd.Series(_books_ratings_.index, index=_books_ratings_['Book-Title'])


# 该函数返回与选取书籍作者&出版社最为相似的5本。
def get_recommendations(_book, cosine_simm = cosine_simm):
    idx = indices[_book]
    sim_scores = list(enumerate(cosine_simm[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    return _books_ratings_['Book-Title'].iloc[book_indices].tolist()


# 创建一个函数来调用算法参数或者第二个侧边栏中下拉框的选项

if alg == '作者 & 出版社':
    st.title('推荐这些~~~')
    list2 = get_recommendations(book)

    for b in list2:
        P = df1[df1['Book-Title'] == b]['Image-URL-M'].to_string(index=False).lstrip()
        rate = authors[authors['Book-Title'] == b]['Book-Rating'].to_string(index=False).lstrip()
        rate = float(rate)
        rate = truncate(rate, 2)

        dd1 = df3[df3['Book-Title'] == b]['Book-Author'].tolist()
        dd2 = df3[df3['Book-Title'] == b]['Publisher'].tolist()
        dd3 = df3[df3['Book-Title'] == b]['Year-Of-Publication'].tolist()
        dd4 = df3[df3['Book-Title'] == b]['ISBN'].tolist()

        y = dd1[0]
        z = dd2[0]
        t = dd3[0]
        w = dd4[0]

        st.write(f'## {b}')
        col1, mid, col2, mid, col3 = st.columns([10, 1, 10, 1, 10])
        with col1:
            st.image(P)
        with col2:
            st.write(y)
            st.write(z)
            st.text(t)
            st.text(w)
        with col3:
            st.write('Rating:')
            st.error(rate)


elif alg == '根据投票选出最相似的书籍推荐':
    st.title('推荐这些~~~')
    list2 = find_similar_books(book)

    for b in list2:
        P = df1[df1['Book-Title'] == b]['Image-URL-M'].to_string(index=False).lstrip()
        rate = authors[authors['Book-Title'] == b]['Book-Rating'].to_string(index=False).lstrip()
        rate = float(rate)
        rate = truncate(rate, 2)

        dd1 = df3[df3['Book-Title'] == b]['Book-Author'].tolist()
        dd2 = df3[df3['Book-Title'] == b]['Publisher'].tolist()
        dd3 = df3[df3['Book-Title'] == b]['Year-Of-Publication'].tolist()
        dd4 = df3[df3['Book-Title'] == b]['ISBN'].tolist()

        y = dd1[0]
        z = dd2[0]
        t = dd3[0]
        w = dd4[0]

        st.write(f'## {b}')
        col1, mid, col2, mid, col3 = st.columns([10, 1, 10, 1, 10])
        with col1:
            st.image(P)
        with col2:
            st.write(y)
            st.write(z)
            st.text(t)
            st.text(w)
        with col3:
            st.write('Rating:')
            st.warning(rate)


else:
    st.title('推荐这些~~~')
    list2 = find_nonsimilar_books(book)

    for b in list2:
        P = df1[df1['Book-Title'] == b]['Image-URL-M'].to_string(index=False).lstrip()
        rate = authors[authors['Book-Title'] == b]['Book-Rating'].to_string(index=False).lstrip()
        rate = float(rate)
        rate = truncate(rate, 2)

        dd1 = df3[df3['Book-Title'] == b]['Book-Author'].tolist()
        dd2 = df3[df3['Book-Title'] == b]['Publisher'].tolist()
        dd3 = df3[df3['Book-Title'] == b]['Year-Of-Publication'].tolist()
        dd4 = df3[df3['Book-Title'] == b]['ISBN'].tolist()

        y = dd1[0]
        z = dd2[0]
        t = dd3[0]
        w = dd4[0]

        st.write(f'## {b}')
        col1, mid, col2, mid, col3 = st.columns([10, 1, 10, 1, 10])
        with col1:
            st.image(P)
        with col2:
            st.write(y)
            st.write(z)
            st.text(t)
            st.text(w)
        with col3:
            st.write('Rating:')
            st.info(rate)
