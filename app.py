import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew
import plotly.express as px
import plotly.graph_objects as go

# ----- PAGE CONFIG & STYLE -----
st.set_page_config(
    page_title="Real Estate Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .css-18e3th9 {
            padding-top: 2rem;
        }
        .css-1d391kg {
            padding: 2rem;
        }
        .css-1kyxreq {
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ----- SIDEBAR: FILE UPLOAD & NAVIGATION -----
st.sidebar.title("📊 Navigimi")
uploaded_file = st.sidebar.file_uploader("Ngarko një Excel tjetër:", type=['xlsx'])

tab_labels = [
    'Llojet e të Dhënave',
    'Histogram (Numri i Kutinave)',
    'Histogrami i Çmimit (Kuti prej 100 mijë $)',
    'Interpreto Histogramin',
    'Scatter Çmimi vs Sipërfaqe',
    'Tabela e Frekuencës së Shteteve',
    'Diagrami Pareto',
    'Statistikat Përshkruese të Çmimit',
    'Interpreto Statistikat',
    'Kovarianca & Korrelacioni',
    'Frekuenca e Gjinive',
    'Frekuenca e Lokacionit',
    'Analiza e Moshës',
    'Shfaq Tabelën'
]

tab = st.sidebar.radio("Zgjidh Detyrën", tab_labels)
st.sidebar.markdown(f"**Aktualisht në:** {tab}")

# Optional: Data Dictionary
if st.sidebar.checkbox("ℹ️ Shfaq përshkrimin e kolonave"):
    st.sidebar.markdown("""
    - **Price**: Çmimi i blerjes në €  
    - **Area (ft.)**: Sipërfaqja në këmbë katrore  
    - **Gender**: Gjinia e blerësit (Mashkull, Femër, Firms)  
    - **Country / State**: Lokacioni i blerësit  
    - **Age at time of purchase**: Mosha në momentin e blerjes  
    """)

# ----- LOAD DATA -----
if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Data")
else:
    df = pd.read_excel("RealEstateSalesData.xlsx", sheet_name="Data")

# ----- FUNCTIONS -----
def analyze_column(series):
    dtype = series.dtype
    unique_vals = series.dropna().unique()
    n_unique = len(unique_vals)

    if dtype == 'object' or dtype == 'bool':
        type_of_data = 'Kategorike'
        level = 'Nominale (Binomiale)' if n_unique == 2 else 'Nominale'
    elif np.issubdtype(dtype, np.integer):
        type_of_data = 'Numerike (Diskrete)'
        level = 'Interval' if "vit" in series.name.lower() else 'Raport'
    elif np.issubdtype(dtype, np.floating):
        type_of_data = 'Numerike (Vazhdueshme)'
        level = 'Raport'
    else:
        type_of_data = 'E panjohur'
        level = 'E panjohur'

    return type_of_data, level, f"Vlera unike: {n_unique}, dtype: {dtype}"

# ----- APP TITLE -----
st.title("📈 Analiza e të Dhënave të Shitjeve të Pasurive të Paluajtshme")

# ----- TABS LOGIC -----
if tab == 'Llojet e të Dhënave':
    st.header("📌 Llojet e të Dhënave")
    col = st.selectbox("Zgjidh një kolonë për analizë:", df.columns)
    if col:
        type_of_data, level, details = analyze_column(df[col])
        st.write(f"**Lloji i të dhënave:** {type_of_data}")
        st.write(f"**Niveli i matjes:** {level}")
        st.write(details)

elif tab == 'Histogram (Numri i Kutinave)':
    st.header("📊 Histogram (Numri i Kutinave)")
    color_col = st.selectbox(
        "Ngjyra e histogramit (variabël kategorike):",
        [None] + [col for col in df.columns if df[col].dtype == 'object'],
        index=0
    )
    max_bins = min(300, df['Price'].nunique())
    num_bins = st.slider("Numri i kutinave:", 5, max_bins, 50)
    with st.spinner("Duke gjeneruar histogramin..."):
        fig = px.histogram(
            df, x='Price', nbins=num_bins,
            color=color_col if color_col else None,
            title='Shpërndarja e Çmimeve',
            labels={'Price': 'Çmimi (€)'}
        )
        fig.update_layout(bargap=0.01)
        st.plotly_chart(fig, use_container_width=True)

elif tab == 'Histogrami i Çmimit (Kuti prej 100 mijë $)':
    st.header("📊 Histogrami i Çmimit (Kuti prej 100 mijë $)")
    min_price = df['Price'].min()
    max_price = df['Price'].max()
    bin_size = 100000
    bin_edges = np.arange(min_price, max_price + bin_size, bin_size)
    fig = px.histogram(df, x='Price', nbins=len(bin_edges) - 1)
    fig.update_traces(xbins=dict(start=min_price, end=bin_edges[-1], size=bin_size))
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=bin_edges,
            ticktext=[f'${x:,.0f}' for x in bin_edges]
        ),
        title='Shpërndarja e Çmimeve (kuti prej 100 mijë $)',
        bargap=0.05
    )
    st.plotly_chart(fig, use_container_width=True)

elif tab == 'Interpreto Histogramin':
    st.header("🧠 Interpretimi i Histogramit")
    st.write(
        "Histogrami tregon shpërndarjen e çmimeve të pronave. Pikat e larta tregojnë intervale të zakonshme të çmimeve, "
        "ndërsa boshllëqet ose bishtat e gjatë mund të tregojnë vlera të jashtëzakonshme ose çmime të rralla."
    )

elif tab == 'Scatter Çmimi vs Sipërfaqe':
    st.header("📍 Scatter: Çmimi vs Sipërfaqe")
    fig = px.scatter(df, x='Area (ft.)', y='Price', labels={
        'Area (ft.)': 'Sipërfaqja (ft²)',
        'Price': 'Çmimi (€)'
    })
    st.plotly_chart(fig, use_container_width=True)

elif tab == 'Tabela e Frekuencës së Shteteve':
    st.header("📋 Tabela e Frekuencës së Shteteve")
    country_col = 'Country' if 'Country' in df.columns else df.columns[-1]
    freq = df[country_col].value_counts().reset_index()
    freq.columns = ['Shteti', 'Frekuenca Absolute']
    freq['Frekuenca Relative'] = (freq['Frekuenca Absolute'] / freq['Frekuenca Absolute'].sum()).round(2)
    freq['Frekuenca Kumulative'] = freq['Frekuenca Absolute'].cumsum()
    st.dataframe(freq, use_container_width=True)

elif tab == 'Diagrami Pareto':
    st.header("📊 Diagrami Pareto i Blerësve sipas Shtetit")
    country_col = 'Country'
    freq = df[country_col].value_counts().reset_index()
    freq.columns = ['Shteti', 'Frekuenca']
    freq['% Kumulative'] = freq['Frekuenca'].cumsum() / freq['Frekuenca'].sum() * 100

    fig = go.Figure()
    fig.add_bar(x=freq['Shteti'], y=freq['Frekuenca'], name='Frekuenca')
    fig.add_scatter(x=freq['Shteti'], y=freq['% Kumulative'], name='% Kumulative', yaxis='y2')

    fig.update_layout(
        title='Diagrami Pareto i Blerësve sipas Shtetit',
        yaxis=dict(title='Frekuenca'),
        yaxis2=dict(title='% Kumulative', overlaying='y', side='right'),
        xaxis=dict(title='Shteti')
    )
    st.plotly_chart(fig, use_container_width=True)

elif tab == 'Statistikat Përshkruese të Çmimit':
    st.header("📉 Statistikat Përshkruese për Çmimin")
    price = df['Price'].dropna()
    st.write(f"**Mesatarja:** {price.mean():,.2f} €")
    st.write(f"**Mediana:** {price.median():,.2f} €")
    st.write(f"**Moda:** {price.mode().iloc[0]:,.2f} €")
    st.write(f"**Anëshmëria:** {skew(price):.2f}")
    st.write(f"**Varianca:** {price.var():,.2f}")
    st.write(f"**Devijimi Standard:** {price.std():,.2f}")

elif tab == 'Interpreto Statistikat':
    st.header("🧠 Interpretimi i Statistikave")
    st.write(
        "Mesatarja dhe mediana japin masa të tendencës qendrore për çmimet e pronave. "
        "Nëse mesatarja është më e lartë se mediana, shpërndarja është e shtrembëruar djathtas. "
        "Varianca dhe devijimi standard matin përhapjen e çmimeve."
    )

elif tab == 'Kovarianca & Korrelacioni':
    st.header("📈 Kovarianca dhe Korrelacioni")
    price, area = df['Price'], df['Area (ft.)']
    cov = np.cov(price, area)[0, 1]
    corr = np.corrcoef(price, area)[0, 1]
    st.write(f"**Kovarianca:** {cov:,.2f}")
    st.write(f"**Korrelacioni:** {corr:.2f}")

elif tab == 'Frekuenca e Gjinive':
    st.header("🧍 Frekuenca e Gjinive")
    if 'Gender' not in df.columns:
        st.error("Kolona 'Gender' mungon.")
    else:
        gender_series = df['Gender'].str.strip().str.capitalize().replace({
            'Male': 'Mashkull', 'Female': 'Femër', 'Firms': 'Firms'
        })
        categories = ['Mashkull', 'Femër', 'Firms']
        freq = gender_series.value_counts().reindex(categories, fill_value=0).reset_index()
        freq.columns = ['Gjinia', 'Frekuenca']
        total = freq['Frekuenca'].sum()
        freq['Frekuenca relative'] = (freq['Frekuenca'] / total * 100).round(1).astype(str) + '%'
        pie_data = freq[freq['Frekuenca'] > 0]
        fig = px.pie(pie_data, names='Gjinia', values='Frekuenca', hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(freq)

elif tab == 'Frekuenca e Lokacionit':
    st.header("🌍 Frekuenca e Lokacionit")
    if 'State' not in df.columns or 'Country' not in df.columns:
        st.error("Kolonat 'State' ose 'Country' mungojnë.")
    else:
        location_df = df[['Country', 'State']].copy()
        location_df['State'] = location_df.apply(
            lambda row: row['State'] if row['Country'] == 'USA' else 'Asnjë (jashtë)', axis=1
        )
        freq = location_df['State'].value_counts().reset_index()
        freq.columns = ['Shteti', 'Frekuenca']
        st.dataframe(freq)

elif tab == 'Analiza e Moshës':
    st.header("👥 Analiza e Moshës")
    age_col = 'Age at time of purchase'
    if age_col not in df.columns:
        st.error("Kolona 'Age' mungon.")
    else:
        age_series = df[age_col]
        bins = [18, 25, 35, 45, 55, 65, np.inf]
        labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        age_groups = pd.cut(age_series, bins=bins, labels=labels)
        freq = age_groups.value_counts().reindex(labels, fill_value=0).reset_index()
        freq.columns = ['Grupi Moshor', 'Frekuenca']
        st.bar_chart(freq.set_index('Grupi Moshor'))
        st.dataframe(freq)

elif tab == 'Shfaq Tabelën':
    st.header("📄 Tabela e Plotë e të Dhënave")
    with st.expander("Kliko për të parë të gjitha të dhënat"):
        st.dataframe(df, use_container_width=True)
