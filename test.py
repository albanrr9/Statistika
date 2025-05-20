from dash import Dash, html, dash_table, dcc, Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import skew, mode

# Load dataset
df = pd.read_excel("RealEstateSalesData.xlsx", sheet_name="Data")

# Define tab options
tab_options = [
    {'label': 'Data Types', 'value': 'task1'},
    {'label': 'Histogram (Max Bins)', 'value': 'task2'},
    {'label': 'Price Histogram ($100k bins)', 'value': 'task3'},
    {'label': 'Interpret Histogram', 'value': 'task4'},
    {'label': 'Price vs Area Scatter', 'value': 'task5'},
    {'label': 'Country Frequency Table', 'value': 'task6'},
    {'label': 'Pareto Diagram', 'value': 'task7'},
    {'label': 'Price Descriptive Stats', 'value': 'task8'},
    {'label': 'Interpret Stats', 'value': 'task9'},
    {'label': 'Covariance & Correlation', 'value': 'task10'},
    {'label': 'Gender Frequency', 'value': 'gender_pie'},
    {'label': 'Location Frequency', 'value': 'location_pareto'},
    {'label': 'Age Analysis', 'value': 'age_analysis'},
    {'label': 'View Table', 'value': 'table'},
]

# Dash App
external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
]

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.Div([
        dcc.Tabs(
    id='task-selector',
    value='task1',
    children=[
        dcc.Tab(
            label='Data Types', value='task1',
            className='tab', selected_className='tab--selected'
        ),
        dcc.Tab(
            label='Histogram (Max Bins)', value='task2',
            className='tab', selected_className='tab--selected'
        ),
        dcc.Tab(
            label='Price Histogram', value='task3',
            className='tab', selected_className='tab--selected'
        ),
        dcc.Tab(
            label='Interpret Histogram', value='task4',
            className='tab', selected_className='tab--selected'
        ),
        dcc.Tab(
            label='Scatter Plot', value='task5',
            className='tab', selected_className='tab--selected'
        ),
        dcc.Tab(
            label='Country Frequency Table', value='task6',
            className='tab', selected_className='tab--selected'
        ),
        dcc.Tab(
            label='Pareto Diagram', value='task7',
            className='tab', selected_className='tab--selected'
        ),
        dcc.Tab(
            label='Price Descriptive Stats', value='task8',
            className='tab', selected_className='tab--selected'
        ),
        dcc.Tab(
            label='Interpret Stats', value='task9',
            className='tab', selected_className='tab--selected'
        ),
        dcc.Tab(
            label='Covariance & Correlation', value='task10',
            className='tab', selected_className='tab--selected'
        ),
        dcc.Tab(
            label='Gender Frequency', value='gender_pie',
            className='tab', selected_className='tab--selected'
        ),
        dcc.Tab(
            label='Location Frequency', value='location_pareto',
            className='tab', selected_className='tab--selected'
        ),
        dcc.Tab(
            label='Age Analysis', value='age_analysis',
            className='tab', selected_className='tab--selected'
        ),
        dcc.Tab(
            label='View Table', value='table',
            className='tab', selected_className='tab--selected'
        ),
    ],
    className='mb-4'
),


        html.Div(id='customization-panel'),
        html.Div(id='content'),
        html.Div(id='dummy-div', style={'display': 'none'}),
    ], className="container")
])

def analyze_column(series):
    dtype = series.dtype
    unique_vals = series.dropna().unique()
    n_unique = len(unique_vals)

    if dtype == 'object' or dtype == 'bool':
        type_of_data = 'Categorical'
        if n_unique == 2:
            level = 'Nominal (Binomial)'
        else:
            level = 'Nominal'
    elif np.issubdtype(dtype, np.integer):
        type_of_data = 'Numerical (Discrete)'
        if "year" in series.name.lower():
            level = 'Interval'
        else:
            level = 'Ratio'
    elif np.issubdtype(dtype, np.floating):
        type_of_data = 'Numerical (Continuous)'
        level = 'Ratio'
    else:
        type_of_data = 'Unknown'
        level = 'Unknown'

    return type_of_data, level, f"Unique values: {n_unique}, dtype: {dtype}"

@app.callback(
    Output('customization-panel', 'children'),
    Input('task-selector', 'value')
)
def show_controls(task):
    if task == 'task1':
        return html.Div([
            html.Label("Select a column to analyze:"),
            dcc.Dropdown(
                options=[{"label": col, "value": col} for col in df.columns],
                id="column-dropdown"
            ),
            html.Br(),
            html.Div(id="output-info")
        ])
    elif task == 'task2':
        return html.Div([
            html.P("Histogram will use the maximum number of bins (272)."),
            html.Label("Histogram Color"),
            dcc.Dropdown(
                id='hist-color',
                options=[{'label': col, 'value': col} for col in df.columns if df[col].dtype == 'object'],
                value=None,
                placeholder='Select a categorical variable for color',
            ),
        ])
    return None

@app.callback(
    Output('output-info', 'children'),
    Input('column-dropdown', 'value')
)
def update_column_info(col):
    if not col:
        return ""
    type_of_data, level, details = analyze_column(df[col])
    return html.Div([
        html.P(f"Type of data: {type_of_data}"),
        html.P(f"Level of measurement: {level}"),
        html.P(details)
    ])

@app.callback(
    Output('dummy-div', 'children'),
    Input('task-selector', 'value')
)
def initialize(task):
    return None

@app.callback(
    Output('content', 'children'),
    [
        Input('task-selector', 'value'),
        Input('hist-color', 'value')
    ]
)
def update_content(task, color):
    if task == 'task2':
        # Histogram with max bins (272)
        fig = px.histogram(
            df,
            x='Price',
            nbins=272,
            color=color,
            title='Price Distribution (272 bins)',
            labels={'Price': 'Price ($)'}
        )
        fig.update_layout(
            bargap=0.01,
            xaxis=dict(
                tickmode='auto',
                tickformat='$,.0f',
                nticks=20
            )
        )
        return dcc.Graph(figure=fig)

    elif task == 'task3':
        # Histogram with $100,000 bins
        min_price = df['Price'].min()
        max_price = df['Price'].max()
        bin_size = 100000
        bin_edges = np.arange(min_price, max_price + bin_size, bin_size)
        fig = px.histogram(
            df,
            x='Price',
            nbins=len(bin_edges)-1,
            title=f'Price Distribution (${bin_size:,} bins starting from ${min_price:,.0f})',
            labels={'Price': 'Price ($)'}
        )
        fig.update_traces(xbins=dict(
            start=min_price,
            end=bin_edges[-1],
            size=bin_size
        ))
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=bin_edges,
                ticktext=[f'${x:,.0f}' for x in bin_edges]
            ),
            bargap=0.05
        )
        return dcc.Graph(figure=fig)

    elif task == 'task4':
        # Interpretation for histogram (static or could be dynamic)
        return html.Div([
            html.H4("Interpretation of Histogram"),
            html.P("The histogram shows the distribution of property prices. Peaks indicate common price ranges, while gaps or long tails may suggest outliers or rare price points. The shape (e.g., skewed, symmetric) provides insight into market trends.")
        ])

    elif task == 'task5':
        # Scatter plot Price vs Area
        area = df['Area (ft.)']
        fig = px.scatter(
            df,
            x='Area (ft.)',
            y='Price',
            title='Scatter Plot: Price vs Area',
            labels={'Area (ft.)': 'Area (sq ft)', 'Price': 'Price ($)'}
        )
        return html.Div([
            dcc.Graph(figure=fig),
            html.H4("Interpretation"),
            html.P("This scatter plot visualizes the relationship between property area and price. A positive trend suggests that larger properties tend to have higher prices. Outliers may indicate unusually priced properties for their size.")
        ])

    elif task == 'task6':
        # Frequency table for countries
        country_col = 'Country' if 'Country' in df.columns else df.columns[-1]  # fallback
        freq = df[country_col].value_counts().reset_index()
        freq.columns = ['Country', 'Absolute Frequency']
        freq['Relative Frequency'] = freq['Absolute Frequency'] / freq['Absolute Frequency'].sum()
        freq['Cumulative Frequency'] = freq['Absolute Frequency'].cumsum()
        return dash_table.DataTable(
            data=freq.to_dict('records'),
            columns=[{"name": i, "id": i} for i in freq.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
            style_data={'whiteSpace': 'normal', 'height': 'auto'}
        )

    elif task == 'task7':
        # Pareto diagram for countries
        country_col = 'Country' if 'Country' in df.columns else df.columns[-1]
        freq = df[country_col].value_counts().reset_index()
        freq.columns = ['Country', 'Frequency']
        freq = freq.sort_values(by='Frequency', ascending=False)
        freq['Cumulative %'] = freq['Frequency'].cumsum() / freq['Frequency'].sum() * 100

        fig = go.Figure()
        fig.add_bar(x=freq['Country'], y=freq['Frequency'], name='Frequency')
        fig.add_scatter(x=freq['Country'], y=freq['Cumulative %'], name='Cumulative %', yaxis='y2')

        fig.update_layout(
            title='Pareto Diagram of Buyers by Country',
            yaxis=dict(title='Frequency', range=[0,204.5]),
            yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 105]),
            xaxis=dict(title='Country')
        )
        return dcc.Graph(figure=fig)

    elif task == 'task8':
        # Descriptive statistics for Price
        price = df['Price'].dropna()
        mean_ = price.mean()
        median_ = price.median()
        mode_ = price.mode().iloc[0] if not price.mode().empty else np.nan
        skewness_ = skew(price)
        variance_ = price.var()
        std_ = price.std()
        return html.Div([
            html.H4("Descriptive Statistics for Price"),
            html.P(f"Mean: ${mean_:,.2f}"),
            html.P(f"Median: ${median_:,.2f}"),
            html.P(f"Mode: ${mode_:,.2f}"),
            html.P(f"Skewness: {skewness_:.2f}"),
            html.P(f"Variance: {variance_:,.2f}"),
            html.P(f"Standard Deviation: {std_:,.2f}")
        ])

    elif task == 'task9':
        # Interpretation for descriptive stats
        return html.Div([
            html.H4("Interpretation of Descriptive Statistics"),
            html.P("The mean and median provide measures of central tendency for property prices. If the mean is higher than the median, the distribution is right-skewed, indicating some high-value properties. Skewness quantifies this asymmetry. Variance and standard deviation measure the spread of prices; higher values indicate more variability in the market.")
        ])

    elif task == 'task10':
        # Covariance and correlation between Price and Area
        price = df['Price']
        area = df['Area (ft.)']
        cov = np.cov(price, area)[0, 1]
        corr = np.corrcoef(price, area)[0, 1]
        return html.Div([
            html.H4("Covariance and Correlation between Price and Area"),
            html.P(f"Covariance: {cov:,.2f}"),
            html.P(f"Correlation Coefficient: {corr:.2f}"),
            html.P("A positive correlation indicates that as area increases, price tends to increase as well. This should be consistent with the scatter plot in Task 5.")
        ])

    elif task == 'gender_pie':
    # Prepare frequency table
        gender_col = 'Gender'
        if gender_col not in df.columns:
            return html.Div([
                html.H5("Gender Frequency & Pie Chart"),
                html.P("The 'Gender' column was not found in the dataset.", className="text-danger")
            ])

        # Map any 'M'/'F' to 'Male'/'Female' for safety
        gender_series = df[gender_col].replace({'M': 'Male', 'F': 'Female'})

        # Only count 'Male', 'Female', 'Firms'
        categories = ['Male', 'Female', 'Firms']
        freq = gender_series.value_counts().reindex(categories, fill_value=0).reset_index()
        freq.columns = ['Gender', 'Frequency']
        total = freq['Frequency'].sum()
        freq['Relative frequency'] = (freq['Frequency'] / total * 100).round().astype(int).astype(str) + '%'

        # Add total row
        total_row = pd.DataFrame([{'Gender': 'Total', 'Frequency': total, 'Relative frequency': '100%'}])
        freq_table = pd.concat([freq, total_row], ignore_index=True)

        # Pie chart (exclude 'Total' row)
        pie_data = freq[freq['Gender'] != 'Total']
        color_map = {'Male': '#c44d58', 'Female': '#a3c86d', 'Firms': '#22313f'}

        fig = px.pie(
            pie_data,
            names='Gender',
            values='Frequency',
            color='Gender',
            color_discrete_map=color_map,
            hole=0,
        )
        fig.update_traces(textinfo='percent+label', textfont_size=18)

        return html.Div([
            html.H5("Gender"),
            html.H6("Frequency distribution table", className="mt-3"),
            dash_table.DataTable(
                data=freq_table.to_dict('records'),
                columns=[{"name": i, "id": i} for i in freq_table.columns],
                style_table={'width': '350px', 'margin-bottom': '20px'},
                style_cell={'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': 16},
                style_header={'fontWeight': 'bold', 'borderBottom': '2px solid #22313f'},
                style_data_conditional=[
                    {'if': {'row_index': len(freq_table)-1}, 'fontWeight': 'bold'}
                ]
            ),
            dcc.Graph(figure=fig, style={'width': '600px', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div(
                "N.B. Firms have no gender. However, we need to add them to this pie chart, as otherwise, we will get a wrong interpretation of the data.",
                className="mt-4",
                style={'fontSize': 15, 'fontStyle': 'italic'}
            )
        ])

    elif task == 'location_pareto':
        country_col = 'Country'
        state_col = 'State'

        if state_col not in df.columns or country_col not in df.columns:
            return html.Div([
                html.H5("Location Frequency & Pareto Chart"),
                html.P("Required columns 'State' or 'Country' not found in the dataset.", className="text-danger")
            ])

        # Create a clean copy
        location_df = df[[country_col, state_col]].copy()

        # Fill missing or non-USA states with 'None (abroad)'
        location_df['State'] = location_df.apply(
            lambda row: row[state_col] if row[country_col] == 'USA' else 'None (abroad)', axis=1
        )

        # Frequency distribution
        freq = location_df['State'].value_counts().reset_index()
        freq.columns = ['State', 'Frequency']
        total = freq['Frequency'].sum()
        freq['Relative frequency'] = (freq['Frequency'] / total * 100).round().astype(int).astype(str) + '%'
        freq['Cumulative frequency'] = freq['Frequency'].cumsum()

        # Calculate US-only cumulative frequency
        us_states = location_df[location_df['State'] != 'None (abroad)']['State'].value_counts()
        us_states = us_states.reindex(freq['State']).fillna(0).astype(int)
        cumulative_us = us_states.cumsum().astype(int)

        freq['Cumulative US only'] = freq['State'].map(cumulative_us).fillna('')
        freq['Cumulative US only'] = freq['Cumulative US only'].astype(str)

        print(cumulative_us)
        print(freq['State'].tolist())


        # Add total row
        total_row = pd.DataFrame([{
            'State': 'Total',
            'Frequency': total,
            'Relative frequency': '100%',
            'Cumulative frequency': '',
            'Cumulative US only': ''
        }])
        freq_table = pd.concat([freq, total_row], ignore_index=True)

        # Pareto data: exclude 'None (abroad)'
        pareto_data = freq[freq['State'] != 'None (abroad)']
        pareto_data['Cumulative %'] = (pareto_data['Frequency'].cumsum() / pareto_data['Frequency'].sum() * 100).round(1)

        # Plot
        fig = go.Figure()
        fig.add_bar(
            x=pareto_data['State'],
            y=pareto_data['Frequency'],
            name='Frequency',
            marker_color='#22313f'
        )
        fig.add_scatter(
            x=pareto_data['State'],
            y=pareto_data['Cumulative %'],
            name='Cumulative %',
            mode='lines+markers',
            line=dict(color='orange', width=2),
            yaxis='y2'
        )
        fig.update_layout(
            title='Segmentation of US clients by State',
            xaxis=dict(title='State'),
            yaxis=dict(title='Frequency'),
            yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 100]),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        return html.Div([
            html.H5("Location"),
            html.H6("Frequency distribution table", className="mt-3"),
            dash_table.DataTable(
                data=freq_table.to_dict('records'),
                columns=[{"name": i, "id": i} for i in freq_table.columns],
                style_table={'width': '600px', 'margin-bottom': '20px'},
                style_cell={'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': 16},
                style_header={'fontWeight': 'bold', 'borderBottom': '2px solid #22313f'},
                style_data_conditional=[{
                    'if': {'row_index': len(freq_table)-1}, 'fontWeight': 'bold'
                }]
            ),
            dcc.Graph(figure=fig, style={'width': '800px', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div(
                "N.B. Clients from outside the USA are grouped under 'None (abroad)'. They're included in the frequency table but excluded from the US-only cumulative count and Pareto diagram.",
                className="mt-4",
                style={'fontSize': 15, 'fontStyle': 'italic'}
            )
        ])

    elif task == 'age_analysis':
        # --- Prepare Age Data ---
        age_col = 'Age at time of purchase'
        if age_col not in df.columns:
            return html.Div([
                html.H5("Age Analysis"),
                html.P("The 'Age' column was not found in the dataset.", className="text-danger")
            ])

        # Remove nulls and firms if present
        age_series = df[age_col]
        if 'Gender' in df.columns:
            age_series = age_series[df['Gender'] != 'Firms']

        # Define bins and labels
        bins = [18, 25, 35, 45, 55, 65, np.inf]
        labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        age_groups = pd.cut(age_series, bins=bins, labels=labels, right=True, include_lowest=True)

        # Frequency table
        freq = age_groups.value_counts().reindex(labels, fill_value=0).reset_index()
        freq.columns = ['Age Group', 'Frequency']
        total = freq['Frequency'].sum()
        freq['Relative frequency'] = (freq['Frequency'] / total * 100).round().astype(int).astype(str) + '%'
        total_row = pd.DataFrame([{'Age Group': 'Total', 'Frequency': total, 'Relative frequency': '100%'}])
        freq_table = pd.concat([freq, total_row], ignore_index=True)

        # Descriptive stats
        stats = {
            'Mean': age_series.mean(),
            'Median': age_series.median(),
            'Mode': age_series.mode().iloc[0] if not age_series.mode().empty else np.nan,
            'Skew': skew(age_series.dropna()),
            'Variance': age_series.var(),
            'St. dev.': age_series.std()
        }

        # Histogram of raw ages
        hist_fig = px.histogram(
            age_series.dropna(), nbins=30,
            title='Histogram of Age',
            labels={'value': 'Age', 'count': 'Frequency'}
        )
        hist_fig.update_layout(bargap=0.05)

        # Bar chart of grouped ages
        bar_fig = px.bar(
            freq.iloc[:-1], x='Age Group', y='Frequency',
            title='Age Group Distribution',
            labels={'Frequency': 'Count', 'Age Group': 'Age Group'}
        )
        bar_fig.update_layout(yaxis=dict(range=[0, freq['Frequency'].max() + 5]))

        # Layout
        return html.Div([
            html.H4("Age"),
            html.H6("Frequency distribution table"),
            dash_table.DataTable(
                data=freq_table.to_dict('records'),
                columns=[{"name": i, "id": i} for i in freq_table.columns],
                style_table={'width': '350px', 'margin-bottom': '20px'},
                style_cell={'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': 16},
                style_header={'fontWeight': 'bold', 'borderBottom': '2px solid #22313f'},
                style_data_conditional=[
                    {'if': {'row_index': len(freq_table)-1}, 'fontWeight': 'bold'}
                ]
            ),
            html.Div([
                dcc.Graph(figure=hist_fig, style={'width': '40%', 'display': 'inline-block'}),
                dcc.Graph(figure=bar_fig, style={'width': '40%', 'display': 'inline-block', 'marginLeft': '5%'})
            ], style={'display': 'flex', 'flexDirection': 'row'}),
            html.Div([
                html.P(f"Mean: {stats['Mean']:.2f}"),
                html.P(f"Median: {stats['Median']:.2f}"),
                html.P(f"Mode: {stats['Mode']:.2f}"),
                html.P(f"Skew: {stats['Skew']:.2f}"),
                html.P(f"Variance: {stats['Variance']:.2f}"),
                html.P(f"St. dev.: {stats['St. dev.']:.2f}")
            ], style={'marginTop': '20px', 'fontWeight': 'bold'}),
            html.Div([
                html.P("The first graph represents the frequency distribution of the variable age. We have a high bar at 0 as that is the null values associated with corporate clients."),
                html.P("The second graph is a histogram, based on the intervals from the original table. It is built using the data from the frequency distribution table above. Firms are not included in it."),
                html.P("Note: if you reorder the Spreadsheet 365RE some values may change")
            ], style={'marginTop': '20px', 'fontSize': 15})
        ])

    elif task == 'table':
        return dash_table.DataTable(
            data=df.to_dict('records'),
            page_size=25,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
            style_data={'whiteSpace': 'normal', 'height': 'auto'}
        )
    return None

if __name__ == '__main__':
    app.run(debug=True)