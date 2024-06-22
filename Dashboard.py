import streamlit as st
import pandas as pd
import warnings
import plotly.express as px

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")


# Inject the custom CSS into Streamlit
st.title("Amazon Sales Dashboard")

df = pd.read_csv("AmazonSalesReport_cleaned.csv", low_memory=False)

# Streamlit app
df['Date'] = df['Date'].str.strip()

st.sidebar.header('Filters')

# Try converting with the expected format
df['Date'] = pd.to_datetime(df['Date'], format='ISO8601', )

# Debug: Print the DataFrame after date parsing





    # Get the minimum and maximum dates
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

    # Date range filter
start_date = st.sidebar.date_input('Start date', min_date)
end_date = st.sidebar.date_input('End date', max_date)

status_options = ["All"] + df['Status'].unique().tolist()
sales_channel_options = ["All"] + df['Sales Channel '].unique().tolist()
status_filter = st.sidebar.selectbox('Select Status',status_options)
sales_channel_filter = st.sidebar.selectbox('Select Sales Channel', sales_channel_options)

    # Ensure the selected dates are within the valid range
if start_date > end_date:
    st.error('Error: End date must fall after start date.')
else:
    filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df['Status'] == status_filter]



    if sales_channel_filter != "All":
        filtered_df = filtered_df[filtered_df['Sales Channel '] == sales_channel_filter]
    aggregated_df = filtered_df.groupby('Date').agg({'Amount': 'sum', 'Qty': 'sum'}).reset_index()



    sales_fig = px.line(aggregated_df, x='Date', y='Amount', title='Sales Over Time')

    sales_fig.update_layout( title='Sales Over Time', xaxis_title='Date',
    yaxis_title='Amount',)
    st.plotly_chart(sales_fig)
    st.divider()
    col1, col2 = st.columns(2)
    with col1:

        if aggregated_df.empty:
            st.warning("No data available for the selected date range.")
        else:
            category_bar = px.bar(filtered_df.groupby('Category').agg({'Amount': 'sum'}).reset_index(),
                                  x='Category', y='Amount', title='Amount by Category')
            category_bar.update_layout( title='Amount by Category', xaxis_title='Category',
    yaxis_title='Amount',)
            st.plotly_chart(category_bar)


    # Quantity over time
    with col2:

        if aggregated_df.empty:
            st.warning("No data available for the selected date range.")
        else:
            status_pie = px.pie(df, names='Status', title='Status Distribution')

            status_pie.update_layout(title='Status Distribution',)
            st.plotly_chart(status_pie)

        # Filter the dataframe based on the selected date range
    col3, col4 = st.columns(2)
    st.divider()

    # Sales over time









    with col3:
        amount_hist = px.histogram(filtered_df, x='Amount', nbins=20, title='Distribution of Amount')

        amount_hist.update_layout( title='Distribution of Amount', xaxis_title='Amount',
    yaxis_title='count',)
        st.plotly_chart(amount_hist)

    with col4:
        ship_state_amount = filtered_df.groupby('ship-state').agg({'Amount': 'sum'}).reset_index()
        top_5_ship_states = ship_state_amount.nlargest(5, 'Amount')
        bar_fig = px.bar(top_5_ship_states, x='ship-state', y='Amount',
                         title='Top 5 Ship-States by Amount',
                         labels={'ship-service-level': 'Ship-State', 'Amount': 'Total Amount'})

        bar_fig.update_layout(title='Top 5 Ship-States by Amount', xaxis_title='ship-state',
    yaxis_title='Amount',)
        st.plotly_chart(bar_fig)

    amount_box = px.box(filtered_df, x='Status', y='Amount', title='Amount by Status')

    amount_box.update_layout(title='Amount by Status', xaxis_title='Status',
    yaxis_title='Amount',)
    st.plotly_chart(amount_box)
    st.divider()

    ship_state_size_amount = filtered_df.groupby(['ship-state', 'Size']).agg({'Amount': 'sum'}).reset_index()
    heatmap_data = ship_state_size_amount.pivot(index='Size', columns='ship-state', values='Amount').fillna(0)
    heatmap_fig = px.imshow(heatmap_data,
                            labels=dict(index="Ship-State", columns="Size", color="Amount"),
                            x=heatmap_data.columns,
                            y=heatmap_data.index,
                            title='Heatmap: Ship-State vs Size in Amount',
                            color_continuous_scale='Viridis')

    heatmap_fig.update_layout(title='Heatmap: Ship-State vs Size in Amount',)
    st.plotly_chart(heatmap_fig)
    st.divider()

    ship_state_category_amount = filtered_df.groupby(['ship-state', 'Category']).agg({'Amount': 'sum'}).reset_index()
    heatmap_data1 = ship_state_category_amount.pivot(index='Category', columns='ship-state', values='Amount').fillna(0)
    heatmap_fig2 = px.imshow(heatmap_data1,
                            labels=dict(index="Ship-State", columns="Category", color="Amount"),
                            x=heatmap_data1.columns,
                            y=heatmap_data1.index,
                            title='Heatmap: Category vs Size in Amount',
                            color_continuous_scale='Viridis')

    heatmap_fig2.update_layout(title='Heatmap: Category vs Size in Amount',)
    st.plotly_chart(heatmap_fig2)