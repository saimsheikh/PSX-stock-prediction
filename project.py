import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load the data from the Excel file
df = pd.read_excel('stock_prices(final).xlsx')
df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime type



# Create the Dash application
app = dash.Dash(__name__)

# Define the layout of the application
app.layout = html.Div([
    html.H1("PSX Stock Analysis and Prediction (SYS)"),
    dcc.Dropdown(
        id='data-dropdown',
        options=[
            {'label': 'Select an option', 'value': ''},
            {'label': 'Data of 2023', 'value': '2023'},
            {'label': 'All Data (2022 to May 2023)', 'value': 'all'}
        ],
        value=''
    ),
    html.Div(id='plot-container')
])

# Define the callback function to update the plot based on the dropdown selection
@app.callback(
    Output('plot-container', 'children'),
    [Input('data-dropdown', 'value')]
)
def update_plots(selected_option):
    if selected_option == '2023':
        # Filter the data for 2023
        filtered_data = df[df['Date'].dt.year == 2023]
        # Prepare the data for regression
        # Prepare the data for regression
        X = filtered_data[['Open', 'High', 'Low', 'Volume']]  # Input features
        y =filtered_data['Close']  # Target variable

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the regression model
        reg_model = LinearRegression()
        reg_model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = reg_model.predict(X_test)
        residuals = y_test - y_pred
        residual_fig = go.Figure(data=go.Scatter(x=y_test, y=residuals, mode='markers'))
        residual_fig.update_layout(title='Residual Plot', xaxis_title='Actual Close Value', yaxis_title='Residuals')

        # Create error histogram
        error_fig = go.Figure(data=go.Histogram(x=residuals, nbinsx=20))
        error_fig.update_layout(title='Error Histogram', xaxis_title='Residuals', yaxis_title='Frequency')

        # Create line plot of predicted values and actual values
        prediction_fig = go.Figure()
        prediction_fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['Close'], mode='lines', name='Actual Close Value'))
        prediction_fig.add_trace(go.Scatter(x=filtered_data.loc[X_test.index, 'Date'], y=y_pred, mode='lines', name='Predicted Close Value'))
        prediction_fig.update_layout(title='Predicted vs Actual Close Values', xaxis_title='Date', yaxis_title='Close Value')

        # Convert the figures to JSON for serialization
        residual = residual_fig.to_json()
        error = error_fig.to_json()
        prediction = prediction_fig.to_json()

        # Create line plot
        line_fig = px.line(filtered_data, x='Date', y='Close', title='Stock Close Price in 2023')

        scatter_fig = px.scatter(filtered_data, x='Date', y='Close', title='Stock Close Price in 2023', color='Open')

        # Create regression plot
        reg_fig = px.scatter(filtered_data, x='Date', y='Close', trendline='ols',color='Volume', title='Regression Plot')
        reg_fig.update_traces(marker=dict(size=8))

        # Create bar plot
        bar_fig = px.bar(filtered_data, x='Date', y='Volume', title='Stock Volume in 2023')

        # Create hit plot (assuming 'High' and 'Low' columns are present)
        hit_fig = go.Figure(data=[go.Candlestick(x=filtered_data['Date'],
                                                 open=filtered_data['Open'],
                                                 high=filtered_data['High'],
                                                 low=filtered_data['Low'],
                                                 close=filtered_data['Close'])])
        hit_fig.update_layout(title='Stock High, Low, Open, Close in 2023')

        return [
            dcc.Graph(figure=line_fig),
            dcc.Graph(figure=scatter_fig),
            dcc.Graph(figure=reg_fig),
            dcc.Graph(figure=bar_fig),
            dcc.Graph(figure=hit_fig),
            html.H2("Our Prediction"),
            dcc.Graph(figure=prediction_fig),
            dcc.Graph(figure=residual_fig),
            dcc.Graph(figure=error_fig)
        ]
    elif selected_option == 'all':
        # Create line plot for all data
        line_fig_all = px.line(df, x='Date', y='Close', title='Stock Close Price (2022 to May 2023)')

        scatter_fig_all= px.scatter(df, x='Date', y='Close', title='Stock Close Price in 2023', color='Open')

        # Create regression plot for all data
        reg_fig_all = px.scatter(df, x='Date', y='Close', trendline='ols',color='Volume', title='Regression Plot')
        reg_fig_all.update_traces(marker=dict(size=8))

        # Create bar plot for all data
        bar_fig_all = px.bar(df, x='Date', y='Volume', title='Stock Volume (2022 to May 2023)')

        # Create hit plot for all data (assuming 'High' and 'Low' columns are present)
        hit_fig_all = go.Figure(data=[go.Candlestick(x=df['Date'],
                                                     open=df['Open'],
                                                     high=df['High'],
                                                     low=df['Low'],
                                                     close=df['Close'])])
        hit_fig_all.update_layout(title='Stock High, Low, Open, Close (2022 to May 2023)')
        # Prepare the data for regression
        X = df[['Open', 'High', 'Low', 'Volume']]  # Input features
        y = df['Close']  # Target variable

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the regression model
        reg_model = LinearRegression()
        reg_model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = reg_model.predict(X_test)
        residuals = y_test - y_pred
        residual_fig = go.Figure(data=go.Scatter(x=y_test, y=residuals, mode='markers'))
        residual_fig.update_layout(title='Residual Plot', xaxis_title='Actual Close Value', yaxis_title='Residuals')

        # Create error histogram
        error_fig = go.Figure(data=go.Histogram(x=residuals, nbinsx=20))
        error_fig.update_layout(title='Error Histogram', xaxis_title='Residuals', yaxis_title='Frequency')

        # Create line plot of predicted values and actual values
        prediction_fig = go.Figure()
        prediction_fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Actual Close Value'))
        prediction_fig.add_trace(go.Scatter(x=df.loc[X_test.index, 'Date'], y=y_pred, mode='lines', name='Predicted Close Value'))
        prediction_fig.update_layout(title='Predicted vs Actual Close Values', xaxis_title='Date', yaxis_title='Close Value')

        # Convert the figures to JSON for serialization
        residual_json = residual_fig.to_json()
        error_json = error_fig.to_json()
        prediction_json = prediction_fig.to_json()

        return [
            dcc.Graph(figure=line_fig_all),
            dcc.Graph(figure=scatter_fig_all),
            dcc.Graph(figure=reg_fig_all),
            dcc.Graph(figure=bar_fig_all),
            dcc.Graph(figure=hit_fig_all),
            html.H2("Our Prediction"),
            dcc.Graph(figure=prediction_fig),
            dcc.Graph(figure=residual_fig),
            dcc.Graph(figure=error_fig)
        ]
    else:
        return []  # Return an empty list if no option is selected



# Run the Dash application
if __name__ == '__main__':
    app.run_server(debug=True)
