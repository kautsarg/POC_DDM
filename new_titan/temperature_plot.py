import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

def load_temperature_data(file_path):
    """
    Load temperature data from CSV file and convert time to datetime objects
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert time column to datetime
    # Assuming the first column contains time in HH:MM:SS format
    time_col = df.columns[0]  # Get the first column name
    
    # Convert time strings to datetime objects
    # We'll use a base date and add the time
    base_date = datetime(2024, 1, 1)  # Use a base date
    times = []
    
    for time_str in df[time_col]:
        # Parse the time string (remove quotes if present)
        time_str = str(time_str).strip('"')
        try:
            # Parse HH:MM:SS format
            time_parts = time_str.split(':')
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = int(time_parts[2])
            
            # Create datetime object
            dt = base_date + timedelta(hours=hours, minutes=minutes, seconds=seconds)
            times.append(dt)
        except:
            # If parsing fails, use index as time
            times.append(base_date + timedelta(seconds=len(times)))
    
    df['datetime'] = times
    
    return df

def plot_temperature_data():
    """
    Load and plot temperature data from both CSV files using Plotly
    """
    # File paths
    file1_path = r"C:\Users\Lacewing-01\OneDrive - ProtonDx\Data\Temperature Experiment\higher_starter_temp.csv"
    file2_path = r"C:\Users\Lacewing-01\OneDrive - ProtonDx\Data\Temperature Experiment\temp_fresh.csv"
    
    try:
        # Load data from both files
        print("Loading temperature data...")
        df1 = load_temperature_data(file1_path)
        df2 = load_temperature_data(file2_path)
        
        print(f"Loaded {len(df1)} data points from higher_starter_temp.csv")
        print(f"Loaded {len(df2)} data points from temp_fresh.csv")
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add traces for higher starter temp data
        fig.add_trace(go.Scatter(
            x=df1['datetime'],
            y=df1['Channel 1 Last (C)'],
            mode='lines',
            name='Higher Starter Temp (Last)',
            line=dict(width=2, color='blue'),
            opacity=0.8
        ))
        
        fig.add_trace(go.Scatter(
            x=df1['datetime'],
            y=df1['Channel 1 Ave. (C)'],
            mode='lines',
            name='Higher Starter Temp (Average)',
            line=dict(width=2, color='blue', dash='dash'),
            opacity=0.8
        ))
        
        # Add traces for temp fresh data
        fig.add_trace(go.Scatter(
            x=df2['datetime'],
            y=df2['Channel 1 Last (C)'],
            mode='lines',
            name='Temp Fresh (Last)',
            line=dict(width=2, color='red'),
            opacity=0.8
        ))
        
        fig.add_trace(go.Scatter(
            x=df2['datetime'],
            y=df2['Channel 1 Ave. (C)'],
            mode='lines',
            name='Temp Fresh (Average)',
            line=dict(width=2, color='red', dash='dash'),
            opacity=0.8
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Temperature Comparison: Higher Starter Temp vs Temp Fresh',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title='Time',
            yaxis_title='Temperature (°C)',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            width=1000,
            height=600,
            margin=dict(r=150)  # Add margin for legend
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Add statistics as annotations
        stats_text = f"""Statistics:<br>
Higher Starter Temp: {df1['Channel 1 Last (C)'].mean():.2f}°C avg, {df1['Channel 1 Last (C)'].min():.2f}°C min, {df1['Channel 1 Last (C)'].max():.2f}°C max<br>
Temp Fresh: {df2['Channel 1 Last (C)'].mean():.2f}°C avg, {df2['Channel 1 Last (C)'].min():.2f}°C min, {df2['Channel 1 Last (C)'].max():.2f}°C max"""
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="wheat",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
        
        # Show the plot
        fig.show()
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("="*50)
        print("Higher Starter Temp:")
        print(f"  Average: {df1['Channel 1 Last (C)'].mean():.3f}°C")
        print(f"  Min: {df1['Channel 1 Last (C)'].min():.3f}°C")
        print(f"  Max: {df1['Channel 1 Last (C)'].max():.3f}°C")
        print(f"  Std Dev: {df1['Channel 1 Last (C)'].std():.3f}°C")
        
        print("\nTemp Fresh:")
        print(f"  Average: {df2['Channel 1 Last (C)'].mean():.3f}°C")
        print(f"  Min: {df2['Channel 1 Last (C)'].min():.3f}°C")
        print(f"  Max: {df2['Channel 1 Last (C)'].max():.3f}°C")
        print(f"  Std Dev: {df2['Channel 1 Last (C)'].std():.3f}°C")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    plot_temperature_data()
