import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

def get_theme_colors():
    is_dark = st.session_state.get("theme", "Light (Default)") == "Dark"
    return {
        "bg": "#1e293b" if is_dark else "#ffffff",
        "text": "#f8fafc" if is_dark else "#0f172a",
        "primary": "#3b82f6" if is_dark else "#1e3a8a",
        "grid": "#334155" if is_dark else "#e2e8f0"
    }

def create_risk_gauge(risk_score, risk_level):
    """Creates a beautiful risk gauge chart."""
    colors = get_theme_colors()
    
    # Determine color based on level
    if risk_level == "Low Risk":
        bar_color = "#10b981" # green
    elif risk_level == "Moderate Risk":
        bar_color = "#f59e0b" # yellow/orange
    else:
        bar_color = "#ef4444" # red
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score", 'font': {'size': 24, 'color': colors['text'], 'family': 'Outfit'}},
        number = {'suffix': "%", 'font': {'color': colors['text'], 'family': 'Outfit'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': colors['grid']},
            'bar': {'color': bar_color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': colors['grid'],
            'steps': [
                {'range': [0, 33], 'color': 'rgba(16, 185, 129, 0.1)'},
                {'range': [33, 66], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [66, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_score * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text'], 'family': 'Outfit'},
        margin=dict(l=20, r=20, t=50, b=20),
        height=250
    )
    return fig

def create_probabilities_bar_chart(probabilities):
    """Creates a horizontal bar chart for class probabilities."""
    colors = get_theme_colors()
    
    labels = list(probabilities.keys())
    values = [val * 100 for val in probabilities.values()]
    
    # Sort for better visualization
    sorted_pairs = sorted(zip(labels, values), key=lambda x: x[1])
    sorted_labels = [p[0] for p in sorted_pairs]
    sorted_values = [p[1] for p in sorted_pairs]
    
    fig = go.Figure(go.Bar(
        x=sorted_values,
        y=sorted_labels,
        orientation='h',
        marker=dict(
            color=colors['primary'],
            line=dict(color=colors['primary'], width=1)
        )
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text'], 'family': 'Outfit'},
        margin=dict(l=10, r=10, t=10, b=10),
        height=200,
        xaxis=dict(
            showgrid=True,
            gridcolor=colors['grid'],
            range=[0, 100],
            title="Probability (%)"
        ),
        yaxis=dict(
            showgrid=False
        )
    )
    return fig

def create_trend_chart(dates, values, title="Trends"):
    """Creates a line chart for historical trends."""
    colors = get_theme_colors()
    
    fig = go.Figure(go.Scatter(
        x=dates, 
        y=values,
        mode='lines+markers',
        line=dict(color=colors['primary'], width=3),
        marker=dict(size=8, color=colors['primary'], line=dict(width=2, color=colors['bg'])),
        fill='tozeroy',
        fillcolor=f"{colors['primary']}33" # 20% opacity
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=colors['text'], family='Outfit')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text'], 'family': 'Outfit'},
        xaxis=dict(showgrid=True, gridcolor=colors['grid']),
        yaxis=dict(showgrid=True, gridcolor=colors['grid']),
        margin=dict(l=40, r=20, t=40, b=30),
        height=300
    )
    return fig

def create_distribution_donut(labels, values, title="Distribution"):
    """Creates a donut chart for categorical distributions."""
    colors = get_theme_colors()
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.6,
        marker=dict(colors=["#10b981", "#f59e0b", "#ef4444"])
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=colors['text'], family='Outfit')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text'], 'family': 'Outfit'},
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig
