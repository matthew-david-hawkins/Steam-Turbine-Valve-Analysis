import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from iapws import IAPWS97 as steam_prop

#Define critical pressure ratio
alpha = 0.55

#Define a function that calculates the density for a given pressure and temperature
def calc_rho(p, t):
    
    #convert psia to MPA
    pressure = p*0.00689476

    #convert farenheit to Kelvin
    temp = (t - 32)*5/9 + 273.15

    #get the density 
    rho = steam_prop(P=pressure, T=temp).rho
    
    #convert density from kg/M^3 to lb/ft^3
    rho = rho*0.062428
    
    return rho

def generate_table(dataframe, max_rows=30):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

design_df = pd.read_csv('Resources/design_data.csv')

# Calculate the Inlet Density and Equivalent J = F/sqrt(rho*dp) based on input data
# Create empty lists to hold calculations
rho = []
factor = []

# Convert Dataframe to lists
f = design_df["Steam Flow (Design)"].to_list()
tp = design_df["Throttle Pressure (Design)"].to_list()
fsp = design_df["First Stage Pressure (Design)"].to_list()
t = design_df["Main Steam Temperature (Design)"].to_list()

# For each data point, calculate inlet density and equivalent J
for i in range(len(design_df)):
    rho.append(calc_rho(tp[i], t[i]))
    
    # If pressure drop is choked use alpha instead of total pressure drop
    if (tp[i] - fsp[i]) > alpha:
        dp = alpha*tp[i]
    else:
        dp = tp[i] - fsp[i]
        
    factor.append(f[i]/(rho[i]*dp)**0.5)

# Add columns for inlet density and Equivalent J
design_df["Inlet Density (Design)"] = rho
design_df["Equivalent J (Design)"] = factor

# Plot equivalent J vs Governor Demand
x_axis = design_df.loc[ design_df["Governor Demand (Design)"] > 0, "Governor Demand (Design)"].to_list()
y_axis = design_df.loc[ design_df["Governor Demand (Design)"] > 0, "Equivalent J (Design)"].to_list()

# Insert point (0,0)
x_axis.insert(0,0)
y_axis.insert(0,0)

markdown_text = '''
Plant Bowen's Governor Demand Data is plotted here.

Enjoy!
'''

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': "#111111",
    'text': '#7FDBFF'
}

app.layout = html.Div(children=[
   
   html.H1(
        children='Turbine Governor Valve Simulation',
        style={
            'textAlign': 'center'
        }
    ),

    html.Div(children=markdown_text, style={
        'textAlign': 'center'
    }),
        
        html.Label('Dropdown'),
        dcc.Dropdown(
            options=[
                {'label': 'Equivalent J', 'value': 'eqff'},
                {'label': 'Steam Flow', 'value': 'flow'},
                {'label': 'Generator MWG', 'value': 'mwg'},
                {'label': 'Main Steam Temp', 'value': 'temp'},
                {'label': 'Throttle Pressure', 'value': 'tp'},
                {'label': 'First Stage Pressure', 'value': 'fsp'},
                {'label': 'Inlet Density', 'value': 'rho'},
            ],
            value='eqff'
        ),

    dcc.Graph(
            id='gen-mwg-vs-gov-dmd',
            figure={
                'data': [
                    go.Scatter(
                        x=x_axis,
                        y=y_axis,
                        text='myText',
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                    )
                ],
                'layout': go.Layout(
                    xaxis={'title': 'Governor Demand'},
                    yaxis={'title': 'Equivalent J'},
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                    legend={'x': 0, 'y': 1},
                    hovermode='closest'
                )
                }
        ),
    
    html.Label('Outlier Elimination'),
    dcc.Slider(
        min=0,
        max=9,
        marks={i: str(i*0.01) for i in range(1, 10)},
        value=5,
    ),

    html.Label('Curve Fit'),
    dcc.RadioItems(
        options=[
            {'label': 'Linear', 'value': 'NYC'},
            {'label': 'x^2', 'value': 'MTL'},
            {'label': 'piece-wise', 'value': 'SF'}
        ],
        value='MTL'
    ),

    html.H4(children='Bowen Turbine Data'),
    generate_table(design_df)
])

if __name__=='__main__':
    app.run_server(debug=True)
    # app.run_server(dev_tools_hot_reload=False)