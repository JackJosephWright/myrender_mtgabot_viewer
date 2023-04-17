import dash
from dash import dcc
from dash import html
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dash import dash_table
from dash.dependencies import Input, Output
from datetime import datetime, timedelta
import plotly.express as px
##os.chdir('tests')
import mysql.connector
import time
db_config = {
    "host" : "mtgabot.mysql.database.azure.com",
    "username" : "masterUsername",
    "password" : "password123!@#",
    "database" : "game_records"
        }
def db_cursor():
    try:
        cnx = mysql.connector.connect(**db_config)
        cursor = cnx.cursor()
        return cursor, cnx
    except Exception as e:
        print("Error connecting to database:", e)
        return None, None
cursor, cnx = db_cursor()
query = """
        SELECT gd.uid, gd.gold, gd.gems
        FROM game_data gd
        INNER JOIN (
          SELECT uid, MAX(date) AS max_date
          FROM game_data
          WHERE gold IS NOT NULL AND gems IS NOT NULL
          GROUP BY uid
        ) AS subquery
        ON gd.uid = subquery.uid AND gd.date = subquery.max_date"""

cursor.execute(query)
account_values = cursor.fetchall()

def gem_converter(gems):
    return gems*(10000/1500)

account_values = [(x[0], x[1], gem_converter(x[2])) for x in account_values]

df = pd.DataFrame(account_values, columns = ['uid', 'gold', 'gems'])
df['total'] = df['gold'] + df['gems']
## multiply by 11/10000 to convert to dollars
df['total_dollars'] = df['total']*(11/10000)
df = df.drop('total', axis = 1)
df = df.drop('gems', axis = 1)
df = df.drop('gold', axis = 1)

num_games = 0  # initialize to 0 in case of error
#print sum of df['total_dollars']
print('total dollars: ', df['total_dollars'].sum())
total_value_txt = "Total dollars: " + str(df['total_dollars'].sum())


try:
    cursor, cnx = db_cursor()
    query = """
            SELECT COUNT(*) AS num_games
            FROM game_data
            WHERE date >= NOW() - INTERVAL 2 HOUR
            """
    cursor.execute(query)
    num_games  = cursor.fetchall()

    ## extract just the digit from the tuple inside the list num_games
    num_games = num_games[0][0]
    print('number of games played in the last 2 hours: ', num_games)
except Exception as e:
    print("Error getting number of games:", e)

games_text = "Number of games played in the last 2 hours: " + str(num_games)

#count of today table where column wins > 0 divided by count of today table

cursor, cnx = db_cursor()
query = """
SELECT COUNT(*) / (SELECT COUNT(*) FROM today) * 100 AS percentage
FROM today
WHERE wins > 0;
"""
cursor.execute(query)
win_percentage = cursor.fetchall()


pct = win_percentage[0][0]
print('percent_complete: ', pct, '%')





# Set the time range to the last 2 hours
start_time = datetime.now() - timedelta(hours=2)
end_time = datetime.now()

cursor, cnx = db_cursor()
query = f"""
SELECT COUNT(*) AS total_games, SUM(result=1) AS wins
FROM game_data
WHERE date BETWEEN '{start_time}' AND '{end_time}'
"""

cursor.execute(query)
result = cursor.fetchone()

total_games = result[0]
total_wins = result[1]
win_percentage = total_wins / total_games if total_games > 0 else 0
## rount win_percentage to 2 decimal places
win_percentage = round(win_percentage, 2)


pct_text = "Percent of accounts leveled: " + str(pct) + '% with a ' + str(win_percentage) + '% win rate'
cursor, cnx = db_cursor()
query = """SELECT ai.uid
FROM account_info ai
LEFT JOIN (
    SELECT uid
    FROM game_data
    WHERE result = 1 AND date >= NOW() - INTERVAL 24 HOUR
) AS subquery
ON ai.uid = subquery.uid
WHERE subquery.uid IS NULL;"""

cursor.execute(query)
accounts = cursor.fetchall()

accounts = [x[0] for x in accounts]

account_attention_text = "Accounts that need attention: " + str(accounts)


## low win accounts
query  = """SELECT ai.uname
FROM game_data gd
INNER JOIN account_info ai ON gd.uid = ai.uid
WHERE gd.date >= DATE_SUB(NOW(), INTERVAL 2 DAY)
GROUP BY gd.uid, ai.uname
HAVING AVG(gd.result) < 0.2;"""

cursor.execute(query)
low_win = cursor.fetchall()


low_win_text = "Accounts with low win rate: " + str(low_win)
app = dash.Dash(__name__)
server = app.server


import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Calculate start and end times for the last 2 hours
end_time = datetime.now()
start_time = end_time - timedelta(hours=2)

# Create a dictionary that maps each deck name to a specific color
deck_colors = {
    'deck_red.PNG': '#E41A1C',
    'deck_green.PNG': '#4DAF4A',
    'deck_black.PNG': '#984EA3',
    'deck_blue.PNG': '#377EB8',
    'deck_gray.PNG': '#AAAAAA',
    'deck_white.PNG': '#FF7F00',
}

# Query the database to get win rates for the last 2 hours
cursor, cnx = db_cursor()
query = f"""
    SELECT SUBSTRING_INDEX(deck, '_', -1) AS color,
           COUNT(*) AS total_games,
           SUM(result=1) AS total_wins,
           SUM(result=1)/COUNT(*) AS win_rate
    FROM game_data
    WHERE date BETWEEN '{start_time}' AND '{end_time}'
    GROUP BY color;
"""
cursor.execute(query)
data = cursor.fetchall()

# Create a list of the deck names and win rates from your data
decks, win_rates = zip(*[(deck_color, win_rate) for (color, total_games, total_wins, win_rate) in data for deck_color in deck_colors if color in deck_color])

# Sort the deck names and win rates in descending order by win rate
decks, win_rates = zip(*sorted(zip(decks, win_rates), key=lambda x: x[1], reverse=True))

# Create a bar chart with custom colors based on the deck name
fig, ax = plt.subplots()
ax.bar(decks, win_rates, width=0.7, color=[deck_colors.get(deck, '#CCCCCC') for deck in decks])

# Add grid lines
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Set the title and axis labels
ax.set_title('Win Rates by Deck (Last 2 Hours)', fontsize=16)
ax.set_xlabel('Deck', fontsize=14)
ax.set_ylabel('Win Rate (%)', fontsize=14)

# Rotate the x-axis labels if needed
plt.xticks(rotation=45)

# Increase font size of tick labels
ax.tick_params(axis='both', labelsize=12)

# Set background color
ax.set_facecolor('#F0F0F0')

## last accounts table

cursor, cnx = db_cursor()
query = """SELECT
  gd1.uid,
  ai.uname,
  COUNT(*) AS num_games,
  SUM(gd1.result) / COUNT(*) AS win_rate,
  TIMEDIFF(MAX(gd1.date), MIN(gd1.date)) AS time_span
FROM
  game_data gd1
  JOIN (
    SELECT uid, MAX(date) AS max_date
    FROM game_data
    GROUP BY uid
    ORDER BY max_date DESC
    LIMIT 5
  ) gd2 ON gd1.uid = gd2.uid
  JOIN account_info ai ON gd1.uid = ai.uid
WHERE
  gd1.date >= DATE_SUB(gd2.max_date, INTERVAL 2 HOUR)
GROUP BY
  gd1.uid;
"""

cursor.execute(query)

last_used = cursor.fetchall()
# convert the result to a Pandas DataFrame
last_used_df = pd.DataFrame(last_used, columns=['uid', 'uname', 'num_games', 'win_rate', 'time_span'])
last_used_df['time_span'] = pd.to_timedelta(last_used_df['time_span']).astype(str).apply(lambda x: x.strip('0 days ')).apply(lambda x: x[:-3] if x.endswith(':00') else x)


print(last_used_df)
# create the Dash table
app.layout = html.Div([
    html.H1("Player Account Values"),
    html.P(games_text),
    html.P(pct_text),
    html.P(account_attention_text),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in last_used_df.columns],
        data=last_used_df.to_dict('records'),
    ),
    html.P(low_win_text),
    html.P(total_value_txt),
    dcc.Graph(id="graph"),
    dcc.Interval(
    id='interval-component',
    interval=1*1000,  # in milliseconds
    n_intervals=1  # only update on page load
)
    
])

@app.callback(
    Output("graph", "figure"),
    [Input("interval-component", "n_intervals")]
)

def update_graph(n):
    fig = px.bar(df, x='uid', y='total_dollars')
    fig.update_layout(
        title='Player Account Values',
        xaxis_title='User ID',
        yaxis_title='Total Dollars'
    )
    fig.update_traces(
        marker_color='rgb(158,202,225)',
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5, 
        opacity=0.6
    )
    fig.update_xaxes(tickangle=90, tickfont=dict(size=8))

    return fig
    
'''import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd

@app.callback(
    Output("graph2", "figure"),
    [Input("interval-component", "n_intervals")]
)
def update_graph2(n):
    # Calculate start and end times for the last 2 hours
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=2)

    # Query the database to get win rates for the last 2 hours
    cursor, cnx = Agent.db_cursor()
    query = f"""
        SELECT SUBSTRING_INDEX(deck, '_', -1) AS color,
               COUNT(*) AS total_games,
               SUM(result=1) AS total_wins,
               SUM(result=1)/COUNT(*) AS win_rate
        FROM game_data
        WHERE date BETWEEN '{start_time}' AND '{end_time}'
        GROUP BY color;
    """
    cursor.execute(query)
    data = cursor.fetchall()

    # Create a pandas dataframe from the data
    df = pd.DataFrame(data, columns=['color', 'total_games', 'total_wins', 'win_rate'])
    df['deck'] = df['color'].str.extract(r'(\w+)\.')

    # Create a bar chart using Plotly Express
    fig = px.bar(df, x='deck', y='win_rate', color='deck', color_discrete_map={'red': '#E41A1C', 'green': '#4DAF4A', 'black': '#984EA3', 'blue': '#377EB8', 'gray': '#AAAAAA', 'white': '#FF7F00'})
    fig.update_layout(
        title='Win Rates and Total Games by Deck (Last 2 Hours)',
        xaxis_title='Deck',
        yaxis_title='Win Rate (%)'
    )
    fig.update_traces(
        text=df['total_games'],
        textposition='outside'
    )

    return fig
'''
if __name__ == '__main__':
    app.run_server(debug=True)
