    #P1

    import plotly.express as px
    from plotly import graph_objects as go
    from plotly import express as px
    # Creating the Figure instance
    fig = px.line(x=[1, 2], y=[3, 4])
    
    # printing the figure instance
    print(fig)
    Figure({
     'data': [{'hovertemplate': 'x=%{x}
y=%{y}',
     'legendgroup': '',
     'line': {'color': '#636efa', 'dash': 'solid'},
     'mode': 'lines',
     'name': '',
     'orientation': 'v',
     'showlegend': False,
     'type': 'scatter',
     'x': array([1, 2], dtype=int64),
     'xaxis': 'x',
     'y': array([3, 4], dtype=int64),
     'yaxis': 'y'}],
     'layout': {'legend': {'tracegroupgap': 0},
     'margin': {'t': 60},
     'template': '...',
     'xaxis': {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text':
    'x'}},
     'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text':
    'y'}}}
    })
    
    
    #LineChat
    
    import plotly.express as px
    
    # using the iris dataset
    df = px.data.iris()
    # plotting the line chart
    fig = px.line(df, y="sepal_width")
    #plot(data, config={'displayModeBar'=True})
    # showing the plot
    fig.show()
    
    
    EX:
    import plotly.express as px
    # using the iris dataset
    df = px.data.iris()
    # plotting the line chart
    fig = px.line(df, y="sepal_width", line_group='species')
    # showing the plot
    fig.show()
    
    
    #BarChat
    import plotly.express as px
    # Loading the data
    df = px.data.tips()
    # Creating the bar chart
    fig = px.bar(df, x='day', y="total_bill")
    fig.show()
    In [8]:
    df.head()
    
    EX:
    import plotly.express as px
    # Loading the data
    df = px.data.tips()
    # Creating the bar chart
    fig = px.bar(df, x='day', y="total_bill", color='sex', facet_row='time',
    facet_col='sex')
    fig.show()
    
    
    
    #Scatter Plot
    
    import plotly.express as px
    # using the dataset
    df = px.data.tips()
    # plotting the scatter chart
    fig = px.scatter(df, x='total_bill', y="tip")
    # showing the plot
    fig.show()
    
    Ex:
    import plotly.express as px
    # using the dataset
    df = px.data.tips()
    # plotting the scatter chart
    fig = px.scatter(df, x='total_bill', y="tip", color='time',
     symbol='sex', size='size', facet_row='day',
     facet_col='time')
    # showing the plot
    fig.show()
    
    
    
    #Histogram
    
    import plotly.express as px
    # using the dataset
    df = px.data.tips()
    # plotting the histogram
    fig = px.histogram(df, x="total_bill")
    # showing the plot
    fig.show()
    
    
    
    EX:
    
    import plotly.express as px
    # using the dataset
    df = px.data.tips()
    # plotting the histogram
    fig = px.histogram(df, x="total_bill", color='sex',
     nbins=50, histnorm='percent',
     barmode='group')
    # showing the plot
    fig.show()
    
    
    #PieChat
    
    import plotly.express as px
    # Loading the iris dataset
    df = px.data.tips()
    fig = px.pie(df, values="total_bill", names="day")
    fig.show()
    
    import plotly.express as px
    # Loading the iris dataset
    df = px.data.tips()
    fig = px.pie(df, values="total_bill", names="day",
     color_discrete_sequence=px.colors.sequential.RdBu,
     opacity=0.8, hole=0.5)
    fig.show()
    
    
    #BoxPlot
    import plotly.express as px
    # using the dataset
    df = px.data.tips()
    # plotting the boxplot
    fig = px.box(df, x="day", y="tip")
    # showing the plot
    fig.show()
    
    EX:
    import plotly.express as px
    # using the dataset
    df = px.data.tips()
    # plotting the boxplot
    fig = px.box(df, x="day", y="tip", color='sex',
     facet_row='time', boxmode='group',
     notched=True)
    # showing the plot
    fig.show()
    
    
    
    #violin Plot
    
    import plotly.express as px
    # using the dataset
    df = px.data.tips()
    # plotting the violin plot
    fig = px.violin(df, x="day", y="tip")
    # showing the plot
    fig.show()
    
    EX:
    import plotly.express as px
    # using the dataset
    df = px.data.tips()
    # plotting the violin plot
    fig = px.violin(df, x="day", y="tip", color='sex',
     facet_row='time', box=True)
    # showing the plot
    fig.show()
    
    
    #3D Scatter Plot
    
    
    import plotly.express as px
    # data to be plotted
    df = px.data.tips()
    # plotting the figure
    fig = px.scatter_3d(df, x="total_bill", y="sex", z="tip")
    fig.show()
    
    EX:
    import plotly.express as px
    # data to be plotted
    df = px.data.tips()
    # plotting the figure
    fig = px.scatter_3d(df, x="total_bill", y="sex", z="tip", color='day',
     size='total_bill', symbol='time')
    fig.show()
    
    
    #P2 and
    #P3
    #PieChat
    # Data Frame plotting
    from pandas import DataFrame
    import matplotlib.pyplot as plt
    Data = {'Tasks': [300,500,700],
     'Task Type' : ['Tasks Pending','Tasks Ongoing','Tasks Completed']
     }
    df = DataFrame(Data)
    df.set_index('Task Type', inplace=True)
    # autopct has extra % at the end as escape, as % is interpreted as
    formatting string begin by default.
    # Only pie chart needs labels to be data frame index
    df.plot.pie(y='Tasks', figsize=(10,10),autopct='%1.1f%%', startangle=90)
    
    import numpy as np
    import matplotlib.pyplot as plt
    # if using a Jupyter notebook, include:
    %matplotlib inline
    # Pie chart, where the slices will be ordered and plotted counterclockwise:
    labels = ['Civil', 'Electrical', 'Mechanical', 'Chemical']
    sizes = [15, 50, 45, 10]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.axis('equal') # Equal aspect ratio ensures the pie chart is circular.
    ax.set_title('Engineering Diciplines')
    
    #Subplots
    plt.figure(figsize=(20,10))
    plt.subplot(2,2,1)
    plt.bar(range(1,6), np.random.randint(1,20,5))
    plt.title("2,2,1")
    plt.subplot(2,2,2)
    plt.bar(range(1,6), np.random.randint(1,20,5))
    plt.title("2,2,2")
    plt.subplot(2,2,3)
    # s is the size of dot
    plt.scatter(range(1,6), np.random.randint(1,20,5), s=100, color="r")
    plt.title("2,2,3")
    plt.subplot(2,2,4)
    plt.plot(range(1,6), np.random.randint(1,20,5), marker='o', color='g',
    linestyle='--')
    plt.title("2,2,4")
    Out[43]:
    Text(0.5, 1.0, '2,2,4')
    plt.bar(range(1,6), np.random.randint(1,20,5), width=0.5)
    plt.scatter(range(1,6), np.random.randint(1,20,5), s=200, color="r")
    plt.plot(range(1,6), np.random.randint(1,20,5), marker='o', color='g',
    linestyle='--')
    
    #Seaborn
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    In [19]:
    os.chdir('D:/AnitaRJ/DATA SCIENCE/MScI_DataSci_Practicals/Practical7')
    cars_data=pd.read_csv('Toyota.csv',index_col=0,na_values=["??","????"])
    cars_data.size
    Out[19]:
    14360
    In [16]:
    cars_data.dropna(axis=0,inplace=True)
    cars_data.size
    Out[16]:
    10960
    In [22]:
    cars_data=pd.read_csv('Toyota.csv')
    cars_data.head()
    
    cars_data=pd.read_csv('Toyota.csv',index_col=0)
    cars_data.head()
    
    
    
    #Scatter plot
    
    sns.set(style="darkgrid")
    sns.regplot(x=cars_data['Age'],y=cars_data['Price'])
    #It estimates and plots a regression model relating the x and y variables
    
    
    #Scatter plot of Price vs Age without the regression fit line
    sns.regplot(x=cars_data['Age'],y=cars_data['Price'],fit_reg=False)
    #Scatter plot of Price vs Age by customizing the appearance of markers
    sns.regplot(x=cars_data['Age'], y=cars_data['Price'], marker="*",
    fit_reg=False)
    
    sns.lmplot(x='Age', y='Price', data=cars_data, fit_reg=False,
    hue='FuelType', legend=True, palette="Set1")
    
    
    #Box and whiskers plot
    
    
    sns.boxplot(x=cars_data['FuelType'],y=cars_data["Price"])
    
    #Heatmap¶
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    In [17]:
    data=np.random.randint(1,100,size=(10,10))
    print("The data to be plotted: \n")
    print(data)
    The data to be plotted:
    [[ 7 46 51 1 51 45 57 23 57 82]
    [85 83 92 48 26 62 8 76 28 94]
    [63 4 14 54 80 28 48 74 67 4]
    [28 61 19 25 29 38 99 54 65 82]
    [31 92 58 93 43 51 94 37 32 50]
    [59 36 1 23 64 16 32 42 56 2]
    [83 12 98 91 55 22 63 79 53 21]
    [32 61 32 83 38 87 44 10 35 60]
    [76 55 99 58 13 21 43 93 56 31]
    [90 60 13 72 41 31 63 76 44 87]]
    In [18]:
    #Plotting Heatmap
    hm=sns.heatmap(data=data)
    
    plt.show()
    
    hm = sns.heatmap(data=data,
     vmin='30',
     vmax='70')
    plt.show()
    
