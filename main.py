'''import essential packages'''
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.express as px
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.metrics import roc_curve, precision_recall_curve
import plotly.graph_objs as go
import plotly.offline as pyo
import random
import xgboost
from sklearn.manifold import TSNE

"""Check the directory"""
print(os.getcwd())

'''read the csv file but drop the lines where there is bad data(too many columns'''
df = pd.read_csv('depression.csv', on_bad_lines= 'skip', sep=r'	')
codebook = open('codebook.txt', 'r').read()
'''print the formatted csv file and the accompanying codebook'''
print(df)
print(codebook)
'''show first five rows'''
print(df.head())
'''needed to check all rows'''
pd.set_option('display.max_columns', None)
print(df)
'''null check'''
print(df.isnull().sum())
'''basic description of full data'''
print(df.describe())

#data cleaning
'''removing rows for people who took too long to answer'''
df = df[ df['testelapse'] <= df['testelapse'].quantile(0.975) ]
df = df[ df['testelapse'] >= df['testelapse'].quantile(0.025) ]
df = df[ df['surveyelapse'] <= df['surveyelapse'].quantile(0.975) ]
df = df[ df['surveyelapse'] >= df['surveyelapse'].quantile(0.025) ]
'''replacing extreme ages with medians'''
median = df.loc[df['age'] <=80, 'age'].median()
df.loc[df.age > 80, 'age'] = np.nan
df.fillna(median,inplace=True)
'''show the new dataset'''
print(df.head(2))
#Data manipulation
'''creating bin groups for ages'''
age_group = [
    'below 20',
    '20 to 24',
    '25 to 29',
    '30 to 34',
    '35 to 39',
    '40 to 49',
    '50 to 59',
    'above 60',
]

def label_age(row):
    if row['age'] < 20:
        return age_group[0]
    elif row['age'] < 25:
        return age_group[1]
    elif row['age'] < 30:
        return age_group[2]
    elif row['age'] < 35:
        return age_group[3]
    elif row['age'] < 40:
        return age_group[4]
    elif row['age'] < 50:
        return age_group[5]
    elif row['age'] < 60:
        return age_group[6]
    elif row['age'] > 60:
        return age_group[7]
'''apply a lambda function'''
df['age_group'] = df.apply(lambda row: label_age(row), axis=1)
print(df.head(2))
'''display the shape in formatted output'''
fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Indicator(mode = "number", value = df.shape[0], number={'font':{'color': '#E58F65','size':100}}, title = {"text": "🧑‍🤝‍🧑 Participants<br><span style='font-size:0.8em;color:gray'>Took the test</span>"}, domain = {'x': [0, 0.5], 'y': [0.6, 1]}))
fig.add_trace(go.Indicator(mode = "number", value = df.shape[1], number={'font':{'color': '#E58F65','size':100}}, title = {"text": "❓ Questions<br><span style='font-size:0.8em;color:gray'>On the test</span>"}, domain = {'x': [0.5, 1], 'y': [0, 0.4]}))
fig.show()

#Adding target DASS columns
'''Severites of Depression, Anxiety, Stress are categorized to:
#0. Normal
#1. Mild
#2. Moderate
#3. Severe
#4. Extremely severe'''

DASS_keys = {'Depression': [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42],
             'Anxiety': [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41],
             'Stress': [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]}

DASS_bins = {'Depression': [(0, 10), (10, 14), (14, 21), (21, 28)],
             'Anxiety': [(0, 8), (8, 10), (10, 15), (15, 20)],
             'Stress': [(0, 15), (15, 19), (19, 26), (26, 34)]}

for name, keys in DASS_keys.items():
    # Subtract one to match definition of DASS score in source
    df[name] = (df.filter(regex='Q(%s)A' % '|'.join(map(str, keys))) - 1).sum(axis=1)

    bins = DASS_bins[name]
    bins.append((DASS_bins[name][-1][-1], df[name].max() + 1))
    bins = pd.IntervalIndex.from_tuples(bins, closed='left')
    df[name + '_cat'] = np.arange(len(bins))[pd.cut(df[name], bins=bins).cat.codes]

dass = df[DASS_keys.keys()]
dass_cat = df[[k + '_cat' for k in DASS_keys.keys()]]

df[[k + '_cat' for k in DASS_keys.keys()] + list(DASS_keys.keys())].head()

print(DASS_keys)

# Add personality types to data
personality_types = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'EmotionalStability', 'Openness']

# Invert some entries
tipi = df.filter(regex='TIPI\d+').copy()
tipi_inv = tipi.filter(regex='TIPI(2|4|6|8|10)').apply(lambda d: 7 - d)
tipi[tipi.columns.intersection(tipi_inv.columns)] = tipi_inv

# Calculate scores
for idx, pt in enumerate( personality_types ):
    df[pt] = tipi[['TIPI{}'.format(idx + 1), 'TIPI{}'.format(6 + idx)]].mean(axis=1)

personalities = df[personality_types]
character = pd.concat([dass, personalities], axis=1)
character.head()

print(df.head(2))

##Data Visualisation
def make_pie_chart(data, series, title):
    temp_series = data[ series ].value_counts()
        # what we want to show in our charts

    labels = ( np.array(temp_series.index) )
    sizes = ( np.array( ( temp_series / temp_series.sum() ) *100) )

    trace = go.Pie(labels=labels,
                   values=sizes)
    layout= go.Layout(
        title= title,
        title_font_size= 24,
        #title_font_color= 'red',
        #title_x= 0.45,
    )
    fig = go.Figure(data= [trace],
                    layout=layout)

    fig.show()
'''Age breakdown'''
make_pie_chart(df, 'age_group', 'Distribution by Age')

'''Gender breakdown'''
temp = df.copy()
temp['gender'].replace({
    1: "Male",
    2: "Female",
    3: "Non-binary",
    0: "Unanswered",
},
    inplace=True)

make_pie_chart(temp, 'gender', 'Distribution by Gender')

'''Education breakdown'''
temp = df.copy()
temp['education'].replace({
    1: "Less than high school",
    2: "High school",
    3: "University degree",
    4: 'Graduate degree',
    0: "Unanswered",
},
    inplace=True)

make_pie_chart(temp, 'education', 'Distribution by Education')

'''Relegion breakdown'''
temp = df.copy()
temp['religion'].replace({
    1: "Agnostic",
    2: "Atheist",
    3: "Buddhist",
    4: 'Christian',
    5: 'Christian',
    6: 'Christian',
    7: 'Christian',
    8: 'Hindu',
    9: 'Jewish',
    10: 'Muslim',
    11: 'Sikh',
    12: 'Other',
    0: 'Unanswered',
},
    inplace=True)

make_pie_chart(temp, 'religion', 'Distribution by Religion')

'''Race breakdown'''
temp = df.copy()
temp['race'].replace({
    10: "Asian",
    20: "Arab",
    30: "Black",
    40: 'Indigenous',
    50: 'Native American',
    60: 'White',
    70: 'Other',
    0: 'Unanswered',
},
    inplace=True)

make_pie_chart(temp, 'race', 'Distribution by Race')

'''Marriage status breakdown'''
temp = df.copy()
temp['married'].replace({
    1: "Never",
    2: "Currently married",
    3: 'Previously married',
    0: 'Unanswered',
},
    inplace=True)

make_pie_chart(temp, 'married', 'Distribution by Marriage Status')

#Correlation matrices

def plot_correlation(df, cmap='RdBu_r'):
    size = len(df.columns)
    fig, ax = plt.subplots(figsize=(1.3 * size, 1. * size))
    corr = df.corr()

    im = ax.matshow(corr, cmap=cmap)
    for (i, j), z in np.ndenumerate(corr):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar(im)
    ax.tick_params(labelsize=14)
    plt.show()

character = pd.concat([dass, personalities], axis=1)
plot_correlation(character, cmap='viridis')

# Drop test data
df_select = df[df.columns.drop(list(df.filter(regex='(Q\d+[AIE])|(TIPI\d+)|(VCL\d+)')))]
df_select = df_select.drop(columns=['source', 'introelapse', 'testelapse', 'surveyelapse'])

# Categorize
object_cols = df_select.columns[df_select.dtypes == object]
df_select[object_cols] = df_select[object_cols].astype('category').apply(lambda x: x.cat.codes)

plot_correlation(df_select, cmap='viridis')


def make_corr_map(data, title, zmin=-1, zmax=1, height=600, width=800):
    """
    data: Your dataframe.
    title: Title for the correlation matrix.
    zmin: Minimum number for color scale. (-1 to 1). Default = -1.
    zmax: Maximum number for color scale. (-1 to 1). Default = 1.
    height: Default = 600
    width: Default = 800
    """

    data = data.corr()
    mask = np.triu(np.ones_like(data, dtype=bool))
    rLT = data.mask(mask)

    heat = go.Heatmap(
        z=rLT,
        x=rLT.columns.values,
        y=rLT.columns.values,
        zmin=zmin,
        # Sets the lower bound of the color domain
        zmax=zmax,
        # Sets the upper bound of color domain
        xgap=1,  # Sets the horizontal gap (in pixels) between bricks
        ygap=1,
        colorscale='RdBu'
    )

    title = title

    layout = go.Layout(
        title_text=title,
        title_x=0.5,
        width=width,
        height=height,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed'
    )

    fig = go.Figure(data=[heat], layout=layout)
    return fig.show()

df['score'] = df.iloc[:, 0:42].sum(axis= 1)
df['mean_score'] = df['score'] / 42

make_corr_map(df.iloc[:, 42:],
              'Correlation heatmap for DASS score and variables',
              zmax=0.8, zmin=-0.8,
              height= 900)

#Visualisation for significant variables
def make_box_plot_by_age(df, color):
    fig = px.box(
        df,
        x= 'age_group',
        y= 'mean_score',
        color= color,
        category_orders= {
            "age_group": ['below 20',
                          '20 to 24',
                          '25 to 29',
                          '30 to 34',
                          '35 to 39',
                          '40 to 49',
                          '50 to 59',
                          'above 60'],
            color: ordering,
            #temp[ color ].unique(),
        },
    )

    fig.update_layout(
        title= f'DASS Score, by age, by { color }',
        title_x = 0.5,
        title_font_size= 20,
        height= 600,
        width= 900,
        showlegend= True
    )

    fig.show()
'''Gender by age'''
temp = df.copy()
temp['gender'].replace({
    1: "Male",
    2: "Female",
    3: "Non-binary",
    0: "Unanswered",
},
    inplace=True)

ordering = [
    'Male',
    'Female',
    'Non-binary',
    'Unanswered',
]

make_box_plot_by_age(temp, 'gender')

'''Education by age'''
temp = df.copy()
temp['education'].replace({
    1: "Less than high school",
    2: "High school",
    3: "University degree",
    4: 'Graduate degree',
    0: "Unanswered",
},
    inplace=True)

ordering = [
    "Less than high school",
    "High school",
    "University degree",
    'Graduate degree',
    "Unanswered",
]

make_box_plot_by_age(temp, 'education')

'''Orientation by age'''
temp = df.copy()
temp['orientation'].replace({
    1: "Heterosexual",
    2: "Bisexual",
    3: "Homosexual",
    4: 'Asexual',
    5: 'Other',
    0: 'Unanswered',
},
    inplace=True)

temp2 = df.copy()
temp2['orientation'].replace({
    1: "Heterosexual",
    2: "Non-heterosexual",
    3: "Non-heterosexual",
    4: 'Non-heterosexual',
    5: 'Non-heterosexual',
    0: 'Unanswered',
},
    inplace=True)

ordering = [
    "Heterosexual",
    #     "Bisexual",
    #     "Homosexual",
    #     "Asexual",
    #     "Other",
    "Non-heterosexual",
    "Unanswered",
]

make_box_plot_by_age(temp2, 'orientation')

def get_from_last_space(line, idx):
    to_return = ''
    while idx > 0 and line[idx] != ' ':
        idx -= 1
        to_return += line[idx]
    return to_return[::-1]

def get_to_next_eq(line, idx):
    to_return = ''
    while idx < len(line) - 1 and line[idx + 1] != '=':
        to_return += line[idx]
        idx += 1
    return to_return.split('=')[0]

def parse_codebook(codebook_text):
    q_dict = {}
    for line in codebook_text.split('\n'):
        if '.' in line and len(line.split('=')) > 2:
            question = line.split('.')[0]
            answers = ''.join(line.split('.')[1:])
            for idx in range(len(answers)):
                if answers[idx] == '=':
                    key = get_from_last_space(answers, idx).lstrip()
                    if len(key) != 0 and '(' == key[0]: key = key[1:]
                    val = get_to_next_eq(answers, idx + 1)
                    if val.endswith(', '): val = val[:-2]
                    if question not in q_dict: q_dict[question] = {}
                    if key not in q_dict[question]: q_dict[question][key] = val
    return q_dict

def map_questions(codebook_text):
    q_dict = {}
    for line in codebook_text.split('\n'):
        q_text = ''
        if '\t' in line:
            question = line.split('\t')[0]
            q_text = line.split('\t')[1]
        elif 'TIPI' in line:
            question = line.split(' ')[0]
            q_text = ' '.join(line.split(' ')[1:])
        elif 'VCL' in line:
            question = line.split(' ')[0]
            q_text = ' '.join(line.split(' ')[1:])
        if len(q_text) > 2:
            q_dict[question] = q_text
    return q_dict

# print(codebook)
q_dict = parse_codebook(codebook)
q_mapping = map_questions(codebook)
# print(q_mapping)

from IPython.display import display, HTML
display(HTML('<pre style="color:DimGrey !important">' +
             '\n'.join(codebook.split('\n')[:10]) + '<br><br>...<br><br><br><b>see original file: '
                                                    'codebook.txt..</b><br>' + '</pre>'))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec

sns.set_style("whitegrid")


def plot_barchart(df, col, order=None, x_label=None, y1_label=None, y2_label=None, title=None, figsize=(20, 4),
                  plot_percentage=False, display_table=False, barh=False, colors=None, padding=None):
    try:
        plt.close()
    except:
        pass
    plt.style.use('seaborn-poster')
    g = df[col].value_counts(ascending=True).to_dict()
    ticks = list(g.keys())
    data = list(g.values())
    ind = np.arange(len(data))
    fig = plt.figure(tight_layout=True)
    ax = plt.subplot(111)
    # ax.set_yticks(ind * 0.5)
    ax.set_yticks(ind)
    bar = ax.barh(ind, data, 0.8, color=colors)
    # ax.set_yticks(ind * 0.5)
    for i, rect in enumerate(bar.patches):
        h = rect.get_height()
        w = rect.get_width()
        y = rect.get_y()
        ax.annotate(f"{w}", (w, y + h / 2), color='k', size=12, ha='left', va='center')
    r = ax.set_yticklabels(ticks, ha='left')
    fig.set_size_inches(figsize[0], figsize[1], forward=True)
    plt.draw()
    yax = ax.get_yaxis()
    if padding is None:
        pad = max(T.label.get_window_extent().width for T in yax.majorTicks)
    else:
        pad = padding
    yax.set_tick_params(pad=pad)
    ax.grid(False)
    plt.title('\n\n' + col)
    plt.draw()
    plt.show()

categs = [i for i in df.columns if len(df[i].unique()) < 10 or i in df.select_dtypes(exclude='number').columns]
max_of_maxes = -1
plt_df = df.copy()
for col in categs:
    if col in q_dict: plt_df[col] = plt_df[col].astype('str').map(q_dict[col])
    max_size = max([len(i) for i in plt_df[col].astype('str').tolist()])
    max_of_maxes = max(max_of_maxes, max_size)

plt_df = df.copy()
max_pad_of_maxes = -1

import random

train_df = df

from sklearn.model_selection import train_test_split
from category_encoders.target_encoder import TargetEncoder

FOLDS = 5
SMOOTH = 0.001
SPLIT = 'interleaved'

encoder = TargetEncoder()

targets = ['Depression', 'Depression' + '_cat', 'Anxiety', 'Anxiety' + '_cat', 'Stress', 'Stress' + '_cat']
X = train_df.drop(targets, axis=1)
y = train_df['Depression']
y = (((y - y.mean()) / y.std()) > 0.0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)

for col in [i for i in df.columns if i not in categs and i not in targets]:
    X_train[col] = encoder.fit_transform(X_train[col], y_train)
    X_test[col] = encoder.transform(X_test[col])

from sklearn.metrics import roc_auc_score


def training(model, X_train, y_train, X_test, y_test, model_name):
    t1 = time.time()

    model.fit(X_train.select_dtypes(include='number'), y_train)
    predicts = model.predict(X_test[X_train.select_dtypes(include='number').columns])
    roc = roc_auc_score(y_test, predicts)

    t2 = time.time()
    training_time = t2 - t1

    print("\t\t\t--- Model:", model_name, "---")
    print("ROC: ", roc, "\t\t\t", "Training time:", training_time, "\n")

from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier

ridge = Ridge(fit_intercept = True)

svr = SVR(kernel='rbf', gamma='scale', C=1, epsilon=0.3)

knc =  KNeighborsClassifier(n_neighbors=3)

m = [ridge,svr,knc]
mn = ["Ridge","SVR","K Neighbors Classifier"]

for i in range(0,len(m)):
    training(model=m[i], X_train=X_train, y_train=y_train, X_test=X_test,y_test=y_test, model_name=mn[i])

features = list(X_train.select_dtypes(include = 'number').columns)

dtrain = xgboost.DMatrix(X_train[features], y_train)
dval   = xgboost.DMatrix(X_test[features], y_test)

params1 = {'max_depth' : 7,'max_leaves' : 15,'objective' : 'binary:logistic','grow_policy' : 'lossguide', 'eta' : 0.7, 'eval_metric':'auc'}
evallist = [(dval, 'validation'), (dtrain, 'train')]
num_round=50

model1 = xgboost.train(params1, dtrain, num_round, evallist)

predicts = model1.predict(xgboost.DMatrix(X_test[features]))
roc = roc_auc_score(y_test, predicts)
roc

from sklearn.metrics import accuracy_score

cu_score = accuracy_score(y_test, predicts > 0.5)
print(cu_score)

# xgboost for feature importance on a classification problem
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=39775, n_features=180, n_informative=30, n_redundant=150, random_state=1)
# define the model
model = XGBClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

# predict labels
y_pred = model.predict(X_test)

# compare with true labels
cfm = confusion_matrix(y_test, y_pred, normalize='true')

# plot size
fig, ax = plt.subplots(figsize=(10,10))
# print confusion matrix
s = sb.heatmap(cfm,
                annot=True,
                cmap=['#ff0000', '#09AA00'],
                center=0.8,
                fmt='.1%',
                linewidths=.5,
                cbar_kws={'format': FuncFormatter(lambda x, pos: '{:.0%}'.format(x))}, #'label': 'Percentage'
                linecolor='white',
                ax=ax)
# set labels
s.set(xlabel='Predict', ylabel='True')
s.set(title='Confusion Matrix')

ticks_label = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']

s.set(xticklabels=ticks_label, yticklabels=ticks_label)

# predict labels
y_pred = model.predict(X_test)

# compare with true labels
cfm = confusion_matrix(y_test, y_pred, normalize='true')

# plot size
fig, ax = plt.subplots(figsize=(10,10))
# print confusion matrix
s = sb.heatmap(cfm,
                annot=True,
                cmap=['#ff0000', '#09AA00'],
                center=0.8,
                fmt='.1%',
                linewidths=.5,
                cbar_kws={'format': FuncFormatter(lambda x, pos: '{:.0%}'.format(x))}, #'label': 'Percentage'
                linecolor='white',
                ax=ax)
# set labels