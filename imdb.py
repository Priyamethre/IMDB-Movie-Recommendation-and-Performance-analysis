# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from wordcloud import WordCloud
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


#pip install pandas numpy scikit-learn streamlit surprise


# Load dataset
df = pd.read_csv(r"C:\Users\priya\OneDrive\Desktop\PRIYA\PROJECT\IMDB MOVIES\imdb_movies.csv\imdb_movies.csv")

df.info()

df.head()


# EDA

df.columns

# Summary statistics
df.describe()

df.shape

#EDA using Autoviz

import sweetviz as sv
sweet_report = sv.analyze(df)

#Saving results to HTML file
sweet_report.show_html('sweet_report.html')



df['status'].value_counts()
df['genre'].value_counts()
df['orig_lang'].value_counts()


# Check for missing values and duplicates
print("Missing values:")
print(df.isnull().sum())

# THERE ARE MISSING VALUES IN  AND CREW(54)

print("\nDuplicates:", df.duplicated().sum())



#renaming the column names
df = df.rename(columns={"names":"Movie_Name",
                                               "date_x":"Release_Date",
                                               "score":"User_Rating",
                                               "genre":"Genre",
                                               "overview":"Overview",
                                               "crew":"Crew",
                                               "orig_title":"Original_Title",
                                               "status":"Release_Status",
                                               "orig_lang":"Original_Language",
                                               "budget_x":"Movie_Budget",
                                               "revenue":"Revenue",
                                               "country":"Country_Code"})


#list the column labels of the table
df.columns

# Converting the Column "Release_Date" into datetime format

#returns the unique values of the column "Release_Date" from movies_df
df["Release_Date"].unique()

df["Release_Date"] = pd.to_datetime(df["Release_Date"].str.strip(), format='mixed', dayfirst=False)
#checking the datatype
df["Release_Date"].dtype

df.info()


#extracting year from the column "Release_Date"
df["Year"] = df["Release_Date"].dt.year
df.head()


# Cleaning the Column "Genre"
#returns the unique values of the column "Genre" from movies_df
df["Genre"].unique()
#replacing "\xa0" to "" in the column "Genre"
df["Genre"] = df["Genre"].str.replace("\xa0","")
#returns the unique values of the column "Genre" from movies_df
df["Genre"].unique()


# Extracting Country Names from the Column "Country"

#returns the unique values of the column "Country" from movies_df
df["Country_Code"].unique()

#install the pycountry library
# !pip install pycountry
#import the required module
import pycountry


#defining a function get_country_name to convert country codes to country names
def get_country_name(country_code):
    try:
        return pycountry.countries.get(alpha_2 = country_code).name
    except:
        return None

#applying the function to my dataset movies_df
df["Country_Name"] = df["Country_Code"].apply(get_country_name)

#display the rows where "Country_Name" column is null
df.loc[df["Country_Name"].isna()]

country_code_mapping = {
    'SU': 'Soviet Union',
    'XC': 'Czech Republic'
}

def get_country_name(country_code):
    try:
        # Check in the custom mapping first
        if country_code in country_code_mapping:
            return country_code_mapping[country_code]
        # Otherwise, use pycountry
        return pycountry.countries.get(alpha_2=country_code).name
    except:
        return "Unknown"

#Applying the updated function to my dataset movies_df
df['Country_Name'] = df['Country_Code'].apply(get_country_name)

# Now all the null country name values are filled

# Inspect the results

#display the first 5 rows of the data
df.head()


# Removing Irrelevant Columns
#drop the column "Overview", "Original_Title" since it is irrelevant
df.drop(["Original_Title"], axis = 1, inplace = True)
#display the first 5 rows of the data
df.head()


print("Missing values:")
print(df.isnull().sum())


df.info()

# Cleaning the Column "Crew"

#display the column "Crew"
df["Crew"]

#counting missing values in the column "Crew"
df["Crew"].isnull().sum()

#fill in the missing values with "" in the column "Crew"
df["Crew"] = df["Crew"].fillna("")

# Function to extract main actor and star power
def extract_main_star_power(crew):
    actors = crew.split(', ')
    main_actor = actors[0] if len(actors) > 0 else ''
    star_power = ', '.join(actors[1:]) if len(actors) > 1 else ''
    return main_actor, star_power

# Apply function to 'Crew' column and create new columns
df[['Main_Actor', 'Star_Power']] = df.apply(lambda row: extract_main_star_power(row['Crew']),
                                                          axis=1, result_type="expand")
# Drop the original 'Crew' column if not needed
df.drop('Crew', axis=1, inplace=True)
df.head()

df.info()

print("Missing values:")
print(df.isnull().sum())

# now our data is cleaned and missing values are treated


# DATA VISUALIZATION


plt.figure(figsize=(18, 6))

# User Rating
plt.subplot(1, 3, 1)
df['User_Rating'].hist(bins=30, edgecolor='k')
plt.xlabel('User Rating')
plt.ylabel('Frequency')
plt.title('Distribution of User Ratings')

# Movie Budget
plt.subplot(1, 3, 2)
df['Movie_Budget'].hist(bins=30, edgecolor='k')
plt.xlabel('Movie Budget')
plt.ylabel('Frequency')
plt.title('Distribution of Movie Budgets')

# Revenue
plt.subplot(1, 3, 3)
df['Revenue'].hist(bins=30, edgecolor='k')
plt.xlabel('Revenue')
plt.ylabel('Frequency')
plt.title('Distribution of Revenues')

plt.tight_layout()
plt.show()

""" From the above histogram distribution, we can infer that:

1. User Ratings:-

   * Distribution: Most movies fall within a mid to high rating range. This suggests that either the majority of movies are relatively well-received, or that rating inflation could be at play.
   * Outliers: If there are any points far from the majority, they could indicate extremely well-loved or poorly received films.

2. Movie Budgets:-

   * Cluster at Lower Budgets: Many films are made on smaller budgets, indicating perhaps a trend towards more indie or low-cost productions.
   * Long Tail of High Budgets: The few films with extremely high budgets are likely blockbuster productions. These outliers could skew the average budget, so consider using the median as a better central measure.

3. Revenues:-

   * Skewed Distribution: Most movies earn modest revenues, with a few making significantly higher amounts. This highlights how a few big hits can dominate the box office.
   """
   
   
# Plotting Box Plot to Highlight Outliers


plt.figure(figsize=(18, 6))

# User Rating
plt.subplot(1, 3, 1)
df.boxplot(['User_Rating'], color="b")
plt.ylabel('User Rating')
plt.title('Box Plot of User Ratings')

# Movie Budget
plt.subplot(1, 3, 2)
df.boxplot(['Movie_Budget'], color="b")
plt.ylabel('Movie Budget')
plt.title('Box Plot of Movie Budgets')

# Revenue
plt.subplot(1, 3, 3)
df.boxplot(['Revenue'], color="b")
plt.ylabel('Revenue')
plt.title('Box Plot of Revenues')

plt.tight_layout()
plt.show()


""" From the above box plot, we observe that:

1. User Ratings:-

   * Clustered Mid to High Ratings: A concentration around mid to high ratings suggests the majority of movies are well-received by audiences. This is typical for IMDb, as people tend to rate movies they enjoyed watching.
   * Outliers: Very high or low ratings can point to extremely popular or unpopular movies, which are interesting case studies in audience preferences.

2. Movie Budgets:-

   * Heavily Clustered at Lower Budgets: Indicates a large number of movies are made with modest budgets. This could be due to the rise of indie films and lower-budget productions.
   * Long Tail with High Budgets: Few movies have exceptionally high budgets. These are likely blockbuster films, which are interesting for their economic impact on the industry.

3. Revenues:-

   * Majority of Films Earn Moderate Revenues: Most movies earn in the mid-range, showing the typical financial performance of many productions.
   * Significant Outliers with High Revenues: Blockbusters or highly successful films earning significantly more, which are key drivers of industry profits.


"""


#  Plotting KDE Plots for Smooth Distributions

plt.figure(figsize=(18, 6))

# User Rating
plt.subplot(1, 3, 1)
sns.kdeplot(df['User_Rating'])
plt.xlabel('User Rating')
plt.ylabel('Density')
plt.title('KDE Plot of User Ratings')

# Movie Budget
plt.subplot(1, 3, 2)
sns.kdeplot(df['Movie_Budget'])
plt.xlabel('Budget')
plt.ylabel('Density')
plt.title('KDE Plot of Movie Budgets')

# Revenue
plt.subplot(1, 3, 3)
sns.kdeplot(df['Revenue'])
plt.xlabel('Revenue')
plt.ylabel('Density')
plt.title('KDE Plot of Revenues')

plt.tight_layout()
plt.show()


"""   From the above plots, we can derive:

1. User Ratings:-

   * Peak Ratings: Most movies cluster around higher ratings, indicating a generally positive reception from audiences.
   * Spread: There's a smooth distribution, but with fewer movies receiving lower ratings, indicating fewer poorly-rated films.

2. Movie Budgets:-

   * Lower Budgets Cluster: A significant number of movies have lower budgets, highlighting a trend towards more cost-effective productions.
   * Long Tail: Some movies have exceptionally high budgets, likely big blockbusters. These outliers indicate the variance in production scales.

3. Revenues:-

   * Revenue Distribution: Similar to budgets, revenues are mostly in a moderate range with a few outliers having very high revenues.
   * Skewness: The distribution might be skewed to the right, suggesting that while most movies earn modest amounts, a few earn significantly more.

"""

########### Identifying and Handling Outliers

## For the column "Revenue"

# Calculate IQR for Column "Revenue"

Q1 = df['Revenue'].quantile(0.25)
Q3 = df['Revenue'].quantile(0.75)
IQR = Q3 - Q1

# Define Outlier Boundaries

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify Outliers
outliers = df[(df['Revenue'] < lower_bound) | (df['Revenue'] > upper_bound)]

# Decide on Outlier Treatment

#review outliers to decide on removal or further investigation
outliers

# Based on our analysis, we decided to retain the outliers in the "Revenue" column as they represent blockbuster movies, which are critical to our analysis.

## For the column "Movie_Budget"

# Calculate IQR for Column "Movie_Budget"

Q1 = df['Movie_Budget'].quantile(0.25)
Q3 = df['Movie_Budget'].quantile(0.75)
IQR = Q3 - Q1

# Define Outlier Boundaries

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify Outliers

outliers1 = df[(df['Movie_Budget'] < lower_bound) | (df['Movie_Budget'] > upper_bound)]

# Decide on Outlier Treatment

#review outliers to decide on removal or further investigation
outliers1

# Based on our analysis, we decided to retain the outliers in the "Movie_Budget" column as they represent blockbuster movies, which are critical to our analysis. These high and low budget films could give us valuable insights into the outliers that make waves in the box office.

## For the column "User_Rating"

# Calculate IQR for Column "User_Rating"

Q1 = df['User_Rating'].quantile(0.25)
Q3 = df['User_Rating'].quantile(0.75)
IQR = Q3 - Q1

# Define Outlier Boundaries

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify Outliers

outliers2 = df[(df['User_Rating'] < lower_bound) | (df['User_Rating'] > upper_bound)]

# Decide on Outlier Treatment

#review outliers to decide on removal or further investigation
outliers2

#Based on our analysis, we decided to retain the outliers in the "User_Rating" column as User ratings are crucial for understanding audience reception. So if we keep the outliers, we’ll capture those films that sparked extreme reactions, for better or worse.

#returns the summary of the dataframe
df.info()


#####  Standardization

# For the Column "Revenue"
#calculating the mean of "Revenue" column of movies_df
mean_revenue = df["Revenue"].mean()

#calculating the standard deviation of "Revenue" column of movies_df
std_revenue = df["Revenue"].std()

#standardizing the "Revenue" column
df["Revenue_Standardized"] = (df["Revenue"] - mean_revenue)/std_revenue
df["Revenue_Standardized"]


# Result:
# *Positive Standardized Values: Movies that have higher than average revenue.
# *Negative Standardized Values: Movies that earned less than the average.
# *Magnitude: The further a standardized value is from zero, the more it deviates from the mean revenue. For example, a revenue standardized value of 1.030504 means this movie's revenue is slightly above average, while -0.828481 indicates below average.


# For the Column "Movie_Budget"

#calculating the mean of "Movie_Budget" column of movies_df
mean_budget = df["Movie_Budget"].mean()
#calculating the standard deviation of "Movie_Budget" column of movies_df
std_budget = df["Movie_Budget"].std()
#standardizing the "Movie_Budget" column
df["Movie_Budget_Standardized"] = (df["Movie_Budget"] - mean_budget)/std_budget
df["Movie_Budget_Standardized"]

# Result:
# *Positive Standardized Values: Films that had higher than average budgets.
# *Negative Standardized Values: Films that were produced on a tighter budget.
# *Magnitude: Indicates how far the production budget deviated from the average.


# For the Column "User_Rating"
#calculating the mean of "User_Ratimg" column of movies_df
mean_rating = df["User_Rating"].mean()

#calculating the standard deviation of "User_Rating" column of movies_df
std_rating =df["User_Rating"].std()

#standardizing the "User_Rating" column
df["User_Rating_Standardized"] = (df["User_Rating"] - mean_rating)/std_rating
df["User_Rating_Standardized"]

#Result:
# *Positive Standardized Values: Movies that received better than average ratings.
#*Negative Standardized Values: Movies that didn't fare as well with audiences.
# *Magnitude: Shows how much the audience reception varied from the norm.

# Summary:
# High Revenue, High Budget, High Rating: Likely blockbuster hits that were well-received and heavily funded.
# High Revenue, Low Budget, High Rating: Highly successful low-budget films.
# Low Revenue, High Budget, Low Rating: Potentially commercial flops.

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Correlation between Revenue, Movie Budget and User Rating
# Calculate correlation matrix
correlation_matrix = df[['Revenue_Standardized', 'Movie_Budget_Standardized', 'User_Rating_Standardized']].corr()
correlation_matrix

# The correlation matrix shows the following relationships:

# Revenue_Standardized vs. Movie_Budget_Standardized → 0.676
# Strong positive correlation → Higher budgets are generally associated with higher revenues.
# Revenue and Budget show a strong positive correlation, indicating that big-budget films generally yield higher revenues. This aligns with the notion that larger investments in production, marketing, and distribution typically boost a film’s financial success. However, it’s essential to analyze if there’s a point of diminishing returns—where increasing the budget further doesn’t proportionally increase revenue.

# Revenue_Standardized vs. User_Rating_Standardized → 0.087
# Weak positive correlation → Higher user ratings have a slight tendency to correspond with higher revenue, but the relationship is very weak.
# The weak positive correlation between Revenue and User Ratings suggests that audience ratings aren’t a reliable predictor of a film’s financial success. This could be due to various factors, such as genre popularity, star power, or marketing campaigns, which can attract large audiences irrespective of the film’s quality.

# Movie_Budget_Standardized vs. User_Rating_Standardized → -0.056
# Weak negative correlation → Higher movie budgets tend to have slightly lower user ratings, but the relationship is almost negligible.
# Finally, the negative correlation between Movie Budget and User Ratings implies that higher budgets don’t guarantee better reception from audiences. This might indicate that creativity and storytelling often outshine the financial investment in a film. Some low-budget movies resonate more with viewers due to their unique approach or emotional depth.

#  Insights:
# The strongest correlation is between movie budget and revenue, suggesting that higher-budget movies tend to make more money.
# User ratings have very little impact on both revenue and movie budget

################### Trend Analysis
df.info()

#plotting line plot in "Movie_Budget" and "Revenue" over Time
plt.figure(figsize=(14, 7))
sns.lineplot(x=df["Year"], y=df['Revenue_Standardized'], label='Revenue')
sns.lineplot(x=df["Year"], y=df['Movie_Budget_Standardized'], label='Budget')
plt.xlabel('Year')
plt.ylabel('Amount')
plt.title('Trends in Movie Budgets and Revenues Over Time')
plt.legend()
plt.show()

""" Therefore, we observe from the plot of trend analysis is that:

1. Early 20th Century:

   * 1920s Surge: A noticeable increase in both budgets and revenues, reflecting the golden age of cinema.

2. Mid-20th Century:

   * 1930s-1950s Stability: A period of relative steadiness in budgets and revenues, possibly due to the economic impact of the Great Depression and WWII.

3. Post-1960s Increase:

   * 1960s Onwards: Increased variability with an upward trend, likely due to advances in technology, changes in audience preferences, and global market expansion.

4. Recent Spike:

   * Around 2020: A significant spike, perhaps driven by blockbuster films, streaming services, and globalized audiences.
Broader Insights

Economic Impact: Reflects historical events' influence on the film industry.

Technological Evolution: Shows how advancements have led to increased budgets and revenues.

Market Expansion: Indicates the growing global reach and consumption of films.



"""


### Top 10 Movies with high revenues

# Sort by 'Revenue' column in descending order
sorted_movies_df = df.sort_values(by='Revenue', ascending=False)

#get the top 10 rows
top_10_movies = sorted_movies_df.head(10)

# Display movie names and revenues
top_10_movies[['Movie_Name', 'Revenue','Country_Name',"Main_Actor"]]

# The movie Avatar is the record holder for the highest-grossing film worldwide, amassing a total revenue of two billion nine hundred twenty-three million seven hundred six thousand dollars. It is followed by "Avengers: Endgame" and "Avatar: The Way of Water" which have garnered revenues of approximately 2.8 billion and 2.32 billion dollars, respectively.



##Top Actors who generated most revenues

# Group by actor and aggregate revenue, movie_name, and country_name
actor_revenue = df.groupby("Main_Actor")["Revenue"].sum().reset_index()
#sorting the dataframe by "Revenue" in descending order
top_10_actors = actor_revenue.sort_values(by="Revenue",ascending=False).head(10)
# Display the results
top_10_actors[["Main_Actor",'Revenue']]

# Frank Welker is the actor who has generated the highest revenue, amounting to sixty-two billion two hundred seventy-five million six hundred seventy thousand dollars.



#########  Identify Categories of Movies based on Revenue and Budget

#calculating the median value of "Revenue" column
revenue_median = df["Revenue"].median()
#calculating the median value of "Budget" column
budget_median = df["Movie_Budget"].median()

# High Revenue, Low Budget
high_rev_low_budget = df[(df['Revenue'] > revenue_median) & 
                                (df['Movie_Budget'] <= budget_median)].sort_values(by='Revenue',
                                            ascending=False).head(10)

# High Revenue, High Budget
high_rev_high_budget = df[(df['Revenue'] > revenue_median) &
                                 (df['Movie_Budget'] > budget_median)].sort_values(by='Revenue',
                                            ascending=False).head(10)

# Low Revenue, Low Budget
low_rev_low_budget = df[(df['Revenue'] <= revenue_median) &
                               (df['Movie_Budget'] <= budget_median)].sort_values(by='Revenue',
                                           ascending=True).head(10)

# Low Revenue, High Budget
low_rev_high_budget = df[(df['Revenue'] <= revenue_median) &
                                (df['Movie_Budget'] > budget_median)].sort_values(by='Revenue',
                                            ascending=True).head(10)
#Displaying the results

print("High Revenue, Low Budget:")
high_rev_low_budget[['Movie_Name', 'Revenue', 'Movie_Budget','Country_Name','Main_Actor']]

# High Revenue, Low Budget:
# * Wolf Warrior 2: With a relatively low budget of  29.7M,it brought in a whopping 870.3M. Shows the power of international markets, especially for Chinese films.
# * E.T. the Extra-Terrestrial and Star Wars: Both are classics that performed exceptionally well despite their modest budgets, underscoring how compelling stories and solid marketing can drive massive returns.


print("\nHigh Revenue, High Budget:")
high_rev_high_budget[['Movie_Name', 'Revenue', 'Movie_Budget','Country_Name','Main_Actor']]

#High Revenue, High Budget:
# * Avatar: Topped the list with a staggering 2.92B in revenue against a 237M budget. High investment in technology and storytelling paid off spectacularly.

print("\nLow Revenue, Low Budget:")
low_rev_low_budget[['Movie_Name', 'Revenue', 'Movie_Budget','Country_Name','Main_Actor']]

# Low Revenue, Low Budget:
#* Films like Teen Beach Movie and Adulterers had very low budgets but didn't generate revenue, indicating a need for better marketing or broader appeal.

print("\nLow Revenue, High Budget:")
low_rev_high_budget[['Movie_Name', 'Revenue', 'Movie_Budget','Country_Name','Main_Actor']]

# Low Revenue, High Budget:
# * The Old Guard and Artemis Fowl had substantial budgets but didn't generate revenue, showcasing the risks involved with high-budget productions that don't hit the mark with audiences.

#Patterns and Insights:
#* Cost Efficiency: Many movies prove you don't need a huge budget to make significant profits. Strong narratives and efficient production can lead to high returns.
#* Blockbuster Strategy: High-budget movies, while riskier, can lead to massive revenues. Investment in technology, special effects, and marketing often pay off.
#* Geographical Influence: Films from various regions (like China) show that the global market plays a crucial role in a movie's financial success.


####### Genre Analysis
df.head()

#aggregating data by "Genre"
genre_analysis = df.groupby("Genre").agg({"Revenue":"mean",
                                                  "Movie_Budget":"mean",
                                                  "User_Rating":"mean"}).reset_index()
# Sort by average revenue and select top 20 genres
top_genres = genre_analysis.sort_values(by="Revenue", ascending=False).head(20)

plt.figure(figsize=(20, len(top_genres) * 0.6))
sns.barplot(x='Revenue', y='Genre', data=top_genres, palette='viridis', hue=None, legend=False)

# Add value labels
for index, value in enumerate(top_genres["Revenue"]):
    plt.text(value, index, f'{value:.2f}', color='black', va="center")

plt.title('Top 20 Average Revenue by Genre')
plt.xlabel('Average Revenue')
plt.ylabel('Genre')

plt.show()

""" 
Adventure, Action, Science Fiction, Fantasy is the highest-grossing genre combination, with an average revenue of over $2 billion. This suggests that big-budget, high-stakes, visually spectacular films in these genres tend to perform exceptionally well at the box office.

Erotic Documentary surprisingly holds the second spot with an average revenue of around $1.89 billion — possibly due to niche audience demand or viral popularity.

Mixed Genre Combinations (like Animation, Action, Comedy, Mystery, Crime, Fantasy) are very prominent in the top 10. This highlights the success of cross-genre storytelling that appeals to broader audiences.

Science Fiction and Fantasy appear frequently across the top entries — suggesting that audiences are particularly drawn to imaginative and visually impressive stories.

TV Movie, Animation, Science Fiction, Action, Adventure, Comedy, Drama, Fantasy, Music ranks high, indicating that multi-genre TV movies with strong production values can generate significant revenue.

Crime and Thriller combinations also appear multiple times, suggesting that suspenseful, edge-of-the-seat narratives remain popular among viewers.

Family-oriented Genres (like Adventure, Animation, Comedy) hold a solid position in the top 20, showing that family-friendly content continues to have consistent commercial success.

High-Concept Genres like War, Horror, and Mystery are also present, indicating that audiences are drawn to emotionally engaging and intense themes.

"""

# Selecting top 20 genres by average budget
top_20_budget = genre_analysis.sort_values(by='Movie_Budget', ascending=False).head(20)

plt.figure(figsize=(18, 8))
sns.barplot(x="Movie_Budget",
            y="Genre",
            hue="Genre",  # Set hue to Genre
            data=top_20_budget,
            palette="plasma",
            legend=False)  # Disable legend

plt.title("Top 20 Average Budget by Genre")
plt.xlabel("Average Budget")
plt.ylabel("Genre")

# Displaying values on bars
for index, value in enumerate(top_20_budget["Movie_Budget"]):
    plt.text(value, index, f'{value:,.2f}', va='center')

plt.show()


""" 
Highest Average Budget Genres:
The genre Fantasy, Action, Comedy leads with the highest average budget of $250M.
Adventure, Action, Science Fiction, Fantasy follows closely at $245M — suggesting that high-budget action and fantasy genres dominate production spending.
Animation and Mixed Genres:

Animation appears multiple times in high-budget categories, often mixed with genres like Adventure, Comedy, Crime, and Family — highlighting that animated and mixed-genre movies require substantial investment.
Thriller and Action's Strong Presence:

Genres like Thriller, Crime and Action, Science Fiction, Fantasy, Family have consistent high budgets (~$220M–$235M) — reflecting the complex production costs of high-stakes action sequences and special effects.
Romance and Comedy in the Mix:

TV Movie, Romance, Drama and Comedy, Animation, Family, Action, Adventure are budgeted over $200M — indicating that even non-action genres can have significant production costs when combined with animation and adventure.
Balanced Mix Across Genres:

The list shows a balanced mix of action, adventure, animation, and fantasy dominating high-budget projects — suggesting that successful genres often rely on a blend of multiple elements rather than sticking to a single category.

"""


# Group by release country and calculate average revenue and average rating
country_stats = df.groupby('Country_Name').agg({'Revenue': 'mean', 'User_Rating': 'mean'}).sort_values(by='Revenue', ascending=False)
country_stats['Revenue'] = country_stats['Revenue']/1000000
# Plot the average revenue by country
plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)
country_stats['Revenue'].plot(kind='bar', color='violet')
plt.title('Average Revenue by Country')
plt.ylabel('Average Revenue in millions')

country_stats = country_stats.sort_values(by='User_Rating', ascending=False)
# Plot the average rating by country
plt.subplot(2, 1, 2)
country_stats['User_Rating'].plot(kind='bar', color='lightgreen')
plt.title('Average User Rating by Country')
plt.ylabel('Average User Rating')
plt.xlabel('Country')

plt.tight_layout()
plt.show()

print(country_stats)


###### What are the most common term mentioned in movie overviews?

# Perform text analysis to identify recurring themes or conditions mentioned in movie descriptions.
from nltk.corpus import stopwords
import nltk

# Tokenize and clean the overview text
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
df['tokens'] = df['Overview'].apply(lambda x: [word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words])

# Flatten the list of tokens and count occurrences
all_tokens = [token for sublist in df['tokens'].tolist() for token in sublist]
token_counts = Counter(all_tokens)

# Display the most common tokens
print("Most common tokens", token_counts.most_common(10))

# Flatten the list of tokens
all_tokens = ' '.join([token for sublist in df['tokens'] for token in sublist])

# Create the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_tokens)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Movie Overviews')
plt.show()

#These words provide insight of movies. Terms like "one," "life," "new," "young," and "world" suggest that many movie plots center around individual journeys, life-changing events, and exploration. Words like "family," "man," "two," and "find" imply that relationships, personal quests, and discoveries are frequent plot . They are common storytelling motif in the film industry, familiar themes lie adventure, personal growth, and interperson problems.



"""
What languages are most commonly used in successful movies?
Analyze the original languages of movies to see which languages are associated with higher revenues and ratings.
"""
df.info()

# Explode the list of languages into separate rows
df_exploded = df.explode('Original_Language')

# Group by original language and calculate average rating and revenue
language_stats = df_exploded.groupby('Original_Language').agg({'User_Rating': 'mean', 'Revenue': 'mean'}).sort_values(by='Revenue', ascending=False)
# Sort by Revenue (descending) for the pink chart
language_stats = language_stats.sort_values(by='Revenue', ascending=False)

# Plot the average revenue by original language
plt.figure(figsize=(12,12))
plt.subplot(2, 1, 1)
language_stats['Revenue'].plot(kind='bar', color='pink')
plt.title('Average Revenue by Original Language')
plt.ylabel('Average Revenue (in dollars)')

language_stats = language_stats.sort_values(by='User_Rating', ascending=False)
# Plot the average rating by original language
plt.subplot(2, 1, 2)
language_stats['User_Rating'].plot(kind='bar', color='skyblue')
plt.title('Average User Rating by Original Language')
plt.ylabel('Average User Rating')
plt.xlabel('Original Language')

plt.tight_layout()
plt.show()

print(language_stats)
"""
Highest Revenue: English, Serbo-Croatian, and Catalan generate the highest average revenue, suggesting strong market performance.
Highest User Rating: Irish, Kurdish, and Croatian have the highest average user ratings, indicating strong audience reception.
Revenue vs. Rating Gap: High-rated languages (like Irish) don’t always correspond to high revenue, highlighting different audience engagement factors.
Low Revenue and Rating: Macedonian and Hungarian show low average revenue and user ratings, indicating limited market impact.
"""

####Exporting Clean Data
#save the data in the new file
df.to_csv("IMDB_Movies_Data.csv")


# ############ MODEL BUILDING

#pip install --upgrade pillow


df.columns

import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create an index column
df['index'] = range(0, len(df))

# Select the required columns for feature combination
selected_features = ['Genre', 'Main_Actor', 'Original_Language']

# Fill missing values with an empty string
for feature in selected_features:
    df[feature] = df[feature].fillna('')

# Combine selected features into one column
df['combined_features'] = df['Genre'] + ' ' + df['Main_Actor'] + ' ' + df['Original_Language']

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the combined features
feature_vectors = vectorizer.fit_transform(df['combined_features'])

# Calculate cosine similarity
similarity = cosine_similarity(feature_vectors)

def recommend_movies():
    movie_name = input("Enter the name of a movie you like: ")

    # Create a list of all movie titles
    list_of_all_titles = df['Movie_Name'].tolist()

    # Find closest match using difflib
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        print("\n❌ Movie not found! Please try again.\n")
        return

    close_match = find_close_match[0]
    print(f"\n✅ Closest match found: {close_match}\n")

    # Get the index of the matched movie
    index_of_the_movie = df[df['Movie_Name'] == close_match]['index'].values[0]

    # Generate similarity scores for the matched movie
    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    # Sort movies based on similarity score (highest first)
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    print('🎬 **Movies recommended for you:**\n')

    recommended_movies = []
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = df.loc[df['index'] == index, 'Movie_Name'].values[0]
        if title_from_index not in recommended_movies and len(recommended_movies) < 10:
            recommended_movies.append(title_from_index)

    for i, movie in enumerate(recommended_movies):
         similarity_score = similarity[index_of_the_movie][df[df['Movie_Name'] == movie]['index'].values[0]]
         print(f"{i + 1}. {movie} – Similarity Score: {similarity_score:.2f}")

# Run the function
recommend_movies()











