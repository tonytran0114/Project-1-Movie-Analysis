# Data Import and First Inspection


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```


```python
df = pd.read_csv("movies_complete.csv")

```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>tagline</th>
      <th>release_date</th>
      <th>genres</th>
      <th>belongs_to_collection</th>
      <th>original_language</th>
      <th>budget_musd</th>
      <th>revenue_musd</th>
      <th>production_companies</th>
      <th>...</th>
      <th>vote_average</th>
      <th>popularity</th>
      <th>runtime</th>
      <th>overview</th>
      <th>spoken_languages</th>
      <th>poster_path</th>
      <th>cast</th>
      <th>cast_size</th>
      <th>crew_size</th>
      <th>director</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>862</td>
      <td>Toy Story</td>
      <td>NaN</td>
      <td>1995-10-30</td>
      <td>Animation|Comedy|Family</td>
      <td>Toy Story Collection</td>
      <td>en</td>
      <td>30.0</td>
      <td>373.554033</td>
      <td>Pixar Animation Studios</td>
      <td>...</td>
      <td>7.7</td>
      <td>21.946943</td>
      <td>81.0</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>English</td>
      <td>&lt;img src='http://image.tmdb.org/t/p/w185//uXDf...</td>
      <td>Tom Hanks|Tim Allen|Don Rickles|Jim Varney|Wal...</td>
      <td>13</td>
      <td>106</td>
      <td>John Lasseter</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8844</td>
      <td>Jumanji</td>
      <td>Roll the dice and unleash the excitement!</td>
      <td>1995-12-15</td>
      <td>Adventure|Fantasy|Family</td>
      <td>NaN</td>
      <td>en</td>
      <td>65.0</td>
      <td>262.797249</td>
      <td>TriStar Pictures|Teitler Film|Interscope Commu...</td>
      <td>...</td>
      <td>6.9</td>
      <td>17.015539</td>
      <td>104.0</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>English|Français</td>
      <td>&lt;img src='http://image.tmdb.org/t/p/w185//vgpX...</td>
      <td>Robin Williams|Jonathan Hyde|Kirsten Dunst|Bra...</td>
      <td>26</td>
      <td>16</td>
      <td>Joe Johnston</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15602</td>
      <td>Grumpier Old Men</td>
      <td>Still Yelling. Still Fighting. Still Ready for...</td>
      <td>1995-12-22</td>
      <td>Romance|Comedy</td>
      <td>Grumpy Old Men Collection</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Warner Bros.|Lancaster Gate</td>
      <td>...</td>
      <td>6.5</td>
      <td>11.712900</td>
      <td>101.0</td>
      <td>A family wedding reignites the ancient feud be...</td>
      <td>English</td>
      <td>&lt;img src='http://image.tmdb.org/t/p/w185//1FSX...</td>
      <td>Walter Matthau|Jack Lemmon|Ann-Margret|Sophia ...</td>
      <td>7</td>
      <td>4</td>
      <td>Howard Deutch</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31357</td>
      <td>Waiting to Exhale</td>
      <td>Friends are the people who let you be yourself...</td>
      <td>1995-12-22</td>
      <td>Comedy|Drama|Romance</td>
      <td>NaN</td>
      <td>en</td>
      <td>16.0</td>
      <td>81.452156</td>
      <td>Twentieth Century Fox Film Corporation</td>
      <td>...</td>
      <td>6.1</td>
      <td>3.859495</td>
      <td>127.0</td>
      <td>Cheated on, mistreated and stepped on, the wom...</td>
      <td>English</td>
      <td>&lt;img src='http://image.tmdb.org/t/p/w185//4wjG...</td>
      <td>Whitney Houston|Angela Bassett|Loretta Devine|...</td>
      <td>10</td>
      <td>10</td>
      <td>Forest Whitaker</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11862</td>
      <td>Father of the Bride Part II</td>
      <td>Just When His World Is Back To Normal... He's ...</td>
      <td>1995-02-10</td>
      <td>Comedy</td>
      <td>Father of the Bride Collection</td>
      <td>en</td>
      <td>NaN</td>
      <td>76.578911</td>
      <td>Sandollar Productions|Touchstone Pictures</td>
      <td>...</td>
      <td>5.7</td>
      <td>8.387519</td>
      <td>106.0</td>
      <td>Just when George Banks has recovered from his ...</td>
      <td>English</td>
      <td>&lt;img src='http://image.tmdb.org/t/p/w185//lf9R...</td>
      <td>Steve Martin|Diane Keaton|Martin Short|Kimberl...</td>
      <td>12</td>
      <td>7</td>
      <td>Charles Shyer</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>44686</th>
      <td>439050</td>
      <td>Subdue</td>
      <td>Rising and falling between a man and woman</td>
      <td>NaN</td>
      <td>Drama|Family</td>
      <td>NaN</td>
      <td>fa</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>4.0</td>
      <td>0.072051</td>
      <td>90.0</td>
      <td>Rising and falling between a man and woman.</td>
      <td>فارسی</td>
      <td>&lt;img src='http://image.tmdb.org/t/p/w185//pfC8...</td>
      <td>Leila Hatami|Kourosh Tahami|Elham Korda</td>
      <td>3</td>
      <td>9</td>
      <td>Hamid Nematollah</td>
    </tr>
    <tr>
      <th>44687</th>
      <td>111109</td>
      <td>Century of Birthing</td>
      <td>NaN</td>
      <td>2011-11-17</td>
      <td>Drama</td>
      <td>NaN</td>
      <td>tl</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Sine Olivia</td>
      <td>...</td>
      <td>9.0</td>
      <td>0.178241</td>
      <td>360.0</td>
      <td>An artist struggles to finish his work while a...</td>
      <td>NaN</td>
      <td>&lt;img src='http://image.tmdb.org/t/p/w185//xZkm...</td>
      <td>Angel Aquino|Perry Dizon|Hazel Orencio|Joel To...</td>
      <td>11</td>
      <td>6</td>
      <td>Lav Diaz</td>
    </tr>
    <tr>
      <th>44688</th>
      <td>67758</td>
      <td>Betrayal</td>
      <td>A deadly game of wits.</td>
      <td>2003-08-01</td>
      <td>Action|Drama|Thriller</td>
      <td>NaN</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>American World Pictures</td>
      <td>...</td>
      <td>3.8</td>
      <td>0.903007</td>
      <td>90.0</td>
      <td>When one of her hits goes wrong, a professiona...</td>
      <td>English</td>
      <td>&lt;img src='http://image.tmdb.org/t/p/w185//eGga...</td>
      <td>Erika Eleniak|Adam Baldwin|Julie du Page|James...</td>
      <td>15</td>
      <td>5</td>
      <td>Mark L. Lester</td>
    </tr>
    <tr>
      <th>44689</th>
      <td>227506</td>
      <td>Satan Triumphant</td>
      <td>NaN</td>
      <td>1917-10-21</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yermoliev</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.003503</td>
      <td>87.0</td>
      <td>In a small town live two brothers, one a minis...</td>
      <td>NaN</td>
      <td>&lt;img src='http://image.tmdb.org/t/p/w185//aorB...</td>
      <td>Iwan Mosschuchin|Nathalie Lissenko|Pavel Pavlo...</td>
      <td>5</td>
      <td>2</td>
      <td>Yakov Protazanov</td>
    </tr>
    <tr>
      <th>44690</th>
      <td>461257</td>
      <td>Queerama</td>
      <td>NaN</td>
      <td>2017-06-09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.163015</td>
      <td>75.0</td>
      <td>50 years after decriminalisation of homosexual...</td>
      <td>English</td>
      <td>&lt;img src='http://image.tmdb.org/t/p/w185//oxFE...</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>Daisy Asquith</td>
    </tr>
  </tbody>
</table>
<p>44691 rows × 22 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 44691 entries, 0 to 44690
    Data columns (total 22 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   id                     44691 non-null  int64  
     1   title                  44691 non-null  object 
     2   tagline                20284 non-null  object 
     3   release_date           44657 non-null  object 
     4   genres                 42586 non-null  object 
     5   belongs_to_collection  4463 non-null   object 
     6   original_language      44681 non-null  object 
     7   budget_musd            8854 non-null   float64
     8   revenue_musd           7385 non-null   float64
     9   production_companies   33356 non-null  object 
     10  production_countries   38835 non-null  object 
     11  vote_count             44691 non-null  float64
     12  vote_average           42077 non-null  float64
     13  popularity             44691 non-null  float64
     14  runtime                43179 non-null  float64
     15  overview               43740 non-null  object 
     16  spoken_languages       41094 non-null  object 
     17  poster_path            44467 non-null  object 
     18  cast                   42502 non-null  object 
     19  cast_size              44691 non-null  int64  
     20  crew_size              44691 non-null  int64  
     21  director               43960 non-null  object 
    dtypes: float64(6), int64(3), object(13)
    memory usage: 7.5+ MB
    


```python

```
