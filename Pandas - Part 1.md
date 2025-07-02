```python
import pandas as pd
import matplotlib as plt
```


```python
world = pd.read_csv('Data for python Course/2018-2019 Happiness.csv') #import world happiness dataset
airbnb = pd.read_csv('Data for python Course/airbnb.csv')
salaries = pd.read_csv('Data for python Course/salaries.csv')
sales = pd.read_csv('Data for python Course/web_sales.csv')
```


```python
sales.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 60398 entries, 0 to 60397
    Data columns (total 11 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   ProductKey             60398 non-null  int64  
     1   product                60398 non-null  object 
     2   product_subcategory    60398 non-null  object 
     3   OrderDateKey           60398 non-null  int64  
     4   CalendarYear           60398 non-null  int64  
     5   SalesTerritoryRegion   60398 non-null  object 
     6   SalesTerritoryCountry  60398 non-null  object 
     7   SalesOrderNumber       60398 non-null  object 
     8   SalesAmount            60398 non-null  float64
     9   TaxAmt                 60398 non-null  float64
     10  Freight                60398 non-null  float64
    dtypes: float64(3), int64(3), object(5)
    memory usage: 5.1+ MB
    


```python
sales.SalesOrderNumber
```




    0        SO43697
    1        SO43698
    2        SO43699
    3        SO43700
    4        SO43701
              ...   
    60393    SO75122
    60394    SO75122
    60395    SO75123
    60396    SO75123
    60397    SO75123
    Name: SalesOrderNumber, Length: 60398, dtype: object




```python
sales.SalesAmount
```




    0        3578.2700
    1        3399.9900
    2        3399.9900
    3         699.0982
    4        3399.9900
               ...    
    60393      21.9800
    60394       8.9900
    60395      21.9800
    60396     159.0000
    60397       8.9900
    Name: SalesAmount, Length: 60398, dtype: float64




```python
airbnb.columns
```




    Index(['Host Id', 'Host Since', 'Name', 'Neighbourhood', 'Property Type',
           'Review Scores Rating (bin)', 'Room Type', 'Zipcode', 'Beds',
           'Number of Records', 'Number Of Reviews', 'Price',
           'Review Scores Rating'],
          dtype='object')




```python
airbnb[['Property Type','Room Type','Beds']]
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
      <th>Property Type</th>
      <th>Room Type</th>
      <th>Beds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Apartment</td>
      <td>Private room</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apartment</td>
      <td>Private room</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Apartment</td>
      <td>Private room</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>30473</th>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>30474</th>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>30475</th>
      <td>Other</td>
      <td>Private room</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>30476</th>
      <td>Apartment</td>
      <td>Private room</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>30477</th>
      <td>House</td>
      <td>Private room</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>30478 rows × 3 columns</p>
</div>




```python
airbnb['Property Type']
```




    0        Apartment
    1        Apartment
    2        Apartment
    3        Apartment
    4        Apartment
               ...    
    30473    Apartment
    30474    Apartment
    30475        Other
    30476    Apartment
    30477        House
    Name: Property Type, Length: 30478, dtype: object




```python
salaries.columns
```




    Index(['work_year', 'experience_level', 'employment_type', 'job_title',
           'salary', 'salary_currency', 'salary_in_usd', 'employee_residence',
           'remote_ratio', 'company_location', 'company_size'],
          dtype='object')




```python
salaries[['experience_level','job_title','salary_in_usd']]
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
      <th>experience_level</th>
      <th>job_title</th>
      <th>salary_in_usd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EX</td>
      <td>Data Science Director</td>
      <td>212000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EX</td>
      <td>Data Science Director</td>
      <td>190000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MI</td>
      <td>Business Intelligence Engineer</td>
      <td>43064</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MI</td>
      <td>Business Intelligence Engineer</td>
      <td>43064</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SE</td>
      <td>Machine Learning Engineer</td>
      <td>245700</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8800</th>
      <td>SE</td>
      <td>Data Scientist</td>
      <td>412000</td>
    </tr>
    <tr>
      <th>8801</th>
      <td>MI</td>
      <td>Principal Data Scientist</td>
      <td>151000</td>
    </tr>
    <tr>
      <th>8802</th>
      <td>EN</td>
      <td>Data Scientist</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>8803</th>
      <td>EN</td>
      <td>Business Data Analyst</td>
      <td>100000</td>
    </tr>
    <tr>
      <th>8804</th>
      <td>SE</td>
      <td>Data Science Manager</td>
      <td>94665</td>
    </tr>
  </tbody>
</table>
<p>8805 rows × 3 columns</p>
</div>




```python
salaries['salary_in_usd'].mean()
```




    np.float64(149488.26564452017)




```python
salaries['salary_in_usd'].max()
```




    np.int64(615201)




```python
salaries['salary_in_usd'].min()
```




    np.int64(15000)




```python
salaries['salary_in_usd'].sum()
```




    np.int64(1316244179)




```python
sales
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
      <th>ProductKey</th>
      <th>product</th>
      <th>product_subcategory</th>
      <th>OrderDateKey</th>
      <th>CalendarYear</th>
      <th>SalesTerritoryRegion</th>
      <th>SalesTerritoryCountry</th>
      <th>SalesOrderNumber</th>
      <th>SalesAmount</th>
      <th>TaxAmt</th>
      <th>Freight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>310</td>
      <td>Road-150 Red, 62</td>
      <td>Road Bikes</td>
      <td>20101229</td>
      <td>2010</td>
      <td>Canada</td>
      <td>Canada</td>
      <td>SO43697</td>
      <td>3578.2700</td>
      <td>286.2616</td>
      <td>89.4568</td>
    </tr>
    <tr>
      <th>1</th>
      <td>346</td>
      <td>Mountain-100 Silver, 44</td>
      <td>Mountain Bikes</td>
      <td>20101229</td>
      <td>2010</td>
      <td>France</td>
      <td>France</td>
      <td>SO43698</td>
      <td>3399.9900</td>
      <td>271.9992</td>
      <td>84.9998</td>
    </tr>
    <tr>
      <th>2</th>
      <td>346</td>
      <td>Mountain-100 Silver, 44</td>
      <td>Mountain Bikes</td>
      <td>20101229</td>
      <td>2010</td>
      <td>Northwest</td>
      <td>United States</td>
      <td>SO43699</td>
      <td>3399.9900</td>
      <td>271.9992</td>
      <td>84.9998</td>
    </tr>
    <tr>
      <th>3</th>
      <td>336</td>
      <td>Road-650 Black, 62</td>
      <td>Road Bikes</td>
      <td>20101229</td>
      <td>2010</td>
      <td>Southwest</td>
      <td>United States</td>
      <td>SO43700</td>
      <td>699.0982</td>
      <td>55.9279</td>
      <td>17.4775</td>
    </tr>
    <tr>
      <th>4</th>
      <td>346</td>
      <td>Mountain-100 Silver, 44</td>
      <td>Mountain Bikes</td>
      <td>20101229</td>
      <td>2010</td>
      <td>Australia</td>
      <td>Australia</td>
      <td>SO43701</td>
      <td>3399.9900</td>
      <td>271.9992</td>
      <td>84.9998</td>
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
    </tr>
    <tr>
      <th>60393</th>
      <td>485</td>
      <td>Fender Set - Mountain</td>
      <td>Fenders</td>
      <td>20140128</td>
      <td>2014</td>
      <td>Canada</td>
      <td>Canada</td>
      <td>SO75122</td>
      <td>21.9800</td>
      <td>1.7584</td>
      <td>0.5495</td>
    </tr>
    <tr>
      <th>60394</th>
      <td>225</td>
      <td>AWC Logo Cap</td>
      <td>Caps</td>
      <td>20140128</td>
      <td>2014</td>
      <td>Canada</td>
      <td>Canada</td>
      <td>SO75122</td>
      <td>8.9900</td>
      <td>0.7192</td>
      <td>0.2248</td>
    </tr>
    <tr>
      <th>60395</th>
      <td>485</td>
      <td>Fender Set - Mountain</td>
      <td>Fenders</td>
      <td>20140128</td>
      <td>2014</td>
      <td>Canada</td>
      <td>Canada</td>
      <td>SO75123</td>
      <td>21.9800</td>
      <td>1.7584</td>
      <td>0.5495</td>
    </tr>
    <tr>
      <th>60396</th>
      <td>486</td>
      <td>All-Purpose Bike Stand</td>
      <td>Bike Stands</td>
      <td>20140128</td>
      <td>2014</td>
      <td>Canada</td>
      <td>Canada</td>
      <td>SO75123</td>
      <td>159.0000</td>
      <td>12.7200</td>
      <td>3.9750</td>
    </tr>
    <tr>
      <th>60397</th>
      <td>225</td>
      <td>AWC Logo Cap</td>
      <td>Caps</td>
      <td>20140128</td>
      <td>2014</td>
      <td>Canada</td>
      <td>Canada</td>
      <td>SO75123</td>
      <td>8.9900</td>
      <td>0.7192</td>
      <td>0.2248</td>
    </tr>
  </tbody>
</table>
<p>60398 rows × 11 columns</p>
</div>




```python
sales.sum()
```




    ProductKey                                                        26427624
    product                  Road-150 Red, 62Mountain-100 Silver, 44Mountai...
    product_subcategory      Road BikesMountain BikesMountain BikesRoad Bik...
    OrderDateKey                                                 1215795730492
    CalendarYear                                                     121575273
    SalesTerritoryRegion     CanadaFranceNorthwestSouthwestAustraliaSouthwe...
    SalesTerritoryCountry    CanadaFranceUnited StatesUnited StatesAustrali...
    SalesOrderNumber         SO43697SO43698SO43699SO43700SO43701SO43702SO43...
    SalesAmount                                                  29358677.2207
    TaxAmt                                                        2348694.2301
    Freight                                                        733969.6091
    dtype: object




```python
salaries.min()
```




    work_year                     2020
    experience_level                EN
    employment_type                 CT
    job_title             AI Architect
    salary                       14000
    salary_currency                AUD
    salary_in_usd                15000
    employee_residence              AD
    remote_ratio                     0
    company_location                AD
    company_size                     L
    dtype: object




```python
salaries.max()
```




    work_year                                        2023
    experience_level                                   SE
    employment_type                                    PT
    job_title             Staff Machine Learning Engineer
    salary                                       30400000
    salary_currency                                   ZAR
    salary_in_usd                                  615201
    employee_residence                                 ZA
    remote_ratio                                      100
    company_location                                   ZA
    company_size                                        S
    dtype: object




```python
sales.max()
```




    ProductKey                                      606
    product                  Women's Mountain Shorts, S
    product_subcategory                           Vests
    OrderDateKey                               20140128
    CalendarYear                                   2014
    SalesTerritoryRegion                 United Kingdom
    SalesTerritoryCountry                 United States
    SalesOrderNumber                            SO75123
    SalesAmount                                 3578.27
    TaxAmt                                     286.2616
    Freight                                     89.4568
    dtype: object




```python
sales.mean(numeric_only=True)
```




    ProductKey      4.375579e+02
    OrderDateKey    2.012973e+07
    CalendarYear    2.012902e+03
    SalesAmount     4.860869e+02
    TaxAmt          3.888695e+01
    Freight         1.215222e+01
    dtype: float64




```python
4.860869e+02
```




    486.0869




```python
sales.columns
```




    Index(['ProductKey', 'product', 'product_subcategory', 'OrderDateKey',
           'CalendarYear', 'SalesTerritoryRegion', 'SalesTerritoryCountry',
           'SalesOrderNumber', 'SalesAmount', 'TaxAmt', 'Freight'],
          dtype='object')




```python
sales['SalesAmount'].std()
```




    np.float64(928.4898919808043)




```python
sales['SalesAmount'].mean()
```




    np.float64(486.0869105053147)




```python
airbnb['Beds'].mean()
```




    np.float64(1.5300891652683184)




```python
airbnb['Beds'].std()
```




    np.float64(1.0153587174803154)




```python
airbnb['Beds'].count()
```




    np.int64(30393)




```python
airbnb.shape
```




    (30478, 13)




```python
airbnb.count()
```




    Host Id                       30478
    Host Since                    30475
    Name                          30478
    Neighbourhood                 30478
    Property Type                 30475
    Review Scores Rating (bin)    22155
    Room Type                     30478
    Zipcode                       30344
    Beds                          30393
    Number of Records             30478
    Number Of Reviews             30478
    Price                         30478
    Review Scores Rating          22155
    dtype: int64




```python
sales.describe()
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
      <th>ProductKey</th>
      <th>OrderDateKey</th>
      <th>CalendarYear</th>
      <th>SalesAmount</th>
      <th>TaxAmt</th>
      <th>Freight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>60398.000000</td>
      <td>6.039800e+04</td>
      <td>60398.000000</td>
      <td>60398.000000</td>
      <td>60398.000000</td>
      <td>60398.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>437.557932</td>
      <td>2.012973e+07</td>
      <td>2012.902298</td>
      <td>486.086911</td>
      <td>38.886954</td>
      <td>12.152217</td>
    </tr>
    <tr>
      <th>std</th>
      <td>118.088390</td>
      <td>4.745050e+03</td>
      <td>0.477666</td>
      <td>928.489892</td>
      <td>74.279193</td>
      <td>23.212248</td>
    </tr>
    <tr>
      <th>min</th>
      <td>214.000000</td>
      <td>2.010123e+07</td>
      <td>2010.000000</td>
      <td>2.290000</td>
      <td>0.183200</td>
      <td>0.057300</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>359.000000</td>
      <td>2.013040e+07</td>
      <td>2013.000000</td>
      <td>7.950000</td>
      <td>0.636000</td>
      <td>0.198800</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>479.000000</td>
      <td>2.013071e+07</td>
      <td>2013.000000</td>
      <td>29.990000</td>
      <td>2.399200</td>
      <td>0.749800</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>529.000000</td>
      <td>2.013102e+07</td>
      <td>2013.000000</td>
      <td>539.990000</td>
      <td>43.199200</td>
      <td>13.499800</td>
    </tr>
    <tr>
      <th>max</th>
      <td>606.000000</td>
      <td>2.014013e+07</td>
      <td>2014.000000</td>
      <td>3578.270000</td>
      <td>286.261600</td>
      <td>89.456800</td>
    </tr>
  </tbody>
</table>
</div>




```python
2.012973e+07	
```




    20129730.0




```python
airbnb.describe(percentiles=[0.1,0.6,0.9])
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
      <th>Host Id</th>
      <th>Review Scores Rating (bin)</th>
      <th>Zipcode</th>
      <th>Beds</th>
      <th>Number of Records</th>
      <th>Number Of Reviews</th>
      <th>Price</th>
      <th>Review Scores Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.047800e+04</td>
      <td>22155.000000</td>
      <td>30344.000000</td>
      <td>30393.000000</td>
      <td>30478.0</td>
      <td>30478.000000</td>
      <td>30478.000000</td>
      <td>22155.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.273171e+07</td>
      <td>90.738659</td>
      <td>10584.854831</td>
      <td>1.530089</td>
      <td>1.0</td>
      <td>12.018735</td>
      <td>163.589737</td>
      <td>91.993230</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.190270e+07</td>
      <td>9.059519</td>
      <td>921.299397</td>
      <td>1.015359</td>
      <td>0.0</td>
      <td>21.980703</td>
      <td>197.785454</td>
      <td>8.850373</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.000000e+02</td>
      <td>20.000000</td>
      <td>1003.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>10%</th>
      <td>7.274437e+05</td>
      <td>80.000000</td>
      <td>10009.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>60.000000</td>
      <td>80.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.551693e+06</td>
      <td>90.000000</td>
      <td>10065.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>3.000000</td>
      <td>125.000000</td>
      <td>94.000000</td>
    </tr>
    <tr>
      <th>60%</th>
      <td>1.216061e+07</td>
      <td>95.000000</td>
      <td>11205.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>6.000000</td>
      <td>150.000000</td>
      <td>96.000000</td>
    </tr>
    <tr>
      <th>90%</th>
      <td>3.243044e+07</td>
      <td>100.000000</td>
      <td>11237.000000</td>
      <td>3.000000</td>
      <td>1.0</td>
      <td>35.000000</td>
      <td>275.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.303307e+07</td>
      <td>100.000000</td>
      <td>99135.000000</td>
      <td>16.000000</td>
      <td>1.0</td>
      <td>257.000000</td>
      <td>10000.000000</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sales.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 60398 entries, 0 to 60397
    Data columns (total 11 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   ProductKey             60398 non-null  int64  
     1   product                60398 non-null  object 
     2   product_subcategory    60398 non-null  object 
     3   OrderDateKey           60398 non-null  int64  
     4   CalendarYear           60398 non-null  int64  
     5   SalesTerritoryRegion   60398 non-null  object 
     6   SalesTerritoryCountry  60398 non-null  object 
     7   SalesOrderNumber       60398 non-null  object 
     8   SalesAmount            60398 non-null  float64
     9   TaxAmt                 60398 non-null  float64
     10  Freight                60398 non-null  float64
    dtypes: float64(3), int64(3), object(5)
    memory usage: 5.1+ MB
    


```python
sales.describe(include='object')
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
      <th>product</th>
      <th>product_subcategory</th>
      <th>SalesTerritoryRegion</th>
      <th>SalesTerritoryCountry</th>
      <th>SalesOrderNumber</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>60398</td>
      <td>60398</td>
      <td>60398</td>
      <td>60398</td>
      <td>60398</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>130</td>
      <td>17</td>
      <td>10</td>
      <td>6</td>
      <td>27659</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Water Bottle - 30 oz.</td>
      <td>Tires and Tubes</td>
      <td>Australia</td>
      <td>United States</td>
      <td>SO70714</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>4244</td>
      <td>17332</td>
      <td>13345</td>
      <td>21344</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
sales[['SalesTerritoryCountry','product_subcategory']].value_counts(ascending = False)
```




    SalesTerritoryCountry  product_subcategory
    United States          Tires and Tubes        6062
    Australia              Tires and Tubes        3300
    United States          Bottles and Cages      2951
    Canada                 Tires and Tubes        2729
    Australia              Road Bikes             2603
                                                  ... 
    France                 Bike Racks               25
    Germany                Bike Racks               24
                           Bike Stands              20
    France                 Bike Stands              19
    Germany                Shorts                    9
    Name: count, Length: 102, dtype: int64




```python
airbnb[['Neighbourhood','Property Type']].value_counts()
```




    Neighbourhood  Property Type  
    Manhattan      Apartment          15433
    Brooklyn       Apartment           9740
    Queens         Apartment           1659
    Brooklyn       House               1202
    Queens         House                522
    Brooklyn       Loft                 502
    Manhattan      Loft                 227
    Bronx          Apartment            218
    Manhattan      House                168
    Bronx          House                110
    Staten Island  House                 88
    Brooklyn       Townhouse             79
                   Bed & Breakfast       76
    Manhattan      Bed & Breakfast       60
                   Condominium           56
    Staten Island  Apartment             52
    Manhattan      Townhouse             45
    Queens         Bed & Breakfast       36
    Brooklyn       Condominium           30
    Manhattan      Other                 23
    Brooklyn       Dorm                  19
    Queens         Loft                  17
    Brooklyn       Other                 16
    Queens         Townhouse              8
    Bronx          Loft                   6
                   Bed & Breakfast        6
    Queens         Dorm                   6
                   Other                  6
    Manhattan      Dorm                   6
    Queens         Boat                   5
                   Condominium            5
                   Camper/RV              5
                   Villa                  4
    Manhattan      Villa                  3
    Queens         Bungalow               3
    Brooklyn       Tent                   3
    Bronx          Condominium            3
                   Townhouse              2
    Manhattan      Boat                   2
                   Treehouse              2
    Staten Island  Bed & Breakfast        2
    Brooklyn       Treehouse              2
    Staten Island  Townhouse              2
                   Other                  2
    Brooklyn       Camper/RV              1
                   Villa                  1
                   Chalet                 1
                   Bungalow               1
                   Boat                   1
                   Lighthouse             1
    Manhattan      Cabin                  1
                   Castle                 1
                   Camper/RV              1
                   Hut                    1
                   Tent                   1
    Queens         Cabin                  1
                   Hut                    1
    Staten Island  Loft                   1
    Name: count, dtype: int64




```python
airbnb.columns
```




    Index(['Host Id', 'Host Since', 'Name', 'Neighbourhood', 'Property Type',
           'Review Scores Rating (bin)', 'Room Type', 'Zipcode', 'Beds',
           'Number of Records', 'Number Of Reviews', 'Price',
           'Review Scores Rating'],
          dtype='object')




```python
salaries.columns
```




    Index(['work_year', 'experience_level', 'employment_type', 'job_title',
           'salary', 'salary_currency', 'salary_in_usd', 'employee_residence',
           'remote_ratio', 'company_location', 'company_size'],
          dtype='object')




```python

```


```python
salaries[['job_title','salary_in_usd','experience_level']].value_counts()
```




    job_title          salary_in_usd  experience_level
    Data Engineer      160000         SE                  51
    Applied Scientist  136000         SE                  48
    Data Scientist     160000         SE                  44
    Data Engineer      130000         SE                  44
    Data Scientist     140000         SE                  38
                                                          ..
    AI Developer       64781          EN                   1
                       60000          SE                   1
                       53984          SE                   1
                       50000          EN                   1
    AI Architect       305100         SE                   1
    Name: count, Length: 4107, dtype: int64




```python

```


```python

```


```python

```
