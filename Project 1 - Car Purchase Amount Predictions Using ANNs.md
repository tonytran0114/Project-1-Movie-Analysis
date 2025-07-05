# CODE TO PREDICT CAR PURCHASING DOLLAR AMOUNT USING ANNs (REGRESSION TASK)
# Dr. Ryan Ahmed @STEMplicity
![image.png](image.png)



# PROBLEM STATEMENT

You are working as a car salesman and you would like to develop a model to predict the total dollar amount that customers are willing to pay given the following attributes: 
- Customer Name
- Customer e-mail
- Country
- Gender
- Age
- Annual Salary 
- Credit Card Debt 
- Net Worth 

The model should predict: 
- Car Purchase Amount 

# STEP #0: LIBRARIES IMPORT



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

# STEP #1: IMPORT DATASET


```python
car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')

```


```python
car_df
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
      <th>Customer Name</th>
      <th>Customer e-mail</th>
      <th>Country</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Salary</th>
      <th>Credit Card Debt</th>
      <th>Net Worth</th>
      <th>Car Purchase Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Martina Avila</td>
      <td>cubilia.Curae.Phasellus@quisaccumsanconvallis.edu</td>
      <td>Bulgaria</td>
      <td>0</td>
      <td>41.851720</td>
      <td>62812.09301</td>
      <td>11609.380910</td>
      <td>238961.2505</td>
      <td>35321.45877</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Harlan Barnes</td>
      <td>eu.dolor@diam.co.uk</td>
      <td>Belize</td>
      <td>0</td>
      <td>40.870623</td>
      <td>66646.89292</td>
      <td>9572.957136</td>
      <td>530973.9078</td>
      <td>45115.52566</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Naomi Rodriquez</td>
      <td>vulputate.mauris.sagittis@ametconsectetueradip...</td>
      <td>Algeria</td>
      <td>1</td>
      <td>43.152897</td>
      <td>53798.55112</td>
      <td>11160.355060</td>
      <td>638467.1773</td>
      <td>42925.70921</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jade Cunningham</td>
      <td>malesuada@dignissim.com</td>
      <td>Cook Islands</td>
      <td>1</td>
      <td>58.271369</td>
      <td>79370.03798</td>
      <td>14426.164850</td>
      <td>548599.0524</td>
      <td>67422.36313</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cedric Leach</td>
      <td>felis.ullamcorper.viverra@egetmollislectus.net</td>
      <td>Brazil</td>
      <td>1</td>
      <td>57.313749</td>
      <td>59729.15130</td>
      <td>5358.712177</td>
      <td>560304.0671</td>
      <td>55915.46248</td>
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
    </tr>
    <tr>
      <th>495</th>
      <td>Walter</td>
      <td>ligula@Cumsociis.ca</td>
      <td>Nepal</td>
      <td>0</td>
      <td>41.462515</td>
      <td>71942.40291</td>
      <td>6995.902524</td>
      <td>541670.1016</td>
      <td>48901.44342</td>
    </tr>
    <tr>
      <th>496</th>
      <td>Vanna</td>
      <td>Cum.sociis.natoque@Sedmolestie.edu</td>
      <td>Zimbabwe</td>
      <td>1</td>
      <td>37.642000</td>
      <td>56039.49793</td>
      <td>12301.456790</td>
      <td>360419.0988</td>
      <td>31491.41457</td>
    </tr>
    <tr>
      <th>497</th>
      <td>Pearl</td>
      <td>penatibus.et@massanonante.com</td>
      <td>Philippines</td>
      <td>1</td>
      <td>53.943497</td>
      <td>68888.77805</td>
      <td>10611.606860</td>
      <td>764531.3203</td>
      <td>64147.28888</td>
    </tr>
    <tr>
      <th>498</th>
      <td>Nell</td>
      <td>Quisque.varius@arcuVivamussit.net</td>
      <td>Botswana</td>
      <td>1</td>
      <td>59.160509</td>
      <td>49811.99062</td>
      <td>14013.034510</td>
      <td>337826.6382</td>
      <td>45442.15353</td>
    </tr>
    <tr>
      <th>499</th>
      <td>Marla</td>
      <td>Camaron.marla@hotmail.com</td>
      <td>marlal</td>
      <td>1</td>
      <td>46.731152</td>
      <td>61370.67766</td>
      <td>9391.341628</td>
      <td>462946.4924</td>
      <td>45107.22566</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 9 columns</p>
</div>



# STEP #2: VISUALIZE DATASET


```python
sns.pairplot(car_df)
```




    <seaborn.axisgrid.PairGrid at 0x2706c6c3f50>




    
![png](output_9_1.png)
    


# STEP #3: CREATE TESTING AND TRAINING DATASET/DATA CLEANING


```python
X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)
```


```python

```


```python
X
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
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Salary</th>
      <th>Credit Card Debt</th>
      <th>Net Worth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>41.851720</td>
      <td>62812.09301</td>
      <td>11609.380910</td>
      <td>238961.2505</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>40.870623</td>
      <td>66646.89292</td>
      <td>9572.957136</td>
      <td>530973.9078</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>43.152897</td>
      <td>53798.55112</td>
      <td>11160.355060</td>
      <td>638467.1773</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>58.271369</td>
      <td>79370.03798</td>
      <td>14426.164850</td>
      <td>548599.0524</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>57.313749</td>
      <td>59729.15130</td>
      <td>5358.712177</td>
      <td>560304.0671</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>0</td>
      <td>41.462515</td>
      <td>71942.40291</td>
      <td>6995.902524</td>
      <td>541670.1016</td>
    </tr>
    <tr>
      <th>496</th>
      <td>1</td>
      <td>37.642000</td>
      <td>56039.49793</td>
      <td>12301.456790</td>
      <td>360419.0988</td>
    </tr>
    <tr>
      <th>497</th>
      <td>1</td>
      <td>53.943497</td>
      <td>68888.77805</td>
      <td>10611.606860</td>
      <td>764531.3203</td>
    </tr>
    <tr>
      <th>498</th>
      <td>1</td>
      <td>59.160509</td>
      <td>49811.99062</td>
      <td>14013.034510</td>
      <td>337826.6382</td>
    </tr>
    <tr>
      <th>499</th>
      <td>1</td>
      <td>46.731152</td>
      <td>61370.67766</td>
      <td>9391.341628</td>
      <td>462946.4924</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 5 columns</p>
</div>




```python
y = car_df['Car Purchase Amount']
y.shape
```




    (500,)




```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

```


```python
scaler.data_max_
```




    array([1.e+00, 7.e+01, 1.e+05, 2.e+04, 1.e+06])




```python
scaler.data_min_
```




    array([    0.,    20., 20000.,   100., 20000.])




```python
print(X_scaled[:,0])
```

    [0. 0. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 0. 0. 0. 1. 0. 1. 1. 0. 1. 0. 1. 1.
     0. 0. 0. 0. 1. 1. 1. 1. 1. 0. 0. 0. 1. 1. 0. 1. 0. 1. 0. 1. 1. 0. 1. 0.
     1. 0. 1. 0. 0. 1. 1. 0. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 0.
     0. 1. 1. 0. 1. 1. 0. 0. 1. 1. 1. 0. 1. 1. 1. 0. 1. 1. 1. 1. 1. 0. 1. 1.
     0. 1. 0. 0. 1. 1. 1. 0. 0. 0. 0. 1. 1. 1. 1. 1. 0. 0. 1. 1. 1. 0. 0. 0.
     1. 0. 1. 0. 0. 0. 0. 0. 1. 0. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1. 0. 1. 0. 1.
     0. 0. 1. 0. 1. 1. 1. 1. 0. 0. 1. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 1. 1. 1.
     0. 1. 1. 1. 0. 1. 0. 0. 1. 0. 1. 1. 1. 0. 1. 1. 0. 1. 0. 0. 1. 1. 1. 1.
     0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 1. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 0.
     1. 1. 0. 0. 1. 1. 0. 1. 1. 1. 1. 0. 1. 0. 1. 0. 1. 1. 1. 1. 1. 0. 0. 1.
     0. 1. 0. 1. 1. 0. 1. 0. 1. 0. 0. 0. 1. 1. 0. 1. 1. 1. 1. 0. 0. 0. 1. 1.
     0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0.
     0. 0. 0. 1. 1. 1. 1. 1. 1. 0. 0. 1. 0. 0. 0. 1. 1. 1. 0. 1. 0. 0. 1. 0.
     0. 1. 0. 1. 0. 1. 1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 1. 1. 0. 1. 0. 1. 0. 0.
     0. 1. 0. 1. 1. 1. 0. 0. 1. 1. 0. 0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 1. 0. 1.
     1. 0. 1. 1. 1. 0. 1. 1. 1. 0. 1. 0. 1. 0. 1. 1. 0. 0. 0. 1. 0. 0. 1. 1.
     0. 1. 0. 1. 1. 0. 1. 1. 1. 1. 0. 1. 0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 1.
     0. 1. 0. 1. 1. 0. 0. 1. 1. 0. 1. 1. 0. 0. 1. 0. 1. 0. 1. 0. 0. 0. 0. 0.
     1. 1. 1. 1. 0. 0. 0. 0. 1. 0. 1. 0. 1. 0. 1. 1. 1. 0. 1. 1. 0. 0. 1. 1.
     0. 1. 1. 0. 0. 1. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 1. 1. 0. 1. 0. 1. 1. 0.
     0. 1. 1. 0. 0. 1. 1. 0. 0. 0. 1. 1. 0. 0. 0. 0. 1. 1. 1. 1.]
    


```python
y.shape
```




    (500,)




```python
y = y.values.reshape(-1,1)
```


```python
y.shape
```




    (500, 1)




```python
y_scaled = scaler.fit_transform(y)

```


```python
y_scaled
```




    array([[0.37072477],
           [0.50866938],
           [0.47782689],
           [0.82285018],
           [0.66078116],
           [0.67059152],
           [0.28064374],
           [0.54133778],
           [0.54948752],
           [0.4111198 ],
           [0.70486638],
           [0.46885649],
           [0.27746526],
           [0.56702642],
           [0.57056385],
           [0.61996151],
           [0.46217916],
           [0.49157341],
           [0.50188722],
           [0.64545808],
           [0.59339372],
           [0.48453965],
           [0.53860366],
           [0.53007738],
           [0.50814651],
           [0.49841668],
           [0.3966416 ],
           [0.56467566],
           [0.6950749 ],
           [0.49287831],
           [0.12090943],
           [0.50211776],
           [0.80794216],
           [0.62661214],
           [0.43394857],
           [0.60017103],
           [0.42223485],
           [0.01538345],
           [0.37927499],
           [0.64539707],
           [0.51838974],
           [0.45869677],
           [0.26804521],
           [0.2650104 ],
           [0.84054134],
           [0.84401542],
           [0.35515157],
           [0.406246  ],
           [0.40680623],
           [0.55963883],
           [0.2561583 ],
           [0.77096325],
           [0.55305289],
           [0.5264948 ],
           [0.3236476 ],
           [0.55070832],
           [0.54057623],
           [0.45669016],
           [0.41053254],
           [0.33433524],
           [0.39926954],
           [0.5420261 ],
           [0.57366948],
           [0.43793831],
           [0.46897896],
           [0.61908354],
           [0.55076214],
           [0.48846357],
           [0.61560519],
           [0.56891394],
           [0.30761974],
           [0.56863909],
           [0.46343171],
           [0.50787179],
           [0.61959897],
           [0.59095889],
           [0.45573492],
           [0.49908055],
           [0.4155271 ],
           [0.45382041],
           [0.41407692],
           [0.45713635],
           [0.30133556],
           [0.47019791],
           [0.42256447],
           [0.14863766],
           [0.50939895],
           [0.38056275],
           [0.59067519],
           [0.05486922],
           [0.4219045 ],
           [0.59466257],
           [0.23904164],
           [0.72775122],
           [0.63486085],
           [0.43668833],
           [0.74075381],
           [0.42962519],
           [0.61231998],
           [0.46743215],
           [0.68227386],
           [0.19270023],
           [0.34705263],
           [0.31747576],
           [0.72480623],
           [0.51744133],
           [0.36343414],
           [0.36116341],
           [0.26178477],
           [0.64750739],
           [0.56538749],
           [0.70197943],
           [0.68037083],
           [0.60481233],
           [0.29916352],
           [0.81860421],
           [0.47053558],
           [0.45706646],
           [0.47313925],
           [0.35945319],
           [0.46779854],
           [0.46357095],
           [0.71008706],
           [0.59721993],
           [0.64444254],
           [0.53723155],
           [0.78016463],
           [0.39792327],
           [0.61500514],
           [0.49297475],
           [0.59931944],
           [0.4118826 ],
           [0.43333307],
           [0.43771229],
           [0.33988067],
           [0.55806565],
           [0.54497514],
           [0.42831636],
           [0.344062  ],
           [0.33380674],
           [0.75865395],
           [0.28768775],
           [0.49885366],
           [0.3893741 ],
           [0.62895125],
           [0.52080458],
           [0.41555485],
           [0.54839351],
           [0.72143981],
           [0.42155708],
           [0.26493265],
           [0.54372318],
           [0.46981253],
           [0.31409216],
           [0.46999496],
           [0.32165106],
           [0.24646921],
           [0.41088501],
           [0.42863953],
           [0.40442699],
           [0.67782275],
           [0.52751195],
           [0.49091635],
           [0.65623527],
           [0.47160596],
           [0.44900269],
           [0.16412978],
           [0.37237754],
           [0.38187033],
           [0.41101838],
           [0.45107076],
           [0.26605566],
           [0.48907732],
           [0.68212351],
           [0.45217002],
           [0.56409653],
           [0.45444407],
           [0.78232624],
           [0.28242388],
           [0.3059129 ],
           [0.41919878],
           [0.42720002],
           [0.33251345],
           [0.69078257],
           [0.63925743],
           [0.38927052],
           [0.42988917],
           [0.47856826],
           [0.73050372],
           [0.52648517],
           [0.67013948],
           [0.47569515],
           [0.40675569],
           [0.50997782],
           [0.665203  ],
           [0.58387492],
           [0.57353959],
           [0.31967601],
           [0.56647918],
           [0.52378641],
           [0.38150608],
           [0.48259225],
           [0.44592089],
           [0.60117759],
           [0.50035241],
           [0.55660425],
           [0.51838459],
           [0.6457801 ],
           [0.33068236],
           [0.46773647],
           [0.64966102],
           [0.54907635],
           [0.48459001],
           [0.50109082],
           [0.40485272],
           [0.5465529 ],
           [0.5048829 ],
           [0.53018684],
           [0.66991531],
           [0.46018938],
           [0.73406295],
           [0.39864179],
           [0.53369389],
           [0.66841888],
           [0.51422109],
           [0.26233016],
           [0.52661271],
           [0.28171911],
           [0.5965593 ],
           [0.46494647],
           [0.61485077],
           [0.49905236],
           [0.52189581],
           [0.69345654],
           [0.47873651],
           [0.58735466],
           [0.53534616],
           [0.56901367],
           [0.47883335],
           [0.49908428],
           [0.5257114 ],
           [0.53305254],
           [0.66900144],
           [0.47568675],
           [0.61005611],
           [0.3540794 ],
           [0.72905982],
           [0.80505204],
           [0.31289637],
           [0.523032  ],
           [0.67567861],
           [0.2138602 ],
           [0.56450168],
           [0.39568902],
           [0.4845291 ],
           [0.28298776],
           [0.55421359],
           [0.34170423],
           [0.45531219],
           [0.56811431],
           [0.5972613 ],
           [0.31338011],
           [0.48730944],
           [0.55352142],
           [0.63399262],
           [0.41795296],
           [0.39544824],
           [0.40771621],
           [0.45521229],
           [0.81533714],
           [0.04981603],
           [0.43026944],
           [0.61562087],
           [0.6268025 ],
           [0.60728039],
           [0.41838955],
           [0.54964825],
           [0.71104101],
           [0.37903726],
           [0.45118709],
           [0.60183344],
           [0.62002621],
           [0.33560612],
           [0.28757249],
           [0.6825565 ],
           [0.58368485],
           [0.45880771],
           [0.52693631],
           [0.54380447],
           [0.8715253 ],
           [0.65554063],
           [0.63167261],
           [0.4352791 ],
           [0.50332973],
           [0.5343264 ],
           [0.27381426],
           [0.41053523],
           [0.47531745],
           [0.2911385 ],
           [0.76110147],
           [0.76406707],
           [0.46931782],
           [0.49948317],
           [0.81820046],
           [0.18438195],
           [0.43693203],
           [0.66298048],
           [0.56960734],
           [0.47179899],
           [0.39556023],
           [0.60375334],
           [0.37628608],
           [0.43511174],
           [0.37719946],
           [0.47698891],
           [1.        ],
           [0.72573208],
           [0.71491172],
           [0.43107389],
           [0.69917902],
           [0.61949147],
           [0.58685749],
           [0.71302854],
           [0.19197549],
           [0.4526464 ],
           [0.62671101],
           [0.38794277],
           [0.48597146],
           [0.3119255 ],
           [0.3172683 ],
           [0.31103807],
           [0.51220225],
           [0.22891454],
           [0.43505067],
           [0.60902435],
           [0.43537481],
           [0.51912329],
           [0.30740999],
           [0.42849005],
           [0.36167369],
           [0.20447774],
           [0.27793926],
           [0.70558126],
           [0.58012487],
           [0.3754806 ],
           [0.52673735],
           [0.32804493],
           [0.56449711],
           [0.56829414],
           [0.45672673],
           [0.21439436],
           [0.49893066],
           [0.72380375],
           [0.47597174],
           [0.53430602],
           [0.69953618],
           [0.40905353],
           [0.42634618],
           [0.64234067],
           [0.4237175 ],
           [0.54907212],
           [0.5222931 ],
           [0.30935321],
           [0.37643596],
           [0.56429808],
           [0.56275857],
           [0.39694511],
           [0.53113416],
           [0.61816285],
           [0.29231919],
           [0.73184274],
           [0.4362737 ],
           [0.41613807],
           [0.67273869],
           [0.76168793],
           [0.65775822],
           [0.3854533 ],
           [0.61236387],
           [0.58164331],
           [0.39802597],
           [0.54529522],
           [0.28930803],
           [0.72629843],
           [0.38204913],
           [0.68033622],
           [0.60453629],
           [0.37815253],
           [0.47470876],
           [0.65035196],
           [0.24788603],
           [0.63371047],
           [0.54888405],
           [0.48791067],
           [0.46027877],
           [0.76253593],
           [0.30644588],
           [0.79707352],
           [0.40664378],
           [0.47773971],
           [0.19154167],
           [0.86759109],
           [0.48228989],
           [0.41040247],
           [0.30168732],
           [0.77280198],
           [0.50863303],
           [0.49805458],
           [0.14824364],
           [0.57734658],
           [0.7427002 ],
           [0.4615406 ],
           [0.52679628],
           [0.39967091],
           [0.34882593],
           [0.3051776 ],
           [0.60642838],
           [0.30614901],
           [0.4287247 ],
           [0.41091372],
           [0.44492764],
           [0.74688327],
           [0.5558489 ],
           [0.43795845],
           [0.571387  ],
           [0.31561446],
           [0.54534475],
           [0.37724541],
           [0.47754279],
           [0.55657526],
           [0.51540401],
           [0.32481193],
           [0.32687852],
           [0.37288737],
           [0.28900791],
           [0.6538108 ],
           [0.46675557],
           [0.58506904],
           [0.36510463],
           [0.49152498],
           [0.42443705],
           [0.45278122],
           [0.21316327],
           [0.4747199 ],
           [0.42114943],
           [0.27669569],
           [0.60775236],
           [0.81194719],
           [0.47759537],
           [0.56687473],
           [0.25779114],
           [0.54746234],
           [0.71808681],
           [0.51086565],
           [0.        ],
           [0.52129727],
           [0.33756622],
           [0.56035434],
           [0.51865585],
           [0.43805795],
           [0.3726359 ],
           [0.2895323 ],
           [0.41187412],
           [0.499023  ],
           [0.59220313],
           [0.61366712],
           [0.73808769],
           [0.27413582],
           [0.26177748],
           [0.54900684],
           [0.26992761],
           [0.85449963],
           [0.55003734],
           [0.39949626],
           [0.50001154],
           [0.36815842],
           [0.6502447 ],
           [0.55469987],
           [0.37779655],
           [0.38757338],
           [0.62128001],
           [0.62041473],
           [0.17564949],
           [0.50726309],
           [0.65320953],
           [0.6691568 ],
           [0.54146119],
           [0.45760058],
           [0.33173992],
           [0.46457217],
           [0.7118085 ],
           [0.4556686 ],
           [0.61669253],
           [0.71996731],
           [0.54592485],
           [0.77729956],
           [0.56199216],
           [0.31678049],
           [0.77672238],
           [0.51326977],
           [0.50855247]])



# STEP#4: TRAINING THE MODEL


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)
```


```python
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

model = Sequential()
model.add(Dense(25, input_dim=5, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

```

    C:\Users\GGPC\miniforge3\Lib\site-packages\keras\src\layers\core\dense.py:93: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">25</span>)                  │             <span style="color: #00af00; text-decoration-color: #00af00">150</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">25</span>)                  │             <span style="color: #00af00; text-decoration-color: #00af00">650</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                   │              <span style="color: #00af00; text-decoration-color: #00af00">26</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">826</span> (3.23 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">826</span> (3.23 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
model.compile(optimizer='adam', loss='mean_squared_error')

```


```python
epochs_hist = model.fit(X_train, y_train, epochs=20, batch_size=25,  verbose=1, validation_split=0.2)

```

    Epoch 1/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - loss: 0.0448 - val_loss: 0.0368
    Epoch 2/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0298 - val_loss: 0.0206
    Epoch 3/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0172 - val_loss: 0.0132
    Epoch 4/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0120 - val_loss: 0.0119
    Epoch 5/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0108 - val_loss: 0.0097
    Epoch 6/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0090 - val_loss: 0.0083
    Epoch 7/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0086 - val_loss: 0.0083
    Epoch 8/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0065 - val_loss: 0.0072
    Epoch 9/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0067 - val_loss: 0.0066
    Epoch 10/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0058 - val_loss: 0.0060
    Epoch 11/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0059 - val_loss: 0.0058
    Epoch 12/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0044 - val_loss: 0.0048
    Epoch 13/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0039 - val_loss: 0.0045
    Epoch 14/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0038 - val_loss: 0.0041
    Epoch 15/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0038 - val_loss: 0.0036
    Epoch 16/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0029 - val_loss: 0.0031
    Epoch 17/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0023 - val_loss: 0.0029
    Epoch 18/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0025 - val_loss: 0.0023
    Epoch 19/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0021 - val_loss: 0.0022
    Epoch 20/20
    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0018 - val_loss: 0.0019
    

# STEP#5: EVALUATING THE MODEL 


```python
print(epochs_hist.history.keys())

```

    dict_keys(['loss', 'val_loss'])
    


```python
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])

```




    <matplotlib.legend.Legend at 0x2707c127890>




    
![png](output_31_1.png)
    



```python
# Gender, Age, Annual Salary, Credit Card Debt, Net Worth

X_Testing = np.array([[1, 50, 50000, 10985, 629312]])
```


```python
y_predict = model.predict(X_Testing)
y_predict.shape
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
    




    (1, 1)




```python
print('Expected Purchase Amount=', y_predict[:,0])
```

    Expected Purchase Amount= [225386.47]
    

# EXCELLENT JOB! NOW YOU'VE MASTERED THE USE OF ANNs FOR REGRESSION TASKS! 
