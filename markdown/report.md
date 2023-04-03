---
layout: wide_default
---

# Comprehensive report on Large-Cap Firms' Political Attitude
#### Due Date 3/24/2023 @ 4 P.M.
Brooks Walsh (Brooks377)


**Hypothesis**: Firms that are more in tune with the "political climate" have better immediate returns after filing a 10-K.

## Abstract
This project uses an objective metric for sentiment (sentiment score) to measure the apparent sentiment of 10-K's, and then correlates that score with stock return data around the filing date of the 10-K. When finding sentiment scores, I take note of the difference between a traditional bag-of-words approach and the new machine learning approach. After thorough analysis, it is determined that the machine learning method is more effective for determining sentiment, but more data should be used to verify these results.<br>
The second half of this project finds sentiment values based on special topics to attempt to correlate those sentiment scores with stock returns around the 10-K filing date, in a similar process to the one I just described. The findings of this analysis did not provide enough correlation data to make meaningful predictions about future stock returns. However, there were many conclusions to be drawn from the standard deviations, weak correlations, and means of the resulting scores. Overall, this report does not meaningfully support or rebut my hypothesis, as no strong correlations were found.

# Data

### The Sample
- The sample consists of 488 of the 503 stocks in the current S&P 500
    - There are few enough firms missing that this data will be a great indicator
        - The missing firms are listed below
- Some of the firms in the study were not in the S&P 500 during the period for which we have returns
    - About 20-25 firms are swapped each year
        - Best I could find for historical constituency data was this [link](https://raw.githubusercontent.com/leosmigel/analyzingalpha/master/2019-09-18-sp500-historical-components-and-changes/sp500_history.csv) (Only shows swaps through 2021)
        

### Total Missing Values in Final Dataframe: 15
- Note: It is definitely possible to replace these ticker names to get additional data, but I felt my sample sufficed
1. First Republic Bank (FRC); did not have any 10-K listed on SEC website
2. GE HealthCare (GEHC); Split from GE on January 4 2023, so there is no 2022 10-K
3. Brown-Forman Corporation Class B (BF.B); class B stock, dropped
4. Berkshire Hathaway Inc Class B (BRK.B); class B stock, dropped
5. Ball Corp (BALL); ticker changed from BLL
6. Elevance Health Inc (ELV); ticker changed from ANTM
7. Gen Digital Inc (GEN); ticker changed from NLOK
8. Meta Platforms Inc (META); ticker changed from FB
9. Paramount Global Class B (PARA); ticker changed from VIAC
10. Warner Bros Discovery Inc (WBD); ticker changed from DISCA
11. Welltower Inc (WELL); no discernible reason... best guess is date issue
- 12 to 15 are missing because I used the code found below to find cumulative returns (1+(t+3)) * (1+(t+4)) *...* (1+(t+10)) - 1:
    - This code creates a rolling window, but if there is not a t+10, t+9, etc. return value, the cumulative return will not calculate
12. A
13. AMAT
14. AVGO
15. NDSN

### Calculating Cumulative Return Values
To calculate return values, I used the df_returns.csv file in the build/inputs folder
- This dataset contains 2587065 values originally 

Saving the file this way created an additional index that I had to drop using the following code:
```python
df_returns.drop('Unnamed: 0', axis=1, inplace=True)
```
Next, I filtered the data set to only include return values for the firms in the scope of this report
- Originally it was at this point that I dropped duplicate dates to ensure one return value per day (Which results in losing 1331 returns (or .975% of the data))
    - However, as the professor correctly noted, we did not have enough data to know which return to drop
        - Thankfully, the professor cleaned the data for us, and duplicate dates are no longer an issue; so I removed my attempted fix
```python
df_returns_500_bc = df_returns[df_returns['ticker'].isin(sp500_wDate['Symbol'])]
```
Finally, I used the following lines of code (using groupby/rolling) to calculate the cumulative returns
- (1+t) * (1+(t+1)) * (1+(t+2)) - 1
```python
df_returns_500['return_t_t2'] = (
    df_returns_500.assign(R=1+df_returns_500['ret'])
    .groupby('ticker')['R']
    .rolling(3)
    .apply(np.prod, raw=True)
    .shift(-2)
    .reset_index(level=0, drop=True)
    .sub(1)
)
```
- (1+(t+3)) * (1+(t+4)) *...* (1+(t+10)) - 1
```python
df_returns_500['return_t3_t10'] = (
    df_returns_500.assign(R=1+df_returns_500['ret'])
    .groupby('ticker')['R']
    .rolling(11, min_periods=11)
    .apply(lambda x: np.prod(x[3:]), raw=True)
    .reset_index(level=0, drop=True)
    .sub(1)
    .shift(-10)
)
```

### Sentiment Scores: Getting 10-K's
To calculate general sentiment scores, I am looking to find the 10-K HTML files that correspond with the 488 firms in the sample
- I grabbed the HTML files using the code contained in the build/download_text_files.ipynb notebook file
    - These hmtl files are stored in the following directory as a .zip: build/10k_files/10k_files.zip
        - After fixes, 498 10-K's are downloaded; the only 2 missing do not have a 10-K filed on the SEC website
    
To work around a major problem where many 10-K's were not downloading, and to fix erroneous filings, I used CIK instead of ticker when looping into the SEC EDGAR downloader.
- for example, this code produces the Heico 10-K, when it should produce one for Agilent Technologies
```python
dl.get("10-K", "A", amount=1, after="2022-01-01", before="2022-12-31")
```
Because of the switch to CIK values, I had to drop the duplicate stocks from the 503 in the original sample:
```python
sp500.drop_duplicates(subset='CIK', keep='first', inplace=True)
```

The following is the code that grabs 10-K's from SEC EDGAR and downloads them into folder trees, which I then zip into the desired zip folder
```python
for firm in tqdm(sp500['CIK'].astype(str)):

        symbol = sp500.loc[sp500['CIK'] == int(firm), 'Symbol'].values[0]
        pattern = 'sec-edgar-filings/'+firm.zfill(10)+'/10-K/*/*.html'
        firm_files = fnmatch.filter(file_list, pattern)
                
        if len(firm_files) == 0:
            dl.get("10-K", firm, amount=1, after="2022-01-01", before="2022-12-31")
```

### Sentiment Scores: Getting 10-K Filing Date + Adding Returns
In order to correctly relate 10-K filing's to returns, I will need the filing date of each
- I started by creating lists of Accession numbers and CIK's by using re.search on the file paths:
<br>
```python
# grab accession number from file paths
acc_pattern = r"\d{10}-\d{2}-\d{6}"
acc_num_list = [re.search(acc_pattern, file_name).group() for file_name in file_list]

# grab the CIK from file paths
CIK_pattern = r"\d{10}"
file_CIK_list = [re.search(CIK_pattern, file_name).group() for file_name in file_list]
```

I then used the following for loop to place the CIK and Accession number into the correct place in the url, and pulled 10-K dates using a CSS selector, adding the dates directly to the dataframe:
<br>
```python
for index, row in tqdm(CIK_ACC.iterrows()):
        cik = row["CIK"]
        accession_number = row["Accession"]
        url = f'https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}-index.html'
        r = session.get(url)
        filing_date = r.html.find('div.formContent > div:nth-child(1) > div:nth-child(2)', first= True).text
        CIK_ACC.loc[index, 'Filing_Date'] = filing_date

        sleep(.15) # for SEC timeout
```

To ensure the data is consistent with my desired sample: I removed the 2 stocks with no 10-K filings, then added back the 3 duplicate stocks I dropped earlier using a merge:
```python
# first remove the 2 stocks that did not have 10-K filings
sp500_m2 = sp500[["Symbol", "CIK"]]
sp500_m2 = sp500_m2.loc[~sp500_m2['Symbol'].isin(['FRC', 'GEHC'])]
sp500_m2.reset_index(drop=True, inplace=True)

# then merge it with DF that only has only CIK's with relevant 10-K's attached
CIK_ACC_noTic = CIK_ACC.drop("Symbol", axis=1)
sp500_wDate = sp500_m2.merge(CIK_ACC_noTic, on="CIK")
```

Once my dates are collected, and return variables are measured, I merge the 2 dataframes together:
```python
# merge return data into dataframe with dates
sp500_ret = sp500_wDate.merge(df_returns_500_merge.rename(columns={'ticker':'Symbol', 'date':'Filing_Date'}),
                              how="left",
                              on=['Symbol','Filing_Date'],
                             validate="1:1")
```

### Sentiment Scores: HTML Parsing and Variable Creation
Between the last step and this one, I used multiple tests to find the stocks that did not have enough data to perform analysis on, and I dropped them, resulting in 488 rows of cumulative return variables and dates. Now, I use the CIK's of my desired 488 stocks to calculate different sentiment scores using re.findall and the NEAR_regex() function (scores are standardized by document length):
- an example of what goes through the loop (finding positive sentiment score using LM lexicon):
```python
LM = pd.read_csv('inputs/LM_MasterDictionary_1993-2021.csv')
LM_positive_U = LM.query('Positive > 0')['Word'].to_list()
LM_positive = [elem.lower() for elem in LM_positive_U]
LM_positive = ['(' + '|'.join(LM_positive) + ')']

sentiment_pos_LM = []

for firm in tqdm(sp500_ret_parse['CIK'].astype(str)):

        # get a list of possible files for this firm
        firm_folder = "sec-edgar-filings/" + firm.zfill(10) + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder)
        if len(possible_files) == 0:
            continue

        fpath = possible_files[0]  # the first match is the path to the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")

        # Cleaning the html

        soup = BeautifulSoup(html, "lxml-xml")

        # Delete the hidden XBRL
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        # clean the data for parsing
        soup_txt = soup.get_text().lower()
        soup_txt = re.sub(r'\W',' ',soup_txt)
        soup_txt = re.sub(r'\s+',' ',soup_txt)
        doc_length = len(soup_txt.split())
        LM_pos_regex = len(re.findall(NEAR_regex(LM_positive, partial=False) ,soup_txt))
        sentiment_pos_LM.append((LM_pos_regex/doc_length) * 100)
```

There is only one difference to the code above when I am creating my sentiment scores on special topics; I add an additional element to the list for the regex to see topics in proximity to positive/negative words. Example code is below:
- I will discuss the reasoning for my topics/words in the next section
```python
diversity_var_str = ['(diversity|diverse|racial|races|ethnicity|equitable|inclusion|culture)']
diversity_Pvar_strlist = GHR_positive + diversity_var_str
DIV_pos_regex = len(re.findall(NEAR_regex(diversity_Pvar_strlist, max_words_between=12, partial=False) ,soup_txt))
DIV_pos_list.append((DIV_pos_regex/doc_length) * 10000)
```
To finish creating my output data, I created a new dataframe using the lists of all the different sentiment scores, merged it with the frame that has dates and returns, and saved it to the output folder for analysis:
```python
# merge sentiment score data with sp500_ret_parse
sp500_nfinal = sp500_ret_parse.merge(sent_scores, how="left", on="Symbol", validate="1:1")
# drop unnecessary data
sp500_final = sp500_nfinal.drop(['CIK', 'Accession'], axis=1).copy()
# final result goes to outputs folder
sp500_final.to_csv('outputs/sp500_final.csv', index=False)
```

*Of note: Throughout the mechanical description, I do not mention when I do simple tasks like reset the index, or change formatting; all the relevant code is included in "_build/build_sample.ipynb"*

### Sentiment Scores: Topic Discussion
The three topics I chose for analysis are diversity, politics, and the environment. These are broad terms with lots of connotations, so let me break it down. 

**Diversity**<br>
Totally contradicting what I just said, the topic of diversity in a business setting is actually quite simple to parse, and often has static meaning. For the most part, diversity is mentioned in relation to some kind of "commitment" that a firm makes in general terms to the community. On social media, people give very little credence to these posts, but I was curious if investor sentiment changes depending on the "official stance" of the company on diversity. In choosing the words for this topic, I used generally broad, legal-style language to invoke the most hits on the regex.

**Politics**<br>
So when I say politics, I don't mean who they vote for; instead I am referring to the sentiment surrounding political situations in the US and around the world. Political situations such as new regulations, laws, geopolitical strife, etc.. In the context of this report, I am focusing on whether or not high-cap firms display any generalized sentiment at all towards regulation, and whether investors respond to that sentiment.

**Environment**<br>
*Admittedly, there will be some overlap between Politics and Environment when it comes to Regex hits*; it is due to regulation being a common factor. Because of this, I had to be careful to avoid so much overlap that the data is meaningless when selecting words for this list. I focused on clean energy topics and progressive environmental thinking. In the context of this report, I would like to gauge if firms are talking about the environment only in the context of regulations, or if they sometimes inspire hope in their 10-K's, and how investors react to this.

**Main Focus**<br>
The three topics I selected for this study are not randomly selected, as they all connect on the idea of being in tune with an ever-changing world with increased regulation. Diversity and the environment are becoming increasingly popular things to discuss, and politics is what ties it all together. Based on the results of this study, I am hoping to conclude whether having a positive sentiment towards "seemingly-regulated", trendy but heated topics will have tangible benefits to a firm in the short run, even if the results of that sentiment are never shown.

### Final Analysis Sample


```python
# required imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
# read dataframe from output folder
df = pd.read_csv("../_build/outputs/sp500_final.csv")
df.describe()
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
      <th>return_t_t2</th>
      <th>return_t3_t10</th>
      <th>LM Positive</th>
      <th>LM Negative</th>
      <th>GHR Positive</th>
      <th>GHR Negative</th>
      <th>Diversity P-score</th>
      <th>Diversity N-score</th>
      <th>Politics P-score</th>
      <th>Politics N-score</th>
      <th>Environment P-score</th>
      <th>Environment N-score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>488.000000</td>
      <td>488.000000</td>
      <td>488.000000</td>
      <td>488.000000</td>
      <td>488.000000</td>
      <td>488.000000</td>
      <td>488.000000</td>
      <td>488.000000</td>
      <td>488.000000</td>
      <td>488.000000</td>
      <td>488.000000</td>
      <td>488.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.004316</td>
      <td>-0.008571</td>
      <td>0.498610</td>
      <td>1.588406</td>
      <td>2.391322</td>
      <td>2.591227</td>
      <td>1.152977</td>
      <td>0.739452</td>
      <td>4.127687</td>
      <td>9.261984</td>
      <td>2.260266</td>
      <td>3.751491</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.054755</td>
      <td>0.064650</td>
      <td>0.132205</td>
      <td>0.367764</td>
      <td>0.348096</td>
      <td>0.338930</td>
      <td>0.794027</td>
      <td>0.535682</td>
      <td>1.956397</td>
      <td>3.671565</td>
      <td>2.608593</td>
      <td>4.085639</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.447499</td>
      <td>-0.288483</td>
      <td>0.122637</td>
      <td>0.660856</td>
      <td>0.796599</td>
      <td>0.895284</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.740764</td>
      <td>1.878169</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.025323</td>
      <td>-0.048139</td>
      <td>0.409171</td>
      <td>1.329653</td>
      <td>2.185601</td>
      <td>2.397706</td>
      <td>0.611078</td>
      <td>0.375162</td>
      <td>2.682415</td>
      <td>6.642683</td>
      <td>0.779966</td>
      <td>1.432961</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.001155</td>
      <td>-0.010189</td>
      <td>0.490398</td>
      <td>1.561861</td>
      <td>2.410403</td>
      <td>2.590050</td>
      <td>1.003079</td>
      <td>0.634550</td>
      <td>3.765500</td>
      <td>8.798603</td>
      <td>1.394714</td>
      <td>2.412533</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.029103</td>
      <td>0.029081</td>
      <td>0.565090</td>
      <td>1.782317</td>
      <td>2.604553</td>
      <td>2.780330</td>
      <td>1.477999</td>
      <td>0.961974</td>
      <td>5.287384</td>
      <td>11.119435</td>
      <td>2.640966</td>
      <td>4.124739</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.348567</td>
      <td>0.332299</td>
      <td>1.089875</td>
      <td>3.018505</td>
      <td>3.798221</td>
      <td>3.802972</td>
      <td>5.595524</td>
      <td>3.618076</td>
      <td>10.811932</td>
      <td>24.741341</td>
      <td>19.225879</td>
      <td>28.177610</td>
    </tr>
  </tbody>
</table>
</div>



**Summary Stats Discussion**<br>
Above is the descriptive summary statistics for the final data sample that I am using for analysis. The first four sentiment statistics (LM/GHR positive/negative) are calculated as a percent of the total words in the document that are positive or negative words, so they are a gauge of the overall tone of each 10-K. The remaining six sentiment scores are **NOT** calculated as a percent. Instead these variables are calculated as a score, standardized by document length, that represents the amount of times a topic was discussed positively or negatively (using BHR lexicon). As for the special topic sentiment scores, I am surprised by the magnitude, but not by the results. As an example for proof that these are logical scores: it makes sense that most firms only discuss diversity in a neutral tone, not being overly positive or negative. In comparison, it is extremely common to lose out on opportunities, or realize losses because of regulations (political or environmental), and so the sentiment being negative for both is expected. Additionally, the degree to which companies talk about the environment in general is higher than diversity (environment is a safer topic) but lower than politics/regulation (can't avoid talking about regulation in a firm). 

# Results

## Correlation Data for each Sentiment Score


```python
# Figure 1
df.corr(numeric_only=True).iloc[:2].drop(columns=["return_t_t2","return_t3_t10"])
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
      <th>LM Positive</th>
      <th>LM Negative</th>
      <th>GHR Positive</th>
      <th>GHR Negative</th>
      <th>Diversity P-score</th>
      <th>Diversity N-score</th>
      <th>Politics P-score</th>
      <th>Politics N-score</th>
      <th>Environment P-score</th>
      <th>Environment N-score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>return_t_t2</th>
      <td>-0.081006</td>
      <td>-0.004184</td>
      <td>0.042264</td>
      <td>0.076401</td>
      <td>-0.030131</td>
      <td>-0.002085</td>
      <td>-0.033906</td>
      <td>-0.004680</td>
      <td>0.093036</td>
      <td>0.105852</td>
    </tr>
    <tr>
      <th>return_t3_t10</th>
      <td>-0.044273</td>
      <td>-0.127429</td>
      <td>-0.030688</td>
      <td>0.050327</td>
      <td>-0.065398</td>
      <td>-0.016134</td>
      <td>0.140764</td>
      <td>0.158657</td>
      <td>0.274920</td>
      <td>0.282851</td>
    </tr>
  </tbody>
</table>
</div>



## Correlation Heatmap


```python
# Figure 2
corr = df.drop(columns=['Symbol','Filing_Date']).corr()

fig, ax = plt.subplots(figsize=(9,9))
plt.title("Correlation between Sentiment and Returns")
ax = sns.heatmap(corr, 
                 center=0,square=True,
                 cmap=sns.diverging_palette(230, 20, as_cmap=True),
                 mask=np.triu(np.ones_like(corr, dtype=bool)),
                 cbar_kws={"shrink": .5},
                )
```


    
![png](output_20_0.png)
    


- Interesting to note that each score is most strongly associated with the negative/positive counterpart to that score. This means that some significant overlap was occurring, which I expected would happen.

## Comparing the Correlations of Environment Scores with t3-t10 returns
- Strongest correlations to returns


```python
# Figure 3
f3 = sns.jointplot(data=df,
                  x="Environment P-score", y="return_t3_t10", kind='reg') 
f3.fig.suptitle('Correlation Between Environment P-score and Return')
f3.fig.subplots_adjust(top=0.95)
```


    
![png](output_23_0.png)
    



```python
# figure 4
f4 = sns.jointplot(data=df,
                  x="Environment N-score", y="return_t3_t10", kind='reg') 
f4.fig.suptitle('Correlation Between Environment N-score and Return')
f4.fig.subplots_adjust(top=0.95)
```


    
![png](output_24_0.png)
    


# Discussion Topics

### Compare / contrast the relationship between the returns variable and the two “LM Sentiment” variables (positive and negative) with the relationship between the returns variable and the two “ML Sentiment” variables (positive and negative). Focus on the patterns of the signs of the relationships and the magnitudes.

The LM sentiment variables are created using the list of positive/negative words that is 86,531 words long. Created as a comprehensive and all inclusive list for sentiment analysis, this list of words was created by people and based on a previous list that is criticized and debunked called H4N. ML (GHR) sentiment variables are created the same way as LM variables, except the list of positive/negative words is only 94 words long, and this list was created by a machine learning algorithm. Despite this difference in length, my analysis (Figure 1) shows that the ML (GHR) sentiment variables are better correlated with both windows of returns in nearly all categories. LM variables are using a much longer list of possible words (920 times longer), and yet they are receiving less hits when parsing the 10-K. This would indicate that, for modern 10-K's, the ML (GHR) list of words is a better indicator of stock price.

### If your comparison/contrast conflicts with Table 3 of the Garcia, Hu, and Rohrer paper (ML_JFE.pdf, in the repo), discuss and brainstorm possible reasons why you think the results may differ. If your patterns agree, discuss why you think they bothered to include so many more firms and years and additional controls in their study? (It was more work than we did on this midterm, so why do it to get to the same point?)

The patterns shown in my analysis are congruent with those that appeared in the GHR paper. To quote the paper, "LM sentiment scores are barely associated with the stock price reactions during the release of 10-K statements" (Page 534). All of the correlation data for LM sentiment scores are negative (Figure 1), which would indicate a negative correlation; however, the magnitude of the correlations are minimal, and few conclusions can be drawn besides comparison. For comparison, the ML lists showed a mostly positive correlation with returns. While the magnitude of this correlation is similarly insignificant, the positive correlation, and consistency across time frames makes ML (GHR) the preferred list of words for analysis. So why did they bother do include so many more firms and years and controls? The best answer here is: for safety. While some conclusions can surely be drawn from the data I collected for this report, overall, it requires a larger timeframe and more 10-K's to prove that any significant relationships exist. If the correlation data showed stronger/weaker relationships with less points of data, we would have to question the validity of that report. For the ML (GHR) paper, it is worth noting that they are doing a similar analysis, but using it to predict future stock prices, and prediction is out of the scope of a project/sample the size of mine. The moral is, more data is always better.

### Discuss your 3 “contextual” sentiment measures. Do they have a relationship with returns that looks “different enough” from zero to investigate further? If so, make an economic argument for why sentiment in that context can be value relevant.

**Diversity & Politics**<br>
The diversity and political sentiment scores showed almost no correlation, or actually, a negative correlation. Unfortunately for my analysis, these relationships are essentially random. That being said, there is a significant outlier in terms of correlation and that is political sentiment score's relationship with returns three to ten days out. Interestingly, it is not correlated with whether the firm talks positively or negatively about politics or regulation, but rather whether they mention the situations at all. As is shown in Figure 2 (heat map), it is both the positive and negative sentiment regarding the environment that have positive correlations with returns (t3-t10). One possibility is that firms who are spending more time learning and writing about regulations and the current political climate will be more suited to adapt when regulations arrive.

**The Environment**<br>
The environment sentiment scores show a pretty distinct positive relationship (in comparison with relevant data) when plotted against returns for day 4 to day 11. There was a less distinct, but also notable positive relationship with immediate returns from day 1 to day 3 (Figure 2). One possible reason for these relationships is similar to the reason I stated for political climate; it could be that firms who update themselves regularly to adapt to shifting environmental regulation see better returns when investors see that they are continuing to commit themselves to environmental goals.

**Overall**<br>
All 3 of my special topic sentiment variables do not have a very notable difference between their positive and negative scores. This is partly due to my choice of words in the regex, as part of my goal in this project was to determine whether there were significant differences between the way firms talk about politics/environment and the regulations surrounding those topics. As can be seen in the summary statistics above Figure 1, the negative sentiment scores for both environment and politics had significantly higher standard deviations than their positive counterpart. This is logical and telling, because we would expect firms to be more hesitant to be critical of regulation or say that it is hurting them. This hesitancy would lead to some firms mentioning regulation in negative light significantly more than others, who are more reserved in language (or less impacted). To drive the point home, take notice of the diversity sentiment score's standard deviation. This standard deviation is so much smaller than the other two because firms are likely to say what is expected of them on the topic and nothing more, to avoid being controversial. 

### Is there a difference in the sign and magnitude? Speculate on why or why not.

The ML(GHR) sentiment measures are mostly correlated positively with the return values, but the correlation is very small, and nearly insignificant. The difference in magnitude is extremely insignificant between the two return measures, especially for the list of negative words. The negative word list includes words such as: impacts, affected, and happened. While this list has been proven to be effective, the ambiguity of words such as "happened" could lead to false positives, or overlap between positive and negative sentiment. Despite this lack of clear evidence for a trend, the difference between ML positive and negative scores is significant enough to warrant further exploration. While there isn't time in this report to dive extremely deep on the topic, I will speculate as to why there is such a difference in the return correlation value of positive and negative sentiment, and why it is only for the t+3 7 day returns. One explanation for this difference is, in the perspective of firms, if your 10-K has lots of positive words in it, it doesn't mean much because all firms try to show off a little in 10-K's (at least language-wise). However, firms will tend to avoid saying anything negative about their status unless they absolutely must. So in this case, it is probable that firms who release lots of negative sentiment see their stock price go down, leading to the positive correlation. In contrast, when firms release lots of positive sentiment, investors see it as business as usual, and the stock price wouldn't react.
