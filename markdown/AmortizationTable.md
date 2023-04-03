---
layouth: wide_default
---

```python
# just spend an hour fixing a problem that didn't exist. All because Jan 1 2033 is a weekend... love bdays
#
# INFO:
#
# All the bank loan amort calcs I found rounded the interest payment amount, and then calculate the total payyment 
#  based on the total interest + the principal
#
# TO FIX:
#
# Final x-axis tic set to total payment period amount
#  prof suggestion: try mod(12) for months and add elif's for each periodtype
#
# Arguably the bonus payment type: seems standard.(see below in ADD section)
#
# TO ADD:
# A pie graph that shows total amount paid in interest vs pricipal
#
# add a folder for the imported functions and change the import path
#
# add bar graph monthly payment distribution
#
# show final payoff date in/near graph. Adjusted date for bonus
#
# add bonus payment type (default=by_periods) (options: yearly, one-time)
#  for yearly: add optional bonus_date parameter (default=start_date)
#    bonus_date would act double as the one-time payment date parameter
#

import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import calendar
from statistics import mean
import seaborn as sns
import matplotlib.pyplot as plt

# functions created for this project (should create links if I ever share this)
from Required_Functions.validate_date import validate_date
from Required_Functions.business_days import business_days
from Required_Functions.cumulative_values import cumulative_values


def loan_amortization(principal, interest_rate,  term_years, start_date, periodtype="M-30", bonus=0, PLOT=False):
    
    """ 
    Args:
        principal (int or float): The principal amount of the loan.
        interest_rate (int or float): The annual interest rate, expressed as a percent (e.g. 5% = 5).
        term_years (int or float): Loan/Borrowing term in years.
        start_date (str or datetime): The start date of the loan as a string in 'YYYY-MM-DD' format or as a datetime object.
        periodtype (str, default="M-30"): The type of period to use for the loan payments, which can be one of the following:
            'D' (daily)
            'bdays' (daily, only includes business days)
            'W' (weekly)
            'BW' (biweekly)
            'M-30' (months where there is 30 days per month and 360 days per year (30/360))
            'M-Actual' (months where months' lengths are accurate, and there are 360 days per year (Actual/360))
            'Q' (quarterly)
            'S' (semi-annual)
            'Y' (Annual)
        bonus= (int or float, default=0): Optional, additional principal paid per period.
        PLOT= (Bool, default=False): With PLOT set to True, the function will create a folder in the cwd
                                    and download the loan amortization graph as a .png file.
                                - The .png file will have the following naming structure:
                                    - /Loan_Graphs/'Principal_Rate_TermYears_StartDate_PeriodType_bonus.png'
    Returns:
        pandas.DataFrame: A DataFrame containing the amortization schedule for the loan
    """
    # input validation for start_date
    if validate_date(start_date) is False:
        raise TypeError("start_date must be a string in 'YYYY-MM-DD' format or a datetime object")

    # if the date is in the string format, convert it
    if not isinstance(start_date, datetime):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')

    # input type checking for principal, interest_rate, term_years, and bonus
    if not isinstance(principal, (int, float)):
        raise TypeError("Principal amount should be numeric (int or float)")
    if not isinstance(interest_rate, (int, float)):
        raise TypeError("Interest rate should be numeric and in % (int or float)")
    if not isinstance(term_years, (int, float)):
        raise TypeError("term_years should be numeric (int or float)")
    if not isinstance(bonus, (int, float)):
        raise TypeError("bonus should be numeric (int or float)")
    if bonus < 0:
        raise TypeError("bonus should be a positive integer")

    # shift the day forward one when using Daily to assume no payment is made today
    if periodtype == "D":
        start_date = start_date + timedelta(days=1)

    # create end date of term using term_years
    end_date = start_date + relativedelta(years=term_years)

    # create a list of business days for the bday index
    bdays_dates = business_days(start_date, end_date)

    # create a list of weeks for weekly and biweekly index, starting at second week
    week_range = pd.date_range(start=start_date + pd.Timedelta(weeks=1) + pd.Timedelta(days=1), periods=52*term_years, freq='W')
    week_list = [f'{date.week}-{date.year}' for date in week_range]

    # force start and end date to first day of month for month indexing (luv u feb)
    start_date_first = datetime(start_date.year, start_date.month, 1)
    end_date_first = datetime(end_date.year, end_date.month, 1)

    # shift start_date_first forward one month
    if start_date_first.month == 12:
        # handle special case where the month is December
        new_month = 1
        new_year = start_date_first.year + 1
    else:
        new_month = start_date_first.month + 1
        new_year = start_date_first.year
    start_date_p1 = start_date_first.replace(year=new_year, month=new_month)
    # shift end_date_first forward one month
    if end_date_first.month == 12:
        # handle special case where the month is December
        new_month = 1
        new_year = end_date_first.year + 1
    else:
        new_month = end_date_first.month + 1
        new_year = end_date_first.year
    end_date_first_p1 = end_date_first.replace(year=new_year, month=new_month)

    # create month_dates index
    month_dates = [start_date_p1.strftime("%m""-""%Y")]
    month_dates_4D = [start_date_p1]
    current_date = start_date_p1
    while current_date < end_date_first_p1:
        current_date += relativedelta(months=1)
        month_dates_4D.append(current_date)
        month_dates.append(current_date.strftime("%m""-""%Y"))
    # remove last month because 1 is start_date
    month_dates.pop()
    month_dates_4D.pop()

    # create list of days in the month of each date
    days_in_month = [calendar.monthrange(date.year, date.month)[1] for date in month_dates_4D]

    # period-type definition
    if periodtype == 'D':
        periods = int((end_date - start_date).days)
        adjusted_rate = interest_rate / 36525
    elif periodtype == 'bdays':
        periods = len(bdays_dates)
        adjusted_rate = interest_rate / 26100
    elif periodtype == 'W':
        periods = int(52 * term_years)
        adjusted_rate = interest_rate / 5200
    elif periodtype == 'BW':
        periods = int((52 * term_years) / 2)
        adjusted_rate = interest_rate / 2600
    elif periodtype == 'M-30':
        periods = int(12 * term_years)
        adjusted_rate = interest_rate / 1200
    elif periodtype == 'M-Actual':
        periods = int(12 * term_years)
        monthly_rate = [interest_rate / 36000 * days_in_month[i] for i in range(len(month_dates))]
        adjusted_rate = mean(monthly_rate)
    elif periodtype == 'Y':
        periods = term_years
        adjusted_rate = interest_rate / 100
    elif periodtype == 'S':
        periods = int(len(month_dates[1::6]))
        adjusted_rate = interest_rate / 200
    elif periodtype == 'Q':
        periods = int(len(month_dates[1::3]))
        adjusted_rate = interest_rate / 400
    else:
        raise TypeError("periodtype should be one of the following: 'D', 'W', 'BW', 'bdays', 'M-30', 'M-Actual', 'Q', 'S', 'Y'")

    # find payment amount
    monthly_payment = (principal * adjusted_rate / (1 - (1 + adjusted_rate) ** (-periods)))
    monthly_payment_fmt = "{:,.2f}".format(monthly_payment)
    monthly_for_plot = f"{monthly_payment_fmt} + {bonus}"
    actual_payment = (principal * adjusted_rate / (1 - (1 + adjusted_rate) ** (-periods))) + bonus

    # create a list of dates for each payment
    if periodtype == 'M-Actual' or periodtype == 'M-30':
        payment_dates = month_dates
    elif periodtype == 'Y':
        payment_dates = [(start_date + relativedelta(years=1 * i)).year for i in range(periods)]
    elif periodtype == 'bdays':
        payment_dates = bdays_dates
    elif periodtype == 'W':
        payment_dates = week_list
    elif periodtype == 'BW':
        payment_dates = week_list[::2]
    elif periodtype == 'S':
        month_dates.insert(0, start_date.strftime("%m""-""%Y"))
        payment_dates = month_dates[:-6:6]
    elif periodtype == 'Q':
        payment_dates = month_dates[1::3]
    else:
        payment_dates = [start_date + relativedelta(days=(i)) for i in range(periods)]

    # lists for the payment number, payment amount, interest, principal, and balance
    payment_number = list(range(1, periods + 1))
    payment_amount = [actual_payment] * periods
    interest = []
    principal_paid = []
    beg_balance = [principal]
    end_balance = []
    pct_interest = []
    pct_principal = []
    bonus_list = [bonus] * periods

    # interest, principal, and balance for each payment period (exlcuding M-actual)
    if not periodtype == "M-Actual":
        for i in range(periods):
            interest.append(beg_balance[i] * adjusted_rate)
            principal_paid.append((monthly_payment) - interest[i])
            beg_balance.append(beg_balance[i] - principal_paid[i] - bonus)
            end_balance.append(beg_balance[i] - principal_paid[i] - bonus)
            pct_interest.append((interest[i] / payment_amount[i]) * 100)
            pct_principal.append(((principal_paid[i] + bonus) / payment_amount[i]) * 100)
    elif periodtype == "M-Actual":
        for i in range(periods):
            interest.append((beg_balance[i] * monthly_rate[i]))
            principal_paid.append((monthly_payment) - interest[i])
            beg_balance.append(beg_balance[i] - principal_paid[i] - bonus)
            end_balance.append(beg_balance[i] - principal_paid[i] - bonus)
            pct_interest.append((interest[i] / payment_amount[i]) * 100)
            pct_principal.append(((principal_paid[i] + bonus) / payment_amount[i]) * 100)
        principal_paid[-1] = beg_balance[-2]
        payment_amount[-1] = principal_paid[-1] + interest[-1]
        end_balance[-1] = 0

    # if bonus > 0: do fake amortization without bonus for calc of interest saved
    if bonus > 0:
        interest2 = []
        principal_paid2 = []
        beg_balance2 = [principal]
        end_balance2 = []
        if not periodtype == "M-Actual":
            for i in range(periods):
                interest2.append(beg_balance2[i] * adjusted_rate)
                principal_paid2.append((monthly_payment) - interest2[i])
                beg_balance2.append(beg_balance2[i] - principal_paid2[i])
                end_balance2.append(beg_balance2[i] - principal_paid2[i])
        elif periodtype == "M-Actual":
            for i in range(periods):
                interest2.append((beg_balance2[i] * monthly_rate[i]))
                principal_paid2.append((monthly_payment) - interest2[i])
                beg_balance2.append(beg_balance2[i] - principal_paid2[i])
                end_balance2.append(beg_balance2[i] - principal_paid2[i])

    # make the amortization-schedule dataframe
    data = {
        'Payment Number': payment_number,
        'Payment Date': payment_dates,
        'Beginning Balance': beg_balance[:-1],
        'Payment Amount': payment_amount,
        'Bonus': bonus_list,
        'Interest Paid': interest,
        'Principal Paid': principal_paid,
        'Ending Balance': end_balance,
        '% Paid In Interest': pct_interest,
        '% Paid To Principal': pct_principal
    }
    # dataframe creation
    df = pd.DataFrame(data)

    # truncate df with bonus
    if bonus > 0:
        index_balance = (df['Ending Balance'] <= 0).idxmax()
        df = df.iloc[:index_balance + 1]
        df["Principal Paid"].iloc[-1] = df["Beginning Balance"].iloc[-1]
        df["Payment Amount"].iloc[-1] = df["Principal Paid"].iloc[-1] + df["Interest Paid"].iloc[-1]
        df["Bonus"].iloc[-1] = 0
        df["Ending Balance"].iloc[-1] = 0
        periods_b4save = periods
        periods = int(len(df.index))
        # find amount saved by extra payment
        amount_saved_nfmt = sum(interest2) - df["Interest Paid"].sum()
        amount_saved = "{:,.2f}".format(amount_saved_nfmt)
        # find periods saved
        periods_saved = periods_b4save - periods

    # create stats for plot
    # create total interest ****
    total_interest_nfmt = df["Interest Paid"].sum()
    total_interest = "{:,.2f}".format(total_interest_nfmt)

    # create total payment
    total_payment_nfmt = total_interest_nfmt + principal
    total_payment = "{:,.2f}".format(total_payment_nfmt)

    # format data for graph
    if bonus > 0:
        start_value = 0
        loan_balance_list = df["Ending Balance"].tolist()
        loan_balance = loan_balance_list.copy()
        loan_balance.insert(0, principal)
        interest_list = df["Interest Paid"].tolist()
        cumulative_interest_list = cumulative_values(interest_list)
        cumulative_interest = cumulative_interest_list.copy()
        cumulative_interest.insert(0, start_value)
        principal_paid_list = df["Principal Paid"].tolist()
        principal_paid_plot = [x + bonus if i < len(principal_paid_list)-1 else x for i, x in enumerate(principal_paid_list)] 
        cumulative_principal_list = cumulative_values(principal_paid_plot)
        cumulative_principal = cumulative_principal_list.copy()
        cumulative_principal.insert(0, start_value)
    else:
        start_value = 0
        loan_balance = end_balance.copy()
        loan_balance.insert(0, principal)
        cumulative_interest_list = cumulative_values(interest)
        cumulative_interest = cumulative_interest_list.copy()
        cumulative_interest.insert(0, start_value)
        cumulative_principal_list = cumulative_values(principal_paid)
        cumulative_principal = cumulative_principal_list.copy()
        cumulative_principal.insert(0, start_value)

    # set index to dates
    df.set_index('Payment Date', inplace=True)
    if periodtype == "M-Actual" or periodtype == "M-30" or periodtype == "Q" or periodtype == "S":
        df.index.name = "Payment Month"
    elif periodtype == "W" or periodtype == "BW":
        df.index.name = "Payment Week"
    elif periodtype == "Y":
        df.index.name = "Payment Year"
    else:
        df.index.name = 'Payment Date'

    # apply formating for dollar signs and two decimals (new df to retain old format)
    df['Payment Amount'] = df['Payment Amount'].apply(lambda x: '${:,.2f}'.format(x))
    df['Interest Paid'] = df['Interest Paid'].apply(lambda x: '${:,.2f}'.format(x))
    df['Principal Paid'] = df['Principal Paid'].apply(lambda x: '${:,.2f}'.format(x))
    df['Beginning Balance'] = df['Beginning Balance'].apply(lambda x: '${:,.2f}'.format(x))
    df['Ending Balance'] = df['Ending Balance'].apply(lambda x: '${:,.2f}'.format(x))
    df['% Paid In Interest'] = df['% Paid In Interest'].apply(lambda x: '{:,.3f}%'.format(x))
    df['% Paid To Principal'] = df['% Paid To Principal'].apply(lambda x: '{:,.3f}%'.format(x))

    plot_data = {
        'Loan Balance': loan_balance,
        'Cumulative Interest': cumulative_interest,
        'Principal Paid': cumulative_principal
    }

    # make plot
    plot = sns.lineplot(data=plot_data)

    # tweak visual aspects of plot
    plot.set_title(f"Loan Amortization Graph ({periodtype})")
    plot.set_xlabel("Payment Number")
    plot.set_ylabel("Amount (in Dollars)")
    plt.grid(True)
    plot.set_xlim(0, periods)
    plot.set_ylim(0)

    # change line color
    lines = plot.lines
    lines[0].set(color='blue', linestyle='-')
    lines[1].set(color='red', linestyle='-')
    lines[2].set(color='green', linestyle='-')

    # make sure legend matches line color
    ax = plot.axes
    handles, labels = ax.get_legend_handles_labels()
    handles[0].set(color='blue', linestyle='-')
    handles[1].set(color='red', linestyle='-')
    handles[2].set(color='green', linestyle='-')
    ax.legend(handles=handles, labels=labels, loc='center left')

    # add the total stats as annotations
    plt.annotate(f'Total Cost of Loan: ${total_payment}', xy=((periods * .3), (principal)), xytext=((periods * .279), (ax.get_ylim()[1] - (ax.get_ylim()[1] * .045))), bbox=dict(facecolor='white', boxstyle='round'))
    plt.annotate(f'Total Interest Paid: ${total_interest}', xy=((periods * .3), (principal)), xytext=((periods * .2815), (ax.get_ylim()[1] - (ax.get_ylim()[1] * .11))), bbox=dict(facecolor='white', boxstyle='round'))
    plt.annotate(f'Payment: ${monthly_for_plot}', xy=((periods * .3), (principal)), xytext=((periods * .31), (ax.get_ylim()[1] - (ax.get_ylim()[1] * .175))), bbox=dict(facecolor='white', boxstyle='round'))

    # add addition annotations if there is bonus
    if bonus > 0:
        plt.annotate(f'Interest Saved w/ Bonus: ${amount_saved}', xy=((periods * .3), (principal)), xytext=(0 - (periods * .015) , (ax.get_ylim()[1] * 1.095)), bbox=dict(facecolor='white', boxstyle='round'))
        plt.annotate(f'Periods Saved w/ Bonus: {periods_saved}', xy=((periods * .3), (principal)), xytext=(periods - (periods * .365) , (ax.get_ylim()[1] * 1.095)), bbox=dict(facecolor='white', boxstyle='round'))

    # tighen layout of plot for saving
    plt.tight_layout()

    # save plot to graphs folder if PLOT=True
    if PLOT is True:
        if not os.path.exists('loan_graphs'):
            os.makedirs('loan_graphs')
        # save plot with filename based on input parameters
        plot_filename = f"loan_graphs/{principal}_{interest_rate}_{term_years}_{start_date.date()}_{periodtype}_bonus{bonus}.png"
        if not os.path.isfile(plot_filename):  # ensure plot doesn't attmept to save twice
            plt.savefig(plot_filename)

    return df
```


```python
schedule = loan_amortization(200000, 3.5, 30, "2023-1-1", PLOT=True)

schedule
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
      <th>Payment Number</th>
      <th>Beginning Balance</th>
      <th>Payment Amount</th>
      <th>Bonus</th>
      <th>Interest Paid</th>
      <th>Principal Paid</th>
      <th>Ending Balance</th>
      <th>% Paid In Interest</th>
      <th>% Paid To Principal</th>
    </tr>
    <tr>
      <th>Payment Month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>02-2023</th>
      <td>1</td>
      <td>$200,000.00</td>
      <td>$898.09</td>
      <td>0</td>
      <td>$583.33</td>
      <td>$314.76</td>
      <td>$199,685.24</td>
      <td>64.953%</td>
      <td>35.047%</td>
    </tr>
    <tr>
      <th>03-2023</th>
      <td>2</td>
      <td>$199,685.24</td>
      <td>$898.09</td>
      <td>0</td>
      <td>$582.42</td>
      <td>$315.67</td>
      <td>$199,369.57</td>
      <td>64.850%</td>
      <td>35.150%</td>
    </tr>
    <tr>
      <th>04-2023</th>
      <td>3</td>
      <td>$199,369.57</td>
      <td>$898.09</td>
      <td>0</td>
      <td>$581.49</td>
      <td>$316.59</td>
      <td>$199,052.98</td>
      <td>64.748%</td>
      <td>35.252%</td>
    </tr>
    <tr>
      <th>05-2023</th>
      <td>4</td>
      <td>$199,052.98</td>
      <td>$898.09</td>
      <td>0</td>
      <td>$580.57</td>
      <td>$317.52</td>
      <td>$198,735.46</td>
      <td>64.645%</td>
      <td>35.355%</td>
    </tr>
    <tr>
      <th>06-2023</th>
      <td>5</td>
      <td>$198,735.46</td>
      <td>$898.09</td>
      <td>0</td>
      <td>$579.65</td>
      <td>$318.44</td>
      <td>$198,417.01</td>
      <td>64.542%</td>
      <td>35.458%</td>
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
      <th>09-2052</th>
      <td>356</td>
      <td>$4,451.42</td>
      <td>$898.09</td>
      <td>0</td>
      <td>$12.98</td>
      <td>$885.11</td>
      <td>$3,566.32</td>
      <td>1.446%</td>
      <td>98.554%</td>
    </tr>
    <tr>
      <th>10-2052</th>
      <td>357</td>
      <td>$3,566.32</td>
      <td>$898.09</td>
      <td>0</td>
      <td>$10.40</td>
      <td>$887.69</td>
      <td>$2,678.63</td>
      <td>1.158%</td>
      <td>98.842%</td>
    </tr>
    <tr>
      <th>11-2052</th>
      <td>358</td>
      <td>$2,678.63</td>
      <td>$898.09</td>
      <td>0</td>
      <td>$7.81</td>
      <td>$890.28</td>
      <td>$1,788.35</td>
      <td>0.870%</td>
      <td>99.130%</td>
    </tr>
    <tr>
      <th>12-2052</th>
      <td>359</td>
      <td>$1,788.35</td>
      <td>$898.09</td>
      <td>0</td>
      <td>$5.22</td>
      <td>$892.87</td>
      <td>$895.48</td>
      <td>0.581%</td>
      <td>99.419%</td>
    </tr>
    <tr>
      <th>01-2053</th>
      <td>360</td>
      <td>$895.48</td>
      <td>$898.09</td>
      <td>0</td>
      <td>$2.61</td>
      <td>$895.48</td>
      <td>$0.00</td>
      <td>0.291%</td>
      <td>99.709%</td>
    </tr>
  </tbody>
</table>
<p>360 rows × 9 columns</p>
</div>




    
![png](output_1_1.png)
    



```python
schedule = loan_amortization(200000, 3.5, 30, "2023-1-1", bonus=200, PLOT=True)

schedule
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
      <th>Payment Number</th>
      <th>Beginning Balance</th>
      <th>Payment Amount</th>
      <th>Bonus</th>
      <th>Interest Paid</th>
      <th>Principal Paid</th>
      <th>Ending Balance</th>
      <th>% Paid In Interest</th>
      <th>% Paid To Principal</th>
    </tr>
    <tr>
      <th>Payment Month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>02-2023</th>
      <td>1</td>
      <td>$200,000.00</td>
      <td>$1,098.09</td>
      <td>200</td>
      <td>$583.33</td>
      <td>$314.76</td>
      <td>$199,485.24</td>
      <td>53.123%</td>
      <td>46.877%</td>
    </tr>
    <tr>
      <th>03-2023</th>
      <td>2</td>
      <td>$199,485.24</td>
      <td>$1,098.09</td>
      <td>200</td>
      <td>$581.83</td>
      <td>$316.26</td>
      <td>$198,968.99</td>
      <td>52.986%</td>
      <td>47.014%</td>
    </tr>
    <tr>
      <th>04-2023</th>
      <td>3</td>
      <td>$198,968.99</td>
      <td>$1,098.09</td>
      <td>200</td>
      <td>$580.33</td>
      <td>$317.76</td>
      <td>$198,451.22</td>
      <td>52.849%</td>
      <td>47.151%</td>
    </tr>
    <tr>
      <th>05-2023</th>
      <td>4</td>
      <td>$198,451.22</td>
      <td>$1,098.09</td>
      <td>200</td>
      <td>$578.82</td>
      <td>$319.27</td>
      <td>$197,931.95</td>
      <td>52.711%</td>
      <td>47.289%</td>
    </tr>
    <tr>
      <th>06-2023</th>
      <td>5</td>
      <td>$197,931.95</td>
      <td>$1,098.09</td>
      <td>200</td>
      <td>$577.30</td>
      <td>$320.79</td>
      <td>$197,411.16</td>
      <td>52.573%</td>
      <td>47.427%</td>
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
      <th>06-2044</th>
      <td>257</td>
      <td>$4,510.96</td>
      <td>$1,098.09</td>
      <td>200</td>
      <td>$13.16</td>
      <td>$884.93</td>
      <td>$3,426.03</td>
      <td>1.198%</td>
      <td>98.802%</td>
    </tr>
    <tr>
      <th>07-2044</th>
      <td>258</td>
      <td>$3,426.03</td>
      <td>$1,098.09</td>
      <td>200</td>
      <td>$9.99</td>
      <td>$888.10</td>
      <td>$2,337.93</td>
      <td>0.910%</td>
      <td>99.090%</td>
    </tr>
    <tr>
      <th>08-2044</th>
      <td>259</td>
      <td>$2,337.93</td>
      <td>$1,098.09</td>
      <td>200</td>
      <td>$6.82</td>
      <td>$891.27</td>
      <td>$1,246.66</td>
      <td>0.621%</td>
      <td>99.379%</td>
    </tr>
    <tr>
      <th>09-2044</th>
      <td>260</td>
      <td>$1,246.66</td>
      <td>$1,098.09</td>
      <td>200</td>
      <td>$3.64</td>
      <td>$894.45</td>
      <td>$152.20</td>
      <td>0.331%</td>
      <td>99.669%</td>
    </tr>
    <tr>
      <th>10-2044</th>
      <td>261</td>
      <td>$152.20</td>
      <td>$152.65</td>
      <td>0</td>
      <td>$0.44</td>
      <td>$152.20</td>
      <td>$0.00</td>
      <td>0.040%</td>
      <td>99.960%</td>
    </tr>
  </tbody>
</table>
<p>261 rows × 9 columns</p>
</div>




    
![png](output_2_1.png)
    



```python
schedule = loan_amortization(200000, 3.5, 30, "2023-1-1", 'M-Actual')

schedule
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
      <th>Payment Number</th>
      <th>Beginning Balance</th>
      <th>Payment Amount</th>
      <th>Bonus</th>
      <th>Interest Paid</th>
      <th>Principal Paid</th>
      <th>Ending Balance</th>
      <th>% Paid In Interest</th>
      <th>% Paid To Principal</th>
    </tr>
    <tr>
      <th>Payment Month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>02-2023</th>
      <td>1</td>
      <td>$200,000.00</td>
      <td>$903.82</td>
      <td>0</td>
      <td>$544.44</td>
      <td>$359.37</td>
      <td>$199,640.63</td>
      <td>60.238%</td>
      <td>39.762%</td>
    </tr>
    <tr>
      <th>03-2023</th>
      <td>2</td>
      <td>$199,640.63</td>
      <td>$903.82</td>
      <td>0</td>
      <td>$601.69</td>
      <td>$302.12</td>
      <td>$199,338.51</td>
      <td>66.573%</td>
      <td>33.427%</td>
    </tr>
    <tr>
      <th>04-2023</th>
      <td>3</td>
      <td>$199,338.51</td>
      <td>$903.82</td>
      <td>0</td>
      <td>$581.40</td>
      <td>$322.41</td>
      <td>$199,016.10</td>
      <td>64.328%</td>
      <td>35.672%</td>
    </tr>
    <tr>
      <th>05-2023</th>
      <td>4</td>
      <td>$199,016.10</td>
      <td>$903.82</td>
      <td>0</td>
      <td>$599.81</td>
      <td>$304.00</td>
      <td>$198,712.09</td>
      <td>66.364%</td>
      <td>33.636%</td>
    </tr>
    <tr>
      <th>06-2023</th>
      <td>5</td>
      <td>$198,712.09</td>
      <td>$903.82</td>
      <td>0</td>
      <td>$579.58</td>
      <td>$324.24</td>
      <td>$198,387.85</td>
      <td>64.126%</td>
      <td>35.874%</td>
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
      <th>09-2052</th>
      <td>356</td>
      <td>$4,404.16</td>
      <td>$903.82</td>
      <td>0</td>
      <td>$12.85</td>
      <td>$890.97</td>
      <td>$3,513.19</td>
      <td>1.421%</td>
      <td>98.579%</td>
    </tr>
    <tr>
      <th>10-2052</th>
      <td>357</td>
      <td>$3,513.19</td>
      <td>$903.82</td>
      <td>0</td>
      <td>$10.59</td>
      <td>$893.23</td>
      <td>$2,619.96</td>
      <td>1.172%</td>
      <td>98.828%</td>
    </tr>
    <tr>
      <th>11-2052</th>
      <td>358</td>
      <td>$2,619.96</td>
      <td>$903.82</td>
      <td>0</td>
      <td>$7.64</td>
      <td>$896.17</td>
      <td>$1,723.79</td>
      <td>0.845%</td>
      <td>99.155%</td>
    </tr>
    <tr>
      <th>12-2052</th>
      <td>359</td>
      <td>$1,723.79</td>
      <td>$903.82</td>
      <td>0</td>
      <td>$5.20</td>
      <td>$898.62</td>
      <td>$825.17</td>
      <td>0.575%</td>
      <td>99.425%</td>
    </tr>
    <tr>
      <th>01-2053</th>
      <td>360</td>
      <td>$825.17</td>
      <td>$827.65</td>
      <td>0</td>
      <td>$2.49</td>
      <td>$825.17</td>
      <td>$0.00</td>
      <td>0.275%</td>
      <td>99.725%</td>
    </tr>
  </tbody>
</table>
<p>360 rows × 9 columns</p>
</div>




    
![png](output_3_1.png)
    



```python
schedule = loan_amortization(200000, 3.5, 30, "2023-1-1", 'M-Actual', bonus=200)

schedule
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
      <th>Payment Number</th>
      <th>Beginning Balance</th>
      <th>Payment Amount</th>
      <th>Bonus</th>
      <th>Interest Paid</th>
      <th>Principal Paid</th>
      <th>Ending Balance</th>
      <th>% Paid In Interest</th>
      <th>% Paid To Principal</th>
    </tr>
    <tr>
      <th>Payment Month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>02-2023</th>
      <td>1</td>
      <td>$200,000.00</td>
      <td>$1,103.82</td>
      <td>200</td>
      <td>$544.44</td>
      <td>$359.37</td>
      <td>$199,440.63</td>
      <td>49.324%</td>
      <td>50.676%</td>
    </tr>
    <tr>
      <th>03-2023</th>
      <td>2</td>
      <td>$199,440.63</td>
      <td>$1,103.82</td>
      <td>200</td>
      <td>$601.09</td>
      <td>$302.72</td>
      <td>$198,937.91</td>
      <td>54.456%</td>
      <td>45.544%</td>
    </tr>
    <tr>
      <th>04-2023</th>
      <td>3</td>
      <td>$198,937.91</td>
      <td>$1,103.82</td>
      <td>200</td>
      <td>$580.24</td>
      <td>$323.58</td>
      <td>$198,414.33</td>
      <td>52.566%</td>
      <td>47.434%</td>
    </tr>
    <tr>
      <th>05-2023</th>
      <td>4</td>
      <td>$198,414.33</td>
      <td>$1,103.82</td>
      <td>200</td>
      <td>$598.00</td>
      <td>$305.82</td>
      <td>$197,908.51</td>
      <td>54.176%</td>
      <td>45.824%</td>
    </tr>
    <tr>
      <th>06-2023</th>
      <td>5</td>
      <td>$197,908.51</td>
      <td>$1,103.82</td>
      <td>200</td>
      <td>$577.23</td>
      <td>$326.58</td>
      <td>$197,381.93</td>
      <td>52.294%</td>
      <td>47.706%</td>
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
      <th>05-2044</th>
      <td>256</td>
      <td>$5,420.72</td>
      <td>$1,103.82</td>
      <td>200</td>
      <td>$16.34</td>
      <td>$887.48</td>
      <td>$4,333.24</td>
      <td>1.480%</td>
      <td>98.520%</td>
    </tr>
    <tr>
      <th>06-2044</th>
      <td>257</td>
      <td>$4,333.24</td>
      <td>$1,103.82</td>
      <td>200</td>
      <td>$12.64</td>
      <td>$891.18</td>
      <td>$3,242.06</td>
      <td>1.145%</td>
      <td>98.855%</td>
    </tr>
    <tr>
      <th>07-2044</th>
      <td>258</td>
      <td>$3,242.06</td>
      <td>$1,103.82</td>
      <td>200</td>
      <td>$9.77</td>
      <td>$894.04</td>
      <td>$2,148.02</td>
      <td>0.885%</td>
      <td>99.115%</td>
    </tr>
    <tr>
      <th>08-2044</th>
      <td>259</td>
      <td>$2,148.02</td>
      <td>$1,103.82</td>
      <td>200</td>
      <td>$6.47</td>
      <td>$897.34</td>
      <td>$1,050.68</td>
      <td>0.587%</td>
      <td>99.413%</td>
    </tr>
    <tr>
      <th>09-2044</th>
      <td>260</td>
      <td>$1,050.68</td>
      <td>$1,053.74</td>
      <td>0</td>
      <td>$3.06</td>
      <td>$1,050.68</td>
      <td>$0.00</td>
      <td>0.278%</td>
      <td>99.722%</td>
    </tr>
  </tbody>
</table>
<p>260 rows × 9 columns</p>
</div>




    
![png](output_4_1.png)
    



```python
schedule = loan_amortization(200000, 6, 2, "2023-1-1", 'D')

schedule
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
      <th>Payment Number</th>
      <th>Beginning Balance</th>
      <th>Payment Amount</th>
      <th>Bonus</th>
      <th>Interest Paid</th>
      <th>Principal Paid</th>
      <th>Ending Balance</th>
      <th>% Paid In Interest</th>
      <th>% Paid To Principal</th>
    </tr>
    <tr>
      <th>Payment Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-01-02</th>
      <td>1</td>
      <td>$200,000.00</td>
      <td>$290.38</td>
      <td>0</td>
      <td>$32.85</td>
      <td>$257.52</td>
      <td>$199,742.48</td>
      <td>11.314%</td>
      <td>88.686%</td>
    </tr>
    <tr>
      <th>2023-01-03</th>
      <td>2</td>
      <td>$199,742.48</td>
      <td>$290.38</td>
      <td>0</td>
      <td>$32.81</td>
      <td>$257.56</td>
      <td>$199,484.91</td>
      <td>11.300%</td>
      <td>88.700%</td>
    </tr>
    <tr>
      <th>2023-01-04</th>
      <td>3</td>
      <td>$199,484.91</td>
      <td>$290.38</td>
      <td>0</td>
      <td>$32.77</td>
      <td>$257.61</td>
      <td>$199,227.31</td>
      <td>11.285%</td>
      <td>88.715%</td>
    </tr>
    <tr>
      <th>2023-01-05</th>
      <td>4</td>
      <td>$199,227.31</td>
      <td>$290.38</td>
      <td>0</td>
      <td>$32.73</td>
      <td>$257.65</td>
      <td>$198,969.66</td>
      <td>11.271%</td>
      <td>88.729%</td>
    </tr>
    <tr>
      <th>2023-01-06</th>
      <td>5</td>
      <td>$198,969.66</td>
      <td>$290.38</td>
      <td>0</td>
      <td>$32.68</td>
      <td>$257.69</td>
      <td>$198,711.97</td>
      <td>11.256%</td>
      <td>88.744%</td>
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
      <th>2024-12-28</th>
      <td>727</td>
      <td>$1,451.17</td>
      <td>$290.38</td>
      <td>0</td>
      <td>$0.24</td>
      <td>$290.14</td>
      <td>$1,161.03</td>
      <td>0.082%</td>
      <td>99.918%</td>
    </tr>
    <tr>
      <th>2024-12-29</th>
      <td>728</td>
      <td>$1,161.03</td>
      <td>$290.38</td>
      <td>0</td>
      <td>$0.19</td>
      <td>$290.19</td>
      <td>$870.84</td>
      <td>0.066%</td>
      <td>99.934%</td>
    </tr>
    <tr>
      <th>2024-12-30</th>
      <td>729</td>
      <td>$870.84</td>
      <td>$290.38</td>
      <td>0</td>
      <td>$0.14</td>
      <td>$290.23</td>
      <td>$580.61</td>
      <td>0.049%</td>
      <td>99.951%</td>
    </tr>
    <tr>
      <th>2024-12-31</th>
      <td>730</td>
      <td>$580.61</td>
      <td>$290.38</td>
      <td>0</td>
      <td>$0.10</td>
      <td>$290.28</td>
      <td>$290.33</td>
      <td>0.033%</td>
      <td>99.967%</td>
    </tr>
    <tr>
      <th>2025-01-01</th>
      <td>731</td>
      <td>$290.33</td>
      <td>$290.38</td>
      <td>0</td>
      <td>$0.05</td>
      <td>$290.33</td>
      <td>$-0.00</td>
      <td>0.016%</td>
      <td>99.984%</td>
    </tr>
  </tbody>
</table>
<p>731 rows × 9 columns</p>
</div>




    
![png](output_5_1.png)
    



```python
schedule = loan_amortization(200000, 6, 2, "2023-1-1", 'bdays')

schedule
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
      <th>Payment Number</th>
      <th>Beginning Balance</th>
      <th>Payment Amount</th>
      <th>Bonus</th>
      <th>Interest Paid</th>
      <th>Principal Paid</th>
      <th>Ending Balance</th>
      <th>% Paid In Interest</th>
      <th>% Paid To Principal</th>
    </tr>
    <tr>
      <th>Payment Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-01-02</th>
      <td>1</td>
      <td>$200,000.00</td>
      <td>$405.90</td>
      <td>0</td>
      <td>$45.98</td>
      <td>$359.93</td>
      <td>$199,640.07</td>
      <td>11.327%</td>
      <td>88.673%</td>
    </tr>
    <tr>
      <th>2023-01-03</th>
      <td>2</td>
      <td>$199,640.07</td>
      <td>$405.90</td>
      <td>0</td>
      <td>$45.89</td>
      <td>$360.01</td>
      <td>$199,280.07</td>
      <td>11.307%</td>
      <td>88.693%</td>
    </tr>
    <tr>
      <th>2023-01-04</th>
      <td>3</td>
      <td>$199,280.07</td>
      <td>$405.90</td>
      <td>0</td>
      <td>$45.81</td>
      <td>$360.09</td>
      <td>$198,919.98</td>
      <td>11.286%</td>
      <td>88.714%</td>
    </tr>
    <tr>
      <th>2023-01-05</th>
      <td>4</td>
      <td>$198,919.98</td>
      <td>$405.90</td>
      <td>0</td>
      <td>$45.73</td>
      <td>$360.17</td>
      <td>$198,559.80</td>
      <td>11.266%</td>
      <td>88.734%</td>
    </tr>
    <tr>
      <th>2023-01-06</th>
      <td>5</td>
      <td>$198,559.80</td>
      <td>$405.90</td>
      <td>0</td>
      <td>$45.65</td>
      <td>$360.26</td>
      <td>$198,199.55</td>
      <td>11.246%</td>
      <td>88.754%</td>
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
      <th>2024-12-26</th>
      <td>519</td>
      <td>$2,028.11</td>
      <td>$405.90</td>
      <td>0</td>
      <td>$0.47</td>
      <td>$405.44</td>
      <td>$1,622.68</td>
      <td>0.115%</td>
      <td>99.885%</td>
    </tr>
    <tr>
      <th>2024-12-27</th>
      <td>520</td>
      <td>$1,622.68</td>
      <td>$405.90</td>
      <td>0</td>
      <td>$0.37</td>
      <td>$405.53</td>
      <td>$1,217.15</td>
      <td>0.092%</td>
      <td>99.908%</td>
    </tr>
    <tr>
      <th>2024-12-30</th>
      <td>521</td>
      <td>$1,217.15</td>
      <td>$405.90</td>
      <td>0</td>
      <td>$0.28</td>
      <td>$405.62</td>
      <td>$811.52</td>
      <td>0.069%</td>
      <td>99.931%</td>
    </tr>
    <tr>
      <th>2024-12-31</th>
      <td>522</td>
      <td>$811.52</td>
      <td>$405.90</td>
      <td>0</td>
      <td>$0.19</td>
      <td>$405.72</td>
      <td>$405.81</td>
      <td>0.046%</td>
      <td>99.954%</td>
    </tr>
    <tr>
      <th>2025-01-01</th>
      <td>523</td>
      <td>$405.81</td>
      <td>$405.90</td>
      <td>0</td>
      <td>$0.09</td>
      <td>$405.81</td>
      <td>$0.00</td>
      <td>0.023%</td>
      <td>99.977%</td>
    </tr>
  </tbody>
</table>
<p>523 rows × 9 columns</p>
</div>




    
![png](output_6_1.png)
    



```python
schedule = loan_amortization(200000, 5, 10, "2023-1-1", 'W')

schedule
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
      <th>Payment Number</th>
      <th>Beginning Balance</th>
      <th>Payment Amount</th>
      <th>Bonus</th>
      <th>Interest Paid</th>
      <th>Principal Paid</th>
      <th>Ending Balance</th>
      <th>% Paid In Interest</th>
      <th>% Paid To Principal</th>
    </tr>
    <tr>
      <th>Payment Week</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2-2023</th>
      <td>1</td>
      <td>$200,000.00</td>
      <td>$488.93</td>
      <td>0</td>
      <td>$192.31</td>
      <td>$296.62</td>
      <td>$199,703.38</td>
      <td>39.332%</td>
      <td>60.668%</td>
    </tr>
    <tr>
      <th>3-2023</th>
      <td>2</td>
      <td>$199,703.38</td>
      <td>$488.93</td>
      <td>0</td>
      <td>$192.02</td>
      <td>$296.91</td>
      <td>$199,406.47</td>
      <td>39.274%</td>
      <td>60.726%</td>
    </tr>
    <tr>
      <th>4-2023</th>
      <td>3</td>
      <td>$199,406.47</td>
      <td>$488.93</td>
      <td>0</td>
      <td>$191.74</td>
      <td>$297.19</td>
      <td>$199,109.28</td>
      <td>39.216%</td>
      <td>60.784%</td>
    </tr>
    <tr>
      <th>5-2023</th>
      <td>4</td>
      <td>$199,109.28</td>
      <td>$488.93</td>
      <td>0</td>
      <td>$191.45</td>
      <td>$297.48</td>
      <td>$198,811.80</td>
      <td>39.157%</td>
      <td>60.843%</td>
    </tr>
    <tr>
      <th>6-2023</th>
      <td>5</td>
      <td>$198,811.80</td>
      <td>$488.93</td>
      <td>0</td>
      <td>$191.17</td>
      <td>$297.76</td>
      <td>$198,514.03</td>
      <td>39.099%</td>
      <td>60.901%</td>
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
      <th>48-2032</th>
      <td>516</td>
      <td>$2,437.61</td>
      <td>$488.93</td>
      <td>0</td>
      <td>$2.34</td>
      <td>$486.59</td>
      <td>$1,951.03</td>
      <td>0.479%</td>
      <td>99.521%</td>
    </tr>
    <tr>
      <th>49-2032</th>
      <td>517</td>
      <td>$1,951.03</td>
      <td>$488.93</td>
      <td>0</td>
      <td>$1.88</td>
      <td>$487.05</td>
      <td>$1,463.97</td>
      <td>0.384%</td>
      <td>99.616%</td>
    </tr>
    <tr>
      <th>50-2032</th>
      <td>518</td>
      <td>$1,463.97</td>
      <td>$488.93</td>
      <td>0</td>
      <td>$1.41</td>
      <td>$487.52</td>
      <td>$976.45</td>
      <td>0.288%</td>
      <td>99.712%</td>
    </tr>
    <tr>
      <th>51-2032</th>
      <td>519</td>
      <td>$976.45</td>
      <td>$488.93</td>
      <td>0</td>
      <td>$0.94</td>
      <td>$487.99</td>
      <td>$488.46</td>
      <td>0.192%</td>
      <td>99.808%</td>
    </tr>
    <tr>
      <th>52-2032</th>
      <td>520</td>
      <td>$488.46</td>
      <td>$488.93</td>
      <td>0</td>
      <td>$0.47</td>
      <td>$488.46</td>
      <td>$-0.00</td>
      <td>0.096%</td>
      <td>99.904%</td>
    </tr>
  </tbody>
</table>
<p>520 rows × 9 columns</p>
</div>




    
![png](output_7_1.png)
    



```python
schedule = loan_amortization(200000, 5, 10, "2023-1-1", 'BW')

schedule
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
      <th>Payment Number</th>
      <th>Beginning Balance</th>
      <th>Payment Amount</th>
      <th>Bonus</th>
      <th>Interest Paid</th>
      <th>Principal Paid</th>
      <th>Ending Balance</th>
      <th>% Paid In Interest</th>
      <th>% Paid To Principal</th>
    </tr>
    <tr>
      <th>Payment Week</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2-2023</th>
      <td>1</td>
      <td>$200,000.00</td>
      <td>$978.22</td>
      <td>0</td>
      <td>$384.62</td>
      <td>$593.61</td>
      <td>$199,406.39</td>
      <td>39.318%</td>
      <td>60.682%</td>
    </tr>
    <tr>
      <th>4-2023</th>
      <td>2</td>
      <td>$199,406.39</td>
      <td>$978.22</td>
      <td>0</td>
      <td>$383.47</td>
      <td>$594.75</td>
      <td>$198,811.65</td>
      <td>39.201%</td>
      <td>60.799%</td>
    </tr>
    <tr>
      <th>6-2023</th>
      <td>3</td>
      <td>$198,811.65</td>
      <td>$978.22</td>
      <td>0</td>
      <td>$382.33</td>
      <td>$595.89</td>
      <td>$198,215.75</td>
      <td>39.084%</td>
      <td>60.916%</td>
    </tr>
    <tr>
      <th>8-2023</th>
      <td>4</td>
      <td>$198,215.75</td>
      <td>$978.22</td>
      <td>0</td>
      <td>$381.18</td>
      <td>$597.04</td>
      <td>$197,618.72</td>
      <td>38.967%</td>
      <td>61.033%</td>
    </tr>
    <tr>
      <th>10-2023</th>
      <td>5</td>
      <td>$197,618.72</td>
      <td>$978.22</td>
      <td>0</td>
      <td>$380.04</td>
      <td>$598.19</td>
      <td>$197,020.53</td>
      <td>38.850%</td>
      <td>61.150%</td>
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
      <th>43-2032</th>
      <td>256</td>
      <td>$4,863.02</td>
      <td>$978.22</td>
      <td>0</td>
      <td>$9.35</td>
      <td>$968.87</td>
      <td>$3,894.15</td>
      <td>0.956%</td>
      <td>99.044%</td>
    </tr>
    <tr>
      <th>45-2032</th>
      <td>257</td>
      <td>$3,894.15</td>
      <td>$978.22</td>
      <td>0</td>
      <td>$7.49</td>
      <td>$970.73</td>
      <td>$2,923.41</td>
      <td>0.766%</td>
      <td>99.234%</td>
    </tr>
    <tr>
      <th>47-2032</th>
      <td>258</td>
      <td>$2,923.41</td>
      <td>$978.22</td>
      <td>0</td>
      <td>$5.62</td>
      <td>$972.60</td>
      <td>$1,950.81</td>
      <td>0.575%</td>
      <td>99.425%</td>
    </tr>
    <tr>
      <th>49-2032</th>
      <td>259</td>
      <td>$1,950.81</td>
      <td>$978.22</td>
      <td>0</td>
      <td>$3.75</td>
      <td>$974.47</td>
      <td>$976.34</td>
      <td>0.384%</td>
      <td>99.616%</td>
    </tr>
    <tr>
      <th>51-2032</th>
      <td>260</td>
      <td>$976.34</td>
      <td>$978.22</td>
      <td>0</td>
      <td>$1.88</td>
      <td>$976.34</td>
      <td>$-0.00</td>
      <td>0.192%</td>
      <td>99.808%</td>
    </tr>
  </tbody>
</table>
<p>260 rows × 9 columns</p>
</div>




    
![png](output_8_1.png)
    



```python
schedule = loan_amortization(200000, 4, 10, "2023-1-1", 'Q', bonus=300)

schedule
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
      <th>Payment Number</th>
      <th>Beginning Balance</th>
      <th>Payment Amount</th>
      <th>Bonus</th>
      <th>Interest Paid</th>
      <th>Principal Paid</th>
      <th>Ending Balance</th>
      <th>% Paid In Interest</th>
      <th>% Paid To Principal</th>
    </tr>
    <tr>
      <th>Payment Month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>03-2023</th>
      <td>1</td>
      <td>$200,000.00</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$2,000.00</td>
      <td>$4,091.12</td>
      <td>$195,608.88</td>
      <td>31.293%</td>
      <td>68.707%</td>
    </tr>
    <tr>
      <th>06-2023</th>
      <td>2</td>
      <td>$195,608.88</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,956.09</td>
      <td>$4,135.03</td>
      <td>$191,173.85</td>
      <td>30.606%</td>
      <td>69.394%</td>
    </tr>
    <tr>
      <th>09-2023</th>
      <td>3</td>
      <td>$191,173.85</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,911.74</td>
      <td>$4,179.38</td>
      <td>$186,694.47</td>
      <td>29.912%</td>
      <td>70.088%</td>
    </tr>
    <tr>
      <th>12-2023</th>
      <td>4</td>
      <td>$186,694.47</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,866.94</td>
      <td>$4,224.17</td>
      <td>$182,170.29</td>
      <td>29.212%</td>
      <td>70.788%</td>
    </tr>
    <tr>
      <th>03-2024</th>
      <td>5</td>
      <td>$182,170.29</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,821.70</td>
      <td>$4,269.42</td>
      <td>$177,600.88</td>
      <td>28.504%</td>
      <td>71.496%</td>
    </tr>
    <tr>
      <th>06-2024</th>
      <td>6</td>
      <td>$177,600.88</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,776.01</td>
      <td>$4,315.11</td>
      <td>$172,985.77</td>
      <td>27.789%</td>
      <td>72.211%</td>
    </tr>
    <tr>
      <th>09-2024</th>
      <td>7</td>
      <td>$172,985.77</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,729.86</td>
      <td>$4,361.26</td>
      <td>$168,324.50</td>
      <td>27.067%</td>
      <td>72.933%</td>
    </tr>
    <tr>
      <th>12-2024</th>
      <td>8</td>
      <td>$168,324.50</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,683.25</td>
      <td>$4,407.87</td>
      <td>$163,616.63</td>
      <td>26.337%</td>
      <td>73.663%</td>
    </tr>
    <tr>
      <th>03-2025</th>
      <td>9</td>
      <td>$163,616.63</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,636.17</td>
      <td>$4,454.95</td>
      <td>$158,861.68</td>
      <td>25.601%</td>
      <td>74.399%</td>
    </tr>
    <tr>
      <th>06-2025</th>
      <td>10</td>
      <td>$158,861.68</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,588.62</td>
      <td>$4,502.50</td>
      <td>$154,059.17</td>
      <td>24.857%</td>
      <td>75.143%</td>
    </tr>
    <tr>
      <th>09-2025</th>
      <td>11</td>
      <td>$154,059.17</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,540.59</td>
      <td>$4,550.53</td>
      <td>$149,208.65</td>
      <td>24.105%</td>
      <td>75.895%</td>
    </tr>
    <tr>
      <th>12-2025</th>
      <td>12</td>
      <td>$149,208.65</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,492.09</td>
      <td>$4,599.03</td>
      <td>$144,309.61</td>
      <td>23.346%</td>
      <td>76.654%</td>
    </tr>
    <tr>
      <th>03-2026</th>
      <td>13</td>
      <td>$144,309.61</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,443.10</td>
      <td>$4,648.02</td>
      <td>$139,361.59</td>
      <td>22.580%</td>
      <td>77.420%</td>
    </tr>
    <tr>
      <th>06-2026</th>
      <td>14</td>
      <td>$139,361.59</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,393.62</td>
      <td>$4,697.50</td>
      <td>$134,364.09</td>
      <td>21.806%</td>
      <td>78.194%</td>
    </tr>
    <tr>
      <th>09-2026</th>
      <td>15</td>
      <td>$134,364.09</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,343.64</td>
      <td>$4,747.48</td>
      <td>$129,316.61</td>
      <td>21.024%</td>
      <td>78.976%</td>
    </tr>
    <tr>
      <th>12-2026</th>
      <td>16</td>
      <td>$129,316.61</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,293.17</td>
      <td>$4,797.95</td>
      <td>$124,218.65</td>
      <td>20.234%</td>
      <td>79.766%</td>
    </tr>
    <tr>
      <th>03-2027</th>
      <td>17</td>
      <td>$124,218.65</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,242.19</td>
      <td>$4,848.93</td>
      <td>$119,069.72</td>
      <td>19.436%</td>
      <td>80.564%</td>
    </tr>
    <tr>
      <th>06-2027</th>
      <td>18</td>
      <td>$119,069.72</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,190.70</td>
      <td>$4,900.42</td>
      <td>$113,869.30</td>
      <td>18.630%</td>
      <td>81.370%</td>
    </tr>
    <tr>
      <th>09-2027</th>
      <td>19</td>
      <td>$113,869.30</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,138.69</td>
      <td>$4,952.43</td>
      <td>$108,616.87</td>
      <td>17.817%</td>
      <td>82.183%</td>
    </tr>
    <tr>
      <th>12-2027</th>
      <td>20</td>
      <td>$108,616.87</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,086.17</td>
      <td>$5,004.95</td>
      <td>$103,311.92</td>
      <td>16.995%</td>
      <td>83.005%</td>
    </tr>
    <tr>
      <th>03-2028</th>
      <td>21</td>
      <td>$103,311.92</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$1,033.12</td>
      <td>$5,058.00</td>
      <td>$97,953.92</td>
      <td>16.165%</td>
      <td>83.835%</td>
    </tr>
    <tr>
      <th>06-2028</th>
      <td>22</td>
      <td>$97,953.92</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$979.54</td>
      <td>$5,111.58</td>
      <td>$92,542.34</td>
      <td>15.327%</td>
      <td>84.673%</td>
    </tr>
    <tr>
      <th>09-2028</th>
      <td>23</td>
      <td>$92,542.34</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$925.42</td>
      <td>$5,165.70</td>
      <td>$87,076.64</td>
      <td>14.480%</td>
      <td>85.520%</td>
    </tr>
    <tr>
      <th>12-2028</th>
      <td>24</td>
      <td>$87,076.64</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$870.77</td>
      <td>$5,220.35</td>
      <td>$81,556.29</td>
      <td>13.625%</td>
      <td>86.375%</td>
    </tr>
    <tr>
      <th>03-2029</th>
      <td>25</td>
      <td>$81,556.29</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$815.56</td>
      <td>$5,275.56</td>
      <td>$75,980.73</td>
      <td>12.761%</td>
      <td>87.239%</td>
    </tr>
    <tr>
      <th>06-2029</th>
      <td>26</td>
      <td>$75,980.73</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$759.81</td>
      <td>$5,331.31</td>
      <td>$70,349.42</td>
      <td>11.888%</td>
      <td>88.112%</td>
    </tr>
    <tr>
      <th>09-2029</th>
      <td>27</td>
      <td>$70,349.42</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$703.49</td>
      <td>$5,387.63</td>
      <td>$64,661.80</td>
      <td>11.007%</td>
      <td>88.993%</td>
    </tr>
    <tr>
      <th>12-2029</th>
      <td>28</td>
      <td>$64,661.80</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$646.62</td>
      <td>$5,444.50</td>
      <td>$58,917.29</td>
      <td>10.117%</td>
      <td>89.883%</td>
    </tr>
    <tr>
      <th>03-2030</th>
      <td>29</td>
      <td>$58,917.29</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$589.17</td>
      <td>$5,501.95</td>
      <td>$53,115.35</td>
      <td>9.219%</td>
      <td>90.781%</td>
    </tr>
    <tr>
      <th>06-2030</th>
      <td>30</td>
      <td>$53,115.35</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$531.15</td>
      <td>$5,559.97</td>
      <td>$47,255.38</td>
      <td>8.311%</td>
      <td>91.689%</td>
    </tr>
    <tr>
      <th>09-2030</th>
      <td>31</td>
      <td>$47,255.38</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$472.55</td>
      <td>$5,618.57</td>
      <td>$41,336.82</td>
      <td>7.394%</td>
      <td>92.606%</td>
    </tr>
    <tr>
      <th>12-2030</th>
      <td>32</td>
      <td>$41,336.82</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$413.37</td>
      <td>$5,677.75</td>
      <td>$35,359.06</td>
      <td>6.468%</td>
      <td>93.532%</td>
    </tr>
    <tr>
      <th>03-2031</th>
      <td>33</td>
      <td>$35,359.06</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$353.59</td>
      <td>$5,737.53</td>
      <td>$29,321.53</td>
      <td>5.533%</td>
      <td>94.467%</td>
    </tr>
    <tr>
      <th>06-2031</th>
      <td>34</td>
      <td>$29,321.53</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$293.22</td>
      <td>$5,797.90</td>
      <td>$23,223.63</td>
      <td>4.588%</td>
      <td>95.412%</td>
    </tr>
    <tr>
      <th>09-2031</th>
      <td>35</td>
      <td>$23,223.63</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$232.24</td>
      <td>$5,858.88</td>
      <td>$17,064.75</td>
      <td>3.634%</td>
      <td>96.366%</td>
    </tr>
    <tr>
      <th>12-2031</th>
      <td>36</td>
      <td>$17,064.75</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$170.65</td>
      <td>$5,920.47</td>
      <td>$10,844.28</td>
      <td>2.670%</td>
      <td>97.330%</td>
    </tr>
    <tr>
      <th>03-2032</th>
      <td>37</td>
      <td>$10,844.28</td>
      <td>$6,391.12</td>
      <td>300</td>
      <td>$108.44</td>
      <td>$5,982.68</td>
      <td>$4,561.60</td>
      <td>1.697%</td>
      <td>98.303%</td>
    </tr>
    <tr>
      <th>06-2032</th>
      <td>38</td>
      <td>$4,561.60</td>
      <td>$4,607.21</td>
      <td>0</td>
      <td>$45.62</td>
      <td>$4,561.60</td>
      <td>$0.00</td>
      <td>0.714%</td>
      <td>99.286%</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](output_9_1.png)
    



```python
schedule = loan_amortization(200000, 4, 10, "2023-1-1", 'S')

schedule
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
      <th>Payment Number</th>
      <th>Beginning Balance</th>
      <th>Payment Amount</th>
      <th>Bonus</th>
      <th>Interest Paid</th>
      <th>Principal Paid</th>
      <th>Ending Balance</th>
      <th>% Paid In Interest</th>
      <th>% Paid To Principal</th>
    </tr>
    <tr>
      <th>Payment Month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>01-2023</th>
      <td>1</td>
      <td>$200,000.00</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$4,000.00</td>
      <td>$8,231.34</td>
      <td>$191,768.66</td>
      <td>32.703%</td>
      <td>67.297%</td>
    </tr>
    <tr>
      <th>07-2023</th>
      <td>2</td>
      <td>$191,768.66</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$3,835.37</td>
      <td>$8,395.97</td>
      <td>$183,372.69</td>
      <td>31.357%</td>
      <td>68.643%</td>
    </tr>
    <tr>
      <th>01-2024</th>
      <td>3</td>
      <td>$183,372.69</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$3,667.45</td>
      <td>$8,563.89</td>
      <td>$174,808.80</td>
      <td>29.984%</td>
      <td>70.016%</td>
    </tr>
    <tr>
      <th>07-2024</th>
      <td>4</td>
      <td>$174,808.80</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$3,496.18</td>
      <td>$8,735.17</td>
      <td>$166,073.63</td>
      <td>28.584%</td>
      <td>71.416%</td>
    </tr>
    <tr>
      <th>01-2025</th>
      <td>5</td>
      <td>$166,073.63</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$3,321.47</td>
      <td>$8,909.87</td>
      <td>$157,163.76</td>
      <td>27.155%</td>
      <td>72.845%</td>
    </tr>
    <tr>
      <th>07-2025</th>
      <td>6</td>
      <td>$157,163.76</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$3,143.28</td>
      <td>$9,088.07</td>
      <td>$148,075.69</td>
      <td>25.699%</td>
      <td>74.301%</td>
    </tr>
    <tr>
      <th>01-2026</th>
      <td>7</td>
      <td>$148,075.69</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$2,961.51</td>
      <td>$9,269.83</td>
      <td>$138,805.86</td>
      <td>24.212%</td>
      <td>75.788%</td>
    </tr>
    <tr>
      <th>07-2026</th>
      <td>8</td>
      <td>$138,805.86</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$2,776.12</td>
      <td>$9,455.23</td>
      <td>$129,350.63</td>
      <td>22.697%</td>
      <td>77.303%</td>
    </tr>
    <tr>
      <th>01-2027</th>
      <td>9</td>
      <td>$129,350.63</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$2,587.01</td>
      <td>$9,644.33</td>
      <td>$119,706.30</td>
      <td>21.151%</td>
      <td>78.849%</td>
    </tr>
    <tr>
      <th>07-2027</th>
      <td>10</td>
      <td>$119,706.30</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$2,394.13</td>
      <td>$9,837.22</td>
      <td>$109,869.08</td>
      <td>19.574%</td>
      <td>80.426%</td>
    </tr>
    <tr>
      <th>01-2028</th>
      <td>11</td>
      <td>$109,869.08</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$2,197.38</td>
      <td>$10,033.96</td>
      <td>$99,835.12</td>
      <td>17.965%</td>
      <td>82.035%</td>
    </tr>
    <tr>
      <th>07-2028</th>
      <td>12</td>
      <td>$99,835.12</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$1,996.70</td>
      <td>$10,234.64</td>
      <td>$89,600.48</td>
      <td>16.324%</td>
      <td>83.676%</td>
    </tr>
    <tr>
      <th>01-2029</th>
      <td>13</td>
      <td>$89,600.48</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$1,792.01</td>
      <td>$10,439.33</td>
      <td>$79,161.15</td>
      <td>14.651%</td>
      <td>85.349%</td>
    </tr>
    <tr>
      <th>07-2029</th>
      <td>14</td>
      <td>$79,161.15</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$1,583.22</td>
      <td>$10,648.12</td>
      <td>$68,513.03</td>
      <td>12.944%</td>
      <td>87.056%</td>
    </tr>
    <tr>
      <th>01-2030</th>
      <td>15</td>
      <td>$68,513.03</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$1,370.26</td>
      <td>$10,861.08</td>
      <td>$57,651.94</td>
      <td>11.203%</td>
      <td>88.797%</td>
    </tr>
    <tr>
      <th>07-2030</th>
      <td>16</td>
      <td>$57,651.94</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$1,153.04</td>
      <td>$11,078.30</td>
      <td>$46,573.64</td>
      <td>9.427%</td>
      <td>90.573%</td>
    </tr>
    <tr>
      <th>01-2031</th>
      <td>17</td>
      <td>$46,573.64</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$931.47</td>
      <td>$11,299.87</td>
      <td>$35,273.77</td>
      <td>7.615%</td>
      <td>92.385%</td>
    </tr>
    <tr>
      <th>07-2031</th>
      <td>18</td>
      <td>$35,273.77</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$705.48</td>
      <td>$11,525.87</td>
      <td>$23,747.90</td>
      <td>5.768%</td>
      <td>94.232%</td>
    </tr>
    <tr>
      <th>01-2032</th>
      <td>19</td>
      <td>$23,747.90</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$474.96</td>
      <td>$11,756.39</td>
      <td>$11,991.51</td>
      <td>3.883%</td>
      <td>96.117%</td>
    </tr>
    <tr>
      <th>07-2032</th>
      <td>20</td>
      <td>$11,991.51</td>
      <td>$12,231.34</td>
      <td>0</td>
      <td>$239.83</td>
      <td>$11,991.51</td>
      <td>$0.00</td>
      <td>1.961%</td>
      <td>98.039%</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](output_10_1.png)
    



```python
schedule = loan_amortization(200000, 4, 10, "2023-1-1", 'Y', bonus=300)

schedule
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
      <th>Payment Number</th>
      <th>Beginning Balance</th>
      <th>Payment Amount</th>
      <th>Bonus</th>
      <th>Interest Paid</th>
      <th>Principal Paid</th>
      <th>Ending Balance</th>
      <th>% Paid In Interest</th>
      <th>% Paid To Principal</th>
    </tr>
    <tr>
      <th>Payment Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023</th>
      <td>1</td>
      <td>$200,000.00</td>
      <td>$24,958.19</td>
      <td>300</td>
      <td>$8,000.00</td>
      <td>$16,658.19</td>
      <td>$183,041.81</td>
      <td>32.054%</td>
      <td>67.946%</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>2</td>
      <td>$183,041.81</td>
      <td>$24,958.19</td>
      <td>300</td>
      <td>$7,321.67</td>
      <td>$17,336.52</td>
      <td>$165,405.29</td>
      <td>29.336%</td>
      <td>70.664%</td>
    </tr>
    <tr>
      <th>2025</th>
      <td>3</td>
      <td>$165,405.29</td>
      <td>$24,958.19</td>
      <td>300</td>
      <td>$6,616.21</td>
      <td>$18,041.98</td>
      <td>$147,063.32</td>
      <td>26.509%</td>
      <td>73.491%</td>
    </tr>
    <tr>
      <th>2026</th>
      <td>4</td>
      <td>$147,063.32</td>
      <td>$24,958.19</td>
      <td>300</td>
      <td>$5,882.53</td>
      <td>$18,775.66</td>
      <td>$127,987.66</td>
      <td>23.570%</td>
      <td>76.430%</td>
    </tr>
    <tr>
      <th>2027</th>
      <td>5</td>
      <td>$127,987.66</td>
      <td>$24,958.19</td>
      <td>300</td>
      <td>$5,119.51</td>
      <td>$19,538.68</td>
      <td>$108,148.98</td>
      <td>20.512%</td>
      <td>79.488%</td>
    </tr>
    <tr>
      <th>2028</th>
      <td>6</td>
      <td>$108,148.98</td>
      <td>$24,958.19</td>
      <td>300</td>
      <td>$4,325.96</td>
      <td>$20,332.23</td>
      <td>$87,516.75</td>
      <td>17.333%</td>
      <td>82.667%</td>
    </tr>
    <tr>
      <th>2029</th>
      <td>7</td>
      <td>$87,516.75</td>
      <td>$24,958.19</td>
      <td>300</td>
      <td>$3,500.67</td>
      <td>$21,157.52</td>
      <td>$66,059.23</td>
      <td>14.026%</td>
      <td>85.974%</td>
    </tr>
    <tr>
      <th>2030</th>
      <td>8</td>
      <td>$66,059.23</td>
      <td>$24,958.19</td>
      <td>300</td>
      <td>$2,642.37</td>
      <td>$22,015.82</td>
      <td>$43,743.41</td>
      <td>10.587%</td>
      <td>89.413%</td>
    </tr>
    <tr>
      <th>2031</th>
      <td>9</td>
      <td>$43,743.41</td>
      <td>$24,958.19</td>
      <td>300</td>
      <td>$1,749.74</td>
      <td>$22,908.45</td>
      <td>$20,534.96</td>
      <td>7.011%</td>
      <td>92.989%</td>
    </tr>
    <tr>
      <th>2032</th>
      <td>10</td>
      <td>$20,534.96</td>
      <td>$21,356.36</td>
      <td>0</td>
      <td>$821.40</td>
      <td>$20,534.96</td>
      <td>$0.00</td>
      <td>3.291%</td>
      <td>96.709%</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](output_11_1.png)
    

