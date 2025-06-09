For plotting, use Seaborn with the "colorblind" palette for as much as you can.
When plotting error bars, plot +/- standard error rather than standard deviation or confidence interval.
Sometimes I'll ask you to plot the raw data as a scatter rather than doing error bars.
To fit functions, use scipy and always plot fitted functions with a +/- error.
If you're doing some sort of data processing more than once make a utility for this, and put it in a named .py file in utils/
For example, creating utils/calcein_release.py in order to re-use the code to process calcein release assays as mentioned above.
This code is going to be used in a published nanotech paper, it doesn't have to be Google-quality but it does need to be comprehensible.