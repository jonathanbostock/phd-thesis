This project uses uv for package management.
To run Python commands, use: uv run python <script.py>
To add a new dependency, use: uv add <package-name>
The project is automatically installed in editable mode - avoid doing any adding of os.path.filename + "../../.." to the python path.

If you're doing some sort of data processing more than once make a utility for this, and put it in a named .py file in utils/
For example, creating utils/calcein_release.py in order to re-use the code to process calcein release assays as mentioned above.
This code is going to be used in a published nanotech paper, it doesn't have to be Google-quality but it does need to be comprehensible.

For plotting, use Seaborn with the "colorblind" palette for as much as you can. Where possible, use both shape and colour to distinguish different classes.
Use filled markers with black borders, it looks cleaner.
When formatting, remove the top and right edges of the bounding box, and remove the grey background lines.
When working with numeric options, always use a gradient.
Some of this will end up being repeated.
Put a plotting.py file in utils/ to reduce code overhead and improve consistency if this happens.
When plotting error bars, plot +/- standard error rather than standard deviation or confidence interval.
Sometimes I'll ask you to plot the raw data as a scatter rather than doing error bars.
To fit functions, use scipy and always plot fitted functions with a +/- error.
Save all your plots as a .svg instead of  a .png.
When plotting fitted curves, always keep the curve the same colour as the data.
